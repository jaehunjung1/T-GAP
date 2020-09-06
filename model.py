import dgl
import dgl.function as fn
from dgl.nn.pytorch.softmax import edge_softmax
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class TGAP(nn.Module):
    def __init__(self, args):
        super(TGAP, self).__init__()
        self.args = args
        self.num_out_heads = args.num_out_heads
        self.num_in_heads = args.num_in_heads
        self.out_head_dim = args.node_dim // self.num_out_heads
        self.in_head_dim = args.node_dim // self.num_in_heads

        # Entity, Relation, Timestamp Embeddings
        self.node_embed = NodeEmbedding(args)
        self.edge_embed = nn.Embedding(len(args.relation_vocab), args.node_dim, padding_idx=0)
        self.tau_embed = nn.Embedding(len(args.time_vocab), args.node_dim, padding_idx=0)

        # Linear Layers
        self.W_c = nn.Linear(args.node_dim * 2, args.node_dim)
        self.W_n = nn.Linear(args.node_dim * 2, args.node_dim)
        self.W_h = nn.Linear(args.node_dim, args.node_dim)

        # Attention Heads for Attention Flow
        self.attn_i_outgoing = nn.Parameter(torch.Tensor(1, self.num_out_heads, self.out_head_dim))
        self.attn_j_outgoing = nn.Parameter(torch.Tensor(1, self.num_out_heads, self.out_head_dim))
        self.inattn_i_outgoing = nn.Parameter(torch.Tensor(1, self.num_out_heads, self.out_head_dim))
        self.inattn_j_outgoing = nn.Parameter(torch.Tensor(1, self.num_out_heads, self.out_head_dim))

        # Attention Heads for PGNN
        self.PGNN_attn_i_incoming = nn.Parameter(torch.Tensor(1, self.num_in_heads, self.in_head_dim))
        self.PGNN_attn_j_incoming = nn.Parameter(torch.Tensor(1, self.num_in_heads, self.in_head_dim))

        # Attention Heads for SGNN
        self.SGNN_attn_i_incoming = nn.Parameter(torch.Tensor(1, self.num_in_heads, self.in_head_dim))
        self.SGNN_attn_j_incoming = nn.Parameter(torch.Tensor(1, self.num_in_heads, self.in_head_dim))

        nn.init.xavier_uniform_(self.attn_i_outgoing)
        nn.init.xavier_uniform_(self.attn_j_outgoing)
        nn.init.xavier_uniform_(self.inattn_i_outgoing)
        nn.init.xavier_uniform_(self.inattn_j_outgoing)
        nn.init.xavier_uniform_(self.PGNN_attn_i_incoming)
        nn.init.xavier_uniform_(self.PGNN_attn_j_incoming)
        nn.init.xavier_uniform_(self.SGNN_attn_i_incoming)
        nn.init.xavier_uniform_(self.SGNN_attn_j_incoming)

        # Timestamp Sign Parameters
        self.inattn_past_lin = nn.Linear(args.node_dim, args.node_dim)
        self.inattn_present_lin = nn.Linear(args.node_dim, args.node_dim)
        self.inattn_future_lin = nn.Linear(args.node_dim, args.node_dim)
        self.attn_past_lin = nn.Linear(args.node_dim, args.node_dim)
        self.attn_present_lin = nn.Linear(args.node_dim, args.node_dim)
        self.attn_future_lin =  nn.Linear(args.node_dim, args.node_dim)

        # TODO remove
        # self.gat_att_dropout = nn.Dropout(self.args.dropout // 2)
        # self.gat_inf_dropout = nn.Dropout(self.args.dropout)

    def forward(self, batch):
        batch_size = batch["head"].size(0)

        # Prepare graph
        graph = batch['graph'].local_var()
        if self.training:
            remove_indices = torch.randperm(batch_size)
            graph.remove_edges(batch["example_idx"][remove_indices])
        graph.add_edges(list(range(graph.number_of_nodes())), list(range(graph.number_of_nodes())))
        graph.edata['relation_type'][graph.edata['relation_type'] == 0] = 1
        graph.edata['time'][graph.edata['time'] == 0] = 1
        graph.edata['time'] = (graph.edata['time'].repeat(batch_size, 1) - batch['time'].unsqueeze(1)).t()
        reverse_graph = graph.reverse(share_ndata=True, share_edata=True)

        # Node and edge embedding in graph
        graph.ndata['h_n'] = self.node_embed(graph.ndata['node_idx']).unsqueeze(1).repeat(1, batch_size, 1)
        graph.edata['h_e'] = self.edge_embed(graph.edata['relation_type']).unsqueeze(1)
        graph.edata['tau'] = self.tau_embed(torch.abs(graph.edata['time']) + 1)

        # PGNN Message Passing
        for i in range(1):
            graph.apply_edges(func=self.incoming_inatt_func)
            graph.edata['g_e_incoming'] = graph.edata['g_e_incoming'] \
                .view(-1, batch_size, self.num_in_heads, self.in_head_dim)
            attn_i_incoming = (graph.ndata['h_n']
                               .view(-1, batch_size, self.num_in_heads, self.in_head_dim)
                               * self.PGNN_attn_i_incoming)
            attn_j_incoming = (graph.edata['g_e_incoming']
                               .view(-1, batch_size, self.num_in_heads, self.in_head_dim) *
                               self.PGNN_attn_j_incoming)
            graph.ndata.update({'attn_self_incoming': attn_i_incoming})
            graph.edata.update({'attn_neighbor_incoming': attn_j_incoming})
            graph.apply_edges(fn.v_mul_e('attn_self_incoming', 'attn_neighbor_incoming', 'attn_neighbor_incoming'))
            attn_j_incoming = F.leaky_relu(graph.edata.pop('attn_neighbor_incoming'))
            graph.edata['a_GAT'] = edge_softmax(graph, attn_j_incoming)
            graph.update_all(self.incoming_msg_func, fn.sum('m', 'h_n'))
            graph.ndata['h_n'] = F.leaky_relu(graph.ndata['h_n'].view(-1, batch_size, self.args.node_dim))

        # Attention value at each step
        attn_history = []
        edge_attn_history = []

        head_indices = torch.stack((batch['head'], torch.arange(batch_size).to(self.args.device)), dim=0)

        graph.ndata['g_n'] = torch.zeros((graph.number_of_nodes(), batch_size, self.args.node_dim)) \
            .to(self.args.device)
        graph.ndata['g_n'][tuple(head_indices)] = graph.ndata['h_n'][tuple(head_indices)]
        graph.ndata['a'] = torch.zeros((graph.number_of_nodes(), batch_size, self.num_out_heads)) \
            .to(self.args.device)
        graph.ndata['a'][tuple(head_indices)] = 1

        # Prepare query vector for each example
        query = torch.cat([self.node_embed(batch['head']), self.edge_embed(batch['relation'])], dim=-1)
        query = self.W_c(query)

        # Subgraph indices for attentive GNN
        subgraph_node_list = list(batch['head'].unsqueeze(1))
        subgraph_edge_list = list([] for _ in range(len(subgraph_node_list)))

        for i in range(self.args.num_step):
            subgraph_batch_indices = torch.cat([torch.tensor([i] * len(subgraph_node_list[i]))
                                                for i in range(len(subgraph_node_list))], dim=-1).to(self.args.device)
            subgraph_indices = torch.stack([torch.cat(subgraph_node_list, dim=-1),
                                            subgraph_batch_indices], dim=0)

            graph.ndata['g_n'] = graph.ndata['g_n'].index_put(tuple(subgraph_indices), self.W_n(
                torch.cat((graph.ndata['g_n'][tuple(subgraph_indices)], query[subgraph_batch_indices, :]), dim=1)))

            # Attention Propagation
            graph.apply_edges(func=self.outgoing_edge_func)
            attn_i_outgoing = (graph.ndata['g_n']
                               .view(-1, batch_size, self.num_out_heads, self.out_head_dim) *
                               self.attn_i_outgoing)
            attn_j_outgoing = (graph.edata.pop('g_e_sub_outgoing')
                               .view(-1, batch_size, self.num_out_heads, self.out_head_dim) *
                               self.attn_j_outgoing)
            inattn_i_outgoing = (graph.ndata['g_n']
                                 .view(-1, batch_size, self.num_out_heads, self.out_head_dim) *
                                 self.inattn_i_outgoing)
            inattn_j_outgoing = (graph.edata.pop('g_e_outgoing')
                                 .view(-1, batch_size, self.num_out_heads, self.out_head_dim) *
                                 self.inattn_j_outgoing)

            graph.ndata.update({'attn_i_outgoing': attn_i_outgoing, 'inattn_i_outgoing': inattn_i_outgoing})
            graph.edata.update({'attn_j_outgoing': attn_j_outgoing, 'inattn_j_outgoing': inattn_j_outgoing})
            graph.apply_edges(fn.u_dot_e('attn_i_outgoing', 'attn_j_outgoing', 'tau_attn'))
            graph.apply_edges(fn.u_dot_e('inattn_i_outgoing', 'inattn_j_outgoing', 'tau_inattn'))
            tau = F.leaky_relu(graph.edata.pop('tau_attn')) + F.leaky_relu(graph.edata.pop('tau_inattn'))

            graph.edata['transition'] = edge_softmax(reverse_graph, tau)
            prev_a = graph.ndata['a'].mean(2)
            graph.apply_edges(fn.u_mul_e('a', 'transition', 'a_tilde'))
            graph.update_all(fn.copy_e('a_tilde', 'a_tilde'), fn.sum('a_tilde', 'a'))
            edge_attn_history.append(graph.edata['a_tilde'][:-graph.number_of_nodes()].mean(2))
            # 'a': (num_nodes, batch_size, num_att_heads)

            # Subgraph Sampling
            subgraph, subgraph_node_list, subgraph_edge_list = self.sample_subgraph(graph, prev_a,
                                                                                    graph.edata['a_tilde'].mean(2),
                                                                                    subgraph_node_list,
                                                                                    subgraph_edge_list)

            # SGNN Message Passing
            subgraph.apply_edges(func=self.incoming_att_func)
            subgraph.edata['g_e_incoming'] = subgraph.edata['g_e_incoming'] \
                .view(-1, self.num_out_heads, self.out_head_dim)
            attn_i_incoming = (subgraph.ndata['g_n'].view(-1, self.num_in_heads, self.in_head_dim) *
                               self.SGNN_attn_i_incoming)
            attn_j_incoming = (subgraph.edata['g_e_incoming'] *
                               self.SGNN_attn_j_incoming)
            subgraph.ndata.update({'attn_i_incoming': attn_i_incoming})
            subgraph.edata.update({'attn_j_incoming': attn_j_incoming})
            subgraph.apply_edges(fn.v_mul_e('attn_i_incoming', 'attn_j_incoming', 'attn_j_incoming'))

            attn_j_incoming = F.leaky_relu(subgraph.edata.pop('attn_j_incoming'))
            subgraph.edata['a_GAT'] = edge_softmax(subgraph, attn_j_incoming)
            subgraph.update_all(self.incoming_msg_func, fn.sum('m', 'g_n'))

            # TODO
            # subgraph.ndata['g_n'] = self.gat_inf_dropout(subgraph.ndata['g_n'].view(-1, self.args.node_dim))
            subgraph.ndata['g_n'] = subgraph.ndata['g_n'].view(-1, self.args.node_dim)
            subgraph.ndata['g_n'] += subgraph.ndata['a'].mean(1, keepdim=True) * self.W_h(subgraph.ndata['h_n'])
            subgraph.ndata['g_n'] = F.leaky_relu(subgraph.ndata['g_n'])

            for sub_idx, sub_g in enumerate(dgl.unbatch(subgraph)):
                graph.ndata['g_n'] = graph.ndata['g_n'].index_put((sub_g.ndata['node_idx'], torch.tensor(sub_idx)),
                                                                  sub_g.ndata['g_n'])

            attn_history.append(graph.ndata['a'].mean(2))

        return attn_history

    def sample_subgraph(self, graph, prev_a, a, prev_subgraph_nodes, prev_subgraph_edges):
        """Given node / edge attention distribution, sample subgraph at each step"""
        new_subgraph_nodes = []
        new_subgraph_edges = []
        new_subgraphs = []
        sample_from = [torch.topk(prev_a[:, i],
                                  dim=0, k=min(self.args.num_sample_from, len(prev_subgraph_nodes[i])))[1]
                       for i in range(len(prev_subgraph_nodes))]

        for i, sample_pool in enumerate(sample_from):
            edges = tuple({edge for query_node in sample_pool for edge
                           in np.random.permutation(graph.out_edges(query_node, form='eid'))
                           [:self.args.max_num_neighbor].tolist()})
            topk_edges = torch.tensor(edges)[torch.topk(
                a[edges, i], dim=0, k=min(len(edges), self.args.max_num_neighbor))[1]].to(a.device)

            if len(prev_subgraph_edges[i]) > 0:
                topk_edges = torch.cat([prev_subgraph_edges[i], topk_edges], dim=-1)
            new_subgraph = graph.edge_subgraph(topk_edges)
            new_subgraph.ndata['node_idx'] = new_subgraph.parent_nid.to(a.device)
            new_subgraph.edata['edge_idx'] = new_subgraph.parent_eid.to(a.device)
            new_subgraph.ndata['g_n'] = graph.ndata['g_n'][new_subgraph.parent_nid][:, i]
            new_subgraph.ndata['a'] = graph.ndata['a'][new_subgraph.parent_nid][:, i]
            new_subgraph.ndata['h_n'] = graph.ndata['h_n'][new_subgraph.parent_nid][:, i]
            new_subgraph.edata['h_e'] = graph.edata['h_e'][new_subgraph.parent_eid].squeeze(1)
            new_subgraph.edata['time'] = graph.edata['time'][new_subgraph.parent_eid][:, i]
            new_subgraph.edata['tau'] = graph.edata['tau'][new_subgraph.parent_eid][:, i]
            new_subgraphs.append(new_subgraph)
            new_subgraph_nodes.append(new_subgraph.ndata['node_idx'])
            new_subgraph_edges.append(new_subgraph.edata['edge_idx'])

        return dgl.batch(new_subgraphs), new_subgraph_nodes, new_subgraph_edges

    def outgoing_edge_func(self, edges):
        """Attention propagation message computation"""
        return {
            'g_e_sub_outgoing': edges.dst['g_n'] + edges.data['h_e'] + edges.data['tau'],
            'g_e_outgoing': edges.dst['h_n'] + edges.data['h_e'] + edges.data['tau']
        }

    def incoming_inatt_func(self, edges):
        """PGNN message computation"""
        translational = edges.src['h_n'] + edges.data['h_e'] + edges.data['tau']
        past = self.inattn_past_lin(translational).masked_fill((edges.data['time'] >= 0).unsqueeze(-1), 0)
        present = self.inattn_present_lin(translational).masked_fill((edges.data['time'] != 0).unsqueeze(-1), 0)
        future = self.inattn_future_lin(translational).masked_fill((edges.data['time'] <= 0).unsqueeze(-1), 0)

        return {
            'g_e_incoming': past + present + future
        }

    def incoming_att_func(self, edges):
        """SGNN message computation"""
        translational = edges.src['g_n'] + edges.data['h_e'] + edges.data['tau']
        past = self.attn_past_lin(translational).masked_fill((edges.data['time'] >= 0).unsqueeze(-1), 0)
        present = self.attn_present_lin(translational).masked_fill((edges.data['time'] != 0).unsqueeze(-1), 0)
        future = self.attn_future_lin(translational).masked_fill((edges.data['time'] <= 0).unsqueeze(-1), 0)

        return {
            'g_e_incoming': past + present + future
        }

    def incoming_msg_func(self, edges):
        return {'m': (edges.data['g_e_incoming'] * edges.data['a_GAT'])}


class NodeEmbedding(nn.Module):
    def __init__(self, args):
        super(NodeEmbedding, self).__init__()
        self.args = args
        self.node_dim = args.node_dim
        self.diachronic_dim = int(args.node_dim * args.gamma)

        self.synchronic_embed = nn.Embedding(len(args.entity_vocab), self.node_dim, padding_idx=0)
        self.diachronic_embed = nn.Embedding(len(args.entity_vocab), self.diachronic_dim, padding_idx=0)
        self.diachronic_w = nn.Embedding(len(args.entity_vocab), self.diachronic_dim, padding_idx=0)
        self.diachronic_b = nn.Embedding(len(args.entity_vocab), self.diachronic_dim, padding_idx=0)

    def forward(self, indices, time_indices=None, diachronic=False):
        """
        :param indices: (num_nodes,)
        :param time_indices: (num_nodes,)
        :param diachronic: bool, True to get diachronic embedding
        :return:
        """
        node_embed = self.synchronic_embed(indices)  # (num_nodes, node_dim)

        if diachronic:
            node_embed[:, self.diachronic_dim:] += \
                self.diachronic_embed(indices) * \
                torch.sin(self.diachronic_w(indices) * time_indices.unsqueeze(1) + self.diachronic_b(indices))

        return node_embed

    def time_transform(self, node_embed, indices, time_indices, masked=False):
        """
        Given embedding, transform to diachronic embedding
        :param node_embed: (num_nodes, node_dim) or (num_nodes, batch_size, node_dim)
        :param indices: (num_nodes,)
        :param time_indices:  (num_nodes,)
        :param masked: whether to mask nodes not included in subgraph
        :return:
        """
        diachronic_embed = \
            self.diachronic_embed(indices).unsqueeze(1) * \
            torch.sin(self.diachronic_w(indices).unsqueeze(1).repeat(1, 16, 1) * time_indices.unsqueeze(2)
                      + self.diachronic_b(indices).unsqueeze(1).repeat(1, 16, 1))

        if node_embed.dim() == 2:
            # for incoming attention
            mask = (torch.sum(node_embed, dim=-1) == 0)
            node_embed[:, self.diachronic_dim:] += diachronic_embed
        else:
            # for outgoing attention
            mask = (torch.sum(node_embed, dim=-1) == 0).unsqueeze(-1)
            node_embed[:, :, self.diachronic_dim:] += diachronic_embed

        if masked:
            node_embed.masked_fill(mask, 0)
        return node_embed
