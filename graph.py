import dgl
import torch
from util import Vocab


class TemporalGraph:
    """Temporal Graph Container Class"""
    def __init__(self, train_path, device):
        self.device = device

        with open(train_path, 'r') as f:
            lines = f.read().lower().splitlines()
            lines = map(lambda x: x.split("\t"), lines)

            head_list, relation_list, tail_list, time_list = tuple(zip(*lines))
            self.entity_vocab = Vocab()
            self.relation_vocab = Vocab()
            self.time_vocab = Vocab()
            self.entity_vocab.update(head_list + tail_list)
            self.relation_vocab.update(relation_list)
            self.time_vocab.update(time_list)
            self.entity_vocab.build()
            self.relation_vocab.build()
            self.time_vocab.build(sort_key="time")

            head_list = list(map(lambda x: self.entity_vocab(x), head_list))
            relation_list = list(map(lambda x: self.relation_vocab(x), relation_list))
            tail_list = list(map(lambda x: self.entity_vocab(x), tail_list))
            time_list = list(map(lambda x: self.time_vocab(x), time_list))

        self.graph = dgl.DGLGraph(multigraph=True)
        self.graph.add_nodes(len(self.entity_vocab))
        self.graph.add_edges(head_list, tail_list)
        self.graph.ndata['node_idx'] = torch.arange(self.graph.number_of_nodes())
        self.graph.edata['relation_type'] = torch.tensor(relation_list)
        self.graph.edata['time'] = torch.tensor(time_list)

        print("Graph prepared.")
