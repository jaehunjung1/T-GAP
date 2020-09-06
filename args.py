import os

from datetime import datetime
import argparse
import torch


def set_properties_to_args(args, entity_vocab, relation_vocab, time_vocab):
    args.entity_vocab = entity_vocab
    args.relation_vocab = relation_vocab
    args.time_vocab = time_vocab

    return args


def get_args():
    args = argparse.ArgumentParser(description="T-GAP arguments")
    args.add_argument('--results_dir', default="results/", type=str, help="Result directory.")
    args.add_argument('--run', default=datetime.now().strftime("%y%m%d_%H%M"), type=str, help="Experiment name.")
    args.add_argument('--device', default="cuda", type=str, help="Device to use (cuda or cpu).")
    args.add_argument('--seed', default=999, type=int, help="Random seed.")
    args.add_argument('--dataset', default="data/icews14_aug", type=str, help="Dataset directory.")
    args.add_argument('--test', action='store_true', help="Inference mode.")
    args.add_argument('--ckpt', default='', help="Checkpoint file name.")
    args.add_argument('--epoch', default=20, type=int, help="Number of training epochs.")
    args.add_argument('--batch_size', default=16, type=int, help="Batch size.")
    args.add_argument('--lr', default=5e-4, type=float, help="Learning rate.")
    args.add_argument('--grad_clip', default=3, type=int, help="Gradient clip size.")
    args.add_argument('--patience', default=3, type=int, help="ReduceLROnPlateau patience.")

    args.add_argument('--node_dim', default=100, type=int, help="Node embedding size.")
    args.add_argument('--gamma', default=0.5, type=float)
    args.add_argument('--num_in_heads', default=5, type=int, help="Number of heads for PGNN, SGNN")
    args.add_argument('--num_out_heads', default=5, type=int, help="Number of heads for Attention Flow")
    args.add_argument('--num_step', default=3, type=int, help="Number of propagation steps.")

    args.add_argument('--num_sample_from', default=10, type=int, help="Number of nodes to sample subgraph from.")
    args.add_argument('--max_num_neighbor', default=100, type=int, help="Max number of neighbors to sample.")

    args = args.parse_args()

    args.train_fname = os.path.join(args.dataset, "train.txt")
    args.valid_fname = os.path.join(args.dataset, "valid.txt")
    args.test_fname = os.path.join(args.dataset, "test.txt")

    args.tensorboard_dir = os.path.join(args.results_dir, 'log', args.run)
    args.ckpt_dir = os.path.join(args.results_dir, 'checkpoint', args.run)

    args.device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")

    return args
