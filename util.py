import os
from collections.abc import Iterable
from typing import Tuple

import torch


_MODEL_STATE_DICT = "model_state_dict"
_OPTIMIZER_STATE_DICT = "optimizer_state_dict"
_SCHEDULER_STATE_DICT = "scheduler_state_dict"
_EPOCH = "epoch"
_COUNT = "global_count"


def load_checkpoint(ckpt_path, model, optimizer=None, scheduler=None):
    """Loads checkpoint file"""
    checkpoint = torch.load(ckpt_path)
    model.load_state_dict(checkpoint[_MODEL_STATE_DICT])

    if optimizer:
        optimizer.load_state_dict(checkpoint[_OPTIMIZER_STATE_DICT])
    if scheduler:
        scheduler.load_state_dict(checkpoint[_SCHEDULER_STATE_DICT])
    start_epoch_id = checkpoint[_EPOCH] + 1
    global_count = checkpoint[_COUNT]

    return start_epoch_id, global_count


def save_checkpoint(ckpt_path, model, optimizer, scheduler, epoch_id, global_count, metric):
    """Save state to checkpoint file"""
    torch.save({
        _MODEL_STATE_DICT: model.state_dict(),
        _OPTIMIZER_STATE_DICT: optimizer.state_dict(),
        _SCHEDULER_STATE_DICT: scheduler.state_dict(),
        _EPOCH: epoch_id,
        _COUNT: global_count,
    }, os.path.join(ckpt_path, f"{metric:.5}.{epoch_id}.tar"))


def hits_at_k(predicted_prob, target_indices, k):
    """
    :param predicted_prob: (batch_size, num_nodes)
    :param target_indices: (batch_size, )
    :param k: number of nodes to decode
    :return: number of correct inferences
    """
    topk = torch.topk(predicted_prob, dim=1, k=k)[1]

    return torch.sum(topk == target_indices.unsqueeze(1)).item()


class Vocab(object):
    """Entity / Relation / Timestamp Vocabulary Class"""
    def __init__(self, max_vocab=2**31, min_freq=-1, sp=None):
        if sp is None:
            sp = ['_PAD', '_UNK']
        self.itos = []
        self.stoi = {}
        self.freq = {}
        self.max_vocab, self.min_freq, self.sp = max_vocab, min_freq, sp

    def __len__(self):
        return len(self.itos)

    def __str__(self):
        return 'Total ' + str(len(self.itos)) + str(self.itos[:10])

    def update(self, token):
        if isinstance(token, Iterable):
            for t in token:
                self.freq[t] = self.freq.get(t, 0) + 1
        else:
            self.freq[token] = self.freq.get(token, 0) + 1

    def build(self, sort_key="freq"):
        assert len(self.itos) == 0 and len(self.stoi) == 0, "Build should only be called for initialization."
        self.itos.extend(self.sp)

        freq = sorted(self.freq.items(), key=lambda x: x[1] if sort_key == "freq" else x[0],
                      reverse=(sort_key == "freq"))

        for k, v in freq:
            if len(self.itos) < self.max_vocab and k not in self.sp and v >= self.min_freq:
                self.itos.append(k)
        self.stoi.update(list(zip(self.itos, range(len(self.itos)))))

    def __call__(self, x):
        if isinstance(x, int):
            return self.itos[x]
        else:
            return self.stoi.get(x, self.stoi['_UNK'])


def pad(tensor_list, pad_idx) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pads list of tensors with maximal length, and return stacked tensor / lengths"""
    lens = torch.Tensor([x.size(0) for x in tensor_list]).long()
    max_len = max([x.size(0) for x in tensor_list])

    return torch.stack(
        [torch.cat([x, torch.full([max_len-len(x)] + list(x.shape[1:]), pad_idx).type_as(x)], 0) for x in tensor_list],
        dim=0), lens
