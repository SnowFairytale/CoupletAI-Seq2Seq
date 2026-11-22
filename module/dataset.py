"""Dataset / collate utilities for the couplet seq2seq task.

Pre-processed arrays have the following layout::

    inputs   int32 (N, max_len)     upper couplet ids, padded with PAD
    in_lens  int16 (N,)             number of real characters
    targets  int32 (N, max_len + 2) [BOS] + lower couplet + [EOS], padded
    out_lens int16 (N,)             in_len + 2

Batches are padded dynamically (to the longest sequence in the batch) rather
than to the global maximum, which saves a lot of compute because the average
couplet is only ~9.4 characters long.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

PAD_ID = 0


class CoupletDataset(Dataset):
    def __init__(self, npz_path: str | Path, max_samples: int | None = None,
                 seed: int = 42):
        data = np.load(Path(npz_path))
        inputs = data["inputs"]
        in_lens = data["in_lens"]
        targets = data["targets"]
        out_lens = data["out_lens"]

        if max_samples is not None and max_samples < len(inputs):
            rng = np.random.default_rng(seed)
            idx = rng.choice(len(inputs), size=max_samples, replace=False)
            idx.sort()
            inputs = inputs[idx]
            in_lens = in_lens[idx]
            targets = targets[idx]
            out_lens = out_lens[idx]

        self.inputs = inputs
        self.in_lens = in_lens
        self.targets = targets
        self.out_lens = out_lens

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, index: int):
        src_len = int(self.in_lens[index])
        tgt_len = int(self.out_lens[index])
        src = torch.from_numpy(self.inputs[index, :src_len].astype(np.int64))
        tgt = torch.from_numpy(self.targets[index, :tgt_len].astype(np.int64))
        return src, tgt


def collate_batch(batch: Sequence[Tuple[torch.Tensor, torch.Tensor]],
                  pad_id: int = PAD_ID):
    """Pad a list of (src, tgt) variable-length tensors into a batch."""
    srcs, tgts = zip(*batch)
    src_lens = torch.tensor([len(s) for s in srcs], dtype=torch.long)
    tgt_lens = torch.tensor([len(t) for t in tgts], dtype=torch.long)
    src = pad_sequence(list(srcs), batch_first=True, padding_value=pad_id)
    tgt = pad_sequence(list(tgts), batch_first=True, padding_value=pad_id)
    return src, src_lens, tgt, tgt_lens


def make_collate(pad_id: int = PAD_ID):
    def _collate(batch):
        return collate_batch(batch, pad_id=pad_id)
    return _collate
