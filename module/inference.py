"""Decoding helpers shared by training-time validation, evaluation and demo."""
from __future__ import annotations

from typing import List, Sequence, Tuple

import torch
from torch.nn.utils.rnn import pad_sequence

from .model import BiLSTMTagger, Seq2Seq


@torch.no_grad()
def decode_batch(model: torch.nn.Module, tokenizer, src: torch.Tensor,
                 src_lens: torch.Tensor, model_name: str,
                 beam_size: int = 1) -> Tuple[List[List[int]], List[str]]:
    """Decode one padded batch and return token ids + decoded strings."""
    if isinstance(model, BiLSTMTagger) or model_name == "bilstm_tagger":
        pred_ids = model.generate(src, src_lens)
    elif beam_size and beam_size > 1:
        pred_ids = [
            model.generate_beam(src[i:i + 1], int(src_lens[i]),
                                max_len=int(src_lens[i]) + 2,
                                beam_size=beam_size)
            for i in range(src.size(0))
        ]
    else:
        generated = model.generate_greedy(
            src, src_lens, max_len=int(src_lens.max().item()) + 2)
        pred_ids = [row.tolist() for row in generated]
    pred_texts = [tokenizer.decode(ids) for ids in pred_ids]
    return pred_ids, pred_texts


@torch.no_grad()
def predict_texts(model: torch.nn.Module, tokenizer, texts: Sequence[str],
                  device: torch.device, model_name: str,
                  beam_size: int = 1) -> List[str]:
    """Generate lower couplets for a list of upper-couplet strings."""
    encoded = [tokenizer.encode(text) for text in texts]
    lengths = [len(ids) for ids in encoded]
    max_len = max(lengths) if lengths else 1
    src = torch.full((len(encoded), max_len), tokenizer.pad_id,
                     dtype=torch.long, device=device)
    for i, ids in enumerate(encoded):
        if ids:
            src[i, :len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)
    src_lens = torch.tensor(lengths, dtype=torch.long, device=device)
    _, pred_texts = decode_batch(model, tokenizer, src, src_lens,
                                 model_name=model_name, beam_size=beam_size)
    return pred_texts
