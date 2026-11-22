"""Seq2seq (encoder-decoder) models for couplet generation.

Unlike the provided sample, which tags every upper-couplet character with the
lower-couplet character at the *same position*, the models here read the whole
upper couplet with a bidirectional GRU encoder and then generate the lower
couplet one character at a time with an autoregressive GRU decoder.  The main
model uses additive (Bahdanau) attention over the encoder states.

``BiLSTMTagger`` is a re-implementation of the sample's sequence-labelling
architecture, kept only as a baseline for the report.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

MODEL_NAMES = ("gru_attn", "gru_noattn", "bilstm_tagger")


# --------------------------------------------------------------------------- #
# Encoder
# --------------------------------------------------------------------------- #
class Encoder(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int,
                 num_layers: int, dec_layers: int, dropout: float,
                 pad_id: int = 0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.dec_layers = dec_layers
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.dropout = nn.Dropout(dropout)
        self.rnn = nn.GRU(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        # 2H (forward + backward of the last layer) -> dec_layers * H
        self.bridge = nn.Linear(hidden_dim * 2, dec_layers * hidden_dim)

    def forward(self, src: torch.Tensor, src_lens: torch.Tensor):
        embedded = self.dropout(self.embedding(src))
        packed = pack_padded_sequence(
            embedded, src_lens.cpu(), batch_first=True, enforce_sorted=False)
        outputs, hidden = self.rnn(packed)
        outputs, _ = pad_packed_sequence(
            outputs, batch_first=True, total_length=src.size(1))
        # hidden: (num_layers * 2, B, H); last layer forward / backward
        h = torch.cat([hidden[-2], hidden[-1]], dim=-1)
        h = torch.tanh(self.bridge(h))
        initial_hidden = h.view(h.size(0), self.dec_layers,
                                self.hidden_dim).transpose(0, 1).contiguous()
        return outputs, initial_hidden


# --------------------------------------------------------------------------- #
# Additive attention
# --------------------------------------------------------------------------- #
class AdditiveAttention(nn.Module):
    """Bahdanau-style additive attention (Luong "concat" is equivalent)."""

    def __init__(self, hidden_dim: int, enc_dim: int):
        super().__init__()
        self.W_enc = nn.Linear(enc_dim, hidden_dim, bias=False)
        self.W_dec = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def project(self, enc_out: torch.Tensor) -> torch.Tensor:
        return self.W_enc(enc_out)

    def forward(self, enc_out: torch.Tensor, enc_proj: torch.Tensor,
                dec_h: torch.Tensor, mask: torch.Tensor):
        # dec_h: (B, H); enc_out / enc_proj: (B, T, ...)
        score = self.v(torch.tanh(enc_proj + self.W_dec(dec_h).unsqueeze(1)))
        score = score.squeeze(-1)                       # (B, T)
        score = score.masked_fill(~mask, -1e4)
        weights = F.softmax(score, dim=-1)              # (B, T)
        context = torch.bmm(weights.unsqueeze(1), enc_out).squeeze(1)
        return context, weights


# --------------------------------------------------------------------------- #
# Decoder
# --------------------------------------------------------------------------- #
class Decoder(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int,
                 enc_dim: int, num_layers: int, dropout: float,
                 attention: bool = True, pad_id: int = 0):
        super().__init__()
        self.attention = attention
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.dropout = nn.Dropout(dropout)
        self.attn = AdditiveAttention(hidden_dim, enc_dim) if attention else None
        rnn_input = embed_dim + (enc_dim if attention else 0)
        self.rnn = nn.GRU(
            input_size=rnn_input,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.out = nn.Linear(hidden_dim + (enc_dim if attention else 0), vocab_size)

    def project(self, enc_out: torch.Tensor) -> Optional[torch.Tensor]:
        return self.attn.project(enc_out) if self.attn is not None else None

    def step(self, input_tok: torch.Tensor, hidden: torch.Tensor,
             enc_out: torch.Tensor, enc_proj: Optional[torch.Tensor],
             mask: torch.Tensor):
        """One decoding step.

        Returns (logits, new_hidden, attention_weights).
        """
        embedded = self.dropout(self.embedding(input_tok))       # (B, E)
        if self.attn is not None:
            context, weights = self.attn(enc_out, enc_proj, hidden[-1], mask)
            rnn_input = torch.cat([embedded, context], dim=-1).unsqueeze(1)
        else:
            context, weights = None, None
            rnn_input = embedded.unsqueeze(1)
        output, hidden = self.rnn(rnn_input, hidden)
        output = output.squeeze(1)
        if context is not None:
            output = torch.cat([output, context], dim=-1)
        logits = self.out(output)
        return logits, hidden, weights


# --------------------------------------------------------------------------- #
# Full seq2seq model
# --------------------------------------------------------------------------- #
class Seq2Seq(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 256, hidden_dim: int = 256,
                 enc_layers: int = 2, dec_layers: int = 2, dropout: float = 0.2,
                 attention: bool = True, pad_id: int = 0, bos_id: int = 2,
                 eos_id: int = 3):
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_id = pad_id
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.attention = attention
        self.encoder = Encoder(vocab_size, embed_dim, hidden_dim, enc_layers,
                               dec_layers, dropout, pad_id=pad_id)
        self.decoder = Decoder(vocab_size, embed_dim, hidden_dim, hidden_dim * 2,
                               dec_layers, dropout, attention=attention, pad_id=pad_id)

    # ------------------------------------------------------------------ #
    def encode(self, src: torch.Tensor, src_lens: torch.Tensor):
        enc_out, hidden = self.encoder(src, src_lens)
        mask = src.ne(self.pad_id)
        if self.attention:
            enc_proj = self.decoder.project(enc_out)
        else:
            enc_proj = None
        return enc_out, enc_proj, hidden, mask

    def forward(self, src: torch.Tensor, src_lens: torch.Tensor,
                tgt: torch.Tensor, teacher_forcing_ratio: float = 1.0) -> torch.Tensor:
        """Teacher-forced cross-entropy loss (mean over real target tokens)."""
        enc_out, enc_proj, hidden, mask = self.encode(src, src_lens)
        batch_size, tgt_len = tgt.size()
        input_tok = tgt[:, 0]
        loss_sum = torch.zeros((), device=tgt.device)
        token_count = torch.zeros((), device=tgt.device)

        for t in range(1, tgt_len):
            logits, hidden, _ = self.decoder.step(
                input_tok, hidden, enc_out, enc_proj, mask)
            target = tgt[:, t]
            loss_sum = loss_sum + F.cross_entropy(
                logits, target, ignore_index=self.pad_id, reduction="sum")
            token_count = token_count + target.ne(self.pad_id).sum().to(loss_sum.dtype)

            if teacher_forcing_ratio >= 1.0:
                input_tok = target
            else:
                predicted = logits.argmax(dim=-1)
                keep = torch.rand(batch_size, device=tgt.device) < teacher_forcing_ratio
                input_tok = torch.where(keep, target, predicted)

        return loss_sum / token_count.clamp(min=1.0)

    # ------------------------------------------------------------------ #
    # inference
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def generate_greedy(self, src: torch.Tensor, src_lens: torch.Tensor,
                        max_len: Optional[int] = None,
                        mask_eos_until: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Batched greedy decoding.  Returns (B, steps) ids including EOS.

        ``mask_eos_until`` is an optional per-sample tensor: EOS is forbidden
        while the current step is smaller than that value.  Passing
        ``mask_eos_until=src_lens`` therefore uses the equal-length prior of
        couplets and forces at least ``src_len`` characters.
        """
        enc_out, enc_proj, hidden, mask = self.encode(src, src_lens)
        batch_size = src.size(0)
        if max_len is None:
            max_len = int(src_lens.max().item()) + 2
        input_tok = torch.full((batch_size,), self.bos_id, dtype=torch.long,
                               device=src.device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=src.device)
        outputs: List[torch.Tensor] = []
        eos = torch.full_like(input_tok, self.eos_id)

        for step in range(max_len):
            logits, hidden, _ = self.decoder.step(
                input_tok, hidden, enc_out, enc_proj, mask)
            if mask_eos_until is not None:
                forbid = step < mask_eos_until
                logits[:, self.eos_id] = logits[:, self.eos_id].masked_fill(forbid, -1e4)
            predicted = logits.argmax(dim=-1)
            predicted = torch.where(finished, eos, predicted)
            outputs.append(predicted)
            finished = finished | predicted.eq(self.eos_id)
            input_tok = predicted
            if bool(finished.all()):
                break

        if not outputs:
            return torch.empty((batch_size, 0), dtype=torch.long, device=src.device)
        return torch.stack(outputs, dim=1)

    @torch.no_grad()
    def generate_beam(self, src: torch.Tensor, src_len: int,
                      max_len: Optional[int] = None, beam_size: int = 5,
                      length_penalty: float = 0.6,
                      min_len: int = 0) -> List[int]:
        """Beam search for a single upper couplet.  Returns token ids (no EOS)."""
        device = src.device
        if max_len is None:
            max_len = int(src_len) + 2
        src_lens = torch.tensor([src_len], dtype=torch.long, device=device)
        enc_out, enc_proj, hidden, mask = self.encode(src, src_lens)
        enc_out = enc_out.expand(beam_size, -1, -1).contiguous()
        mask = mask.expand(beam_size, -1).contiguous()
        hidden = hidden.expand(-1, beam_size, -1).contiguous()
        if self.attention:
            enc_proj = enc_proj.expand(beam_size, -1, -1).contiguous()

        input_tok = torch.full((beam_size,), self.bos_id, dtype=torch.long, device=device)
        scores = torch.full((beam_size,), -1e9, dtype=torch.float, device=device)
        scores[0] = 0.0
        sequences: List[List[int]] = [[] for _ in range(beam_size)]
        finished = torch.zeros(beam_size, dtype=torch.bool, device=device)

        for step in range(max_len):
            logits, hidden, _ = self.decoder.step(
                input_tok, hidden, enc_out, enc_proj, mask)
            if step < min_len:
                logits[:, self.eos_id] = -1e4
            log_probs = F.log_softmax(logits.float(), dim=-1)
            if bool(finished.any()):
                log_probs[finished] = -1e9
                log_probs[finished, self.eos_id] = 0.0
            candidates = scores.unsqueeze(1) + log_probs          # (beam, V)
            top_scores, top_ids = candidates.view(-1).topk(beam_size)
            beam_ids = torch.div(top_ids, self.vocab_size, rounding_mode="floor")
            token_ids = top_ids % self.vocab_size
            sequences = [sequences[b] + [t] for b, t in
                         zip(beam_ids.tolist(), token_ids.tolist())]
            hidden = hidden.index_select(1, beam_ids)
            scores = top_scores
            input_tok = token_ids.to(device)
            finished = torch.tensor(
                [bool(seq and seq[-1] == self.eos_id) for seq in sequences],
                dtype=torch.bool, device=device)
            if bool(finished.all()):
                break

        best_score = -float("inf")
        best_tokens: List[int] = []
        for score, seq in zip(scores.tolist(), sequences):
            if self.eos_id in seq:
                content = seq[:seq.index(self.eos_id)]
            else:
                content = seq
            length_penalty_value = ((5.0 + len(content)) / 6.0) ** length_penalty
            normalized = score / length_penalty_value
            if normalized > best_score:
                best_score = normalized
                best_tokens = content
        return best_tokens


# --------------------------------------------------------------------------- #
# Sample-style sequence-labelling baseline
# --------------------------------------------------------------------------- #
class BiLSTMTagger(nn.Module):
    """BiLSTM sequence tagger, mirroring the provided sample architecture."""

    def __init__(self, vocab_size: int, embed_dim: int = 128, hidden_dim: int = 256,
                 num_layers: int = 1, dropout: float = 0.2, pad_id: int = 0):
        super().__init__()
        self.pad_id = pad_id
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.dropout = nn.Dropout(dropout)
        self.rnn = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim // 2,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.linear = nn.Linear(hidden_dim, embed_dim)

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        embedded = self.dropout(self.embedding(src))
        outputs, _ = self.rnn(embedded)
        projected = torch.tanh(self.linear(outputs))
        logits = torch.matmul(projected, self.embedding.weight.t())
        return logits

    @torch.no_grad()
    def generate(self, src: torch.Tensor, src_lens: torch.Tensor) -> List[List[int]]:
        logits = self.forward(src)
        preds = logits.argmax(dim=-1).tolist()
        return [row[:int(length)] for row, length in zip(preds, src_lens.tolist())]


# --------------------------------------------------------------------------- #
# factory
# --------------------------------------------------------------------------- #
def build_model(model_name: str, vocab_size: int, pad_id: int = 0,
                bos_id: int = 2, eos_id: int = 3, **kwargs) -> nn.Module:
    name = model_name.lower()
    if name == "gru_attn":
        return Seq2Seq(vocab_size, attention=True, pad_id=pad_id,
                       bos_id=bos_id, eos_id=eos_id, **kwargs)
    if name == "gru_noattn":
        return Seq2Seq(vocab_size, attention=False, pad_id=pad_id,
                       bos_id=bos_id, eos_id=eos_id, **kwargs)
    if name == "bilstm_tagger":
        return BiLSTMTagger(vocab_size, pad_id=pad_id, **kwargs)
    raise KeyError(f"Unknown model '{model_name}', choose from {MODEL_NAMES}")
