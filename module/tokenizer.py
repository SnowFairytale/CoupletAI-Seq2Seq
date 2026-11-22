"""Character-level tokenizer with explicit BOS/EOS/PAD/UNK tokens.

The original CoupletAI tokenizer has only PAD/UNK and treats ``<s>``/``</s>``
from the vocabulary file as ordinary characters.  A seq2seq model needs a
real sentence-start and sentence-end symbol, so this tokenizer reserves four
special ids in a fixed order and skips the two legacy markers.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Union


class Tokenizer:
    PAD_TOKEN = "[PAD]"
    UNK_TOKEN = "[UNK]"
    BOS_TOKEN = "<s>"
    EOS_TOKEN = "</s>"
    SPECIAL_TOKENS = (PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN)
    LEGACY_SPECIAL_TOKENS = {"<s>", "</s>"}

    def __init__(self, token_to_id: Optional[Dict[str, int]] = None):
        if token_to_id is None:
            token_to_id = {tok: i for i, tok in enumerate(self.SPECIAL_TOKENS)}
        self.token_to_id: Dict[str, int] = dict(token_to_id)
        self.id_to_token: Dict[int, str] = {i: t for t, i in self.token_to_id.items()}

    # ------------------------------------------------------------------ #
    # construction
    # ------------------------------------------------------------------ #
    @classmethod
    def build(cls, vocab_file: Union[str, Path]) -> "Tokenizer":
        """Build from a newline separated vocabulary file (one char per line).

        The two legacy markers ``<s>`` / ``</s>`` in the provided vocabulary are
        ignored, otherwise they would collide with the BOS/EOS ids.
        """
        vocab_file = Path(vocab_file)
        tokenizer = cls()
        with vocab_file.open("r", encoding="utf-8") as f:
            for line in f:
                token = line.rstrip("\r\n")
                if not token or token in tokenizer.token_to_id:
                    continue
                if token in cls.LEGACY_SPECIAL_TOKENS:
                    continue
                tokenizer.token_to_id[token] = len(tokenizer.token_to_id)
        tokenizer.id_to_token = {i: t for t, i in tokenizer.token_to_id.items()}
        return tokenizer

    @classmethod
    def from_vocab_list(cls, tokens: Iterable[str]) -> "Tokenizer":
        tokenizer = cls()
        for token in tokens:
            token = token.rstrip("\r\n")
            if not token or token in tokenizer.token_to_id:
                continue
            if token in cls.LEGACY_SPECIAL_TOKENS:
                continue
            tokenizer.token_to_id[token] = len(tokenizer.token_to_id)
        tokenizer.id_to_token = {i: t for t, i in tokenizer.token_to_id.items()}
        return tokenizer

    # ------------------------------------------------------------------ #
    # serialisation
    # ------------------------------------------------------------------ #
    def save(self, filename: Union[str, Path]) -> None:
        filename = Path(filename)
        filename.parent.mkdir(parents=True, exist_ok=True)
        with filename.open("w", encoding="utf-8") as f:
            json.dump({"token_to_id": self.token_to_id}, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, filename: Union[str, Path]) -> "Tokenizer":
        with Path(filename).open("r", encoding="utf-8") as f:
            obj = json.load(f)
        return cls(obj["token_to_id"])

    # ------------------------------------------------------------------ #
    # properties
    # ------------------------------------------------------------------ #
    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)

    @property
    def pad_id(self) -> int:
        return self.token_to_id[self.PAD_TOKEN]

    @property
    def unk_id(self) -> int:
        return self.token_to_id[self.UNK_TOKEN]

    @property
    def bos_id(self) -> int:
        return self.token_to_id[self.BOS_TOKEN]

    @property
    def eos_id(self) -> int:
        return self.token_to_id[self.EOS_TOKEN]

    # ------------------------------------------------------------------ #
    # conversion
    # ------------------------------------------------------------------ #
    def convert_token_to_id(self, token: str) -> int:
        return self.token_to_id.get(token, self.unk_id)

    def convert_tokens_to_ids(self, tokens: Sequence[str]) -> List[int]:
        return [self.convert_token_to_id(t) for t in tokens]

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        """Encode a string as a list of character ids (whitespace is ignored)."""
        tokens = [ch for ch in text if not ch.isspace()]
        ids = self.convert_tokens_to_ids(tokens)
        if add_bos:
            ids = [self.bos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]
        return ids

    def decode(self, ids: Sequence[int], stop_at_eos: bool = True,
               skip_special: bool = True) -> str:
        pieces: List[str] = []
        for idx in ids:
            idx = int(idx)
            if stop_at_eos and idx == self.eos_id:
                break
            if skip_special and idx in (self.pad_id, self.bos_id):
                continue
            if idx == self.unk_id:
                pieces.append("□")
            else:
                pieces.append(self.id_to_token.get(idx, ""))
        return "".join(pieces)

    def __len__(self) -> int:
        return self.vocab_size

    def __repr__(self) -> str:
        return f"Tokenizer(vocab_size={self.vocab_size})"
