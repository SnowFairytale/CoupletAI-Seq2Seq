"""Convert the raw couplet dataset into padded numpy arrays.

Usage::

    conda activate dip
    cd MySolution
    python preprocess.py --input ../CoupletAI/couplet --output data

The raw files contain one couplet per line with characters separated by spaces.
All pairs in this dataset have equal upper/lower length (verified: 100%), and
the maximum length is 32.  Besides the official test split we carve a
validation split out of the training file (default 20,000 pairs), so that the
test set is only touched once for the final report.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Dict, List, Tuple

import numpy as np
from tqdm import tqdm

from module import Tokenizer


def iter_pairs(in_file: Path, out_file: Path):
    """Yield (upper_chars, lower_chars) lists, skipping malformed lines."""
    with in_file.open("r", encoding="utf-8") as fi, \
            out_file.open("r", encoding="utf-8") as fo:
        for upper, lower in zip(fi, fo):
            yield upper.split(), lower.split()


def count_valid(in_file: Path, out_file: Path, max_len: int) -> Tuple[int, Dict[str, int]]:
    valid = 0
    stats = {"different_length": 0, "too_long": 0, "empty": 0}
    lengths: List[int] = []
    for upper, lower in iter_pairs(in_file, out_file):
        if not upper or not lower:
            stats["empty"] += 1
            continue
        if len(upper) != len(lower):
            stats["different_length"] += 1
            continue
        if len(upper) > max_len:
            stats["too_long"] += 1
            continue
        valid += 1
        lengths.append(len(upper))
    if lengths:
        stats["min_len"] = int(min(lengths))
        stats["max_len"] = int(max(lengths))
        stats["mean_len"] = float(sum(lengths) / len(lengths))
    return valid, stats


def build_arrays(in_file: Path, out_file: Path, tokenizer: Tokenizer,
                 max_len: int, n_valid: int):
    inputs = np.zeros((n_valid, max_len), dtype=np.int32)
    targets = np.zeros((n_valid, max_len + 2), dtype=np.int32)
    in_lens = np.zeros(n_valid, dtype=np.int16)
    out_lens = np.zeros(n_valid, dtype=np.int16)

    get_id = tokenizer.token_to_id.get
    unk_id = tokenizer.unk_id
    bos_id = tokenizer.bos_id
    eos_id = tokenizer.eos_id
    total_chars = 0
    oov_chars = 0

    row = 0
    for upper, lower in tqdm(iter_pairs(in_file, out_file), total=n_valid,
                             desc=f"encode {in_file.parent.name}"):
        if not upper or len(upper) != len(lower) or len(upper) > max_len:
            continue
        length = len(upper)
        targets[row, 0] = bos_id
        for j, ch in enumerate(upper):
            token_id = get_id(ch, unk_id)
            inputs[row, j] = token_id
            total_chars += 1
            if token_id == unk_id:
                oov_chars += 1
        for j, ch in enumerate(lower):
            token_id = get_id(ch, unk_id)
            targets[row, j + 1] = token_id
            total_chars += 1
            if token_id == unk_id:
                oov_chars += 1
        targets[row, length + 1] = eos_id
        in_lens[row] = length
        out_lens[row] = length + 2
        row += 1

    assert row == n_valid, (row, n_valid)
    return inputs, targets, in_lens, out_lens, total_chars, oov_chars


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(Path(__file__).resolve().parent.parent
                                               / "CoupletAI" / "couplet"))
    parser.add_argument("--output", default=str(Path(__file__).resolve().parent / "data"))
    parser.add_argument("--max_len", default=32, type=int)
    parser.add_argument("--val_size", default=20000, type=int)
    parser.add_argument("--seed", default=42, type=int)
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/4] building tokenizer from {input_dir / 'vocabs'}")
    tokenizer = Tokenizer.build(input_dir / "vocabs")
    tokenizer.save(output_dir / "vocab.json")
    print(f"      vocab size = {tokenizer.vocab_size}")

    stats: Dict[str, Dict] = {}
    splits: Dict[str, Tuple] = {}
    oov_total = 0
    char_total = 0

    for split in ("train", "test"):
        in_file = input_dir / split / "in.txt"
        out_file = input_dir / split / "out.txt"
        print(f"[2/4] scanning {split}: {in_file}")
        n_valid, split_stats = count_valid(in_file, out_file, args.max_len)
        print(f"      {n_valid} usable pairs, stats={split_stats}")
        arrays = build_arrays(in_file, out_file, tokenizer, args.max_len, n_valid)
        inputs, targets, in_lens, out_lens, chars, oov = arrays
        splits[split] = (inputs, targets, in_lens, out_lens)
        stats[split] = split_stats
        stats[split]["pairs"] = n_valid
        char_total += chars
        oov_total += oov

    # random validation split carved out of the training file
    train_inputs, train_targets, train_in_lens, train_out_lens = splits["train"]
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(train_inputs))
    val_size = min(args.val_size, len(train_inputs) // 10)
    val_idx = np.sort(perm[:val_size])
    train_idx = np.sort(perm[val_size:])

    print(f"[3/4] saving arrays (train={len(train_idx)}, val={len(val_idx)})")
    t0 = time.time()
    np.savez(output_dir / "train.npz",
             inputs=train_inputs[train_idx], targets=train_targets[train_idx],
             in_lens=train_in_lens[train_idx], out_lens=train_out_lens[train_idx])
    np.savez(output_dir / "val.npz",
             inputs=train_inputs[val_idx], targets=train_targets[val_idx],
             in_lens=train_in_lens[val_idx], out_lens=train_out_lens[val_idx])
    test_inputs, test_targets, test_in_lens, test_out_lens = splits["test"]
    np.savez(output_dir / "test.npz",
             inputs=test_inputs, targets=test_targets,
             in_lens=test_in_lens, out_lens=test_out_lens)

    meta = {
        "max_len": args.max_len,
        "vocab_size": tokenizer.vocab_size,
        "num_train": int(len(train_idx)),
        "num_val": int(len(val_idx)),
        "num_test": int(len(test_inputs)),
        "val_size_requested": args.val_size,
        "seed": args.seed,
        "oov_rate": oov_total / max(char_total, 1),
        "num_oov_chars": int(oov_total),
        "num_chars": int(char_total),
        "splits": stats,
    }
    with (output_dir / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[4/4] done in {time.time() - t0:.1f}s")
    print(json.dumps({k: v for k, v in meta.items() if k != "splits"},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
