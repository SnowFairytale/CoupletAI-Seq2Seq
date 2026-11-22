"""Inspect the submitted checkpoint without importing the project code.

Usage:
    python inspect_checkpoint.py
    python inspect_checkpoint.py --checkpoint gru_attn_seq2seq_best.pt
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=None, type=str,
                        help="checkpoint path; by default the FP32 or FP16 file next to this script")
    parser.add_argument("--no_sha256", action="store_true")
    args = parser.parse_args()

    if args.checkpoint is None:
        base = Path(__file__).resolve().parent
        candidates = [base / "gru_attn_seq2seq_best.pt",
                      base / "gru_attn_seq2seq_best_fp16.pt"]
        path = next((candidate for candidate in candidates if candidate.exists()), None)
        if path is None:
            raise FileNotFoundError("no gru_attn_seq2seq_best*.pt found next to this script")
    else:
        path = Path(args.checkpoint)
        if not path.is_absolute():
            path = Path(__file__).resolve().parent / path
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)

    num_params = sum(value.numel() for value in checkpoint["model_state"].values()
                     if value.is_floating_point())
    print(f"checkpoint      : {path}")
    print(f"size            : {path.stat().st_size / 1024 / 1024:.2f} MB")
    print(f"model_name      : {checkpoint['model_name']}")
    print(f"model_kwargs    : {checkpoint['model_kwargs']}")
    print(f"precision       : {checkpoint.get('precision', 'fp32')}")
    print(f"best epoch      : {checkpoint['epoch']}")
    print(f"best val loss   : {checkpoint['val_loss']:.6f}")
    print(f"parameters      : {num_params:,}")
    print(f"state_dict keys : {len(checkpoint['model_state'])}")
    if not args.no_sha256:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        print(f"sha256          : {digest}")


if __name__ == "__main__":
    main()
