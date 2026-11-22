"""Evaluate a trained couplet model on the validation or test split.

Example::

    python evaluate.py --checkpoint results/checkpoints/gru_attn_best.pt \
        --split test --name gru_attn

Outputs ``results/metrics/<name>_<split>.json`` and a per-sample CSV under
``results/samples``.
"""
from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import csv
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from module import (CoupletDataset, Tokenizer, build_model, calc_bleu,
                    calc_rouge_l, compute_metrics, make_collate, read_json,
                    save_json)
from module.inference import decode_batch


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=str)
    parser.add_argument("--data", default="data", type=str)
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--beam_size", default=1, type=int)
    parser.add_argument("--max_samples", default=None, type=int)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--output", default="results", type=str)
    parser.add_argument("--name", default=None, type=str)
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--device", default=None, type=str)
    return parser.parse_args()


@torch.no_grad()
def teacher_forced_loss(model, loader, device, model_name: str, pad_id: int,
                        use_amp: bool):
    model.eval()
    loss_sum, token_count = 0.0, 0
    for batch in loader:
        src, src_lens, tgt, _ = (t.to(device, non_blocking=True) for t in batch)
        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            if model_name == "bilstm_tagger":
                logits = model(src)
                length = logits.size(1)
                target = tgt[:, 1:1 + length]
                valid = (torch.arange(length, device=device).unsqueeze(0)
                         < src_lens.unsqueeze(1))
                target = target.masked_fill(~valid, pad_id)
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                       target.reshape(-1), ignore_index=pad_id,
                                       reduction="sum")
                tokens = int(valid.sum().item())
            else:
                loss = model(src, src_lens, tgt, teacher_forcing_ratio=1.0)
                tokens = int(tgt[:, 1:].ne(pad_id).sum().item())
        loss_sum += float(loss.item()) * (tokens if model_name != "bilstm_tagger" else 1)
        token_count += tokens
    return loss_sum / max(token_count, 1)


def main() -> None:
    args = get_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    use_amp = device.type == "cuda" and not args.no_amp
    data_dir = Path(args.data)
    output_dir = Path(args.output)
    ckpt_path = Path(args.checkpoint)

    tokenizer = Tokenizer.load(data_dir / "vocab.json")
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model_name = checkpoint["model_name"]
    model = build_model(model_name, tokenizer.vocab_size,
                        pad_id=tokenizer.pad_id, bos_id=tokenizer.bos_id,
                        eos_id=tokenizer.eos_id,
                        **checkpoint["model_kwargs"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    name = args.name or ckpt_path.stem.replace("_best", "").replace("_last", "")
    dataset = CoupletDataset(data_dir / f"{args.split}.npz",
                             max_samples=args.max_samples, seed=42)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        collate_fn=make_collate(tokenizer.pad_id),
                        pin_memory=device.type == "cuda")
    print(f"evaluating {ckpt_path.name} on {args.split} "
          f"({len(dataset)} samples, beam={args.beam_size})")

    loss = teacher_forced_loss(model, loader, device, model_name,
                               tokenizer.pad_id, use_amp)
    ppl = math.exp(min(loss, 20))

    preds, refs, uppers = [], [], []
    rows = []
    for batch in loader:
        src, src_lens, tgt, _ = (t.to(device, non_blocking=True) for t in batch)
        _, pred_texts = decode_batch(model, tokenizer, src, src_lens,
                                     model_name=model_name,
                                     beam_size=args.beam_size)
        for i, pred_text in enumerate(pred_texts):
            length = int(src_lens[i])
            upper = tokenizer.decode(src[i, :length].tolist())
            ref = tokenizer.decode(tgt[i, 1:1 + length].tolist())
            pred_chars = list(pred_text)
            ref_chars = list(ref)
            preds.append(pred_chars)
            refs.append(ref_chars)
            uppers.append(upper)
            rows.append({
                "index": len(rows),
                "upper": upper,
                "reference": ref,
                "prediction": pred_text,
                "bleu4": calc_bleu(pred_chars, ref_chars, 4),
                "rouge_l": calc_rouge_l(pred_chars, ref_chars),
                "exact_match": int(pred_chars == ref_chars),
                "length_match": int(len(pred_chars) == len(ref_chars)),
            })

    metrics = compute_metrics(preds, refs)
    metrics.update({
        "split": args.split,
        "checkpoint": str(ckpt_path),
        "model": model_name,
        "best_epoch": checkpoint.get("epoch"),
        "beam_size": args.beam_size,
        "teacher_forced_loss": loss,
        "teacher_forced_ppl": ppl,
        "num_samples": len(dataset),
    })

    metrics_dir = output_dir / "metrics"
    samples_dir = output_dir / "samples"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    samples_dir.mkdir(parents=True, exist_ok=True)
    save_json(metrics, metrics_dir / f"{name}_{args.split}.json")

    sample_path = samples_dir / f"{name}_{args.split}.csv"
    with sample_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"saved {metrics_dir / f'{name}_{args.split}.json'}")
    print(f"saved {sample_path}")


if __name__ == "__main__":
    main()
