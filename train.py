"""Train a seq2seq couplet model (or the sequence-labelling baseline).

Examples::

    conda activate dip
    cd MySolution
    python train.py --model gru_attn --epochs 10 --batch_size 128
    python train.py --model gru_noattn --epochs 6 --batch_size 128 --run_name gru_noattn
    python train.py --model bilstm_tagger --epochs 5 --batch_size 512

The script logs one row per epoch to ``results/logs/<run>_epoch.csv`` and a
smoothed training loss every ``--log_every`` steps to
``results/logs/<run>_steps.csv``.  The best checkpoint (lowest validation
loss) and the last checkpoint are written to ``results/checkpoints``.
"""
from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from module import (CoupletDataset, Tokenizer, build_model, compute_metrics,
                    count_parameters, make_collate, read_json, save_json,
                    set_seed)
from module.inference import decode_batch


# --------------------------------------------------------------------------- #
# arguments
# --------------------------------------------------------------------------- #
def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Couplet seq2seq training")
    parser.add_argument("--data", default="data", type=str)
    parser.add_argument("--output", default="results", type=str)
    parser.add_argument("--model", default="gru_attn", type=str,
                        choices=["gru_attn", "gru_noattn", "bilstm_tagger"])
    parser.add_argument("--run_name", default=None, type=str)
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--weight_decay", default=0.0, type=float)
    parser.add_argument("--embed_dim", default=256, type=int)
    parser.add_argument("--hidden_dim", default=256, type=int)
    parser.add_argument("--enc_layers", default=2, type=int)
    parser.add_argument("--dec_layers", default=2, type=int)
    parser.add_argument("--dropout", default=0.2, type=float)
    parser.add_argument("--clip", default=3.0, type=float)
    parser.add_argument("--teacher_forcing", default=1.0, type=float)
    parser.add_argument("--label_smoothing", default=0.0, type=float)
    parser.add_argument("--eval_samples", default=1000, type=int,
                        help="validation subset used for BLEU/ROUGE each epoch")
    parser.add_argument("--val_batch_size", default=256, type=int)
    parser.add_argument("--patience", default=3, type=int)
    parser.add_argument("--min_lr", default=1e-5, type=float)
    parser.add_argument("--max_train", default=None, type=int,
                        help="optionally train on a random subset (debug/ablation)")
    parser.add_argument("--limit_batches", default=None, type=int,
                        help="stop each epoch after N batches (smoke test)")
    parser.add_argument("--log_every", default=50, type=int)
    parser.add_argument("--demo_every", default=2, type=int)
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--device", default=None, type=str)
    return parser.parse_args()


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def tagged_loss(logits: torch.Tensor, tgt: torch.Tensor, src_lens: torch.Tensor,
                pad_id: int, reduction: str = "mean"):
    """Aligned tagging loss: source position j predicts lower-couplet char j."""
    length = logits.size(1)
    target = tgt[:, 1:1 + length]
    valid = torch.arange(length, device=tgt.device).unsqueeze(0) < src_lens.unsqueeze(1)
    target = target.masked_fill(~valid, pad_id)
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                           target.reshape(-1), ignore_index=pad_id,
                           reduction=reduction)


def build_dataloader(dataset, batch_size: int, shuffle: bool, pad_id: int,
                     num_workers: int = 0, drop_last: bool = False) -> DataLoader:
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        collate_fn=make_collate(pad_id), num_workers=num_workers,
        pin_memory=torch.cuda.is_available(), drop_last=drop_last,
    )


@torch.no_grad()
def evaluate_loss(model, loader, device, model_name: str, pad_id: int,
                  use_amp: bool) -> float:
    model.eval()
    loss_sum, token_count = 0.0, 0
    for batch in loader:
        src, src_lens, tgt, _ = (t.to(device, non_blocking=True) for t in batch)
        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            if model_name == "bilstm_tagger":
                logits = model(src)
                loss = tagged_loss(logits, tgt, src_lens, pad_id, reduction="sum")
                tokens = int((torch.arange(logits.size(1), device=device)
                              .unsqueeze(0) < src_lens.unsqueeze(1)).sum().item())
            else:
                loss = model(src, src_lens, tgt, teacher_forcing_ratio=1.0)
                tokens = int(tgt[:, 1:].ne(pad_id).sum().item())
        # seq2seq returns the mean loss over the batch, the tagger returns a sum
        loss_sum += float(loss.item()) * (tokens if model_name != "bilstm_tagger" else 1)
        token_count += tokens
    return loss_sum / max(token_count, 1)


@torch.no_grad()
def evaluate_generation(model, loader, tokenizer, device, model_name: str,
                        beam_size: int = 1):
    model.eval()
    preds, refs = [], []
    for batch in loader:
        src, src_lens, tgt, _ = (t.to(device, non_blocking=True) for t in batch)
        _, pred_texts = decode_batch(model, tokenizer, src, src_lens,
                                     model_name=model_name, beam_size=beam_size)
        preds.extend([list(t) for t in pred_texts])
        for row, length in zip(tgt.tolist(), src_lens.tolist()):
            refs.append([tokenizer.id_to_token.get(i, "") for i in row[1:1 + length]])
    return compute_metrics(preds, refs)


def save_checkpoint(path: Path, model, args, model_kwargs, epoch: int,
                    val_loss: float, metrics: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "model_name": args.model,
        "model_kwargs": model_kwargs,
        "epoch": epoch,
        "val_loss": val_loss,
        "metrics": metrics,
        "args": vars(args),
    }, path)


def demo_predictions(model, tokenizer, device, model_name: str) -> None:
    uppers = [
        "春暮偶登楼，上下鱼龙，应惜满湖绿水",
        "一曲笙歌春似海",
        "马齿草焉无马齿",
        "我有诗情堪纵酒",
        "海内存知己",
    ]
    model.eval()
    for upper in uppers:
        ids = tokenizer.encode(upper)
        src = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
        src_lens = torch.tensor([len(ids)], dtype=torch.long, device=device)
        _, pred = decode_batch(model, tokenizer, src, src_lens, model_name)
        print(f"      上联：{upper}\n      下联：{pred[0]}")


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> None:
    args = get_args()
    run_name = args.run_name or args.model
    set_seed(args.seed)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    use_amp = (device.type == "cuda") and not args.no_amp
    data_dir = Path(args.data)
    output_dir = Path(args.output)
    log_dir = output_dir / "logs"
    ckpt_dir = output_dir / "checkpoints"
    log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print(f"device={device}, amp={use_amp}, run={run_name}")
    tokenizer = Tokenizer.load(data_dir / "vocab.json")
    meta = read_json(data_dir / "meta.json")
    pad_id, bos_id, eos_id = tokenizer.pad_id, tokenizer.bos_id, tokenizer.eos_id
    print(f"vocab={tokenizer.vocab_size}, train={meta['num_train']}, "
          f"val={meta['num_val']}, test={meta['num_test']}")

    train_set = CoupletDataset(data_dir / "train.npz", max_samples=args.max_train,
                               seed=args.seed)
    val_set = CoupletDataset(data_dir / "val.npz")
    val_eval_set = CoupletDataset(data_dir / "val.npz",
                                  max_samples=args.eval_samples, seed=args.seed)

    train_loader = build_dataloader(train_set, args.batch_size, True, pad_id,
                                    args.num_workers, drop_last=True)
    val_loader = build_dataloader(val_set, args.val_batch_size, False, pad_id,
                                  args.num_workers)
    val_eval_loader = build_dataloader(val_eval_set, args.val_batch_size, False,
                                       pad_id, args.num_workers)

    if args.model == "bilstm_tagger":
        model_kwargs = dict(embed_dim=args.embed_dim, hidden_dim=args.hidden_dim,
                            num_layers=args.enc_layers, dropout=args.dropout)
    else:
        model_kwargs = dict(embed_dim=args.embed_dim, hidden_dim=args.hidden_dim,
                            enc_layers=args.enc_layers, dec_layers=args.dec_layers,
                            dropout=args.dropout)
    model = build_model(args.model, tokenizer.vocab_size, pad_id=pad_id,
                        bos_id=bos_id, eos_id=eos_id, **model_kwargs).to(device)
    n_params = count_parameters(model)
    print(f"model={args.model}, trainable parameters={n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=1, min_lr=args.min_lr)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    config = {
        "args": vars(args),
        "run_name": run_name,
        "num_parameters": n_params,
        "device": str(device),
        "torch_version": torch.__version__,
        "train_pairs": len(train_set),
        "val_pairs": len(val_set),
        "val_eval_pairs": len(val_eval_set),
        "eos_id": eos_id,
        "bos_id": bos_id,
        "pad_id": pad_id,
    }
    save_json(config, log_dir / f"{run_name}_config.json")

    epoch_csv = log_dir / f"{run_name}_epoch.csv"
    step_csv = log_dir / f"{run_name}_steps.csv"
    epoch_file = epoch_csv.open("w", newline="", encoding="utf-8")
    epoch_writer = csv.writer(epoch_file)
    epoch_writer.writerow([
        "epoch", "train_loss", "val_loss", "train_ppl", "val_ppl",
        "bleu1", "bleu2", "bleu3", "bleu4", "rouge_l",
        "exact_match", "length_match", "char_accuracy",
        "distinct_1", "distinct_2", "lr", "epoch_seconds",
    ])
    step_file = step_csv.open("w", newline="", encoding="utf-8")
    step_writer = csv.writer(step_file)
    step_writer.writerow(["global_step", "epoch", "batch", "loss", "lr", "tokens"])

    best_val_loss = float("inf")
    best_epoch = 0
    best_metrics: dict = {}
    bad_epochs = 0
    global_step = 0
    train_start = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_start = time.time()
        loss_sum, token_count, batch_count = 0.0, 0, 0
        progress = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}",
                        dynamic_ncols=True, file=sys.stdout)

        for step, batch in enumerate(progress, start=1):
            src, src_lens, tgt, _ = (t.to(device, non_blocking=True) for t in batch)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                if args.model == "bilstm_tagger":
                    logits = model(src)
                    loss = tagged_loss(logits, tgt, src_lens, pad_id)
                    tokens = int((torch.arange(logits.size(1), device=device)
                                  .unsqueeze(0) < src_lens.unsqueeze(1)).sum().item())
                else:
                    loss = model(src, src_lens, tgt,
                                 teacher_forcing_ratio=args.teacher_forcing)
                    tokens = int(tgt[:, 1:].ne(pad_id).sum().item())

            scaler.scale(loss).backward()
            if args.clip and args.clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            scaler.step(optimizer)
            scaler.update()

            loss_value = float(loss.item())
            loss_sum += loss_value * tokens
            token_count += tokens
            batch_count += 1
            global_step += 1

            if step % 10 == 0:
                progress.set_postfix(loss=f"{loss_sum / max(token_count, 1):.4f}",
                                     lr=f"{optimizer.param_groups[0]['lr']:.2e}")
            if step % args.log_every == 0:
                step_writer.writerow([global_step, epoch, step, f"{loss_value:.6f}",
                                      f"{optimizer.param_groups[0]['lr']:.8f}", tokens])
                step_file.flush()
            if args.limit_batches and batch_count >= args.limit_batches:
                break

        progress.close()
        train_loss = loss_sum / max(token_count, 1)
        val_loss = evaluate_loss(model, val_loader, device, args.model, pad_id, use_amp)
        gen_metrics = evaluate_generation(model, val_eval_loader, tokenizer, device,
                                          args.model, beam_size=1)
        lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_loss)
        epoch_seconds = time.time() - epoch_start
        train_ppl = math.exp(min(train_loss, 20))
        val_ppl = math.exp(min(val_loss, 20))

        epoch_writer.writerow([
            epoch, f"{train_loss:.6f}", f"{val_loss:.6f}",
            f"{train_ppl:.4f}", f"{val_ppl:.4f}",
            f"{gen_metrics['bleu1']:.6f}", f"{gen_metrics['bleu2']:.6f}",
            f"{gen_metrics['bleu3']:.6f}", f"{gen_metrics['bleu4']:.6f}",
            f"{gen_metrics['rouge_l']:.6f}",
            f"{gen_metrics['exact_match']:.6f}",
            f"{gen_metrics['length_match']:.6f}",
            f"{gen_metrics['char_accuracy']:.6f}",
            f"{gen_metrics['distinct_1']:.6f}", f"{gen_metrics['distinct_2']:.6f}",
            f"{lr:.8f}", f"{epoch_seconds:.1f}",
        ])
        epoch_file.flush()

        print(f"[epoch {epoch}] train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
              f"bleu4={gen_metrics['bleu4']:.4f} rouge_l={gen_metrics['rouge_l']:.4f} "
              f"len_match={gen_metrics['length_match']:.3f} lr={lr:.2e} "
              f"time={epoch_seconds:.1f}s")

        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
            best_epoch = epoch
            best_metrics = {
                "epoch": epoch, "val_loss": val_loss,
                "train_loss": train_loss, "val_ppl": val_ppl,
                **gen_metrics,
            }
            bad_epochs = 0
            save_checkpoint(ckpt_dir / f"{run_name}_best.pt", model, args,
                            model_kwargs, epoch, val_loss, best_metrics)
        else:
            bad_epochs += 1

        save_checkpoint(ckpt_dir / f"{run_name}_last.pt", model, args,
                        model_kwargs, epoch, val_loss, gen_metrics)

        if args.demo_every and epoch % args.demo_every == 0:
            demo_predictions(model, tokenizer, device, args.model)

        if bad_epochs >= args.patience:
            print(f"early stopping after {bad_epochs} epochs without improvement")
            break

    epoch_file.close()
    step_file.close()

    summary = {
        "run_name": run_name,
        "model": args.model,
        "num_parameters": n_params,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "best_metrics": best_metrics,
        "total_train_seconds": time.time() - train_start,
        "args": vars(args),
    }
    save_json(summary, log_dir / f"{run_name}_summary.json")
    print(f"training finished: best epoch={best_epoch}, "
          f"best val loss={best_val_loss:.4f}, "
          f"total time={time.time() - train_start:.1f}s")
    print(json.dumps(best_metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
