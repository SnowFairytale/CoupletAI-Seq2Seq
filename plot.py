"""Plot training curves and test-set comparison figures.

Example::

    python plot.py --runs gru_attn gru_noattn bilstm_tagger --main_run gru_attn
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read_epoch_csv(path: Path) -> Dict[str, List[float]]:
    columns: Dict[str, List[float]] = {}
    with path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            for key, value in row.items():
                if key is None:
                    continue
                try:
                    columns.setdefault(key, []).append(float(value))
                except ValueError:
                    pass
    return columns


def read_step_csv(path: Path):
    steps, losses = [], []
    with path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            steps.append(int(row["global_step"]))
            losses.append(float(row["loss"]))
    return np.array(steps), np.array(losses)


def ema(values: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    out = np.empty_like(values)
    acc = values[0]
    for i, value in enumerate(values):
        acc = alpha * value + (1 - alpha) * acc
        out[i] = acc
    return out


def plot_loss_curves(runs: List[str], log_dir: Path, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for run in runs:
        path = log_dir / f"{run}_epoch.csv"
        if not path.exists():
            continue
        data = read_epoch_csv(path)
        epochs = data["epoch"]
        axes[0].plot(epochs, data["train_loss"], marker="o", ms=3, label=run)
        axes[1].plot(epochs, data["val_loss"], marker="o", ms=3, label=run)
    for ax, title in zip(axes, ["Training loss (teacher forcing)",
                                "Validation loss (teacher forcing)"]):
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Cross-entropy per character")
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "loss_curves.png", dpi=200)
    plt.close(fig)


def plot_metric_curves(runs: List[str], log_dir: Path, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for run in runs:
        path = log_dir / f"{run}_epoch.csv"
        if not path.exists():
            continue
        data = read_epoch_csv(path)
        epochs = data["epoch"]
        axes[0].plot(epochs, data["bleu4"], marker="o", ms=3, label=run)
        axes[1].plot(epochs, data["rouge_l"], marker="o", ms=3, label=run)
    for ax, title, ylabel in zip(
            axes,
            ["Validation BLEU-4 (greedy)", "Validation ROUGE-L (greedy)"],
            ["BLEU-4", "ROUGE-L F1"]):
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "metric_curves.png", dpi=200)
    plt.close(fig)


def plot_step_loss(run: str, log_dir: Path, out_dir: Path) -> None:
    path = log_dir / f"{run}_steps.csv"
    if not path.exists():
        return
    steps, losses = read_step_csv(path)
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.plot(steps, losses, color="tab:blue", alpha=0.2, lw=0.8,
            label="batch loss")
    ax.plot(steps, ema(losses, 0.05), color="tab:red", lw=1.8,
            label="exponential moving average")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Cross-entropy per character")
    ax.set_title(f"Training loss of {run}")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "train_step_loss.png", dpi=200)
    plt.close(fig)


def plot_test_comparison(runs: List[str], metrics_dir: Path, out_dir: Path,
                         suffix: str) -> None:
    names, metrics = [], []
    for run in runs:
        path = metrics_dir / f"{run}_{suffix}.json"
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as f:
            metrics.append(json.load(f))
        names.append(run)
    if not metrics:
        return

    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    x = np.arange(len(names))
    width = 0.36
    axes[0].bar(x - width / 2, [m["bleu4"] for m in metrics], width,
                label="BLEU-4", color=colors[0])
    axes[0].bar(x + width / 2, [m["rouge_l"] for m in metrics], width,
                label="ROUGE-L", color=colors[1])
    axes[1].bar(x - width / 2, [m["length_match"] for m in metrics], width,
                label="Length match", color=colors[2])
    axes[1].bar(x + width / 2, [m["char_accuracy"] for m in metrics], width,
                label="Character accuracy", color=colors[3])
    axes[0].set_title("Generation quality on test set")
    axes[1].set_title("Structural accuracy on test set")
    axes[0].set_ylabel("Score")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=15)
        ax.grid(axis="y", alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "test_comparison.png", dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+",
                        default=["gru_attn", "gru_noattn", "bilstm_tagger"])
    parser.add_argument("--main_run", default="gru_attn")
    parser.add_argument("--logs", default="results/logs")
    parser.add_argument("--metrics", default="results/metrics")
    parser.add_argument("--output", default="results/figures")
    parser.add_argument("--split", default="test")
    args = parser.parse_args()

    log_dir, metrics_dir, out_dir = Path(args.logs), Path(args.metrics), Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_loss_curves(args.runs, log_dir, out_dir)
    plot_metric_curves(args.runs, log_dir, out_dir)
    plot_step_loss(args.main_run, log_dir, out_dir)
    plot_test_comparison(args.runs, metrics_dir, out_dir, args.split)
    print(f"figures written to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
