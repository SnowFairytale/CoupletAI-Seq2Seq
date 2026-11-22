"""Dataset statistics and length-distribution figure.

Example::

    python analyze_data.py --data data --output results
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def describe(lengths: np.ndarray) -> dict:
    return {
        "pairs": int(len(lengths)),
        "mean": float(lengths.mean()),
        "median": float(np.median(lengths)),
        "p25": int(np.percentile(lengths, 25)),
        "p75": int(np.percentile(lengths, 75)),
        "p95": int(np.percentile(lengths, 95)),
        "min": int(lengths.min()),
        "max": int(lengths.max()),
        "ratio_le7": float((lengths <= 7).mean()),
        "ratio_8_11": float(((lengths >= 8) & (lengths <= 11)).mean()),
        "ratio_12_16": float(((lengths >= 12) & (lengths <= 16)).mean()),
        "ratio_gt16": float((lengths > 16).mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data")
    parser.add_argument("--output", default="results")
    args = parser.parse_args()

    data_dir, output_dir = Path(args.data), Path(args.output)
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    lengths = {split: np.load(data_dir / f"{split}.npz")["in_lens"]
               for split in ("train", "val", "test")}
    stats = {split: describe(values) for split, values in lengths.items()}
    metrics_dir = output_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    with (metrics_dir / "data_stats.json").open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    max_len = int(max(values.max() for values in lengths.values()))
    bins = np.arange(1, max_len + 2)
    fig, ax = plt.subplots(figsize=(10, 4))
    width = 0.8 / len(lengths)
    for i, (split, values) in enumerate(lengths.items()):
        counts, _ = np.histogram(values, bins=bins)
        ratio = counts / counts.sum()
        ax.bar(bins[:-1] + i * width, ratio, width=width, label=split)
    ax.set_xlabel("Couplet length (characters)")
    ax.set_ylabel("Proportion")
    ax.set_title("Length distribution of upper couplets")
    ax.set_xticks(range(1, max_len + 1))
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "length_distribution.png", dpi=200)
    plt.close(fig)

    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"saved {fig_dir / 'length_distribution.png'}")


if __name__ == "__main__":
    main()
