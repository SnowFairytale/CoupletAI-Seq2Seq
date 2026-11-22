"""Select qualitative examples from a per-sample evaluation CSV."""
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path


def load_rows(path: Path, max_len: int | None = None):
    with path.open("r", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row["bleu4"] = float(row["bleu4"])
        row["rouge_l"] = float(row["rouge_l"])
    if max_len is not None:
        rows = [r for r in rows if len(r["reference"]) <= max_len]
    return rows


def md_escape(text: str) -> str:
    return text.replace("|", "\\|")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="results/samples/gru_attn_fixed_test.csv")
    parser.add_argument("--output", default="results/samples/examples.md")
    parser.add_argument("--n_best", default=6, type=int)
    parser.add_argument("--n_worst", default=6, type=int)
    parser.add_argument("--n_random", default=6, type=int)
    parser.add_argument("--seed", default=7, type=int)
    args = parser.parse_args()

    rows = [r for r in load_rows(Path(args.csv)) if r["reference"]]
    ranked = sorted(rows, key=lambda r: r["bleu4"], reverse=True)
    best = ranked[:args.n_best]
    worst = ranked[-args.n_worst:]
    rng = random.Random(args.seed)
    chosen = best + rng.sample(rows, min(args.n_random, len(rows))) + worst
    seen = set()
    unique = []
    for row in chosen:
        key = (row["upper"], row["reference"], row["prediction"])
        if key not in seen:
            seen.add(key)
            unique.append(row)

    lines = ["| 上联 | 参考下联 | 模型生成 | BLEU-4 | ROUGE-L |",
             "| --- | --- | --- | ---: | ---: |"]
    for row in unique:
        lines.append(
            f"| {md_escape(row['upper'])} | {md_escape(row['reference'])} | "
            f"{md_escape(row['prediction'])} | {row['bleu4']:.4f} | "
            f"{row['rouge_l']:.4f} |")
    text = "\n".join(lines)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text + "\n", encoding="utf-8")
    print(text)
    print(f"\nsaved {output}")


if __name__ == "__main__":
    main()
