"""Build a Markdown comparison table from evaluation JSON files."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", default="results/metrics")
    parser.add_argument("--names", nargs="+",
                        default=["bilstm_tagger_test", "gru_noattn_test",
                                 "gru_attn_test", "gru_attn_fixed_test"])
    parser.add_argument("--output", default="results/metrics/comparison.md")
    args = parser.parse_args()

    rows = []
    for name in args.names:
        path = Path(args.metrics) / f"{name}.json"
        if not path.exists():
            print(f"missing {path}")
            continue
        with path.open("r", encoding="utf-8") as f:
            rows.append((name, json.load(f)))
    if not rows:
        raise SystemExit("no metric files found")

    header = ("| 模型 / 解码方式 | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | "
              "ROUGE-L | 完全匹配 | 长度一致 | 字级准确率 | distinct-1 | distinct-2 |")
    sep = "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
    lines = [header, sep]
    for name, m in rows:
        lines.append(
            f"| {name} | {m.get('bleu1', 0):.4f} | {m.get('bleu2', 0):.4f} | "
            f"{m.get('bleu3', 0):.4f} | {m.get('bleu4', 0):.4f} | "
            f"{m.get('rouge_l', 0):.4f} | {m.get('exact_match', 0):.4f} | "
            f"{m.get('length_match', 0):.4f} | {m.get('char_accuracy', 0):.4f} | "
            f"{m.get('distinct_1', 0):.4f} | {m.get('distinct_2', 0):.4f} |")
    text = "\n".join(lines) + "\n"
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(text, encoding="utf-8")
    print(text)
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
