"""Estimate the BLEU / ROUGE-L ceiling caused by non-unique references.

Many uppers in the dataset have more than one human-written lower couplet.
Those lowers are all valid, but a single-reference metric compares the model
against only one of them.  This script measures the metric between two
*different human* lowers of the same upper, which gives an empirical
"human disagreement" level for the task.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from module.metrics import calc_bleu, calc_char_accuracy, calc_rouge_l


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(Path(__file__).resolve().parent.parent
                                               / "CoupletAI" / "couplet"))
    parser.add_argument("--output", default="results/metrics/reference_disagreement.json")
    parser.add_argument("--samples", default=3000, type=int)
    parser.add_argument("--seed", default=0, type=int)
    args = parser.parse_args()

    input_dir = Path(args.input)
    groups: dict[str, set[str]] = defaultdict(set)
    with (input_dir / "train" / "in.txt").open("r", encoding="utf-8") as fi, \
            (input_dir / "train" / "out.txt").open("r", encoding="utf-8") as fo:
        for upper, lower in zip(fi, fo):
            upper = "".join(upper.split())
            lower = "".join(lower.split())
            if upper and len(upper) == len(lower):
                groups[upper].add(lower)

    multi = [sorted(lowers) for lowers in groups.values() if len(lowers) >= 2]
    rng = random.Random(args.seed)
    sampled = rng.sample(multi, min(args.samples, len(multi)))
    pairs = [rng.sample(lowers, 2) for lowers in sampled]

    bleu4 = [calc_bleu(list(a), list(b), 4) for a, b in pairs]
    rouge = [calc_rouge_l(list(a), list(b)) for a, b in pairs]
    char_acc = [calc_char_accuracy(list(a), list(b)) for a, b in pairs]
    result = {
        "uppers_with_multiple_human_lowers": len(multi),
        "sampled_pairs": len(pairs),
        "bleu4_mean": sum(bleu4) / len(bleu4),
        "bleu4_median": sorted(bleu4)[len(bleu4) // 2],
        "bleu4_p90": sorted(bleu4)[int(len(bleu4) * 0.9)],
        "rouge_l_mean": sum(rouge) / len(rouge),
        "char_accuracy_mean": sum(char_acc) / len(char_acc),
        "exact_match_rate": sum(1 for a, b in pairs if a == b) / len(pairs),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"saved {output}")


if __name__ == "__main__":
    main()
