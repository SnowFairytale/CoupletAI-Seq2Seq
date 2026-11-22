"""Evaluation metrics used by both training (fast validation) and testing.

BLEU is computed with NLTK's sentence-level implementation and smoothing
method 1, i.e. the same convention as the provided CoupletAI sample.  ROUGE-L
follows the sample implementation (LCS F-measure, beta = 1.2).  In addition we
report a few couplet-specific statistics: exact match, length match,
position-wise character accuracy and corpus-level distinct-n.
"""
from __future__ import annotations

from collections import Counter
from typing import Dict, List, Sequence

from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu, sentence_bleu

_SMOOTHER = SmoothingFunction().method1


def calc_bleu(cand: Sequence[str], ref: Sequence[str], n: int = 4) -> float:
    if not cand or not ref:
        return 0.0
    weights = tuple([1.0 / n] * n + [0.0] * (4 - n))
    try:
        return float(sentence_bleu([list(ref)], list(cand), weights=weights,
                                   smoothing_function=_SMOOTHER))
    except (ZeroDivisionError, ValueError):
        return 0.0


def calc_rouge_l(cand: Sequence[str], ref: Sequence[str], beta: float = 1.2) -> float:
    """LCS based ROUGE-L F-measure (same formula as the sample code)."""
    len_cand, len_ref = len(cand), len(ref)
    if len_cand == 0 or len_ref == 0:
        return 0.0
    lengths = [[0] * (len_ref + 1) for _ in range(len_cand + 1)]
    for i in range(len_cand):
        for j in range(len_ref):
            if cand[i] == ref[j]:
                lengths[i + 1][j + 1] = lengths[i][j] + 1
            else:
                lengths[i + 1][j + 1] = max(lengths[i + 1][j], lengths[i][j + 1])
    lcs = lengths[-1][-1]
    eps = 1e-10
    recall = lcs / (eps + len_ref)
    precision = lcs / (eps + len_cand)
    return float(((1 + beta ** 2) * recall * precision) /
                 (eps + recall + beta ** 2 * precision))


def calc_char_accuracy(cand: Sequence[str], ref: Sequence[str]) -> float:
    """Position-wise accuracy over the reference length (couplets align)."""
    if not ref:
        return 0.0
    correct = sum(1 for a, b in zip(cand, ref) if a == b)
    return correct / len(ref)


def distinct_n(texts: Sequence[Sequence[str]], n: int) -> float:
    total = 0
    unique = set()
    for text in texts:
        grams = [tuple(text[i:i + n]) for i in range(len(text) - n + 1)]
        total += len(grams)
        unique.update(grams)
    return len(unique) / total if total else 0.0


def compute_metrics(preds: Sequence[Sequence[str]], refs: Sequence[Sequence[str]],
                    max_len: int | None = None) -> Dict[str, float]:
    """Aggregate metrics for decoded character sequences.

    ``max_len`` can be supplied so that an EOS rate is meaningful; otherwise
    it is ignored.
    """
    assert len(preds) == len(refs)
    bleu_sums = {n: 0.0 for n in range(1, 5)}
    rouge_sum = 0.0
    exact = 0
    length_match = 0
    char_acc_sum = 0.0
    len_sum = 0

    for pred, ref in zip(preds, refs):
        for n in range(1, 5):
            bleu_sums[n] += calc_bleu(pred, ref, n)
        rouge_sum += calc_rouge_l(pred, ref)
        exact += int(list(pred) == list(ref))
        length_match += int(len(pred) == len(ref))
        char_acc_sum += calc_char_accuracy(pred, ref)
        len_sum += len(pred)

    total = max(len(preds), 1)
    metrics = {
        "bleu1": bleu_sums[1] / total,
        "bleu2": bleu_sums[2] / total,
        "bleu3": bleu_sums[3] / total,
        "bleu4": bleu_sums[4] / total,
        "rouge_l": rouge_sum / total,
        "exact_match": exact / total,
        "length_match": length_match / total,
        "char_accuracy": char_acc_sum / total,
        "distinct_1": distinct_n(preds, 1),
        "distinct_2": distinct_n(preds, 2),
        "avg_pred_len": len_sum / total,
        "num_samples": total,
    }
    try:
        metrics["bleu4_corpus"] = float(corpus_bleu(
            [[list(r)] for r in refs], [list(p) for p in preds],
            weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=_SMOOTHER))
    except Exception:
        metrics["bleu4_corpus"] = 0.0
    return metrics


def ngram_counts(texts: Sequence[Sequence[str]], n: int) -> Counter:
    counter = Counter()
    for text in texts:
        counter.update(tuple(text[i:i + n]) for i in range(len(text) - n + 1))
    return counter
