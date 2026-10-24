"""Evaluation metrics (retrieval + answer quality), kept free of I/O so they are unit-testable."""

from __future__ import annotations

import re
import string
from collections import Counter


# ---------------------------------------------------------------- retrieval
def recall_at_k(ranked: list[str], gold: list[str], k: int) -> float:
    if not gold:
        return float("nan")
    return len(set(ranked[:k]) & set(gold)) / len(set(gold))


def all_gold_at_k(ranked: list[str], gold: list[str], k: int) -> float:
    if not gold:
        return float("nan")
    return float(set(gold) <= set(ranked[:k]))


def mrr(ranked: list[str], gold: list[str]) -> float:
    if not gold:
        return float("nan")
    for i, d in enumerate(ranked, 1):
        if d in gold:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked: list[str], gold: list[str], k: int) -> float:
    import math

    if not gold:
        return float("nan")
    dcg = sum(1.0 / math.log2(i + 1) for i, d in enumerate(ranked[:k], 1) if d in gold)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(set(gold)), k) + 1))
    return dcg / idcg


# ---------------------------------------------------------------- answers (HotpotQA-style normalisation)
def normalize_answer(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def exact_match(pred: str, gold: str) -> float:
    return float(normalize_answer(pred) == normalize_answer(gold))


def f1(pred: str, gold: str) -> float:
    p, g = normalize_answer(pred).split(), normalize_answer(gold).split()
    if not p or not g:
        return float(p == g)
    common = Counter(p) & Counter(g)
    same = sum(common.values())
    if same == 0:
        return 0.0
    precision, recall = same / len(p), same / len(g)
    return 2 * precision * recall / (precision + recall)


def keyword_hit(text: str, keywords: list[str]) -> float:
    """Fraction of required keywords present (case-insensitive, whitespace-insensitive)."""
    if not keywords:
        return float("nan")
    t = " ".join(text.lower().split())
    return sum(1 for k in keywords if " ".join(k.lower().split()) in t) / len(keywords)


def mean(xs) -> float:
    xs = [x for x in xs if x == x]  # drop NaN
    return sum(xs) / len(xs) if xs else float("nan")
