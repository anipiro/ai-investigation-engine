"""Duplicate detection.

- content_hash: exact duplicates after normalisation (case, punctuation, whitespace).
- simhash64 / hamming: near-duplicates (re-exported copies, small edits) — Charikar's SimHash
  over word 3-shingles. Two texts with hamming distance <= 3 are treated as near-duplicates.
"""

from __future__ import annotations

import hashlib
import re

_NON_ALNUM = re.compile(r"[^0-9a-z]+")


def normalize_for_hash(text: str) -> str:
    return _NON_ALNUM.sub(" ", text.lower()).strip()


def content_hash(text: str) -> str:
    return hashlib.sha256(normalize_for_hash(text).encode("utf-8")).hexdigest()


def _h64(s: str) -> int:
    return int.from_bytes(hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest(), "big")


def simhash64(text: str, shingle: int = 3) -> int:
    words = normalize_for_hash(text).split()
    if not words:
        return 0
    grams = [" ".join(words[i : i + shingle]) for i in range(max(1, len(words) - shingle + 1))]
    acc = [0] * 64
    for g in grams:
        h = _h64(g)
        for bit in range(64):
            acc[bit] += 1 if (h >> bit) & 1 else -1
    out = 0
    for bit in range(64):
        if acc[bit] > 0:
            out |= 1 << bit
    return out


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def is_near_duplicate(a: int, b: int, threshold: int = 3) -> bool:
    return hamming(a, b) <= threshold
