"""Our own BM25 sparse encoder (keyword side of hybrid search).

Documents are encoded with the BM25 term-frequency component
    w(t, d) = tf * (k1 + 1) / (tf + k1 * (1 - b + b * |d| / avgdl))
and stored as Qdrant sparse vectors. The collection uses Qdrant's IDF modifier, so the
server multiplies each matched term by idf(t) computed over the *live* collection — which
keeps scores correct as documents are added and deleted without re-encoding the corpus.
Query vectors are binary (1.0 per unique query term), so score(q, d) = sum idf(t) * w(t, d):
exactly Okapi BM25.

Tokens are hashed to 32-bit indices (crc32) so no vocabulary has to be stored.
"""

from __future__ import annotations

import re
import zlib
from collections import Counter
from dataclasses import dataclass

STOPWORDS = frozenset(
    """a an and are as at be been but by can could did do does for from had has have he her his how i if in
    into is it its me my no not of on or our she so than that the their them then there these they this to
    was we were what when where which who whom why will with would you your about after also any before
    between both during each more most other over same some such only own very just which while""".split()
)

_TOKEN = re.compile(r"[a-z0-9]+(?:[-_.][a-z0-9]+)*")


def _stem(tok: str) -> str:
    """Tiny suffix stripper: enough to match plurals / -ing / -ed without a dependency."""
    if len(tok) <= 4 or tok.isdigit():
        return tok
    for suf, rep in (("ies", "y"), ("sses", "ss"), ("ing", ""), ("edly", ""), ("ed", ""), ("es", ""), ("s", "")):
        if tok.endswith(suf) and len(tok) - len(suf) >= 3:
            if suf == "s" and tok.endswith("ss"):
                return tok
            return tok[: -len(suf)] + rep
    return tok


def tokenize(text: str) -> list[str]:
    out: list[str] = []
    for m in _TOKEN.finditer(text.lower()):
        tok = m.group(0)
        if any(c in tok for c in "-_."):
            # Keep compound identifiers ("chg-1182", "v2.3") and their parts.
            out.append(tok)
            out.extend(_stem(p) for p in re.split(r"[-_.]", tok) if p and p not in STOPWORDS)
        elif tok not in STOPWORDS:
            out.append(_stem(tok))
    return out


def term_index(term: str) -> int:
    return zlib.crc32(term.encode("utf-8"))


@dataclass
class SparseVec:
    indices: list[int]
    values: list[float]


class BM25Encoder:
    def __init__(self, k1: float = 1.2, b: float = 0.75, avgdl: float = 120.0):
        self.k1, self.b, self.avgdl = k1, b, avgdl

    def encode_document(self, text: str) -> SparseVec:
        toks = tokenize(text)
        dl = len(toks) or 1
        tf = Counter(toks)
        norm = self.k1 * (1 - self.b + self.b * dl / self.avgdl)
        weights: dict[int, float] = {}
        for term, f in tf.items():
            idx = term_index(term)
            # crc32 collisions are rare; if they happen, add weights.
            weights[idx] = weights.get(idx, 0.0) + f * (self.k1 + 1) / (f + norm)
        return SparseVec(indices=list(weights), values=list(weights.values()))

    def encode_query(self, text: str) -> SparseVec:
        idxs = sorted({term_index(t) for t in tokenize(text)})
        return SparseVec(indices=idxs, values=[1.0] * len(idxs))
