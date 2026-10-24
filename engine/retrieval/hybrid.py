"""Hybrid retrieval: dense + BM25 candidates -> our own fusion and re-ranking pipeline.

Own algorithmic components (not delegated to a library):
  1. weighted Reciprocal Rank Fusion of the dense and sparse rankings
  2. title-match boost — a query that names an entity should surface that entity's document
     (important for multi-hop "bridge" questions)
  3. score blending with an optional cross-encoder (min-max normalised)
  4. document-aware MMR diversification: penalises redundancy (cosine on dense vectors) and
     caps chunks per document so the context covers several sources instead of one.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from engine.retrieval.sparse import BM25Encoder, tokenize
from engine.retrieval.store import Hit, VectorStore

log = logging.getLogger(__name__)

Mode = Literal["dense", "sparse", "hybrid"]


@dataclass
class RetrievalConfig:
    mode: Mode = "hybrid"
    candidates: int = 40
    top_k: int = 8
    rrf_k: int = 60
    dense_weight: float = 1.0
    sparse_weight: float = 1.0
    title_boost: float = 0.5  # in units of a rank-1 RRF contribution
    rerank: bool = True
    rerank_pool: int = 30
    rerank_weight: float = 0.7
    mmr: bool = True
    mmr_lambda: float = 0.75
    max_per_doc: int = 2
    sources: list[str] | None = None
    doc_ids: list[str] | None = None
    collection: str | None = None


@dataclass
class ScoredHit:
    hit: Hit
    score: float
    debug: dict[str, float | int | None] = field(default_factory=dict)


def rrf_fuse(rankings: dict[str, list[Hit]], weights: dict[str, float], k: int = 60) -> dict[str, ScoredHit]:
    """Weighted RRF: score(d) = sum_r w_r / (k + rank_r(d)), ranks are 1-based."""
    fused: dict[str, ScoredHit] = {}
    for name, hits in rankings.items():
        w = weights.get(name, 1.0)
        for rank, h in enumerate(hits, start=1):
            sh = fused.get(h.chunk_id)
            if sh is None:
                sh = fused[h.chunk_id] = ScoredHit(hit=h, score=0.0, debug={"dense_rank": None, "sparse_rank": None})
            if sh.hit.dense is None and h.dense is not None:
                sh.hit.dense = h.dense
            sh.score += w / (k + rank)
            sh.debug[f"{name}_rank"] = rank
            sh.debug[f"{name}_score"] = round(h.score, 4)
    return fused


def title_match_fraction(query_tokens: set[str], title: str) -> float:
    tt = set(tokenize(title))
    if not tt:
        return 0.0
    return len(tt & query_tokens) / len(tt)


def _minmax(xs: list[float]) -> list[float]:
    if not xs:
        return xs
    lo, hi = min(xs), max(xs)
    if hi - lo < 1e-9:
        return [1.0] * len(xs)
    return [(x - lo) / (hi - lo) for x in xs]


def mmr_select(cands: list[ScoredHit], k: int, lam: float, max_per_doc: int) -> list[ScoredHit]:
    """Maximal Marginal Relevance with a per-document cap.

    relevance = normalised fused score; redundancy = max cosine to already selected chunks.
    Candidates without vectors fall back to pure relevance order.
    """
    if not cands:
        return []
    rel = _minmax([c.score for c in cands])
    vecs = []
    for c in cands:
        v = np.asarray(c.hit.dense, dtype=np.float32) if c.hit.dense is not None else None
        if v is not None:
            v = v / (np.linalg.norm(v) or 1.0)
        vecs.append(v)
    selected: list[int] = []
    per_doc: dict[str, int] = {}
    remaining = list(range(len(cands)))
    while remaining and len(selected) < k:
        best_i, best_val = None, -1e9
        for i in remaining:
            if per_doc.get(cands[i].hit.doc_id, 0) >= max_per_doc:
                continue
            red = 0.0
            if vecs[i] is not None:
                for j in selected:
                    if vecs[j] is not None:
                        red = max(red, float(vecs[i] @ vecs[j]))
            val = lam * rel[i] - (1 - lam) * red
            if val > best_val:
                best_i, best_val = i, val
        if best_i is None:
            break
        selected.append(best_i)
        remaining.remove(best_i)
        per_doc[cands[best_i].hit.doc_id] = per_doc.get(cands[best_i].hit.doc_id, 0) + 1
        cands[best_i].debug["mmr"] = round(best_val, 4)
    return [cands[i] for i in selected]


class CrossEncoder:
    def __init__(self, model_name: str, cache_dir: str | None = None):
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        self.name = model_name
        self._model = TextCrossEncoder(model_name=model_name, cache_dir=cache_dir)

    def score(self, query: str, texts: list[str]) -> list[float]:
        return [float(s) for s in self._model.rerank(query, texts, batch_size=32)]


class HybridRetriever:
    def __init__(self, store: VectorStore, embedder, bm25: BM25Encoder, cross_encoder: CrossEncoder | None = None):
        self.store = store
        self.embedder = embedder
        self.bm25 = bm25
        self.cross_encoder = cross_encoder

    def search(self, query: str, cfg: RetrievalConfig | None = None) -> tuple[list[ScoredHit], dict[str, float]]:
        cfg = cfg or RetrievalConfig()
        timings: dict[str, float] = {}
        rankings: dict[str, list[Hit]] = {}
        t0 = time.perf_counter()
        if cfg.mode in ("dense", "hybrid"):
            qv = self.embedder.embed_query(query)
            timings["embed_ms"] = (time.perf_counter() - t0) * 1000
            t1 = time.perf_counter()
            rankings["dense"] = self.store.dense_search(qv, cfg.candidates, cfg.sources, cfg.doc_ids, cfg.collection)
            timings["dense_ms"] = (time.perf_counter() - t1) * 1000
        if cfg.mode in ("sparse", "hybrid"):
            t1 = time.perf_counter()
            rankings["sparse"] = self.store.sparse_search(
                self.bm25.encode_query(query), cfg.candidates, cfg.sources, cfg.doc_ids, cfg.collection
            )
            timings["sparse_ms"] = (time.perf_counter() - t1) * 1000

        fused = rrf_fuse(rankings, {"dense": cfg.dense_weight, "sparse": cfg.sparse_weight}, cfg.rrf_k)
        if cfg.title_boost > 0:
            qtok = set(tokenize(query))
            unit = 1.0 / (cfg.rrf_k + 1)
            for sh in fused.values():
                frac = title_match_fraction(qtok, sh.hit.title)
                if frac >= 0.999:  # whole title named in the query
                    sh.score += cfg.title_boost * unit
                    sh.debug["title_boost"] = 1
        cands = sorted(fused.values(), key=lambda s: s.score, reverse=True)

        if cfg.rerank and self.cross_encoder is not None and cands:
            t1 = time.perf_counter()
            pool = cands[: cfg.rerank_pool]
            ce = self.cross_encoder.score(query, [f"{c.hit.title}. {c.hit.text}" for c in pool])
            ce_n, fz_n = _minmax(ce), _minmax([c.score for c in pool])
            for c, raw, a, b in zip(pool, ce, ce_n, fz_n):
                c.debug["ce"] = round(raw, 4)
                c.debug["fused"] = round(c.score, 5)
                c.score = cfg.rerank_weight * a + (1 - cfg.rerank_weight) * b
            cands = sorted(pool, key=lambda s: s.score, reverse=True) + cands[cfg.rerank_pool :]
            timings["rerank_ms"] = (time.perf_counter() - t1) * 1000

        if cfg.mmr:
            out = mmr_select(cands[: max(cfg.top_k * 4, 20)], cfg.top_k, cfg.mmr_lambda, cfg.max_per_doc)
        else:
            out = cands[: cfg.top_k]
        for sh in out:
            sh.debug["final"] = round(sh.score, 5)
        timings["total_ms"] = (time.perf_counter() - t0) * 1000
        return out, timings
