"""Composition root: builds the store, models, retriever, ingestion service and agent once."""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field

from engine.agent.investigator import Investigator
from engine.cache import TTLCache
from engine.config import Settings
from engine.ingestion.pipeline import IngestionService
from engine.llm.gemini import GeminiClient
from engine.retrieval.embeddings import FastEmbedder
from engine.retrieval.hybrid import CrossEncoder, HybridRetriever, RetrievalConfig
from engine.retrieval.sparse import BM25Encoder
from engine.retrieval.store import VectorStore

log = logging.getLogger(__name__)


@dataclass
class Services:
    settings: Settings
    store: VectorStore
    embedder: object
    retriever: HybridRetriever
    ingestion: IngestionService
    llm: GeminiClient
    investigator: Investigator
    answer_cache: TTLCache
    corpus_version: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def bump_version(self) -> None:
        with self._lock:
            self.corpus_version += 1

    def retrieval_config(self, **overrides) -> RetrievalConfig:
        s = self.settings
        base = dict(candidates=s.retrieve_candidates, top_k=s.final_top_k, rrf_k=s.rrf_k, dense_weight=s.dense_weight,
                    sparse_weight=s.sparse_weight, mmr_lambda=s.mmr_lambda, rerank=self.retriever.cross_encoder is not None)
        base.update({k: v for k, v in overrides.items() if v is not None})
        return RetrievalConfig(**base)


def build_services(settings: Settings, store: VectorStore | None = None, embedder=None, cross_encoder=None,
                   llm: GeminiClient | None = None) -> Services:
    store = store or VectorStore(settings)
    embedder = embedder or FastEmbedder(settings.embedding_model, cache_dir=settings.model_cache_dir)
    if cross_encoder is None and settings.enable_cross_encoder:
        try:
            cross_encoder = CrossEncoder(settings.reranker_model, cache_dir=settings.model_cache_dir)
        except Exception as exc:  # model download failure must not take the API down
            log.warning("cross-encoder unavailable, continuing without it: %s", exc)
            cross_encoder = None
    store.ensure_collections(embedder.dim)
    bm25 = BM25Encoder()
    retriever = HybridRetriever(store, embedder, bm25, cross_encoder)
    ingestion = IngestionService(store, embedder, bm25, settings.chunk_strategy, settings.chunk_max_words, settings.chunk_overlap_words)
    llm = llm or GeminiClient(
        settings.gemini_api_key.get_secret_value() if settings.gemini_api_key else None,
        settings.gemini_model, settings.gemini_fallback_model, settings.llm_timeout_s, settings.llm_max_retries,
        settings.llm_min_interval_s,
        thinking_levels={settings.gemini_model: settings.gemini_thinking_level,
                         settings.gemini_fallback_model: settings.gemini_fallback_thinking_level},
    )
    svc = Services(settings=settings, store=store, embedder=embedder, retriever=retriever, ingestion=ingestion, llm=llm,
                   investigator=None, answer_cache=TTLCache(512, settings.cache_ttl_s))  # type: ignore[arg-type]
    svc.investigator = Investigator(retriever, llm, settings.agent_max_steps, settings.agent_max_context_chunks,
                                    svc.retrieval_config())
    return svc
