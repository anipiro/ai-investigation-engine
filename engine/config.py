"""Runtime configuration, loaded from environment variables (and a local .env file).

Secrets (API keys) are only ever read from the environment; they are never logged
or returned by the API.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- LLM (Google Gemini) ---
    gemini_api_key: SecretStr | None = None
    gemini_model: str = "gemini-3.5-flash"
    gemini_fallback_model: str = "gemini-3.5-flash-lite"
    gemini_thinking_level: str = "low"
    gemini_fallback_thinking_level: str = "minimal"
    llm_timeout_s: float = 60.0
    llm_max_retries: int = 4
    # Free tier is rate limited per minute; we pace calls client-side.
    llm_min_interval_s: float = 0.0

    # --- Vector store ---
    # "qdrant" (local server / Qdrant Cloud / embedded) or "pgvector" (Neon Postgres in production)
    store_backend: Literal["qdrant", "pgvector"] = "qdrant"
    database_url: SecretStr | None = None
    qdrant_url: str | None = None  # e.g. https://xyz.cloud.qdrant.io:6333 ; None -> embedded local mode
    qdrant_api_key: SecretStr | None = None
    qdrant_path: str = "data/qdrant_local"  # used only when qdrant_url is empty
    collection_prefix: str = "aie"

    # --- Embeddings / reranking (local ONNX models via fastembed) ---
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    reranker_model: str = "Xenova/ms-marco-MiniLM-L-6-v2"
    enable_cross_encoder: bool = True
    model_cache_dir: str = "data/models"

    # --- Chunking ---
    chunk_strategy: Literal["recursive", "fixed"] = "recursive"
    chunk_max_words: int = 180
    chunk_overlap_words: int = 30

    # --- Retrieval ---
    retrieve_candidates: int = 40  # per retriever, before fusion
    final_top_k: int = 8
    rrf_k: int = 60
    dense_weight: float = 1.0
    sparse_weight: float = 1.0
    mmr_lambda: float = 0.75
    min_evidence_score: float = 0.0  # cross-encoder logit below which evidence is "weak"

    # --- Agent ---
    agent_max_steps: int = 4
    agent_max_context_chunks: int = 24

    # --- API / ops ---
    admin_token: SecretStr | None = None  # required to modify protected (seed corpus) documents
    protected_sources: list[str] = Field(default_factory=lambda: ["hotpotqa", "postmortems", "northwind"])
    max_upload_mb: int = 10
    ask_rate_per_minute: int = 12
    cache_ttl_s: int = 3600
    log_level: str = "INFO"

    def collection(self, name: str) -> str:
        return f"{self.collection_prefix}_{name}"


@lru_cache
def get_settings() -> Settings:
    return Settings()
