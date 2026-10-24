"""Dense embedding backends.

Default: a local ONNX model through fastembed (BAAI/bge-small-en-v1.5, 384-d). Local embeddings
keep ingestion of 50k+ records free and independent of LLM rate limits; the same model runs on
the free Hugging Face Space CPU at query time. Other fastembed models (e.g. MiniLM) can be
selected for the embedding-model comparison in the evaluation.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from typing import Protocol

import numpy as np


class Embedder(Protocol):
    name: str
    dim: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class FastEmbedder:
    def __init__(self, model_name: str, cache_dir: str | None = None, query_cache_size: int = 2048):
        from fastembed import TextEmbedding

        self.name = model_name
        self._model = TextEmbedding(model_name=model_name, cache_dir=cache_dir)
        self.dim = len(next(iter(self._model.query_embed(["probe"]))))
        self._qcache: OrderedDict[str, list[float]] = OrderedDict()
        self._qcache_size = query_cache_size
        self._lock = threading.Lock()
        self.cache_hits = 0
        self.cache_misses = 0

    def embed_documents(self, texts: list[str], batch_size: int = 16) -> list[list[float]]:
        # Small batches: attention memory grows with batch * seq_len^2 (free-tier RAM is tight).
        return [v.tolist() for v in self._model.passage_embed(texts, batch_size=batch_size)]

    def embed_query(self, text: str) -> list[float]:
        with self._lock:
            if text in self._qcache:
                self._qcache.move_to_end(text)
                self.cache_hits += 1
                return self._qcache[text]
        vec = next(iter(self._model.query_embed([text]))).tolist()
        with self._lock:
            self.cache_misses += 1
            self._qcache[text] = vec
            if len(self._qcache) > self._qcache_size:
                self._qcache.popitem(last=False)
        return vec


class DiskCachedEmbedder:
    """Wraps an embedder with a SQLite cache keyed by sha1(model, text).

    Used by the offline ingestion scripts: re-indexing (another chunking strategy, a different
    Qdrant target, a crash mid-run) only embeds texts that were never embedded before.
    Misses are embedded in length-sorted order, which minimises padding inside each batch.
    """

    def __init__(self, inner, path: str):
        import sqlite3

        self.inner, self.name, self.dim = inner, inner.name, inner.dim
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.execute("CREATE TABLE IF NOT EXISTS emb (k TEXT PRIMARY KEY, v BLOB)")
        self.hits = self.misses = 0

    def _key(self, text: str) -> str:
        import hashlib

        return hashlib.sha1(f"{self.name}\x00{text}".encode("utf-8")).hexdigest()

    def embed_documents(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        keys = [self._key(t) for t in texts]
        found: dict[str, list[float]] = {}
        for i in range(0, len(keys), 500):
            part = keys[i : i + 500]
            q = f"SELECT k, v FROM emb WHERE k IN ({','.join('?' * len(part))})"
            for k, v in self._db.execute(q, part):
                found[k] = np.frombuffer(v, dtype=np.float32).tolist()
        missing = sorted({k: t for k, t in zip(keys, texts) if k not in found}.items(), key=lambda kv: len(kv[1]))
        self.hits += len(texts) - len(missing)
        self.misses += len(missing)
        if missing:
            vecs = self.inner.embed_documents([t for _, t in missing], batch_size=batch_size)
            rows = [(k, np.asarray(v, dtype=np.float32).tobytes()) for (k, _), v in zip(missing, vecs)]
            self._db.executemany("INSERT OR REPLACE INTO emb VALUES (?, ?)", rows)
            self._db.commit()
            for (k, _), v in zip(missing, vecs):
                found[k] = v
        return [found[k] for k in keys]

    def embed_query(self, text: str) -> list[float]:
        return self.inner.embed_query(text)


def cosine(a: list[float] | np.ndarray, b: list[float] | np.ndarray) -> float:
    a, b = np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b)) or 1.0
    return float(a @ b) / denom
