"""Ingestion pipeline: parsed text -> dedup check -> chunks -> dense + sparse vectors -> Qdrant.

Create / update / delete semantics (keyed by doc_id):
  - same doc_id, same content hash  -> "unchanged" (no work)
  - same doc_id, different content  -> "updated"   (old chunks deleted, new ones indexed)
  - new doc_id, content hash already indexed under another id -> "duplicate" (not indexed,
    points to the original)
  - otherwise -> "created"
Near-duplicates (SimHash distance <= 3 against the best dense match) are indexed but reported.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from typing import Any, Iterable

from engine.ingestion.chunking import chunk_text
from engine.ingestion.dedup import content_hash, hamming, simhash64
from engine.retrieval.sparse import BM25Encoder
from engine.retrieval.store import ChunkRecord, VectorStore
from engine.security import injection_signals

log = logging.getLogger(__name__)


def slugify(s: str, max_len: int = 80) -> str:
    s = re.sub(r"[^0-9A-Za-z]+", "-", s).strip("-").lower()
    return s[:max_len] or "doc"


def embedding_text(title: str, section: str | None, text: str) -> str:
    """Contextual chunk header: the title (and section) travel with every chunk."""
    head = title if not section else f"{title} — {section}"
    return f"{head}\n{text}"


@dataclass
class IngestResult:
    doc_id: str
    status: str  # created | updated | unchanged | duplicate
    n_chunks: int = 0
    duplicate_of: str | None = None
    near_duplicate_of: str | None = None
    suspicious_chunks: int = 0
    ms: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


class IngestionService:
    def __init__(self, store: VectorStore, embedder, bm25: BM25Encoder, strategy: str = "recursive",
                 max_words: int = 180, overlap: int = 30):
        self.store, self.embedder, self.bm25 = store, embedder, bm25
        self.strategy, self.max_words, self.overlap = strategy, max_words, overlap

    def build_chunks(self, doc_id: str, title: str, text: str, source: str, extra: dict[str, Any] | None = None,
                     strategy: str | None = None) -> list[ChunkRecord]:
        chunks = chunk_text(text, strategy or self.strategy, self.max_words, self.overlap)
        records = []
        seen_hashes: set[str] = set()
        for c in chunks:
            h = content_hash(c.text)
            if h in seen_hashes:  # identical repeated block inside one document
                continue
            seen_hashes.add(h)
            payload_extra = dict(extra or {})
            signals = injection_signals(c.text)
            if signals:
                payload_extra["suspicious"] = True
                payload_extra["injection_signals"] = signals
            records.append(ChunkRecord(
                chunk_id=f"{doc_id}#{c.index}", doc_id=doc_id, text=c.text, title=title, source=source,
                chunk_index=c.index, section=c.section, page=c.page, extra=payload_extra,
            ))
        return records

    def vectorize(self, records: list[ChunkRecord]) -> None:
        texts = [embedding_text(r.title, r.section, r.text) for r in records]
        for r, v, t in zip(records, self.embedder.embed_documents(texts), texts):
            r.dense = v
            r.sparse = self.bm25.encode_document(t)

    def ingest(self, *, doc_id: str, title: str, text: str, source: str, fmt: str = "text",
               metadata: dict[str, Any] | None = None, url: str | None = None) -> IngestResult:
        t0 = time.perf_counter()
        chash = content_hash(text)
        existing = self.store.get_doc(doc_id)
        if existing and existing.get("content_hash") == chash:
            return IngestResult(doc_id=doc_id, status="unchanged", n_chunks=existing.get("n_chunks", 0))
        if not existing:
            dup = self.store.find_doc_by_hash(chash)
            if dup:
                return IngestResult(doc_id=doc_id, status="duplicate", duplicate_of=dup["doc_id"])

        records = self.build_chunks(doc_id, title, text, source, {"fmt": fmt, "url": url})
        if not records:
            raise ValueError("document produced no chunks")
        self.vectorize(records)

        near_dup = None
        sh = simhash64(text)
        if records[0].dense is not None:
            for h in self.store.dense_search(records[0].dense, 3, with_vectors=False):
                if h.doc_id == doc_id:
                    continue
                other = self.store.get_doc(h.doc_id)
                if other and other.get("simhash") and hamming(sh, int(other["simhash"], 16)) <= 3:
                    near_dup = h.doc_id
                    break

        if existing:
            self.store.delete_doc_chunks(doc_id)
        self.store.upsert_chunks(records)
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        self.store.upsert_doc({
            "doc_id": doc_id, "title": title, "source": source, "fmt": fmt, "url": url,
            "content_hash": chash, "simhash": f"{sh:016x}", "n_chunks": len(records),
            "n_words": len(text.split()), "metadata": metadata or {},
            "created_at": existing.get("created_at", now) if existing else now, "updated_at": now,
            "near_duplicate_of": near_dup,
        })
        return IngestResult(
            doc_id=doc_id, status="updated" if existing else "created", n_chunks=len(records),
            near_duplicate_of=near_dup, suspicious_chunks=sum(1 for r in records if r.extra.get("suspicious")),
            ms=(time.perf_counter() - t0) * 1000,
        )

    def bulk_ingest(self, docs: Iterable[dict[str, Any]], batch_chunks: int = 512, collection: str | None = None,
                    strategy: str | None = None, register_docs: bool = True, progress_every: int = 5000) -> dict[str, int]:
        """Fast path for the seed corpus: corpus builder already removed exact duplicates."""
        stats = {"docs": 0, "chunks": 0}
        buf: list[ChunkRecord] = []
        doc_buf: list[dict[str, Any]] = []
        t0 = time.perf_counter()

        def flush() -> None:
            if not buf:
                return
            self.vectorize(buf)
            self.store.upsert_chunks(buf, collection=collection, wait=False)
            stats["chunks"] += len(buf)
            buf.clear()
            if register_docs:
                self.store.upsert_docs(doc_buf, wait=False)
            doc_buf.clear()

        for d in docs:
            recs = self.build_chunks(d["doc_id"], d["title"], d["text"], d["source"],
                                     {"fmt": d.get("fmt", "text"), "url": d.get("url")}, strategy=strategy)
            buf.extend(recs)
            doc_buf.append({
                "doc_id": d["doc_id"], "title": d["title"], "source": d["source"], "fmt": d.get("fmt", "text"),
                "url": d.get("url"), "content_hash": content_hash(d["text"]), "simhash": f"{simhash64(d['text']):016x}",
                "n_chunks": len(recs), "n_words": len(d["text"].split()), "metadata": d.get("metadata", {}),
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            })
            stats["docs"] += 1
            if len(buf) >= batch_chunks:
                flush()
            if stats["docs"] % progress_every == 0:
                log.info("ingested %d docs / %d chunks (%.0fs)", stats["docs"], stats["chunks"], time.perf_counter() - t0)
        flush()
        return stats
