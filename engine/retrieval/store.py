"""Qdrant persistence layer.

Two collections:
  <prefix>_chunks  named dense vector "dense" (cosine) + sparse vector "bm25" (IDF modifier);
                   payload holds the chunk text and its provenance (doc_id, title, source, page...).
  <prefix>_docs    document registry (one point per document, 1-d placeholder vector);
                   used for listing, de-duplication (content_hash) and update detection.

Works against Qdrant Cloud (QDRANT_URL) or embedded local mode (QDRANT_PATH) with the same code.
"""

from __future__ import annotations

import logging
import uuid
import warnings
from dataclasses import dataclass, field
from typing import Any

from qdrant_client import QdrantClient, models

from engine.config import Settings
from engine.retrieval.sparse import SparseVec

log = logging.getLogger(__name__)
# Embedded (local) Qdrant ignores payload indexes; the same code creates them on the server.
warnings.filterwarnings("ignore", message="Payload indexes have no effect")


def point_id(key: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, key))


@dataclass
class ChunkRecord:
    chunk_id: str
    doc_id: str
    text: str
    title: str
    source: str
    chunk_index: int
    dense: list[float] | None = None
    sparse: SparseVec | None = None
    section: str | None = None
    page: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def payload(self) -> dict[str, Any]:
        p = {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "text": self.text,
            "title": self.title,
            "source": self.source,
            "chunk_index": self.chunk_index,
            "section": self.section,
            "page": self.page,
        }
        p.update(self.extra)
        return p


@dataclass
class Hit:
    chunk_id: str
    doc_id: str
    title: str
    source: str
    text: str
    score: float
    payload: dict[str, Any]
    dense: list[float] | None = None


def _to_hit(p: models.ScoredPoint | models.Record, score: float | None = None) -> Hit:
    pl = p.payload or {}
    dense = None
    if getattr(p, "vector", None):
        v = p.vector
        dense = v.get("dense") if isinstance(v, dict) else v
    return Hit(
        chunk_id=pl.get("chunk_id", str(p.id)),
        doc_id=pl.get("doc_id", ""),
        title=pl.get("title", ""),
        source=pl.get("source", ""),
        text=pl.get("text", ""),
        score=float(score if score is not None else getattr(p, "score", 0.0) or 0.0),
        payload=pl,
        dense=dense,
    )


class VectorStore:
    def __init__(self, settings: Settings, client: QdrantClient | None = None):
        self.s = settings
        if client is not None:
            self.client = client
        elif settings.qdrant_url:
            self.client = QdrantClient(
                url=settings.qdrant_url,
                api_key=settings.qdrant_api_key.get_secret_value() if settings.qdrant_api_key else None,
                timeout=60,
            )
        else:
            self.client = QdrantClient(path=settings.qdrant_path)
        self.chunks = settings.collection("chunks")
        self.docs = settings.collection("docs")

    # ---------- schema ----------
    def ensure_collections(self, dim: int, chunks_name: str | None = None) -> None:
        chunks = chunks_name or self.chunks
        if not self.client.collection_exists(chunks):
            self.client.create_collection(
                chunks,
                vectors_config={"dense": models.VectorParams(size=dim, distance=models.Distance.COSINE, on_disk=True)},
                sparse_vectors_config={"bm25": models.SparseVectorParams(modifier=models.Modifier.IDF)},
                hnsw_config=models.HnswConfigDiff(m=16, ef_construct=100),
                optimizers_config=models.OptimizersConfigDiff(default_segment_number=2),
            )
            for fld in ("doc_id", "source"):
                self.client.create_payload_index(chunks, fld, models.PayloadSchemaType.KEYWORD)
            log.info("created collection %s (dim=%d)", chunks, dim)
        if not self.client.collection_exists(self.docs):
            self.client.create_collection(self.docs, vectors_config=models.VectorParams(size=1, distance=models.Distance.DOT))
            for fld in ("doc_id", "source", "content_hash"):
                self.client.create_payload_index(self.docs, fld, models.PayloadSchemaType.KEYWORD)
            self.client.create_payload_index(
                self.docs,
                "title",
                models.TextIndexParams(type=models.TextIndexType.TEXT, tokenizer=models.TokenizerType.WORD, lowercase=True),
            )

    # ---------- writes ----------
    def upsert_chunks(self, records: list[ChunkRecord], collection: str | None = None, wait: bool = True) -> None:
        points = []
        for r in records:
            vec: dict[str, Any] = {}
            if r.dense is not None:
                vec["dense"] = r.dense
            if r.sparse is not None and r.sparse.indices:
                vec["bm25"] = models.SparseVector(indices=r.sparse.indices, values=r.sparse.values)
            points.append(models.PointStruct(id=point_id(r.chunk_id), vector=vec, payload=r.payload()))
        if points:
            self.client.upsert(collection or self.chunks, points=points, wait=wait)

    def upsert_doc(self, doc: dict[str, Any]) -> None:
        self.client.upsert(self.docs, points=[models.PointStruct(id=point_id(doc["doc_id"]), vector=[1.0], payload=doc)])

    def upsert_docs(self, docs: list[dict[str, Any]], wait: bool = True) -> None:
        if docs:
            self.client.upsert(
                self.docs,
                points=[models.PointStruct(id=point_id(d["doc_id"]), vector=[1.0], payload=d) for d in docs],
                wait=wait,
            )

    def delete_doc(self, doc_id: str) -> None:
        flt = models.Filter(must=[models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id))])
        self.client.delete(self.chunks, points_selector=models.FilterSelector(filter=flt), wait=True)
        self.client.delete(self.docs, points_selector=models.PointIdsList(points=[point_id(doc_id)]), wait=True)

    def delete_doc_chunks(self, doc_id: str) -> None:
        flt = models.Filter(must=[models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id))])
        self.client.delete(self.chunks, points_selector=models.FilterSelector(filter=flt), wait=True)

    # ---------- registry reads ----------
    def get_doc(self, doc_id: str) -> dict[str, Any] | None:
        res = self.client.retrieve(self.docs, ids=[point_id(doc_id)], with_payload=True)
        return res[0].payload if res else None

    def find_doc_by_hash(self, content_hash: str) -> dict[str, Any] | None:
        flt = models.Filter(must=[models.FieldCondition(key="content_hash", match=models.MatchValue(value=content_hash))])
        pts, _ = self.client.scroll(self.docs, scroll_filter=flt, limit=1, with_payload=True)
        return pts[0].payload if pts else None

    def list_docs(
        self, source: str | None = None, title_query: str | None = None, limit: int = 50, cursor: str | None = None
    ) -> tuple[list[dict[str, Any]], str | None]:
        must = []
        if source:
            must.append(models.FieldCondition(key="source", match=models.MatchValue(value=source)))
        if title_query:
            must.append(models.FieldCondition(key="title", match=models.MatchText(text=title_query)))
        pts, nxt = self.client.scroll(
            self.docs, scroll_filter=models.Filter(must=must) if must else None, limit=limit, offset=cursor, with_payload=True
        )
        return [p.payload for p in pts], (str(nxt) if nxt else None)

    def count(self, which: str = "chunks", source: str | None = None) -> int:
        coll = self.chunks if which == "chunks" else self.docs
        flt = None
        if source:
            flt = models.Filter(must=[models.FieldCondition(key="source", match=models.MatchValue(value=source))])
        return self.client.count(coll, count_filter=flt, exact=True).count

    def doc_chunks(self, doc_id: str, limit: int = 200) -> list[Hit]:
        flt = models.Filter(must=[models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id))])
        pts, _ = self.client.scroll(self.chunks, scroll_filter=flt, limit=limit, with_payload=True)
        hits = [_to_hit(p, 0.0) for p in pts]
        return sorted(hits, key=lambda h: h.payload.get("chunk_index", 0))

    def get_chunks(self, chunk_ids: list[str], with_vectors: bool = False) -> list[Hit]:
        if not chunk_ids:
            return []
        res = self.client.retrieve(
            self.chunks, ids=[point_id(c) for c in chunk_ids], with_payload=True, with_vectors=["dense"] if with_vectors else False
        )
        return [_to_hit(p, 0.0) for p in res]

    # ---------- search ----------
    @staticmethod
    def _filter(sources: list[str] | None, doc_ids: list[str] | None) -> models.Filter | None:
        must = []
        if sources:
            must.append(models.FieldCondition(key="source", match=models.MatchAny(any=sources)))
        if doc_ids:
            must.append(models.FieldCondition(key="doc_id", match=models.MatchAny(any=doc_ids)))
        return models.Filter(must=must) if must else None

    def dense_search(
        self, vector: list[float], limit: int, sources: list[str] | None = None, doc_ids: list[str] | None = None,
        collection: str | None = None, with_vectors: bool = True,
    ) -> list[Hit]:
        res = self.client.query_points(
            collection or self.chunks, query=vector, using="dense", limit=limit,
            query_filter=self._filter(sources, doc_ids), with_payload=True,
            with_vectors=["dense"] if with_vectors else False,
        )
        return [_to_hit(p) for p in res.points]

    def sparse_search(
        self, vec: SparseVec, limit: int, sources: list[str] | None = None, doc_ids: list[str] | None = None,
        collection: str | None = None, with_vectors: bool = True,
    ) -> list[Hit]:
        if not vec.indices:
            return []
        res = self.client.query_points(
            collection or self.chunks, query=models.SparseVector(indices=vec.indices, values=vec.values), using="bm25",
            limit=limit, query_filter=self._filter(sources, doc_ids), with_payload=True,
            with_vectors=["dense"] if with_vectors else False,
        )
        return [_to_hit(p) for p in res.points]

    def ping(self) -> bool:
        try:
            self.client.get_collections()
            return True
        except Exception:  # pragma: no cover - network dependent
            return False
