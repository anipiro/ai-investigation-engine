"""PostgreSQL + pgvector backend (production: Neon serverless Postgres, provisioned through Vercel).

Same interface as the Qdrant VectorStore, so the retriever, ingestion and API do not change.

  chunks  dense halfvec(384) with an HNSW cosine index; sparse sparsevec holding our BM25
          term-frequency weights; chunk text + provenance columns; extra payload as jsonb
  docs    document registry (content hash, simhash, counts, metadata)
  term_df document frequency per hashed term, maintained on every insert/delete

Keyword search is exact Okapi BM25 computed in SQL as an inner product:
  score(q, d) = sum_t idf(t) * w(t, d) = <q_idf, d_tf>      (pgvector sparsevec, `<#>` operator)
where the query vector carries idf(t) = ln(1 + (N - df + 0.5) / (df + 0.5)) from term_df.
"""

from __future__ import annotations

import json
import logging
import math
import threading
from typing import Any

from engine.config import Settings
from engine.retrieval.sparse import SparseVec
from engine.retrieval.store import ChunkRecord, Hit

log = logging.getLogger(__name__)

SPARSE_DIM = 1_000_000_000  # pgvector sparsevec dimension limit; crc32 term ids are folded into it
_COLS = "chunk_id, doc_id, source, title, text, chunk_index, section, page, payload"


def fold(idx: int) -> int:
    return idx % (SPARSE_DIM - 1)


class PgVectorStore:
    def __init__(self, settings: Settings, dsn: str | None = None):
        from psycopg_pool import ConnectionPool

        self.s = settings
        self.dsn = dsn or (settings.database_url.get_secret_value() if settings.database_url else None)
        if not self.dsn:
            raise ValueError("DATABASE_URL is required for the pgvector backend")
        self.dim: int | None = None
        self.pool = ConnectionPool(self.dsn, min_size=1, max_size=4, kwargs={"autocommit": True, "prepare_threshold": None},  # pooled (PgBouncer) URLs
                                   configure=self._configure, open=True, timeout=30)
        self._n_docs: int | None = None
        self._lock = threading.Lock()
        # Interface compatibility with VectorStore (scripts refer to these names).
        self.chunks, self.docs = "chunks", "docs"

    @staticmethod
    def _configure(conn) -> None:
        from pgvector.psycopg import register_vector

        try:
            register_vector(conn)
        except Exception:  # extension not created yet (first run)
            pass

    # ---------- schema ----------
    def ensure_collections(self, dim: int, chunks_name: str | None = None) -> None:
        self.dim = dim
        with self.pool.connection() as c:
            c.execute("CREATE EXTENSION IF NOT EXISTS vector")
            from pgvector.psycopg import register_vector

            register_vector(c)
            c.execute(f"""
                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id text PRIMARY KEY, doc_id text NOT NULL, source text NOT NULL, title text NOT NULL,
                    text text NOT NULL, chunk_index int NOT NULL, section text, page int, payload jsonb NOT NULL DEFAULT '{{}}',
                    dense halfvec({dim}) NOT NULL, sparse sparsevec({SPARSE_DIM}))""")
            c.execute("CREATE INDEX IF NOT EXISTS chunks_doc_idx ON chunks (doc_id)")
            c.execute("CREATE INDEX IF NOT EXISTS chunks_source_idx ON chunks (source)")
            c.execute("CREATE TABLE IF NOT EXISTS docs (doc_id text PRIMARY KEY, source text NOT NULL, title text NOT NULL, "
                      "content_hash text NOT NULL, payload jsonb NOT NULL)")
            c.execute("CREATE INDEX IF NOT EXISTS docs_hash_idx ON docs (content_hash)")
            c.execute("CREATE INDEX IF NOT EXISTS docs_source_idx ON docs (source)")
            c.execute("CREATE TABLE IF NOT EXISTS term_df (term bigint PRIMARY KEY, df int NOT NULL)")

    def create_dense_index(self) -> None:
        """Build the HNSW index after bulk loading (much faster than maintaining it row by row)."""
        with self.pool.connection() as c:
            c.execute("SET maintenance_work_mem = '256MB'")
            c.execute("CREATE INDEX IF NOT EXISTS chunks_dense_hnsw ON chunks USING hnsw (dense halfvec_cosine_ops) "
                      "WITH (m = 16, ef_construction = 64)")

    # ---------- writes ----------
    @staticmethod
    def _sparse(vec: SparseVec | None):
        from pgvector import SparseVector

        if vec is None or not vec.indices:
            return None
        folded: dict[int, float] = {}
        for i, v in zip(vec.indices, vec.values):
            folded[fold(i)] = folded.get(fold(i), 0.0) + v
        return SparseVector(folded, SPARSE_DIM)

    def upsert_chunks(self, records: list[ChunkRecord], collection: str | None = None, wait: bool = True) -> None:
        from pgvector import HalfVector

        if not records:
            return
        rows, terms = [], []
        for r in records:
            p = r.payload()
            extra = {k: v for k, v in p.items() if k not in ("chunk_id", "doc_id", "source", "title", "text", "chunk_index", "section", "page")}
            rows.append((r.chunk_id, r.doc_id, r.source, r.title, r.text, r.chunk_index, r.section, r.page, json.dumps(extra),
                         HalfVector(r.dense), self._sparse(r.sparse)))
            if r.sparse:
                terms.extend({fold(i) for i in r.sparse.indices})
        with self.pool.connection() as c, c.transaction():
            # Replacing an existing chunk must first release its term counts.
            ids = [r.chunk_id for r in records]
            self._release_terms(c, "chunk_id = ANY(%s)", (ids,))
            c.execute("DELETE FROM chunks WHERE chunk_id = ANY(%s)", (ids,))
            with c.cursor() as cur:
                cur.executemany(f"INSERT INTO chunks ({_COLS}, dense, sparse) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)", rows)
            self._add_terms(c, terms, +1)
        self._n_docs = None

    @staticmethod
    def _add_terms(c, terms: list[int], sign: int) -> None:
        if not terms:
            return
        counts: dict[int, int] = {}
        for t in terms:
            counts[t] = counts.get(t, 0) + 1
        keys, vals = list(counts), [sign * v for v in counts.values()]
        c.execute("INSERT INTO term_df (term, df) SELECT * FROM unnest(%s::bigint[], %s::int[]) "
                  "ON CONFLICT (term) DO UPDATE SET df = term_df.df + EXCLUDED.df", (keys, vals))

    def _release_terms(self, c, where: str, params: tuple) -> None:
        rows = c.execute(f"SELECT sparse FROM chunks WHERE {where} AND sparse IS NOT NULL", params).fetchall()
        terms = [i for (sv,) in rows for i in sv.indices()]
        self._add_terms(c, terms, -1)

    def upsert_doc(self, doc: dict[str, Any]) -> None:
        self.upsert_docs([doc])

    def upsert_docs(self, docs: list[dict[str, Any]], wait: bool = True) -> None:
        if not docs:
            return
        with self.pool.connection() as c, c.cursor() as cur:
            cur.executemany(
                "INSERT INTO docs (doc_id, source, title, content_hash, payload) VALUES (%s,%s,%s,%s,%s) "
                "ON CONFLICT (doc_id) DO UPDATE SET source=EXCLUDED.source, title=EXCLUDED.title, "
                "content_hash=EXCLUDED.content_hash, payload=EXCLUDED.payload",
                [(d["doc_id"], d["source"], d["title"], d["content_hash"], json.dumps(d, default=str)) for d in docs])

    def delete_doc_chunks(self, doc_id: str) -> None:
        with self.pool.connection() as c, c.transaction():
            self._release_terms(c, "doc_id = %s", (doc_id,))
            c.execute("DELETE FROM chunks WHERE doc_id = %s", (doc_id,))
        self._n_docs = None

    def delete_doc(self, doc_id: str) -> None:
        self.delete_doc_chunks(doc_id)
        with self.pool.connection() as c:
            c.execute("DELETE FROM docs WHERE doc_id = %s", (doc_id,))

    # ---------- registry reads ----------
    def get_doc(self, doc_id: str) -> dict[str, Any] | None:
        with self.pool.connection() as c:
            row = c.execute("SELECT payload FROM docs WHERE doc_id = %s", (doc_id,)).fetchone()
        return row[0] if row else None

    def find_doc_by_hash(self, content_hash: str) -> dict[str, Any] | None:
        with self.pool.connection() as c:
            row = c.execute("SELECT payload FROM docs WHERE content_hash = %s LIMIT 1", (content_hash,)).fetchone()
        return row[0] if row else None

    def list_docs(self, source: str | None = None, title_query: str | None = None, limit: int = 50,
                  cursor: str | None = None) -> tuple[list[dict[str, Any]], str | None]:
        where, params = ["TRUE"], []
        if source:
            where.append("source = %s")
            params.append(source)
        if title_query:
            where.append("title ILIKE %s")
            params.append(f"%{title_query}%")
        if cursor:
            where.append("doc_id > %s")
            params.append(cursor)
        with self.pool.connection() as c:
            rows = c.execute(f"SELECT doc_id, payload FROM docs WHERE {' AND '.join(where)} ORDER BY doc_id LIMIT %s",
                             (*params, limit + 1)).fetchall()
        nxt = rows[limit - 1][0] if len(rows) > limit else None
        return [r[1] for r in rows[:limit]], nxt

    def count(self, which: str = "chunks", source: str | None = None) -> int:
        table = "chunks" if which == "chunks" else "docs"
        with self.pool.connection() as c:
            if source:
                return c.execute(f"SELECT count(*) FROM {table} WHERE source = %s", (source,)).fetchone()[0]
            return c.execute(f"SELECT count(*) FROM {table}").fetchone()[0]

    def _rows_to_hits(self, rows, with_score: bool = True) -> list[Hit]:
        hits = []
        for row in rows:
            chunk_id, doc_id, source, title, text, idx, section, page, payload = row[:9]
            dense = row[9].to_list() if len(row) > 9 and row[9] is not None else None
            score = float(row[10]) if with_score and len(row) > 10 else 0.0
            pl = {"chunk_id": chunk_id, "doc_id": doc_id, "source": source, "title": title, "text": text,
                  "chunk_index": idx, "section": section, "page": page, **(payload or {})}
            hits.append(Hit(chunk_id=chunk_id, doc_id=doc_id, title=title, source=source, text=text, score=score, payload=pl, dense=dense))
        return hits

    def doc_chunks(self, doc_id: str, limit: int = 200) -> list[Hit]:
        with self.pool.connection() as c:
            rows = c.execute(f"SELECT {_COLS} FROM chunks WHERE doc_id = %s ORDER BY chunk_index LIMIT %s", (doc_id, limit)).fetchall()
        return self._rows_to_hits(rows, with_score=False)

    def get_chunks(self, chunk_ids: list[str], with_vectors: bool = False) -> list[Hit]:
        if not chunk_ids:
            return []
        with self.pool.connection() as c:
            rows = c.execute(f"SELECT {_COLS}, dense FROM chunks WHERE chunk_id = ANY(%s)", (chunk_ids,)).fetchall()
        return self._rows_to_hits(rows, with_score=False)

    # ---------- search ----------
    @staticmethod
    def _filter(sources, doc_ids) -> tuple[str, list]:
        where, params = [], []
        if sources:
            where.append("source = ANY(%s)")
            params.append(list(sources))
        if doc_ids:
            where.append("doc_id = ANY(%s)")
            params.append(list(doc_ids))
        return (" WHERE " + " AND ".join(where)) if where else "", params

    def dense_search(self, vector: list[float], limit: int, sources: list[str] | None = None, doc_ids: list[str] | None = None,
                     collection: str | None = None, with_vectors: bool = True) -> list[Hit]:
        from pgvector import HalfVector

        where, params = self._filter(sources, doc_ids)
        q = HalfVector(vector)
        with self.pool.connection() as c, c.transaction():
            c.execute("SET LOCAL hnsw.ef_search = 100")
            c.execute("SET LOCAL hnsw.iterative_scan = relaxed_order")  # keeps filtered queries from returning too few rows
            rows = c.execute(f"SELECT {_COLS}, dense, 1 - (dense <=> %s) AS score FROM chunks{where} "
                             f"ORDER BY dense <=> %s LIMIT %s", (q, *params, q, limit)).fetchall()
        return self._rows_to_hits(rows)

    def n_docs(self) -> int:
        if self._n_docs is None:
            self._n_docs = self.count("chunks")
        return self._n_docs

    def sparse_search(self, vec: SparseVec, limit: int, sources: list[str] | None = None, doc_ids: list[str] | None = None,
                      collection: str | None = None, with_vectors: bool = True) -> list[Hit]:
        from pgvector import SparseVector

        if not vec.indices:
            return []
        terms = sorted({fold(i) for i in vec.indices})
        with self.pool.connection() as c:
            df = dict(c.execute("SELECT term, df FROM term_df WHERE term = ANY(%s) AND df > 0", (terms,)).fetchall())
            n = max(self.n_docs(), 1)
            weights = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}
            if not weights:
                return []
            q = SparseVector(weights, SPARSE_DIM)
            where, params = self._filter(sources, doc_ids)
            # Exact BM25: <#> is the negative inner product. Only rows sharing a term can score > 0.
            rows = c.execute(f"SELECT {_COLS}, dense, -(sparse <#> %s) AS score FROM chunks{where} "
                             f"{'AND' if where else 'WHERE'} sparse IS NOT NULL ORDER BY sparse <#> %s LIMIT %s",
                             (q, *params, q, limit)).fetchall()
        return [h for h in self._rows_to_hits(rows) if h.score > 0]

    def ping(self) -> bool:
        try:
            with self.pool.connection() as c:
                c.execute("SELECT 1")
            return True
        except Exception:
            return False
