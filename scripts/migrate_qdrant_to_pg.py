"""Copy an indexed corpus from Qdrant to Postgres/pgvector without re-embedding anything.

The heavy work (embedding 66k+ chunks) is done once into the local Qdrant index; production runs on
Neon Postgres. This script streams chunks (dense + BM25 sparse vectors + payload) and the document
registry across, recomputes term document frequencies, then builds the HNSW index.

  uv run python scripts/migrate_qdrant_to_pg.py --dsn "$DATABASE_URL_UNPOOLED" [--recreate]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pgvector import HalfVector, SparseVector  # noqa: E402

from engine.config import Settings  # noqa: E402
from engine.retrieval.pgstore import SPARSE_DIM, PgVectorStore, fold  # noqa: E402
from engine.retrieval.store import VectorStore  # noqa: E402

CORE = ("chunk_id", "doc_id", "source", "title", "text", "chunk_index", "section", "page")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dsn", required=True)
    ap.add_argument("--recreate", action="store_true")
    ap.add_argument("--batch", type=int, default=500)
    args = ap.parse_args()

    src = VectorStore(Settings(store_backend="qdrant"))
    dst = PgVectorStore(Settings(store_backend="pgvector"), dsn=args.dsn)
    if args.recreate:
        with dst.pool.connection() as c:
            c.execute("DROP TABLE IF EXISTS chunks, docs, term_df")
    dim = src.client.get_collection(src.chunks).config.params.vectors["dense"].size
    dst.ensure_collections(dim)

    t0 = time.perf_counter()
    df: Counter = Counter()
    offset, n = None, 0
    while True:
        pts, offset = src.client.scroll(src.chunks, limit=args.batch, offset=offset, with_payload=True, with_vectors=True)
        rows = []
        for p in pts:
            pl = p.payload or {}
            sp = p.vector.get("bm25")
            sparse = None
            if sp is not None and sp.indices:
                folded: dict[int, float] = {}
                for i, v in zip(sp.indices, sp.values):
                    folded[fold(i)] = folded.get(fold(i), 0.0) + v
                df.update(folded.keys())
                sparse = SparseVector(folded, SPARSE_DIM)
            extra = {k: v for k, v in pl.items() if k not in CORE}
            rows.append((pl["chunk_id"], pl["doc_id"], pl["source"], pl["title"], pl["text"], pl.get("chunk_index", 0),
                         pl.get("section"), pl.get("page"), json.dumps(extra), HalfVector(p.vector["dense"]), sparse))
        if rows:
            with dst.pool.connection() as c, c.cursor() as cur:
                cur.executemany(
                    "INSERT INTO chunks (chunk_id, doc_id, source, title, text, chunk_index, section, page, payload, dense, sparse) "
                    "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) ON CONFLICT (chunk_id) DO NOTHING", rows)
        n += len(rows)
        print(f"chunks: {n} ({time.perf_counter() - t0:.0f}s)", flush=True)
        if offset is None:
            break

    with dst.pool.connection() as c:
        c.execute("TRUNCATE term_df")
        items = list(df.items())
        for i in range(0, len(items), 20000):
            part = items[i : i + 20000]
            c.execute("INSERT INTO term_df (term, df) SELECT * FROM unnest(%s::bigint[], %s::int[])",
                      ([k for k, _ in part], [v for _, v in part]))
    print(f"term_df: {len(df)} terms")

    offset, m = None, 0
    while True:
        pts, offset = src.client.scroll(src.docs, limit=1000, offset=offset, with_payload=True)
        dst.upsert_docs([p.payload for p in pts if p.payload])
        m += len(pts)
        if offset is None:
            break
    print(f"docs: {m}")

    t1 = time.perf_counter()
    dst.create_dense_index()
    with dst.pool.connection() as c:
        c.execute("ANALYZE chunks")
        size = c.execute("SELECT pg_size_pretty(pg_database_size(current_database()))").fetchone()[0]
    print(f"HNSW index built in {time.perf_counter() - t1:.0f}s; database size {size}; total {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main()
