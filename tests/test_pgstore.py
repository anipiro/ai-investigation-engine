"""Integration tests for the pgvector backend. Opt-in: set PG_TEST_DSN to a Postgres with pgvector.

Each run works in its own throw-away schema, so it is safe to point at a development database.
"""

import os
import uuid

import pytest

from engine.config import Settings
from engine.services import build_services

DSN = os.environ.get("PG_TEST_DSN")
pytestmark = pytest.mark.skipif(not DSN, reason="PG_TEST_DSN not set")


@pytest.fixture
def pg_services(llm):
    import psycopg

    from engine.retrieval.pgstore import PgVectorStore
    from tests.conftest import HashEmbedder

    schema = f"aie_test_{uuid.uuid4().hex[:8]}"
    with psycopg.connect(DSN, autocommit=True) as c:
        c.execute("CREATE EXTENSION IF NOT EXISTS vector")
        c.execute(f"CREATE SCHEMA {schema}")
    sep = "&" if "?" in DSN else "?"
    dsn = f"{DSN}{sep}options=-csearch_path%3D{schema},public"
    s = Settings(_env_file=None, store_backend="pgvector", enable_cross_encoder=False, gemini_api_key="k")
    store = PgVectorStore(s, dsn=dsn)
    try:
        yield build_services(s, store=store, embedder=HashEmbedder(), llm=llm)
    finally:
        store.pool.close()
        with psycopg.connect(DSN, autocommit=True) as c:
            c.execute(f"DROP SCHEMA {schema} CASCADE")


def test_pg_lifecycle_and_search(pg_services):
    ing, store, ret = pg_services.ingestion, pg_services.store, pg_services.retriever
    from engine.retrieval.hybrid import RetrievalConfig

    assert ing.ingest(doc_id="a", title="Ledger runbook", source="upload",
                      text="The ledger pool must stay above 150 connections.").status == "created"
    ing.ingest(doc_id="b", title="Change CHG-1187", source="upload", text="CHG-1187 reduced the pool size from 200 to 80.")
    ing.ingest(doc_id="c", title="Film", source="hotpotqa", text="Ed Wood is a 1994 American film directed by Tim Burton.")
    assert ing.ingest(doc_id="d", title="Copy", source="upload", text="CHG-1187 reduced the pool size from 200 to 80.").status == "duplicate"

    for mode in ("dense", "sparse", "hybrid"):
        hits, _ = ret.search("CHG-1187 pool size", RetrievalConfig(mode=mode, top_k=2, rerank=False))
        assert hits[0].hit.doc_id == "b", mode
    hits, _ = ret.search("pool", RetrievalConfig(mode="sparse", top_k=5, rerank=False, sources=["hotpotqa"]))
    assert hits == []

    docs, nxt = store.list_docs(source="upload", limit=1)
    assert len(docs) == 1 and nxt is not None
    assert store.count("docs") == 3

    assert ing.ingest(doc_id="b", title="Change CHG-1187", source="upload", text="CHG-1187 was rolled back.").status == "updated"
    store.delete_doc("a")
    assert store.get_doc("a") is None and store.doc_chunks("a") == []
    with store.pool.connection() as c:
        # Document frequencies follow deletes and updates: "150" only appeared in the deleted doc.
        from engine.retrieval.pgstore import fold
        from engine.retrieval.sparse import term_index

        df = c.execute("SELECT df FROM term_df WHERE term = %s", (fold(term_index("150")),)).fetchone()
        assert df is None or df[0] == 0
