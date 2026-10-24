from engine.retrieval.hybrid import RetrievalConfig, ScoredHit, mmr_select, rrf_fuse, title_match_fraction
from engine.retrieval.sparse import BM25Encoder, term_index, tokenize
from engine.retrieval.store import Hit


def _hit(cid, doc, title="t", dense=None):
    return Hit(chunk_id=cid, doc_id=doc, title=title, source="s", text=cid, score=1.0, payload={}, dense=dense)


def test_tokenize_keeps_identifiers_and_drops_stopwords():
    toks = tokenize("What did CHG-1187 change in the ledger pools?")
    assert "chg-1187" in toks and "1187" in toks
    assert "the" not in toks and "what" not in toks
    assert "pool" in toks  # plural stemmed


def test_bm25_tf_saturation_and_length_norm():
    enc = BM25Encoder(k1=1.2, b=0.75, avgdl=10)
    one = enc.encode_document("pool")
    many = enc.encode_document("pool pool pool pool pool pool pool pool pool pool")
    w1 = dict(zip(one.indices, one.values))[term_index("pool")]
    w10 = dict(zip(many.indices, many.values))[term_index("pool")]
    assert w10 > w1 and w10 < 2.2  # saturates below k1 + 1
    q = enc.encode_query("pool pool ledger")
    assert sorted(q.indices) == q.indices and q.values == [1.0, 1.0]


def test_rrf_rewards_agreement():
    dense = [_hit("a", "A"), _hit("b", "B"), _hit("c", "C")]
    sparse = [_hit("c", "C"), _hit("a", "A"), _hit("d", "D")]
    fused = rrf_fuse({"dense": dense, "sparse": sparse}, {"dense": 1.0, "sparse": 1.0}, k=60)
    order = sorted(fused, key=lambda k: -fused[k].score)
    assert order[0] == "a"  # rank 1 + rank 2 beats rank 3 + rank 1
    assert fused["d"].debug["dense_rank"] is None and fused["d"].debug["sparse_rank"] == 3


def test_title_match_fraction():
    assert title_match_fraction(set(tokenize("Where was Scott Derrickson born?")), "Scott Derrickson") == 1.0
    assert title_match_fraction(set(tokenize("Ed Wood film")), "Scott Derrickson") == 0.0


def test_mmr_caps_chunks_per_document_and_avoids_redundancy():
    v1, v2 = [1.0, 0.0], [0.0, 1.0]
    cands = [ScoredHit(_hit("a1", "A", dense=v1), 1.0), ScoredHit(_hit("a2", "A", dense=v1), 0.99),
             ScoredHit(_hit("a3", "A", dense=v1), 0.98), ScoredHit(_hit("b1", "B", dense=v2), 0.5)]
    out = mmr_select(cands, k=3, lam=0.7, max_per_doc=2)
    ids = [s.hit.chunk_id for s in out]
    assert "a3" not in ids and "b1" in ids
    assert ids[0] == "a1"


def test_hybrid_search_end_to_end(services):
    ing = services.ingestion
    ing.ingest(doc_id="d1", title="Ledger runbook", text="The ledger-db PgBouncer pool must stay above 150 connections.", source="t")
    ing.ingest(doc_id="d2", title="Film", text="Ed Wood is a 1994 American film directed by Tim Burton.", source="t")
    ing.ingest(doc_id="d3", title="Change CHG-1187", text="CHG-1187 reduced the pool size from 200 to 80.", source="t")
    for mode in ("dense", "sparse", "hybrid"):
        hits, timings = services.retriever.search("pool size CHG-1187", RetrievalConfig(mode=mode, top_k=2, rerank=False))
        assert hits and hits[0].hit.doc_id == "d3", mode
        assert "total_ms" in timings
    hits, _ = services.retriever.search("pool", RetrievalConfig(top_k=5, rerank=False, doc_ids=["d1"]))
    assert {h.hit.doc_id for h in hits} == {"d1"}
