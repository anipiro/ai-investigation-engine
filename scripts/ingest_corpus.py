"""Index the seed corpus into Qdrant (local embedded mode or Qdrant Cloud, from .env).

  uv run python scripts/ingest_corpus.py                       # full corpus, default settings
  uv run python scripts/ingest_corpus.py --prefix aie_fixed --strategy fixed      # chunking comparison
  uv run python scripts/ingest_corpus.py --prefix aie_minilm --embedding-model sentence-transformers/all-MiniLM-L6-v2

Northwind files go through the same parse -> dedup -> ingest path as user uploads, so the run
log shows created / duplicate / near-duplicate handling.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    import truststore

    truststore.inject_into_ssl()
except ImportError:
    pass

from engine.config import Settings  # noqa: E402
from engine.datasets import northwind_doc_id  # noqa: E402
from engine.ingestion.parsers import parse_file  # noqa: E402
from engine.ingestion.pipeline import IngestionService  # noqa: E402
from engine.observability import setup_logging  # noqa: E402
from engine.retrieval.embeddings import DiskCachedEmbedder, FastEmbedder  # noqa: E402
from engine.retrieval.sparse import BM25Encoder  # noqa: E402
from engine.retrieval.store import VectorStore  # noqa: E402

log = logging.getLogger("ingest")


def iter_jsonl(path: Path, limit: int | None, skip: int = 0):
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i < skip:
                continue
            if limit is not None and i >= skip + limit:
                break
            yield json.loads(line)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default=None, help="collection prefix (default from settings)")
    ap.add_argument("--strategy", choices=["recursive", "fixed"], default=None)
    ap.add_argument("--embedding-model", default=None)
    ap.add_argument("--limit", type=int, default=None, help="max docs per JSONL source")
    ap.add_argument("--skip", type=int, default=0, help="skip the first N docs of each JSONL source (resume in segments)")
    ap.add_argument("--only", choices=["hotpotqa", "postmortems", "northwind"], action="append")
    ap.add_argument("--recreate", action="store_true", help="drop the collections first")
    args = ap.parse_args()

    overrides = {k: v for k, v in {"collection_prefix": args.prefix, "chunk_strategy": args.strategy,
                                    "embedding_model": args.embedding_model}.items() if v}
    s = Settings(**overrides)
    setup_logging("INFO")
    store = VectorStore(s)
    if args.recreate:
        for c in (store.chunks, store.docs):
            if store.client.collection_exists(c):
                store.client.delete_collection(c)
    t0 = time.perf_counter()
    (ROOT / "data" / "cache").mkdir(parents=True, exist_ok=True)
    embedder = DiskCachedEmbedder(FastEmbedder(s.embedding_model, cache_dir=s.model_cache_dir),
                                  str(ROOT / "data" / "cache" / "embeddings.sqlite"))
    store.ensure_collections(embedder.dim)
    svc = IngestionService(store, embedder, BM25Encoder(), s.chunk_strategy, s.chunk_max_words, s.chunk_overlap_words)
    log.info("target=%s qdrant=%s strategy=%s model=%s", store.chunks, s.qdrant_url or s.qdrant_path, s.chunk_strategy, s.embedding_model)

    only = set(args.only or ["hotpotqa", "postmortems", "northwind"])
    summary: dict = {}
    for name in ("postmortems", "hotpotqa"):
        if name in only:
            st = svc.bulk_ingest(iter_jsonl(ROOT / "data" / "corpus" / f"{name}.jsonl", args.limit, args.skip))
            summary[name] = st
            log.info("%s done: %s", name, st)
    if "northwind" in only:
        results = []
        for path in sorted((ROOT / "data" / "northwind").iterdir()):
            parsed = parse_file(path.read_bytes(), path.name)
            r = svc.ingest(doc_id=northwind_doc_id(path.name), title=parsed.title, text=parsed.text, source="northwind",
                           fmt=parsed.fmt, metadata=parsed.metadata)
            results.append({"file": path.name, **r.as_dict()})
            log.info("northwind %s -> %s", path.name, r.status)
        summary["northwind"] = results
    summary["seconds"] = round(time.perf_counter() - t0, 1)
    summary["chunks_in_collection"] = store.count("chunks")
    summary["docs_in_registry"] = store.count("docs")
    out = ROOT / "data" / f"ingest_report_{store.chunks}.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "northwind"}, indent=2))


if __name__ == "__main__":
    main()
