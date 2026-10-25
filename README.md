# AI Investigation Engine

An evidence-first question-answering system over a 66,000+ document collection. It does more than "chat with a PDF": it plans multi-step investigations, follows leads across documents, flags contradictions, refuses to answer when the evidence is missing, and cites (and verifies) every claim.

> *"Which system changes caused the payment processing delays in March 2024, which services were affected, and which documents confirm this?"*
> → plans sub-questions → searches the change log, incident report and post-mortem → follows the change ids it discovers → answers with `[S#]` citations, and points out that the incident report and the post-mortem disagree on the duration and on the primary root cause.

- **Live demo:** https://ai-investigation-engine.vercel.app (API docs: `/docs`)
- **Evaluation report:** [EVALUATION.md](EVALUATION.md)

## Contents

1. [Architecture](#architecture)
2. [Data](#data)
3. [Ingestion](#ingestion)
4. [Retrieval](#retrieval)
5. [Agent and answers](#agent-and-answers)
6. [API](#api)
7. [Running locally](#running-locally)
8. [Deployment](#deployment)
9. [Costs, limits and performance](#costs-limits-and-performance)
10. [Design decisions](#design-decisions)
11. [Project layout](#project-layout)

## Architecture

```
                 ┌──────────────────────── FastAPI (Vercel Python function, Fluid Compute) ────────────────────┐
 browser ──HTTP──▶  REST + SSE  ─┬─ /api/documents ──▶ parsers ─▶ chunker ─▶ dedup ─▶ embed + BM25 ─┐         │
  (static UI)     │              │                                                                    ▼         │
                  │              ├─ /api/search ───▶ HybridRetriever ◀── dense + sparse queries ── VectorStore ◀─┼── Neon Postgres
                  │              │                   (RRF, title boost, cross-encoder, MMR)       (pgvector | Qdrant)│   + pgvector (free)
                  │              └─ /api/ask(/stream) ▶ Investigator (plan → search → tool loop → answer → verify)
                  │                                          │                                                  │
                  │                 answer cache · rate limiter · metrics · JSON logs                           │
                  └──────────────────────────────────────────┼──────────────────────────────────────────────────┘
                                                             ▼
                                                 Gemini API (generateContent; function calling, JSON schema)
```

| Layer | Choice | Why |
|---|---|---|
| Backend | Python 3.11, FastAPI, Uvicorn | Typed request models, async uploads, SSE streaming. |
| Vector DB | **Production:** Neon serverless Postgres + pgvector (free tier, provisioned through the Vercel Marketplace). **Development / evaluation:** Qdrant (local server or embedded). Both implement one `VectorStore` interface (`engine/retrieval/store.py`, `engine/retrieval/pgstore.py`). | pgvector: `halfvec` HNSW index for dense search and `sparsevec` for BM25 in one table, one transaction for chunks + term statistics. Qdrant: fast local bulk indexing and experiments. |
| Embeddings | `BAAI/bge-small-en-v1.5` (384-d) via fastembed / ONNX, run locally | Free, no rate limits for 66k+ documents; small enough for a serverless CPU. |
| Keyword search | Own BM25 encoder → sparse vectors (Qdrant: IDF modifier; Postgres: our own `term_df` table and an inner product `sparse <#> q_idf`) | Exact Okapi BM25 with idf over the live collection, so adds/deletes need no re-encoding. |
| Re-ranker | `Xenova/ms-marco-MiniLM-L-6-v2` cross-encoder (ONNX) | Large precision gain for ~0.5–1 s of CPU per query. |
| LLM | Google Gemini `gemini-3.5-flash` (thinking level "low"), fallback `gemini-3.5-flash-lite` (free tier) | Free API, native function calling and JSON-schema output. (Gemini 2.5 models are no longer offered to new API keys; the client also falls back automatically on 404.) |
| UI | Static HTML/CSS/JS served by FastAPI | No build step; the priority is functionality, not visual complexity. |

## Data

`scripts/build_corpus.py` builds a reproducible corpus from public sources (it writes `data/corpus/build_report.json` with these counts):

| Source | Records | Use |
|---|---|---|
| [HotpotQA](https://huggingface.co/datasets/hotpotqa/hotpot_qa), *distractor* config, *validation* split | 73,700 paragraphs → **66,552 unique** (7,047 repeated titles, 40 same-text-different-title duplicates, 61 same-title-different-text conflicts removed) | Multi-hop questions with gold supporting documents → Recall@K / MRR. |
| [danluu/post-mortems](https://github.com/danluu/post-mortems) | **240** incident summaries | Real incident data in the domain of the brief. |
| Synthetic *Northwind Pay* set (`scripts/generate_northwind.py`) | 11 files in 6 formats (pdf, docx, md, html, json, txt) | The brief's own scenario, with planted cross-document links, contradictions, gaps, a prompt injection, an exact and a near duplicate. |

**Why this subset.** The validation split is the only HotpotQA split with gold supporting facts *and* a manageable size: all 7,405 questions' gold and distractor paragraphs are indexed (66k records, above the 50k requirement), so every question is a needle-in-a-66k-haystack problem, while the whole index (~120 MB of vectors and payload) fits the Qdrant Cloud free tier and embeds in about an hour on a laptop CPU. Everything is seeded and deterministic (`random.Random(42)`), so the evaluation is repeatable.

## Ingestion

`parse → normalise → chunk → de-duplicate → embed (dense) + encode (BM25) → upsert`

- **Formats:** `.pdf` (pypdf, keeps page numbers), `.docx` (python-docx, headings become markdown headings, tables kept), `.md`/`.txt`, `.html` (scripts/nav stripped, headings kept), `.json`/`.jsonl` (records flattened). Metadata (title, author, dates, page count, filename) is extracted per format.
- **Chunking — two strategies, compared in the evaluation:**
  - `fixed`: 180-word sliding window, 30-word overlap. Structure-blind.
  - `recursive` (default): section → paragraph → sentence; packs whole units up to 180 words, overlaps by whole trailing sentences, never cuts a sentence, and stores the heading path (`Report > Root cause`) with each chunk.
  - Every chunk is embedded with a *contextual header* (document title and section), which matters for short Wikipedia paragraphs whose subject is only in the title.
- **Duplicates:** exact duplicates are detected on a normalised SHA-256 content hash (re-uploads are `unchanged`, identical content under a new name is `duplicate` and not indexed); near-duplicates are detected with a 64-bit SimHash over word 3-shingles (Hamming ≤ 3) against the nearest dense neighbours and reported; repeated blocks inside one document are dropped.
- **Create / update / delete:** documents are keyed by `doc_id`. A changed upload replaces all of the document's chunks atomically from the user's point of view; delete removes chunks and the registry entry. Seed-corpus documents are protected by an admin token.
- **Prompt-injection screening:** chunks containing instruction-like text are flagged at ingestion (`suspicious`) and shown with a warning.

## Retrieval

Our own ranking pipeline (`engine/retrieval/hybrid.py`) on top of Qdrant's two indexes:

1. **Candidates:** top-40 from dense (cosine HNSW) and top-40 from BM25 (sparse).
2. **Weighted Reciprocal Rank Fusion:** `score = Σ w_r / (60 + rank_r)`.
3. **Title boost:** if the query names a document's full title, add half a rank-1 RRF contribution — the bridge entity of a multi-hop question is usually a title.
4. **Cross-encoder blend:** re-score the top 30 with a cross-encoder and blend min-max normalised scores (0.7 cross-encoder / 0.3 fusion).
5. **Document-aware MMR:** pick the final *k* by maximal marginal relevance (λ = 0.75) on the dense vectors with at most two chunks per document, so the context covers several sources instead of five chunks of one.

The *Search lab* tab shows each stage's ranks and scores for any query.

## Agent and answers

`engine/agent/investigator.py`:

1. **Plan** (JSON-schema output): sub-questions and keyword-rich search queries.
2. **Retrieve** each planned query; results are pooled and labelled `S1…Sn`.
3. **Investigate** (function calling, up to 4 steps): the model sees the evidence so far and may call `search_documents(query, source?)` to follow a newly discovered entity, `read_document(doc_id)` to read a whole document, or `finish(reason)`.
4. **Answer** (JSON schema): `status` (`answered` / `insufficient` / `conflicting`), prose answer with inline `[S#]` citations, short answer, atomic claims with citations, conflicts with the citations on each side, missing information, confidence.
5. **Verify:** unknown citation ids are removed; each claim's lexical support in its cited passages is scored; an "answered" response with no valid citation is downgraded to `insufficient`.

**Grounding and safety.** Evidence is passed inside `<doc id="S#">` tags with any tag-like text neutralised, the system prompt treats document text as data, and flagged chunks are announced as suspected injections. **Failure handling:** retries with backoff (honouring the server's `retryDelay`), fallback model, client-side pacing, typed errors; if the model is unavailable the API still returns the retrieved evidence with status `llm_unavailable` instead of an error. A `simple` mode (one retrieval → answer) is kept as the evaluation baseline.

## API

Interactive docs at `/docs` (OpenAPI).

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/api/documents` | Upload one or more files (multipart `files`). Returns `created` / `updated` / `unchanged` / `duplicate` per file. |
| `GET` | `/api/documents?source=&q=&limit=&cursor=` | List / filter documents (cursor pagination). |
| `GET` | `/api/documents/{doc_id}` | Document metadata and all its chunks. |
| `PUT` | `/api/documents/{doc_id}` | Replace a document's content. |
| `DELETE` | `/api/documents/{doc_id}` | Delete a document. Seed-corpus documents need `X-Admin-Token`. |
| `POST` | `/api/search` | Retrieval only: `{query, mode: dense|sparse|hybrid, rerank, mmr, top_k, sources}` with per-stage debug scores. |
| `POST` | `/api/ask` | `{question, mode: agentic|simple, sources}` → answer, citations, sources, plan, trace, token usage. |
| `POST` | `/api/ask/stream` | Same, as Server-Sent Events: `plan`, `search`, `tool_call`, `finish`, `synthesis`, `result`. |
| `GET` | `/api/health`, `/api/stats`, `/api/metrics` | Dependencies, index size, counters, latency percentiles, LLM tokens and paid-tier cost equivalent. |

## Running locally

Requirements: Python 3.11 and [uv](https://docs.astral.sh/uv/). No GPU.

```bash
uv sync                                   # install
cp .env.example .env                      # add GEMINI_API_KEY; leave QDRANT_URL empty for embedded storage
uv run python scripts/download_data.py    # HotpotQA parquet + post-mortems README (~28 MB)
uv run python scripts/build_corpus.py
uv run python scripts/generate_northwind.py
uv run python scripts/ingest_corpus.py    # ~1 h on a laptop CPU for 66k docs; resumable (embedding cache)
uv run uvicorn main:app --port 8000       # http://localhost:8000
uv run pytest                             # 44 tests, no network or API key needed (+1 pgvector test with PG_TEST_DSN)
```

Embedded Qdrant (`QDRANT_PATH`) is fine for small sets; for the full corpus run a Qdrant server (`docker run -p 6333:6333 qdrant/qdrant` or the single Windows/Linux binary) and set `QDRANT_URL=http://127.0.0.1:6333`.

## Deployment

```
GitHub (code) ─ scripts/stage_vercel.py ─▶ Vercel (Python 3.12 function, Fluid Compute, iad1) ──▶ Neon Postgres + pgvector (us-east-1, free)
                                                  │                                              └─ persistent: chunks, vectors, term_df, document registry
                                                  └──HTTPS──▶ Gemini API (free tier)
```

- **Hosting:** the FastAPI app (`main.py`) is deployed as a single Vercel Python function; the static UI is served by the same app. `scripts/stage_vercel.py` stages only the code and a `requirements.txt` (no data, tests or caches) and `vercel deploy --prod` publishes it. The function runs in `iad1`, next to the Neon database in `us-east-1`.
- **Database:** Neon serverless Postgres with pgvector, provisioned through the Vercel Marketplace (`vercel integration add neon`), which injects `DATABASE_URL` into the project. Tables: `chunks` (halfvec(384) + HNSW cosine index, sparsevec BM25 weights, text and provenance), `docs` (registry), `term_df` (document frequencies). 294 MB for the full corpus.
- **Seeding:** the corpus is embedded once on a workstation into a local Qdrant index (`scripts/ingest_corpus.py`, resumable, embedding cache), then copied to Postgres without re-embedding (`scripts/migrate_qdrant_to_pg.py`, ~9 minutes, HNSW built after the bulk load).
- **Persistence:** all state lives in Postgres, so cold starts, redeploys and scale-to-zero lose nothing. The answer cache and metrics are per instance by design.
- **Models:** the two ONNX models (~150 MB) are downloaded to `/tmp` on a cold start (`MODEL_CACHE_DIR=/tmp/models`); ONNX threads are pinned to the function's vCPU (`ONNX_THREADS=1`).
- **Secrets:** `GEMINI_API_KEY`, `ADMIN_TOKEN` and the Neon `DATABASE_URL` are Vercel environment variables; `.env` / `.env.local` are git-ignored; keys are never logged or returned by the API.
- **Alternative:** the `Dockerfile` runs the same app anywhere (e.g. a Hugging Face Docker Space with `QDRANT_URL` pointing at Qdrant Cloud) — the store backend is selected with `STORE_BACKEND=qdrant|pgvector`.

## Costs, limits and performance

| Item | Free-tier limit | How it is handled |
|---|---|---|
| Vercel Hobby function | 2 GB / 1 vCPU, 300 s max duration, 4.5 MB request body, scale to zero | Agentic answers take 15–40 s and stream progress over SSE; uploads limited to 4 MB (`MAX_UPLOAD_MB`). A cold start (model download + load) adds ~10–20 s to the first request. |
| Neon free | 0.5 GB storage, compute suspends when idle | Full corpus is 294 MB (halfvec halves the dense vector size); first query after suspension takes ~1–2 s longer. |
| Gemini free tier | Per-model requests-per-day quotas (the `gemini-3.5-flash` daily quota was exhausted during evaluation) | Automatic fallback to `gemini-3.5-flash-lite` on 429/503/404; answer cache; per-IP rate limit (12 questions/min); graceful `llm_unavailable` mode that still returns the evidence. |
| Agent cost | 3–6 LLM calls per agentic question (plan, 1–4 tool steps, answer); 1 per simple question | Token usage and paid-tier cost equivalent are reported per answer and in `/api/metrics`; measured numbers are in the evaluation report. |

Measured on the live deployment: dense search 15–90 ms and BM25 (exact, full sparse scan) ~100 ms on Neon; query embedding ~20 ms; the cross-encoder dominates retrieval at 1.5–4 s for 20 pairs on one vCPU (`RERANK_POOL=20`; locally ~1 s for 30 pairs). It can be disabled with `ENABLE_CROSS_ENCODER=false` at a measured cost of ~3 points Recall@5 and ~4 points MRR (see the evaluation).

## Design decisions

- **Local embeddings instead of an embedding API.** Embedding 66k records through a free API would take days of rate-limited calls; a local ONNX model is free, deterministic and also runs in the serverless function at query time.
- **Sparse vectors in Qdrant instead of a separate keyword engine.** One database, one consistency domain, filters shared by both retrievers. The BM25 term weighting is ours; Qdrant only multiplies by IDF.
- **RRF before a cross-encoder.** RRF is robust to the incomparable score scales of cosine and BM25; the cross-encoder then fixes the precision at the top. MMR is applied last because multi-hop questions need *different* documents, not the five best paragraphs about one entity.
- **Plan + bounded tool loop instead of an open-ended agent.** A fixed upper bound (4 steps) keeps latency and free-tier usage predictable; the loop also stops when tools stop returning new evidence. Each step is stateless (evidence re-rendered into the prompt), which avoids provider-specific multi-turn state.
- **Structured answers + post-hoc verification.** Citations are only as good as their check: unknown ids are stripped, weakly supported claims are flagged, and an answer without valid citations cannot be reported as answered.
- **No frameworks (LangChain/LlamaIndex).** The pipeline is short enough to own, and owning it makes every stage observable and testable.

## Project layout

```
engine/
  ingestion/   parsers.py, chunking.py, dedup.py, pipeline.py
  retrieval/   embeddings.py, sparse.py (BM25), store.py (Qdrant), hybrid.py (fusion, re-ranking, MMR)
  llm/         gemini.py (client, retries, fallback)
  agent/       prompts.py, investigator.py, citations.py
  api/app.py   FastAPI routes, SSE, rate limiting
  web/         index.html, app.css, app.js
  config.py, services.py, cache.py, observability.py, security.py, datasets.py
scripts/       build_corpus.py, generate_northwind.py, ingest_corpus.py
eval/          questions.jsonl (49), retrieval_questions.jsonl (300), metrics.py, run_eval.py, results/
tests/         43 unit / integration tests (in-memory Qdrant, fake Gemini transport)
```
