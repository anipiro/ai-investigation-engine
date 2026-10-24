# AI Investigation Engine

An evidence-first question-answering system over a 66,000+ document collection. It does more than "chat with a PDF": it plans multi-step investigations, follows leads across documents, flags contradictions, refuses to answer when the evidence is missing, and cites (and verifies) every claim.

> *"Which system changes caused the payment processing delays in March 2024, which services were affected, and which documents confirm this?"*
> → plans sub-questions → searches the change log, incident report and post-mortem → follows the change ids it discovers → answers with `[S#]` citations, and points out that the incident report and the post-mortem disagree on the duration and on the primary root cause.

- **Live demo:** _see the link at the top of the GitHub repository (Hugging Face Space)_
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
                 ┌──────────────────────────── FastAPI (Hugging Face Space, Docker) ───────────────────────────┐
 browser ──HTTP──▶  REST + SSE  ─┬─ /api/documents ──▶ parsers ─▶ chunker ─▶ dedup ─▶ embed + BM25 ─┐         │
  (static UI)     │              │                                                                    ▼         │
                  │              ├─ /api/search ───▶ HybridRetriever ◀──── dense + sparse queries ── Qdrant ◀───┼── Qdrant Cloud
                  │              │                   (RRF, title boost, cross-encoder, MMR)                     │   (free 1 GB)
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
| Vector DB | Qdrant (Cloud free tier in production; local server or embedded mode in development) | Dense **and** sparse vectors in one collection, payload filters, server-side IDF; same client code in all modes. |
| Embeddings | `BAAI/bge-small-en-v1.5` (384-d) via fastembed / ONNX, run locally | Free, no rate limits for 66k+ documents; small enough for a CPU Space. |
| Keyword search | Own BM25 encoder → Qdrant sparse vectors with IDF modifier | Exact BM25 with idf computed over the live collection, so adds/deletes need no re-encoding. |
| Re-ranker | `Xenova/ms-marco-MiniLM-L-6-v2` cross-encoder (ONNX) | Large precision gain for ~0.5–1 s of CPU per query. |
| LLM | Google Gemini `gemini-2.5-flash`, fallback `gemini-2.5-flash-lite` (free tier) | Free API, native function calling and JSON-schema output. |
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
uv run pytest                             # 43 tests, no network or API key needed
```

Embedded Qdrant (`QDRANT_PATH`) is fine for small sets; for the full corpus run a Qdrant server (`docker run -p 6333:6333 qdrant/qdrant` or the single Windows/Linux binary) and set `QDRANT_URL=http://127.0.0.1:6333`.

## Deployment

```
GitHub (code) ──push──▶ Hugging Face Space (Docker SDK, free CPU: 2 vCPU / 16 GB) ──HTTPS──▶ Qdrant Cloud (free 1 GB cluster)
                                     │                                              └─ persistent: vectors, payload, document registry
                                     └──HTTPS──▶ Gemini API (free tier)
```

- **Image:** `Dockerfile` installs `requirements.txt` and bakes both ONNX models into the image, so a cold start does not download models.
- **Persistence:** all state (chunks, vectors, document registry, uploads) lives in Qdrant Cloud, so restarts and rebuilds of the Space lose nothing. The answer cache is in-memory by design.
- **Secrets:** `GEMINI_API_KEY`, `QDRANT_URL`, `QDRANT_API_KEY`, `ADMIN_TOKEN` are Space *secrets* (environment variables). `.env` is git-ignored; keys are never logged or returned by the API.
- **Seeding:** the corpus is indexed once from a workstation with `scripts/ingest_corpus.py` pointed at Qdrant Cloud (the embedding cache makes this an upload, not a re-embedding).

## Costs, limits and performance

| Item | Free-tier limit | How it is handled |
|---|---|---|
| Hugging Face Space | Sleeps after 48 h without traffic; ~1 min cold start | Documented; models are baked in so a cold start only loads them. The UI shows "Loading models" meanwhile. |
| Qdrant Cloud | 1 GB RAM / 4 GB disk; suspended after a week of inactivity | ~66k × 384-d vectors + sparse vectors + payload ≈ 0.3–0.5 GB. Vectors stored on disk (`on_disk=True`). |
| Gemini free tier | Per-minute and per-day request quotas | Answer cache; per-IP rate limit (12 questions/min); retries with server-provided delay; fallback model; graceful `llm_unavailable` mode. |
| Agent cost | 3–6 LLM calls per agentic question (plan, 1–4 tool steps, answer); 1 per simple question | Token usage and paid-tier cost equivalent are reported per answer and in `/api/metrics`; measured numbers are in the evaluation report. |

Query latency on the Space CPU is dominated by the cross-encoder (~0.5–1 s for 30 pairs) and the LLM calls; retrieval itself (embedding + two Qdrant queries) is ~50–150 ms. The cross-encoder can be disabled with `ENABLE_CROSS_ENCODER=false`.

## Design decisions

- **Local embeddings instead of an embedding API.** Embedding 66k records through a free API would take days of rate-limited calls; a local ONNX model is free, deterministic and also runs in the Space at query time.
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
