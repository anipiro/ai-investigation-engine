"""FastAPI application: REST API + static single-page UI."""

from __future__ import annotations

import json
import logging
import queue
import threading
import time
import uuid
from contextlib import asynccontextmanager
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Literal

from fastapi import Depends, FastAPI, File, Header, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from engine.config import Settings, get_settings
from engine.ingestion.parsers import ParseError, UnsupportedFormatError, parse_file
from engine.ingestion.pipeline import slugify
from engine.observability import METRICS, setup_logging
from engine.services import Services, build_services

log = logging.getLogger("engine.api")
STATIC = Path(__file__).resolve().parent.parent / "web"


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=1000)
    mode: Literal["dense", "sparse", "hybrid"] = "hybrid"
    rerank: bool = True
    mmr: bool = True
    top_k: int = Field(8, ge=1, le=50)
    sources: list[str] | None = None


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    mode: Literal["agentic", "simple"] = "agentic"
    sources: list[str] | None = None
    use_cache: bool = True


class RateLimiter:
    """Per-client sliding window; protects the free-tier LLM quota from a single visitor."""

    def __init__(self, per_minute: int):
        self.per_minute = per_minute
        self.hits: dict[str, deque[float]] = defaultdict(deque)
        self.lock = threading.Lock()

    def check(self, key: str) -> None:
        now = time.monotonic()
        with self.lock:
            q = self.hits[key]
            while q and q[0] < now - 60:
                q.popleft()
            if len(q) >= self.per_minute:
                raise HTTPException(429, f"rate limit: max {self.per_minute} questions per minute")
            q.append(now)


def create_app(services: Services | None = None, settings: Settings | None = None) -> FastAPI:
    settings = settings or (services.settings if services else get_settings())
    setup_logging(settings.log_level)
    state: dict[str, Any] = {"svc": services}

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        # Build models in the background so the port opens immediately on cold start.
        threading.Thread(target=svc, daemon=True).start()
        yield

    app = FastAPI(title="AI Investigation Engine", version="1.0.0", lifespan=lifespan,
                  description="Hybrid retrieval + agentic RAG over 50k+ documents with verified citations.")
    limiter = RateLimiter(settings.ask_rate_per_minute)

    build_lock = threading.Lock()

    def svc() -> Services:
        if state["svc"] is None:
            with build_lock:
                if state["svc"] is None:
                    state["svc"] = build_services(settings)
        return state["svc"]

    @app.middleware("http")
    async def observe(request: Request, call_next):
        rid = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
        t0 = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            log.exception("unhandled error", extra={"request_id": rid, "path": request.url.path})
            response = JSONResponse({"detail": "internal error", "request_id": rid}, status_code=500)
        ms = (time.perf_counter() - t0) * 1000
        if request.url.path.startswith("/api/"):
            route = request.scope.get("route")
            name = getattr(route, "path", request.url.path)
            METRICS.inc(f"http_{response.status_code}")
            METRICS.observe(f"http.{request.method} {name}", ms)
            log.info("request", extra={"request_id": rid, "path": request.url.path, "status": response.status_code, "ms": round(ms)})
        response.headers["x-request-id"] = rid
        return response

    def require_admin_for(doc: dict[str, Any] | None, token: str | None) -> None:
        if doc and doc.get("source") in settings.protected_sources:
            expected = settings.admin_token.get_secret_value() if settings.admin_token else None
            if not expected or token != expected:
                raise HTTPException(403, "this document belongs to the protected seed corpus; X-Admin-Token required")

    async def read_upload(f: UploadFile) -> bytes:
        data = await f.read(settings.max_upload_mb * 1024 * 1024 + 1)
        if len(data) > settings.max_upload_mb * 1024 * 1024:
            raise HTTPException(413, f"{f.filename}: file larger than {settings.max_upload_mb} MB")
        return data

    def ingest_bytes(s: Services, data: bytes, filename: str, doc_id: str | None = None) -> dict[str, Any]:
        try:
            parsed = parse_file(data, filename)
        except UnsupportedFormatError as e:
            raise HTTPException(415, str(e)) from e
        except ParseError as e:
            raise HTTPException(422, f"{filename}: {e}") from e
        res = s.ingestion.ingest(doc_id=doc_id or f"upload::{slugify(Path(filename).stem)}", title=parsed.title,
                                 text=parsed.text, source="upload", fmt=parsed.fmt, metadata=parsed.metadata)
        if res.status in ("created", "updated"):
            s.bump_version()
        return {"filename": filename, "title": parsed.title, "format": parsed.fmt, **res.as_dict()}

    # ------------------------------------------------------------------ system
    @app.get("/api/health")
    def health() -> dict[str, Any]:
        s = state["svc"]
        if s is None:
            return {"status": "starting"}
        return {"status": "ok", "vector_db": "ok" if s.store.ping() else "unreachable", "llm_configured": s.llm.configured,
                "llm_model": s.llm.model, "fallback_model": s.llm.fallback_model, "embedding_model": getattr(s.embedder, "name", "?"),
                "reranker": s.retriever.cross_encoder.name if s.retriever.cross_encoder else None,
                "qdrant": "server" if settings.qdrant_url else "embedded"}

    @app.get("/api/stats")
    def stats(s: Services = Depends(svc)) -> dict[str, Any]:
        by_source = {src: s.store.count("docs", src) for src in [*settings.protected_sources, "upload"]}
        return {"documents": s.store.count("docs"), "chunks": s.store.count("chunks"), "documents_by_source": by_source,
                "corpus_version": s.corpus_version, "answer_cache": s.answer_cache.stats(),
                "embedding_cache": {"hits": getattr(s.embedder, "cache_hits", 0), "misses": getattr(s.embedder, "cache_misses", 0)},
                "chunking": {"strategy": settings.chunk_strategy, "max_words": settings.chunk_max_words, "overlap": settings.chunk_overlap_words}}

    @app.get("/api/metrics")
    def metrics() -> dict[str, Any]:
        return METRICS.snapshot()

    # ------------------------------------------------------------------ documents
    @app.get("/api/documents")
    def list_documents(source: str | None = None, q: str | None = None, limit: int = 50, cursor: str | None = None,
                       s: Services = Depends(svc)) -> dict[str, Any]:
        docs, nxt = s.store.list_docs(source, q, min(max(limit, 1), 200), cursor)
        return {"documents": docs, "next_cursor": nxt}

    @app.get("/api/documents/{doc_id:path}")
    def get_document(doc_id: str, s: Services = Depends(svc)) -> dict[str, Any]:
        doc = s.store.get_doc(doc_id)
        if not doc:
            raise HTTPException(404, "document not found")
        chunks = [{"chunk_id": h.chunk_id, "index": h.payload.get("chunk_index"), "section": h.payload.get("section"),
                   "page": h.payload.get("page"), "text": h.text, "suspicious": bool(h.payload.get("suspicious"))}
                  for h in s.store.doc_chunks(doc_id)]
        return {**doc, "chunks": chunks}

    @app.post("/api/documents", status_code=201)
    async def upload_documents(files: list[UploadFile] = File(...), s: Services = Depends(svc)) -> dict[str, Any]:
        results, errors = [], []
        for f in files[:20]:
            try:
                data = await read_upload(f)
                results.append(ingest_bytes(s, data, f.filename or "upload.txt"))
            except HTTPException as e:
                errors.append({"filename": f.filename, "status": e.status_code, "error": e.detail})
        if errors and not results:
            raise HTTPException(errors[0]["status"], errors[0]["error"])
        return {"results": results, "errors": errors}

    @app.put("/api/documents/{doc_id:path}")
    async def update_document(doc_id: str, file: UploadFile = File(...), x_admin_token: str | None = Header(None),
                              s: Services = Depends(svc)) -> dict[str, Any]:
        doc = s.store.get_doc(doc_id)
        if not doc:
            raise HTTPException(404, "document not found")
        require_admin_for(doc, x_admin_token)
        data = await read_upload(file)
        try:
            parsed = parse_file(data, file.filename or "upload.txt")
        except UnsupportedFormatError as e:
            raise HTTPException(415, str(e)) from e
        except ParseError as e:
            raise HTTPException(422, str(e)) from e
        res = s.ingestion.ingest(doc_id=doc_id, title=parsed.title, text=parsed.text, source=doc["source"],
                                 fmt=parsed.fmt, metadata=parsed.metadata, url=doc.get("url"))
        if res.status != "unchanged":
            s.bump_version()
        return res.as_dict()

    @app.delete("/api/documents/{doc_id:path}")
    def delete_document(doc_id: str, x_admin_token: str | None = Header(None), s: Services = Depends(svc)) -> dict[str, Any]:
        doc = s.store.get_doc(doc_id)
        if not doc:
            raise HTTPException(404, "document not found")
        require_admin_for(doc, x_admin_token)
        s.store.delete_doc(doc_id)
        s.bump_version()
        return {"doc_id": doc_id, "status": "deleted"}

    # ------------------------------------------------------------------ retrieval / QA
    @app.post("/api/search")
    def search(req: SearchRequest, s: Services = Depends(svc)) -> dict[str, Any]:
        cfg = s.retrieval_config(mode=req.mode, top_k=req.top_k, mmr=req.mmr, sources=req.sources)
        cfg.rerank = req.rerank and s.retriever.cross_encoder is not None
        hits, timings = s.retriever.search(req.query, cfg)
        return {"query": req.query, "timings_ms": {k: round(v, 1) for k, v in timings.items()},
                "hits": [{"chunk_id": h.hit.chunk_id, "doc_id": h.hit.doc_id, "title": h.hit.title, "source": h.hit.source,
                          "section": h.hit.payload.get("section"), "text": h.hit.text, "score": round(h.score, 4),
                          "suspicious": bool(h.hit.payload.get("suspicious")), "debug": h.debug} for h in hits]}

    def cache_key(s: Services, req: AskRequest) -> tuple:
        return (" ".join(req.question.lower().split()), req.mode, tuple(sorted(req.sources or [])), s.corpus_version)

    @app.post("/api/ask")
    def ask(req: AskRequest, request: Request, s: Services = Depends(svc)) -> dict[str, Any]:
        key = cache_key(s, req)
        if req.use_cache and (hit := s.answer_cache.get(key)) is not None:
            METRICS.inc("answer_cache_hits")
            return {**hit, "cached": True}
        limiter.check(request.client.host if request.client else "anon")
        result = s.investigator.investigate(req.question, req.mode, req.sources)
        if result["status"] != "llm_unavailable":
            s.answer_cache.set(key, result)
        return {**result, "cached": False}

    @app.post("/api/ask/stream")
    def ask_stream(req: AskRequest, request: Request, s: Services = Depends(svc)) -> StreamingResponse:
        key = cache_key(s, req)
        cached = s.answer_cache.get(key) if req.use_cache else None
        if cached is None:
            limiter.check(request.client.host if request.client else "anon")
        events: queue.Queue = queue.Queue()

        def run() -> None:
            try:
                if cached is not None:
                    events.put({"type": "result", "result": {**cached, "cached": True}})
                    return
                result = s.investigator.investigate(req.question, req.mode, req.sources, on_event=events.put)
                if result["status"] != "llm_unavailable":
                    s.answer_cache.set(key, result)
                events.put({"type": "result", "result": {**result, "cached": False}})
            except Exception as exc:  # surface as an SSE error event, never a broken stream
                log.exception("stream failure")
                events.put({"type": "error", "message": str(exc)[:300]})
            finally:
                events.put(None)

        threading.Thread(target=run, daemon=True).start()

        def gen():
            while True:
                try:
                    ev = events.get(timeout=15)
                except queue.Empty:
                    yield ": keep-alive\n\n"
                    continue
                if ev is None:
                    break
                yield f"event: {ev['type']}\ndata: {json.dumps(ev, ensure_ascii=False, default=str)}\n\n"

        return StreamingResponse(gen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    # ------------------------------------------------------------------ UI
    if STATIC.exists():
        app.mount("/static", StaticFiles(directory=STATIC), name="static")

        @app.get("/", include_in_schema=False)
        def index() -> FileResponse:
            return FileResponse(STATIC / "index.html")

    return app
