"""Shared fixtures: in-memory Qdrant, a deterministic hashing embedder and a scripted fake Gemini."""

from __future__ import annotations

import json
import sys
import zlib
from pathlib import Path

import httpx
import numpy as np
import pytest
from qdrant_client import QdrantClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine.config import Settings  # noqa: E402
from engine.llm.gemini import GeminiClient  # noqa: E402
from engine.retrieval.sparse import tokenize  # noqa: E402
from engine.retrieval.store import VectorStore  # noqa: E402
from engine.services import build_services  # noqa: E402


class HashEmbedder:
    """Bag-of-words hashed into 128 dims: texts sharing words get similar vectors. No model download."""

    name = "hash-test"
    dim = 128

    def _vec(self, text: str) -> list[float]:
        v = np.zeros(self.dim, dtype=np.float32)
        for t in tokenize(text):
            v[zlib.crc32(t.encode()) % self.dim] += 1.0
        n = np.linalg.norm(v)
        return (v / n if n else v + 1e-3).tolist()

    def embed_documents(self, texts, batch_size: int = 16):
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str):
        return self._vec(text)


class FakeGemini:
    """httpx transport that answers generateContent calls from a queue of scripted responses."""

    def __init__(self):
        self.queue: list[tuple[int, dict]] = []
        self.requests: list[dict] = []

    def text(self, payload) -> None:
        txt = payload if isinstance(payload, str) else json.dumps(payload)
        self.queue.append((200, {"candidates": [{"content": {"role": "model", "parts": [{"text": txt}]}, "finishReason": "STOP"}],
                                 "usageMetadata": {"promptTokenCount": 100, "candidatesTokenCount": 20}}))

    def call(self, name: str, args: dict) -> None:
        self.queue.append((200, {"candidates": [{"content": {"role": "model", "parts": [{"functionCall": {"name": name, "args": args}}]},
                                                 "finishReason": "STOP"}], "usageMetadata": {"promptTokenCount": 100, "candidatesTokenCount": 10}}))

    def error(self, status: int, message: str = "err") -> None:
        self.queue.append((status, {"error": {"code": status, "message": message}}))

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append({"url": str(request.url), "body": json.loads(request.content)})
        if not self.queue:
            return httpx.Response(500, json={"error": {"message": "fake queue empty"}})
        status, body = self.queue.pop(0)
        return httpx.Response(status, json=body)


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(_env_file=None, gemini_api_key="test-key", qdrant_url=None, qdrant_path=str(tmp_path / "q"),
                    enable_cross_encoder=False, admin_token="admin-secret", llm_max_retries=2, ask_rate_per_minute=100)


@pytest.fixture
def fake_gemini() -> FakeGemini:
    return FakeGemini()


@pytest.fixture
def llm(fake_gemini, monkeypatch) -> GeminiClient:
    monkeypatch.setattr("engine.llm.gemini.time.sleep", lambda s: None)
    return GeminiClient("test-key", "gemini-2.5-flash", "gemini-2.5-flash-lite", max_retries=2,
                        transport=httpx.MockTransport(fake_gemini.handler))


@pytest.fixture
def services(settings, llm):
    store = VectorStore(settings, client=QdrantClient(":memory:"))
    return build_services(settings, store=store, embedder=HashEmbedder(), llm=llm)
