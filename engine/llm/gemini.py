"""Minimal Gemini REST client with the failure handling a free-tier deployment needs.

- retries with exponential backoff + jitter on 429 / 5xx / timeouts, honouring the server's
  RetryInfo delay when present
- automatic fallback to a second model when the primary stays rate-limited or unavailable
- client-side pacing (minimum interval between calls) to stay under free-tier RPM
- typed errors (LLMError.kind) so callers can degrade gracefully instead of crashing
- token usage metering for cost / observability

Uses the public v1beta generateContent endpoint directly (function calling + JSON schema output).
"""

from __future__ import annotations

import json
import logging
import random
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from engine.observability import METRICS

log = logging.getLogger(__name__)

API = "https://generativelanguage.googleapis.com/v1beta"


class LLMError(RuntimeError):
    def __init__(self, kind: str, message: str, status: int | None = None):
        super().__init__(f"{kind}: {message}")
        self.kind = kind  # auth | rate_limit | unavailable | timeout | bad_request | blocked | not_configured
        self.status = status


@dataclass
class FunctionCall:
    name: str
    args: dict[str, Any]


@dataclass
class LLMResponse:
    text: str
    function_calls: list[FunctionCall]
    content: dict[str, Any]  # raw model turn, appended back verbatim for multi-turn tool use
    model: str
    finish_reason: str | None
    usage: dict[str, int] = field(default_factory=dict)
    latency_ms: float = 0.0

    def json(self) -> Any:
        txt = self.text.strip()
        txt = re.sub(r"^```(?:json)?\s*|\s*```$", "", txt)
        return json.loads(txt)


def _retry_delay_from(body: dict[str, Any]) -> float | None:
    for d in (body.get("error", {}) or {}).get("details", []) or []:
        if d.get("@type", "").endswith("RetryInfo") and d.get("retryDelay"):
            try:
                return float(str(d["retryDelay"]).rstrip("s"))
            except ValueError:
                return None
    return None


class GeminiClient:
    def __init__(self, api_key: str | None, model: str, fallback_model: str | None = None, timeout_s: float = 60.0,
                 max_retries: int = 4, min_interval_s: float = 0.0, transport: httpx.BaseTransport | None = None):
        self.api_key = api_key
        self.model = model
        self.fallback_model = fallback_model
        self.max_retries = max_retries
        self.min_interval_s = min_interval_s
        self._http = httpx.Client(timeout=timeout_s, transport=transport)
        self._lock = threading.Lock()
        self._last_call = 0.0

    @property
    def configured(self) -> bool:
        return bool(self.api_key)

    def _pace(self) -> None:
        if self.min_interval_s <= 0:
            return
        with self._lock:
            wait = self._last_call + self.min_interval_s - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            self._last_call = time.monotonic()

    def generate(self, contents: list[dict[str, Any]], *, system: str | None = None, tools: list[dict] | None = None,
                 json_schema: dict | None = None, temperature: float = 0.1, max_output_tokens: int = 2048,
                 thinking_budget: int | None = 0, purpose: str = "generate") -> LLMResponse:
        if not self.api_key:
            raise LLMError("not_configured", "GEMINI_API_KEY is not set")
        body: dict[str, Any] = {
            "contents": contents,
            "generationConfig": {"temperature": temperature, "maxOutputTokens": max_output_tokens},
        }
        if system:
            body["systemInstruction"] = {"parts": [{"text": system}]}
        if tools:
            body["tools"] = [{"functionDeclarations": tools}]
            body["toolConfig"] = {"functionCallingConfig": {"mode": "AUTO"}}
        if json_schema:
            body["generationConfig"]["responseMimeType"] = "application/json"
            body["generationConfig"]["responseSchema"] = json_schema

        models = [self.model] + ([self.fallback_model] if self.fallback_model and self.fallback_model != self.model else [])
        last_err: LLMError | None = None
        for model in models:
            b = dict(body)
            if thinking_budget is not None and model.startswith("gemini-2.5"):
                b["generationConfig"] = {**body["generationConfig"], "thinkingConfig": {"thinkingBudget": thinking_budget}}
            try:
                return self._call_with_retries(model, b, purpose)
            except LLMError as e:
                last_err = e
                if e.kind not in ("rate_limit", "unavailable", "timeout"):
                    raise
                if model != models[-1]:
                    log.warning("model %s failed (%s); trying fallback", model, e.kind)
                    METRICS.inc("llm_fallbacks")
        assert last_err is not None
        raise last_err

    def _call_with_retries(self, model: str, body: dict[str, Any], purpose: str) -> LLMResponse:
        url = f"{API}/models/{model}:generateContent"
        attempt = 0
        while True:
            attempt += 1
            self._pace()
            t0 = time.perf_counter()
            try:
                r = self._http.post(url, json=body, headers={"x-goog-api-key": self.api_key or ""})
            except httpx.TimeoutException:
                err = LLMError("timeout", f"{model} timed out")
                delay = None
            except httpx.HTTPError as exc:
                err = LLMError("unavailable", f"network error: {exc.__class__.__name__}")
                delay = None
            else:
                latency = (time.perf_counter() - t0) * 1000
                if r.status_code == 200:
                    return self._parse(model, r.json(), latency, purpose)
                try:
                    payload = r.json()
                except ValueError:
                    payload = {}
                msg = (payload.get("error") or {}).get("message", r.text[:200])
                delay = _retry_delay_from(payload)
                if r.status_code in (401, 403):
                    raise LLMError("auth", msg, r.status_code)
                if r.status_code == 429:
                    err = LLMError("rate_limit", msg, 429)
                elif r.status_code >= 500:
                    err = LLMError("unavailable", msg, r.status_code)
                else:
                    raise LLMError("bad_request", msg, r.status_code)
            METRICS.inc(f"llm_errors_{err.kind}")
            if attempt > self.max_retries:
                raise err
            backoff = delay if delay is not None else min(30.0, 1.5 * 2 ** (attempt - 1))
            # A very long server-requested wait (daily quota) is not worth blocking a user request on.
            if backoff > 45:
                raise err
            time.sleep(backoff + random.uniform(0, 0.5))
            log.info("retrying %s after %s (attempt %d)", model, err.kind, attempt)

    @staticmethod
    def _parse(model: str, data: dict[str, Any], latency_ms: float, purpose: str) -> LLMResponse:
        if (data.get("promptFeedback") or {}).get("blockReason"):
            raise LLMError("blocked", f"prompt blocked: {data['promptFeedback']['blockReason']}")
        cands = data.get("candidates") or []
        if not cands:
            raise LLMError("unavailable", "empty response (no candidates)")
        cand = cands[0]
        finish = cand.get("finishReason")
        if finish in ("SAFETY", "PROHIBITED_CONTENT", "BLOCKLIST", "SPII"):
            raise LLMError("blocked", f"response blocked: {finish}")
        content = cand.get("content") or {"role": "model", "parts": []}
        content.setdefault("role", "model")
        texts, calls = [], []
        for p in content.get("parts", []) or []:
            if "functionCall" in p:
                fc = p["functionCall"]
                calls.append(FunctionCall(name=fc.get("name", ""), args=fc.get("args") or {}))
            elif "text" in p and not p.get("thought"):
                texts.append(p["text"])
        um = data.get("usageMetadata") or {}
        usage = {
            "input_tokens": int(um.get("promptTokenCount", 0)),
            "output_tokens": int(um.get("candidatesTokenCount", 0)) + int(um.get("thoughtsTokenCount", 0)),
        }
        METRICS.record_llm(model, purpose, usage["input_tokens"], usage["output_tokens"], latency_ms)
        return LLMResponse(text="".join(texts), function_calls=calls, content=content, model=model,
                           finish_reason=finish, usage=usage, latency_ms=latency_ms)
