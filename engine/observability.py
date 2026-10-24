"""In-process metrics + structured logging.

Kept dependency-free on purpose: the free hosting tier runs a single process, so counters and
bounded latency reservoirs in memory are enough. Exposed at GET /api/metrics.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections import defaultdict, deque
from typing import Any

# Paid-tier prices (USD per 1M input / output tokens) used to report what the free-tier traffic *would* cost.
# Approximate: update from https://ai.google.dev/pricing when models change.
PRICES = {
    "gemini-3.5-flash-lite": (0.10, 0.40),
    "gemini-3.5-flash": (0.30, 2.50),
    "gemini-2.5-flash": (0.30, 2.50),
    "gemini-2.5-flash-lite": (0.10, 0.40),
    "gemini-2.5-pro": (1.25, 10.0),
}


def _price(model: str) -> tuple[float, float]:
    for prefix, p in sorted(PRICES.items(), key=lambda kv: -len(kv[0])):
        if model.startswith(prefix):
            return p
    return (0.30, 2.50)


class Metrics:
    def __init__(self, reservoir: int = 500):
        self._lock = threading.Lock()
        self.started = time.time()
        self.counters: dict[str, int] = defaultdict(int)
        self.latencies: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=reservoir))
        self.llm: dict[str, dict[str, float]] = defaultdict(lambda: {"calls": 0, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0})
        self.llm_purpose: dict[str, int] = defaultdict(int)

    def inc(self, name: str, n: int = 1) -> None:
        with self._lock:
            self.counters[name] += n

    def observe(self, name: str, ms: float) -> None:
        with self._lock:
            self.latencies[name].append(ms)

    def record_llm(self, model: str, purpose: str, tin: int, tout: int, latency_ms: float) -> None:
        pin, pout = _price(model)
        with self._lock:
            m = self.llm[model]
            m["calls"] += 1
            m["input_tokens"] += tin
            m["output_tokens"] += tout
            m["cost_usd"] += tin / 1e6 * pin + tout / 1e6 * pout
            self.llm_purpose[purpose] += 1
            self.latencies[f"llm.{purpose}"].append(latency_ms)

    @staticmethod
    def _pct(xs: list[float], q: float) -> float:
        if not xs:
            return 0.0
        xs = sorted(xs)
        return round(xs[min(len(xs) - 1, int(q * len(xs)))], 1)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            lat = {k: {"n": len(v), "p50_ms": self._pct(list(v), 0.5), "p95_ms": self._pct(list(v), 0.95)} for k, v in self.latencies.items()}
            return {
                "uptime_s": round(time.time() - self.started),
                "counters": dict(self.counters),
                "latency": lat,
                "llm": {k: {**v, "cost_usd": round(v["cost_usd"], 5)} for k, v in self.llm.items()},
                "llm_calls_by_purpose": dict(self.llm_purpose),
            }


METRICS = Metrics()


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        out = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)), "level": record.levelname,
               "logger": record.name, "msg": record.getMessage()}
        for key in ("request_id", "path", "status", "ms"):
            if hasattr(record, key):
                out[key] = getattr(record, key)
        if record.exc_info:
            out["exc"] = self.formatException(record.exc_info)
        return json.dumps(out, ensure_ascii=False)


def setup_logging(level: str = "INFO") -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level)
    for noisy in ("httpx", "httpcore", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
