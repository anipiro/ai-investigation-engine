"""Small thread-safe TTL + LRU cache used for answers and search results.

Keys include the corpus version, so any document create/update/delete invalidates cached
answers implicitly (the version is bumped and old keys are never hit again).
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from typing import Any, Hashable


class TTLCache:
    def __init__(self, max_items: int = 512, ttl_s: float = 3600):
        self.max_items, self.ttl_s = max_items, ttl_s
        self._d: OrderedDict[Hashable, tuple[float, Any]] = OrderedDict()
        self._lock = threading.Lock()
        self.hits = self.misses = 0

    def get(self, key: Hashable) -> Any | None:
        with self._lock:
            item = self._d.get(key)
            if item is None or item[0] < time.monotonic():
                if item is not None:
                    del self._d[key]
                self.misses += 1
                return None
            self._d.move_to_end(key)
            self.hits += 1
            return item[1]

    def set(self, key: Hashable, value: Any) -> None:
        with self._lock:
            self._d[key] = (time.monotonic() + self.ttl_s, value)
            self._d.move_to_end(key)
            while len(self._d) > self.max_items:
                self._d.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._d.clear()

    def stats(self) -> dict[str, int]:
        return {"items": len(self._d), "hits": self.hits, "misses": self.misses}
