"""Cache en memoria con TTL y limite de entradas."""

from __future__ import annotations

import time
from typing import Any, Dict, Optional, Tuple


class TTLCache:
    """Cache simple para el dashboard del MVP.

    Vive dentro del proceso Python. No reemplaza una cache externa para produccion,
    pero concentra la politica de TTL y eviccion fuera de app.py.
    """

    def __init__(self, ttl_seconds: int, max_entries: int) -> None:
        self.ttl_seconds = max(1, int(ttl_seconds))
        self.max_entries = max(1, int(max_entries))
        self.store: Dict[str, Tuple[float, Any]] = {}

    def get(self, key: str) -> Optional[Any]:
        item = self.store.get(key)
        if not item:
            return None
        expires_at, value = item
        if time.time() >= expires_at:
            self.store.pop(key, None)
            return None
        return value

    def prune_expired(self, now_ts: Optional[float] = None) -> None:
        now = time.time() if now_ts is None else now_ts
        expired_keys = [key for key, (expires_at, _value) in self.store.items() if now >= expires_at]
        for key in expired_keys:
            self.store.pop(key, None)

    def set(self, key: str, value: Any) -> None:
        now = time.time()
        self.prune_expired(now)
        if key in self.store:
            self.store.pop(key, None)
        elif len(self.store) >= self.max_entries:
            oldest_key = min(self.store, key=lambda cached_key: self.store[cached_key][0])
            self.store.pop(oldest_key, None)
        self.store[key] = (now + self.ttl_seconds, value)

    def invalidate_prefix(self, prefix: str) -> None:
        keys = [key for key in self.store.keys() if key.startswith(prefix)]
        for key in keys:
            self.store.pop(key, None)

