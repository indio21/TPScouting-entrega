"""Servicios de seguridad livianos usados por la app Flask."""

from __future__ import annotations

import secrets
import threading
import time
from typing import Dict, List, Optional

from flask import abort


class LoginRateLimiter:
    """Rate limiter en memoria para intentos de login del MVP."""

    def __init__(self, window_seconds: int, max_attempts: int) -> None:
        self.window_seconds = max(1, int(window_seconds))
        self.max_attempts = max(1, int(max_attempts))
        self.attempts: Dict[str, List[float]] = {}
        self._lock = threading.Lock()

    def key(self, username: Optional[str], client_ip: str) -> str:
        normalized_username = (username or "").strip().lower() or "-"
        return f"{client_ip}:{normalized_username}"

    def prune(self, now_ts: Optional[float] = None) -> None:
        now_ts = now_ts if now_ts is not None else time.time()
        cutoff = now_ts - self.window_seconds
        expired_keys = []
        for key, attempts in self.attempts.items():
            fresh_attempts = [attempt for attempt in attempts if attempt >= cutoff]
            if fresh_attempts:
                self.attempts[key] = fresh_attempts
            else:
                expired_keys.append(key)
        for key in expired_keys:
            self.attempts.pop(key, None)

    def is_limited(self, username: Optional[str], client_ip: str) -> bool:
        with self._lock:
            self.prune()
            return len(self.attempts.get(self.key(username, client_ip), [])) >= self.max_attempts

    def register_failure(self, username: Optional[str], client_ip: str) -> None:
        with self._lock:
            self.prune()
            key = self.key(username, client_ip)
            attempts = self.attempts.setdefault(key, [])
            attempts.append(time.time())

    def clear(self, username: Optional[str], client_ip: str) -> None:
        with self._lock:
            self.attempts.pop(self.key(username, client_ip), None)


def csrf_token(session_obj) -> str:
    token = session_obj.get("csrf_token")
    if not token:
        token = secrets.token_urlsafe(32)
        session_obj["csrf_token"] = token
    return token


def require_csrf(form, headers, session_obj) -> None:
    token = form.get("csrf_token") or headers.get("X-CSRF-Token")
    if not token or token != session_obj.get("csrf_token"):
        abort(400)


def client_ip_from_request(request_obj) -> str:
    forwarded_for = (request_obj.headers.get("X-Forwarded-For") or "").split(",")[0].strip()
    return forwarded_for or request_obj.remote_addr or "unknown"

