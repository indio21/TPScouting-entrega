"""Servicios de seguridad livianos usados por la app Flask."""

from __future__ import annotations

import secrets
import threading
import time
from typing import Dict, List, Optional
from urllib.parse import unquote, urlsplit

from flask import abort


class LoginRateLimiter:
    """Rate limiter en memoria para intentos de login del MVP."""

    def __init__(self, window_seconds: int, max_attempts: int) -> None:
        self.window_seconds = max(1, int(window_seconds))
        self.max_attempts = max(1, int(max_attempts))
        self.attempts: Dict[str, List[float]] = {}
        self._lock = threading.Lock()

    def key(self, username: Optional[str], client_ip: str) -> str:
        """Agrupa intentos por cuenta sin confiar en cabeceras de proxy.

        ``client_ip`` se conserva en la firma por compatibilidad, pero no forma
        parte de la clave. Así, rotar ``X-Forwarded-For`` o la IP de origen no
        permite seguir probando contraseñas contra el mismo usuario.
        """
        normalized_username = (username or "").strip().lower() or "-"
        return normalized_username

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
    expected_token = session_obj.get("csrf_token")
    if not isinstance(token, str) or not isinstance(expected_token, str):
        abort(400)
    if not secrets.compare_digest(token, expected_token):
        abort(400)


def client_ip_from_request(request_obj) -> str:
    """Devuelve la dirección del peer sin confiar en headers del cliente.

    Si más adelante se configura una cantidad conocida de proxies confiables,
    esa normalización debe hacerse una sola vez en WSGI y documentarse junto al
    despliegue. Hasta entonces, ``X-Forwarded-For`` es entrada no confiable.
    """
    return request_obj.remote_addr or "unknown"


def safe_internal_redirect_target(target: Optional[str]) -> Optional[str]:
    """Acepta solo rutas absolutas internas que el navegador no reinterpretará."""
    if not isinstance(target, str) or not target:
        return None
    if any(ord(character) < 32 or ord(character) == 127 for character in target):
        return None
    if "\\" in target:
        return None

    # Flask decodifica el query string una vez. Revisar también hasta dos capas
    # adicionales evita que separadores codificados lleguen al navegador.
    decoded_target = target
    for _ in range(2):
        decoded_target = unquote(decoded_target)
        if "\\" in decoded_target or decoded_target.startswith("//"):
            return None

    try:
        parsed = urlsplit(target)
    except ValueError:
        return None
    if parsed.scheme or parsed.netloc or not parsed.path.startswith("/"):
        return None
    if parsed.path.startswith("//"):
        return None
    return target

