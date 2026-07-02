"""Smoke test HTTP contra el despliegue real en Render.

Uso:
    python scripts/smoke_render.py --base-url https://tu-servicio.onrender.com

Tambien acepta RENDER_SMOKE_BASE_URL. Si se definen SMOKE_USERNAME y
SMOKE_PASSWORD, valida login y acceso autenticado a /dashboard.
"""

from __future__ import annotations

import argparse
import http.cookiejar
import os
import sys
import urllib.parse
import urllib.request


def _request(opener, base_url: str, path: str, data: bytes | None = None):
    url = urllib.parse.urljoin(base_url.rstrip("/") + "/", path.lstrip("/"))
    request = urllib.request.Request(url, data=data, method="POST" if data is not None else "GET")
    if data is not None:
        request.add_header("Content-Type", "application/x-www-form-urlencoded")
    with opener.open(request, timeout=20) as response:
        body = response.read().decode("utf-8", errors="replace")
        return response.status, body, response.geturl()


def _csrf_from_html(html: str) -> str | None:
    marker = 'name="csrf_token" value="'
    start = html.find(marker)
    if start < 0:
        return None
    start += len(marker)
    end = html.find('"', start)
    return html[start:end] if end > start else None


def run_smoke(base_url: str, username: str | None = None, password: str | None = None) -> int:
    cookie_jar = http.cookiejar.CookieJar()
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cookie_jar))

    checks: list[str] = []
    status, body, _ = _request(opener, base_url, "/health")
    if status != 200 or "ok" not in body.lower():
        raise RuntimeError(f"/health no respondio OK. status={status}, body={body[:160]!r}")
    checks.append("/health OK")

    status, login_html, _ = _request(opener, base_url, "/login")
    if status != 200 or "TPScouting" not in login_html:
        raise RuntimeError(f"/login no renderizo la pantalla esperada. status={status}")
    checks.append("/login OK")

    if username and password:
        csrf = _csrf_from_html(login_html)
        if not csrf:
            raise RuntimeError("No se encontro csrf_token en /login.")
        payload = urllib.parse.urlencode(
            {"username": username, "password": password, "csrf_token": csrf}
        ).encode("utf-8")
        status, _body, final_url = _request(opener, base_url, "/login", payload)
        if status not in {200, 302}:
            raise RuntimeError(f"Login fallo con status={status}.")
        checks.append(f"login OK -> {final_url}")

        status, dashboard_html, _ = _request(opener, base_url, "/dashboard")
        if status != 200 or ("Mesa de scouting" not in dashboard_html and "Estado del plantel" not in dashboard_html):
            raise RuntimeError(f"/dashboard autenticado no renderizo el panel esperado. status={status}")
        checks.append("/dashboard OK")
    else:
        checks.append("login autenticado omitido: faltan SMOKE_USERNAME/SMOKE_PASSWORD")

    for check in checks:
        print(check)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke test contra Render.")
    parser.add_argument("--base-url", default=os.environ.get("RENDER_SMOKE_BASE_URL"))
    parser.add_argument("--username", default=os.environ.get("SMOKE_USERNAME"))
    parser.add_argument("--password", default=os.environ.get("SMOKE_PASSWORD"))
    args = parser.parse_args()

    if not args.base_url:
        print("SKIP: no se definio --base-url ni RENDER_SMOKE_BASE_URL.")
        return 0
    try:
        return run_smoke(args.base_url, args.username, args.password)
    except Exception as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
