"""Blueprints y utilidades de registro de rutas."""

from __future__ import annotations

from typing import Iterable, Sequence, Tuple


EndpointAlias = Tuple[str, str, Sequence[str]]


def register_legacy_endpoint_aliases(app, blueprint_name: str, aliases: Iterable[EndpointAlias]) -> None:
    """Mantiene endpoints historicos mientras las rutas migran a blueprints."""
    for endpoint, rule, methods in aliases:
        view_func = app.view_functions[f"{blueprint_name}.{endpoint}"]
        app.add_url_rule(rule, endpoint=endpoint, view_func=view_func, methods=list(methods))
