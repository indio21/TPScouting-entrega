"""Utilidades compartidas para configuracion de base de datos."""

from __future__ import annotations

import os
from typing import Optional

from sqlalchemy import create_engine, event, inspect, text
from sqlalchemy.engine import Engine

from player_logic import ATTRIBUTE_FIELDS, ATTRIBUTE_MAX_VALUE, ATTRIBUTE_MIN_VALUE


def is_sqlite_url(url: str) -> bool:
    return url.startswith("sqlite:")


def normalize_db_url(url: str, base_dir: Optional[str] = None) -> str:
    """Normaliza URLs para SQLite local y PostgreSQL en despliegue."""
    value = (url or "").strip()
    if not value:
        raise ValueError("Database URL vacia.")

    if value.startswith("postgresql+"):
        return value

    if value.startswith("postgres://"):
        return value.replace("postgres://", "postgresql+psycopg://", 1)

    if value.startswith("postgresql://"):
        return value.replace("postgresql://", "postgresql+psycopg://", 1)

    if base_dir and value.startswith("sqlite:///") and not value.startswith("sqlite:////"):
        rel = value.replace("sqlite:///", "", 1)
        return "sqlite:///" + os.path.join(base_dir, rel).replace("\\", "/")

    return value


def create_app_engine(db_url: str) -> Engine:
    """Crea un engine SQLAlchemy con defaults razonables por dialecto."""
    if is_sqlite_url(db_url):
        engine = create_engine(db_url, connect_args={"check_same_thread": False})

        @event.listens_for(engine, "connect")
        def _set_sqlite_pragma(dbapi_connection, connection_record):
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON;")
            cursor.execute("PRAGMA journal_mode=WAL;")
            cursor.execute("PRAGMA synchronous=NORMAL;")
            cursor.execute("PRAGMA busy_timeout=5000;")
            cursor.close()

        return engine

    return create_engine(db_url, pool_pre_ping=True)


def _clamp_scale_columns(conn, table_name: str, columns: set[str], field_names: tuple[str, ...]) -> None:
    for field_name in field_names:
        if field_name not in columns:
            continue
        conn.execute(
            text(
                f"UPDATE {table_name} "
                f"SET {field_name} = CASE "
                f"WHEN {field_name} IS NULL THEN NULL "
                f"WHEN {field_name} < :min_value THEN :min_value "
                f"WHEN {field_name} > :max_value THEN :max_value "
                f"ELSE {field_name} END"
            ),
            {"min_value": ATTRIBUTE_MIN_VALUE, "max_value": ATTRIBUTE_MAX_VALUE},
        )


def ensure_player_columns(engine: Engine) -> int:
    """Asegura columnas optativas y de auditoría en bases existentes."""
    inspector = inspect(engine)
    table_specs = {
        "players": {
            "national_id": "TEXT" if engine.dialect.name == "sqlite" else "VARCHAR",
            "birth_date": "DATE",
            "photo_url": "TEXT" if engine.dialect.name == "sqlite" else "VARCHAR",
            "created_at": "TIMESTAMP",
            "updated_at": "TIMESTAMP",
        },
        "users": {
            "created_at": "TIMESTAMP",
            "updated_at": "TIMESTAMP",
        },
        "coaches": {
            "created_at": "TIMESTAMP",
            "updated_at": "TIMESTAMP",
        },
        "directors": {
            "created_at": "TIMESTAMP",
            "updated_at": "TIMESTAMP",
        },
        "player_stats": {
            "created_at": "TIMESTAMP",
            "updated_at": "TIMESTAMP",
        },
        "player_attribute_history": {
            "created_at": "TIMESTAMP",
            "updated_at": "TIMESTAMP",
        },
        "matches": {
            "created_at": "TIMESTAMP",
            "updated_at": "TIMESTAMP",
        },
        "player_match_participations": {
            "created_at": "TIMESTAMP",
            "updated_at": "TIMESTAMP",
        },
        "scout_reports": {
            "created_at": "TIMESTAMP",
            "updated_at": "TIMESTAMP",
        },
        "physical_assessments": {
            "created_at": "TIMESTAMP",
            "updated_at": "TIMESTAMP",
        },
        "player_availability": {
            "created_at": "TIMESTAMP",
            "updated_at": "TIMESTAMP",
        },
    }

    added_columns = 0
    with engine.begin() as conn:
        for table_name, columns in table_specs.items():
            if not inspector.has_table(table_name):
                continue
            existing = {col["name"] for col in inspector.get_columns(table_name)}
            for column_name, column_type in columns.items():
                if column_name in existing:
                    continue
                conn.execute(text(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type}"))
                added_columns += 1
            if "created_at" in columns and "updated_at" in columns:
                conn.execute(
                    text(
                        f"UPDATE {table_name} "
                        "SET created_at = COALESCE(created_at, CURRENT_TIMESTAMP), "
                        "updated_at = COALESCE(updated_at, CURRENT_TIMESTAMP)"
                    )
                )
            if table_name in {"players", "player_attribute_history"}:
                _clamp_scale_columns(conn, table_name, existing | set(columns), tuple(ATTRIBUTE_FIELDS))
            elif table_name == "scout_reports":
                _clamp_scale_columns(
                    conn,
                    table_name,
                    existing | set(columns),
                    ("decision_making", "tactical_reading", "mental_profile", "adaptability"),
                )
            elif table_name == "physical_assessments":
                _clamp_scale_columns(conn, table_name, existing | set(columns), ("estimated_speed", "endurance"))

    return added_columns
