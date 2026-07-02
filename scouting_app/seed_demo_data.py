"""Seed idempotente para demo Render.

Carga jugadores sinteticos solo si la base operativa esta vacia. Esta pensado para
despliegues Free donde no se versionan bases SQLite ni se usa una base de
entrenamiento persistente separada.
"""

from __future__ import annotations

import os
from pathlib import Path

from sqlalchemy import func
from sqlalchemy.orm import sessionmaker

from db_utils import create_app_engine, ensure_player_columns, normalize_db_url
from generate_data import DEFAULT_SEED, EVAL_MAX_AGE, EVAL_MIN_AGE, main as generate_demo_data
from models import Base, Player


def _env_flag(name: str, default: str = "0") -> bool:
    return (os.environ.get(name, default).strip().lower() in {"1", "true", "yes", "y", "si", "s", "on"})


def main() -> int:
    if not _env_flag("DEMO_SEED_ON_STARTUP"):
        print("Seed demo omitido: DEMO_SEED_ON_STARTUP no esta habilitado.")
        return 0

    base_dir = Path(__file__).resolve().parent
    db_url = normalize_db_url(os.environ.get("APP_DB_URL", "sqlite:///players_updated_v2.db"), base_dir=str(base_dir))
    player_count = max(1, int(os.environ.get("DEMO_SEED_PLAYERS", "100")))
    seed = int(os.environ.get("DEMO_SEED", str(DEFAULT_SEED)))

    engine = create_app_engine(db_url)
    Base.metadata.create_all(engine)
    ensure_player_columns(engine)
    Session = sessionmaker(bind=engine)
    db_session = Session()
    try:
        existing = db_session.query(func.count(Player.id)).scalar() or 0
    finally:
        db_session.close()

    if existing:
        print(f"Seed demo omitido: la base operativa ya tiene {existing} jugadores.")
        return 0

    generate_demo_data(
        player_count,
        db_url,
        seed=seed,
        min_age=EVAL_MIN_AGE,
        max_age=EVAL_MAX_AGE,
        reset_existing=False,
    )
    print(f"Seed demo completado: {player_count} jugadores generados.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
