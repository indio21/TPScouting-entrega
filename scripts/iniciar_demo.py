#!/usr/bin/env python
"""Prepara y ejecuta la demo local autocontenida para evaluar TPScouting.

La base creada por este script contiene exclusivamente datos sinteticos y queda
ignorada por Git. El comando es idempotente: si la base ya tiene jugadores,
conserva las modificaciones realizadas durante la evaluacion.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict


REPO_ROOT = Path(__file__).resolve().parents[1]
APP_DIR = REPO_ROOT / "scouting_app"
DEFAULT_DB_PATH = APP_DIR / "demo_profesor.db"
DEFAULT_PLAYERS = 60
DEFAULT_SEED = 42
DEFAULT_USERNAME = "profesor_demo"
DEFAULT_PASSWORD = "DemoProfesor123"

if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))


def sqlite_url(db_path: Path) -> str:
    return f"sqlite:///{db_path.resolve().as_posix()}"


def remove_demo_database(db_path: Path) -> None:
    """Elimina solo la base demo indicada y sus sidecars SQLite."""
    resolved = db_path.resolve()
    if resolved.suffix.lower() != ".db":
        raise ValueError("La ruta de demo debe terminar en .db.")
    for candidate in (resolved, Path(f"{resolved}-wal"), Path(f"{resolved}-shm")):
        if candidate.exists():
            candidate.unlink()


def configure_demo_environment(
    db_path: Path,
    players: int,
    seed: int,
    username: str,
    password: str,
) -> str:
    db_url = sqlite_url(db_path)
    os.environ["APP_DB_URL"] = db_url
    os.environ["TRAINING_DB_URL"] = db_url
    os.environ["DEMO_SEED_ON_STARTUP"] = "1"
    os.environ["DEMO_SEED_PLAYERS"] = str(players)
    os.environ["DEMO_SEED"] = str(seed)
    os.environ["ADMIN_USERNAME"] = username
    os.environ["ADMIN_PASSWORD"] = password
    os.environ["AUTO_TRAIN_ON_STARTUP"] = "0"
    os.environ["APP_SECRET_KEY"] = "tp-scouting-demo-local-no-usar-en-produccion"
    return db_url


def prepare_demo(db_path: Path, players: int, seed: int, username: str, password: str) -> Dict[str, int]:
    if players < 1:
        raise ValueError("La cantidad de jugadores debe ser mayor o igual a 1.")

    db_path.parent.mkdir(parents=True, exist_ok=True)
    configure_demo_environment(db_path, players, seed, username, password)

    from create_admin import main as create_admin
    from db_utils import create_app_engine
    from models import (
        PhysicalAssessment,
        Player,
        PlayerAttributeHistory,
        PlayerAvailability,
        PlayerMatchParticipation,
        PlayerStat,
        ScoutReport,
        User,
    )
    from seed_demo_data import main as seed_demo
    from sqlalchemy import func
    from sqlalchemy.orm import sessionmaker
    from werkzeug.security import check_password_hash

    if seed_demo() != 0:
        raise RuntimeError("No se pudieron generar los datos demo.")
    if create_admin() != 0:
        raise RuntimeError("No se pudo crear el administrador demo.")

    engine = create_app_engine(sqlite_url(db_path))
    Session = sessionmaker(bind=engine)
    db_session = Session()
    try:
        all_players = db_session.query(Player).all()
        demo_user = db_session.query(User).filter(User.username == username).one_or_none()
        summary = {
            "players": len(all_players),
            "missing_birth_date": sum(player.birth_date is None for player in all_players),
            "age_mismatches": sum(player.current_age != player.age for player in all_players),
            "missing_category": sum(player.category_year is None for player in all_players),
            "admins": int(db_session.query(func.count(User.id)).filter(User.role == "administrador").scalar() or 0),
            "credentials_match": int(bool(demo_user and check_password_hash(demo_user.password_hash, password))),
            "player_stats": int(db_session.query(func.count(PlayerStat.id)).scalar() or 0),
            "attribute_history": int(db_session.query(func.count(PlayerAttributeHistory.id)).scalar() or 0),
            "match_participations": int(db_session.query(func.count(PlayerMatchParticipation.id)).scalar() or 0),
            "scout_reports": int(db_session.query(func.count(ScoutReport.id)).scalar() or 0),
            "physical_assessments": int(db_session.query(func.count(PhysicalAssessment.id)).scalar() or 0),
            "availability_records": int(db_session.query(func.count(PlayerAvailability.id)).scalar() or 0),
        }
    finally:
        db_session.close()
        engine.dispose()

    if summary["players"] < 1:
        raise RuntimeError("La demo quedo sin jugadores.")
    if summary["missing_birth_date"] or summary["age_mismatches"] or summary["missing_category"]:
        raise RuntimeError(f"La demo no supero el control de edad/nacimiento/categoria: {summary}")
    if summary["admins"] < 1:
        raise RuntimeError("La demo quedo sin un usuario administrador.")
    if not summary["credentials_match"]:
        raise RuntimeError("Las credenciales existentes no coinciden. Usar --recrear para restaurar la demo.")
    related_fields = (
        "player_stats",
        "attribute_history",
        "match_participations",
        "scout_reports",
        "physical_assessments",
        "availability_records",
    )
    if any(summary[field] < 1 for field in related_fields):
        raise RuntimeError(f"Faltan historiales necesarios para evaluar la demo: {summary}")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepara e inicia la demo local de TPScouting.")
    parser.add_argument("--players", type=int, default=DEFAULT_PLAYERS, help="Jugadores sinteticos iniciales (default: 60)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Semilla reproducible (default: 42)")
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH, help="Ruta de la base SQLite local")
    parser.add_argument("--username", default=DEFAULT_USERNAME, help="Usuario administrador local")
    parser.add_argument("--password", default=DEFAULT_PASSWORD, help="Contrasena administrador local")
    parser.add_argument("--host", default="127.0.0.1", help="Host local (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=5000, help="Puerto local (default: 5000)")
    parser.add_argument("--recrear", action="store_true", help="Borra y vuelve a crear solamente la base demo indicada")
    parser.add_argument("--solo-preparar", action="store_true", help="Prepara y verifica los datos sin iniciar Flask")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    db_path = args.db_path.resolve()
    try:
        if args.recrear:
            remove_demo_database(db_path)
        summary = prepare_demo(db_path, args.players, args.seed, args.username, args.password)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print("\nDemo local verificada correctamente.")
    print(f"Base local: {db_path}")
    print(f"Jugadores disponibles: {summary['players']}")
    print("Fechas faltantes: 0 | Edades inconsistentes: 0 | Categorias faltantes: 0")
    print("Estadisticas, partidos, informes y evaluaciones relacionadas: OK")
    print(f"Usuario: {args.username}")
    print(f"Contrasena: {args.password}")

    if args.solo_preparar:
        return 0

    print(f"Abrir: http://{args.host}:{args.port}/")
    print("Para detener la aplicacion, presionar Ctrl+C.\n")
    from app import app

    app.run(host=args.host, port=args.port, debug=False, use_reloader=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
