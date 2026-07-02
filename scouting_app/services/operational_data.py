"""Servicios de calidad y mantenimiento de la base operativa."""

from __future__ import annotations

from datetime import date
from typing import Any, Callable, Dict, Optional, Set, Tuple

from sqlalchemy import func
from sqlalchemy.orm import Session as SQLAlchemySession

from models import Player, PlayerAttributeHistory, PlayerStat
from player_logic import ATTRIBUTE_FIELDS, ATTRIBUTE_MIN_VALUE, default_player_photo_url, is_valid_eval_age


def trim_operational_player_pool(db_session: SQLAlchemySession, max_players: int) -> int:
    """Mantiene la base operativa en un maximo de jugadores evaluables."""
    total = db_session.query(func.count(Player.id)).scalar() or 0
    if total <= max_players:
        return 0

    stat_subq = (
        db_session.query(
            PlayerStat.player_id.label("player_id"),
            func.max(PlayerStat.record_date).label("last_stat_date"),
        )
        .group_by(PlayerStat.player_id)
        .subquery()
    )
    attr_subq = (
        db_session.query(
            PlayerAttributeHistory.player_id.label("player_id"),
            func.max(PlayerAttributeHistory.record_date).label("last_attr_date"),
        )
        .group_by(PlayerAttributeHistory.player_id)
        .subquery()
    )

    rows = (
        db_session.query(
            Player.id,
            Player.age,
            Player.birth_date,
            stat_subq.c.last_stat_date,
            attr_subq.c.last_attr_date,
        )
        .outerjoin(stat_subq, stat_subq.c.player_id == Player.id)
        .outerjoin(attr_subq, attr_subq.c.player_id == Player.id)
        .all()
    )

    def sort_key(row: Any) -> Tuple[int, int, date, int]:
        dates = [item_date for item_date in (row.last_stat_date, row.last_attr_date) if item_date is not None]
        last_activity = max(dates) if dates else date.min
        has_history = 1 if dates else 0
        current_age = Player.calculate_age_from_birth_date(row.birth_date) if row.birth_date else row.age
        in_eval_range = 1 if is_valid_eval_age(int(current_age or 0)) else 0
        return (has_history, in_eval_range, last_activity, row.id)

    rows_sorted = sorted(rows, key=sort_key, reverse=True)
    keep_ids: Set[int] = {row.id for row in rows_sorted[:max_players]}
    drop_ids = [row.id for row in rows_sorted if row.id not in keep_ids]

    if not drop_ids:
        return 0

    db_session.query(PlayerStat).filter(PlayerStat.player_id.in_(drop_ids)).delete(synchronize_session=False)
    db_session.query(PlayerAttributeHistory).filter(PlayerAttributeHistory.player_id.in_(drop_ids)).delete(synchronize_session=False)
    db_session.query(Player).filter(Player.id.in_(drop_ids)).delete(synchronize_session=False)
    return len(drop_ids)


def backfill_player_photo_urls(db_session: SQLAlchemySession) -> int:
    players = (
        db_session.query(Player)
        .filter((Player.photo_url == None) | (Player.photo_url == ""))  # noqa: E711
        .all()
    )
    updated = 0
    for player in players:
        player.photo_url = default_player_photo_url(
            name=player.name,
            national_id=player.national_id,
            fallback=str(player.id),
        )
        updated += 1
    return updated


def compute_operational_data_quality(db_session: SQLAlchemySession, eval_pool_max: int) -> Dict[str, int]:
    """Resume consistencia basica de la base operativa."""
    players = db_session.query(Player).all()
    players_total = len(players)
    invalid_age = sum(1 for player in players if not is_valid_eval_age(int(player.current_age or 0)))
    missing_birth_date = sum(1 for player in players if player.birth_date is None)
    missing_national_id = (
        db_session.query(func.count(Player.id))
        .filter((Player.national_id == None) | (func.trim(Player.national_id) == ""))  # noqa: E711
        .scalar()
        or 0
    )
    missing_name = (
        db_session.query(func.count(Player.id))
        .filter((Player.name == None) | (func.trim(Player.name) == ""))  # noqa: E711
        .scalar()
        or 0
    )
    missing_position = (
        db_session.query(func.count(Player.id))
        .filter((Player.position == None) | (func.trim(Player.position) == ""))  # noqa: E711
        .scalar()
        or 0
    )
    missing_photo_url = (
        db_session.query(func.count(Player.id))
        .filter((Player.photo_url == None) | (func.trim(Player.photo_url) == ""))  # noqa: E711
        .scalar()
        or 0
    )
    orphan_stats = (
        db_session.query(func.count(PlayerStat.id))
        .outerjoin(Player, Player.id == PlayerStat.player_id)
        .filter(Player.id == None)  # noqa: E711
        .scalar()
        or 0
    )
    orphan_attribute_history = (
        db_session.query(func.count(PlayerAttributeHistory.id))
        .outerjoin(Player, Player.id == PlayerAttributeHistory.player_id)
        .filter(Player.id == None)  # noqa: E711
        .scalar()
        or 0
    )
    over_limit_players = max(players_total - eval_pool_max, 0)

    return {
        "players_total": int(players_total),
        "missing_national_id": int(missing_national_id),
        "invalid_age": int(invalid_age),
        "missing_birth_date": int(missing_birth_date),
        "missing_name": int(missing_name),
        "missing_position": int(missing_position),
        "missing_photo_url": int(missing_photo_url),
        "orphan_stats": int(orphan_stats),
        "orphan_attribute_history": int(orphan_attribute_history),
        "over_limit_players": int(over_limit_players),
    }


def _attribute_row_from_player(player: Player) -> Dict[str, int]:
    return {field: int(getattr(player, field) or ATTRIBUTE_MIN_VALUE) for field in ATTRIBUTE_FIELDS}


def _attribute_row_from_entry(entry: PlayerAttributeHistory) -> Dict[str, int]:
    return {field: int(getattr(entry, field) or ATTRIBUTE_MIN_VALUE) for field in ATTRIBUTE_FIELDS}


def sync_player_attribute_history(
    player: Player,
    db_session: SQLAlchemySession,
    note: str = "Sincronizacion automatica de ficha",
) -> bool:
    """Asegura que el ultimo registro del historial tecnico refleje la ficha actual."""
    current_attrs = _attribute_row_from_player(player)
    latest = (
        db_session.query(PlayerAttributeHistory)
        .filter(PlayerAttributeHistory.player_id == player.id)
        .order_by(PlayerAttributeHistory.record_date.desc(), PlayerAttributeHistory.id.desc())
        .first()
    )
    if latest and _attribute_row_from_entry(latest) == current_attrs:
        return False

    entry = PlayerAttributeHistory(
        player_id=player.id,
        record_date=date.today(),
        notes=note,
        **current_attrs,
    )
    db_session.add(entry)
    return True


def sync_attribute_history_baseline(db_session: SQLAlchemySession) -> int:
    """Sincroniza historial tecnico para jugadores con ficha desfasada o sin historial."""
    players = db_session.query(Player).all()
    created = 0
    for player in players:
        if sync_player_attribute_history(player, db_session):
            created += 1
    return created


def cleanup_operational_data(
    db_session: SQLAlchemySession,
    eval_pool_max: int,
    enforce_eval_pool_limit: bool,
    sync_history: Optional[Callable[[SQLAlchemySession], int]] = None,
) -> Dict[str, int]:
    """Limpia inconsistencias legacy sin inventar datos faltantes."""
    invalid_player_ids = []
    for player in db_session.query(Player).all():
        has_required_text = all(
            (value or "").strip()
            for value in (player.name, player.position, player.national_id)
        )
        if not has_required_text or not is_valid_eval_age(int(player.current_age or 0)):
            invalid_player_ids.append(player.id)

    removed_invalid_players = len(invalid_player_ids)
    if invalid_player_ids:
        db_session.query(PlayerStat).filter(PlayerStat.player_id.in_(invalid_player_ids)).delete(synchronize_session=False)
        db_session.query(PlayerAttributeHistory).filter(
            PlayerAttributeHistory.player_id.in_(invalid_player_ids)
        ).delete(synchronize_session=False)
        db_session.query(Player).filter(Player.id.in_(invalid_player_ids)).delete(synchronize_session=False)

    photo_updates = backfill_player_photo_urls(db_session)
    history_updates = (sync_history or sync_attribute_history_baseline)(db_session)

    trimmed = 0
    if enforce_eval_pool_limit:
        trimmed = trim_operational_player_pool(db_session, eval_pool_max)

    quality_after = compute_operational_data_quality(db_session, eval_pool_max)
    return {
        "removed_invalid_players": int(removed_invalid_players),
        "photo_updates": int(photo_updates),
        "history_updates": int(history_updates),
        "trimmed_players": int(trimmed),
        **quality_after,
    }
