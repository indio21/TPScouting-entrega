"""Rutas del dashboard principal."""

from __future__ import annotations

from datetime import date, datetime, timedelta
from types import SimpleNamespace
from typing import Dict, List, Optional

from flask import Blueprint, render_template, request
from sqlalchemy import func
from sqlalchemy.orm import load_only

RECENT_ACTIVITY_DAYS = 30
LOW_AVAILABILITY_THRESHOLD = 70
HIGH_FATIGUE_THRESHOLD = 70
DASHBOARD_TOP_N = 8


def create_dashboard_blueprint(*, deps: SimpleNamespace) -> Blueprint:
    bp = Blueprint("dashboard", __name__)

    Session = deps.Session
    Player = deps.Player
    PlayerStat = deps.PlayerStat
    PlayerAttributeHistory = deps.PlayerAttributeHistory
    Match = deps.Match
    PlayerMatchParticipation = deps.PlayerMatchParticipation
    ScoutReport = deps.ScoutReport
    PhysicalAssessment = deps.PhysicalAssessment
    PlayerAvailability = deps.PlayerAvailability

    def _date_label(value: Optional[date]) -> str:
        return value.strftime("%Y-%m-%d") if value else "Sin registros"

    def _set_latest(activity_map: Dict[int, Optional[date]], player_id: int, value: Optional[date]) -> None:
        if value is None:
            return
        current = activity_map.get(player_id)
        if current is None or value > current:
            activity_map[player_id] = value

    def _latest_by_player(rows: List[object]) -> Dict[int, object]:
        latest: Dict[int, object] = {}
        for row in rows:
            if row.player_id not in latest:
                latest[row.player_id] = row
        return latest

    def _availability_reasons(record) -> List[str]:
        reasons: List[str] = []
        if record.injury_flag:
            reasons.append("Lesion")
        if record.availability_pct is not None and record.availability_pct < LOW_AVAILABILITY_THRESHOLD:
            reasons.append("Disponibilidad baja")
        if record.fatigue_pct is not None and record.fatigue_pct > HIGH_FATIGUE_THRESHOLD:
            reasons.append("Fatiga alta")
        return reasons

    @bp.route("/dashboard")
    @deps.login_required
    def dashboard():
        period = request.args.get("period", "month")
        start_str = request.args.get("start")
        end_str = request.args.get("end")
        role = deps.current_role()
        dashboard_mode = "director" if role == deps.ROLE_DIRECTOR else "scout"
        dashboard_cache_key = f"dashboard:{dashboard_mode}:{period}:{start_str}:{end_str}"
        cached_html = deps.cache_get(dashboard_cache_key)
        if cached_html is not None:
            return cached_html

        today = date.today()
        if period == "custom":
            try:
                start_date = (
                    datetime.strptime(start_str, "%Y-%m-%d").date()
                    if start_str
                    else today - timedelta(days=30)
                )
            except ValueError:
                start_date = today - timedelta(days=30)
            try:
                end_date = (
                    datetime.strptime(end_str, "%Y-%m-%d").date()
                    if end_str
                    else today
                )
            except ValueError:
                end_date = today
        else:
            end_date = today
            delta = {
                "week": 7,
                "month": 30,
                "year": 365,
            }.get(period, 30)
            start_date = end_date - timedelta(days=delta)
        if start_date > end_date:
            start_date, end_date = end_date, start_date
        start_str = start_date.isoformat()
        end_str = end_date.isoformat()

        db = Session()
        pos_rows = db.query(Player.position, func.count(Player.id)).group_by(Player.position).all()
        positions = [deps.display_position_label(row[0]) if row[0] else "Sin posicion" for row in pos_rows]
        pos_values = [row[1] for row in pos_rows]
        total_players = db.query(func.count(Player.id)).scalar() or 0
        avg_attrs = db.query(
            func.avg(Player.pace),
            func.avg(Player.shooting),
            func.avg(Player.passing),
            func.avg(Player.dribbling),
            func.avg(Player.defending),
            func.avg(Player.physical),
            func.avg(Player.vision),
            func.avg(Player.tackling),
            func.avg(Player.determination),
            func.avg(Player.technique),
        ).one()
        avg_labels = [deps.ATTRIBUTE_LABELS[field] for field in deps.ATTRIBUTE_FIELDS]
        avg_values = [round(float(value), 2) if value is not None else 0 for value in avg_attrs]

        players = (
            db.query(Player)
            .options(
                load_only(
                    Player.id,
                    Player.name,
                    Player.age,
                    Player.birth_date,
                    Player.position,
                    Player.pace,
                    Player.shooting,
                    Player.passing,
                    Player.dribbling,
                    Player.defending,
                    Player.physical,
                    Player.vision,
                    Player.tackling,
                    Player.determination,
                    Player.technique,
                    Player.photo_url,
                )
            )
            .all()
        )
        player_map = {player.id: player for player in players}
        player_ids = list(player_map.keys())
        player_avg_rows = db.query(
            PlayerStat.player_id,
            func.avg(PlayerStat.final_score),
        ).group_by(PlayerStat.player_id).all()
        avg_score_map = {
            player_id: (float(avg_score) if avg_score is not None else None)
            for player_id, avg_score in player_avg_rows
        }
        top_potential = []
        category_counts = {
            "Alto potencial": 0,
            "Potencial medio": 0,
            "Bajo potencial": 0,
            "Sin datos": 0,
        }
        if players and deps.get_model() is not None:
            projections = deps.batch_project_players(players, avg_score_map)
            for player in players:
                projection = projections.get(player.id)
                if not projection:
                    continue
                top_potential.append(
                    {
                        "id": player.id,
                        "name": player.name,
                        "age": player.current_age,
                        "juvenile_category": player.category_year,
                        "position": deps.display_position_label(player.position),
                        "probability": projection["combined_prob"],
                        "category": projection["category"],
                    }
                )
                category_counts[projection["category"]] += 1
            top_potential.sort(key=lambda item: item["probability"], reverse=True)
            top_potential = top_potential[:10]
        else:
            category_counts["Sin datos"] = total_players

        high_potential_count = category_counts.get("Alto potencial", 0)
        high_potential_pct = round((high_potential_count / total_players) * 100, 1) if total_players else 0
        top_by_position = []
        seen_positions = set()
        for row in top_potential:
            if row["position"] in seen_positions:
                continue
            seen_positions.add(row["position"])
            top_by_position.append(row)
        top_by_position = top_by_position[:DASHBOARD_TOP_N]

        activity_map: Dict[int, Optional[date]] = {player.id: None for player in players}
        if player_ids:
            activity_sources = [
                db.query(PlayerStat.player_id, func.max(PlayerStat.record_date))
                .filter(PlayerStat.player_id.in_(player_ids))
                .group_by(PlayerStat.player_id)
                .all(),
                db.query(PlayerAttributeHistory.player_id, func.max(PlayerAttributeHistory.record_date))
                .filter(PlayerAttributeHistory.player_id.in_(player_ids))
                .group_by(PlayerAttributeHistory.player_id)
                .all(),
                db.query(ScoutReport.player_id, func.max(ScoutReport.report_date))
                .filter(ScoutReport.player_id.in_(player_ids))
                .group_by(ScoutReport.player_id)
                .all(),
                db.query(PhysicalAssessment.player_id, func.max(PhysicalAssessment.assessment_date))
                .filter(PhysicalAssessment.player_id.in_(player_ids))
                .group_by(PhysicalAssessment.player_id)
                .all(),
                db.query(PlayerAvailability.player_id, func.max(PlayerAvailability.record_date))
                .filter(PlayerAvailability.player_id.in_(player_ids))
                .group_by(PlayerAvailability.player_id)
                .all(),
                db.query(PlayerMatchParticipation.player_id, func.max(Match.match_date))
                .join(Match, PlayerMatchParticipation.match_id == Match.id)
                .filter(PlayerMatchParticipation.player_id.in_(player_ids))
                .group_by(PlayerMatchParticipation.player_id)
                .all(),
            ]
            for rows in activity_sources:
                for player_id, activity_date in rows:
                    _set_latest(activity_map, player_id, activity_date)

        recent_cutoff = today - timedelta(days=RECENT_ACTIVITY_DAYS)
        recent_activity_count = sum(1 for value in activity_map.values() if value and value >= recent_cutoff)
        stale_count = max(total_players - recent_activity_count, 0)
        followup_players = [
            {
                "id": player.id,
                "name": player.name,
                "age": player.current_age,
                "juvenile_category": player.category_year,
                "position": deps.display_position_label(player.position),
                "last_activity": activity_map.get(player.id),
                "last_activity_label": _date_label(activity_map.get(player.id)),
            }
            for player in players
            if activity_map.get(player.id) is None or activity_map[player.id] < recent_cutoff
        ]
        followup_players.sort(key=lambda item: (item["last_activity"] is not None, item["last_activity"] or date.min))
        followup_players = followup_players[:DASHBOARD_TOP_N]

        latest_availability = {}
        latest_stats = {}
        if player_ids:
            availability_rows = (
                db.query(PlayerAvailability)
                .filter(PlayerAvailability.player_id.in_(player_ids))
                .order_by(
                    PlayerAvailability.player_id.asc(),
                    PlayerAvailability.record_date.desc(),
                    PlayerAvailability.id.desc(),
                )
                .all()
            )
            latest_availability = _latest_by_player(availability_rows)
            stat_rows = (
                db.query(PlayerStat)
                .filter(PlayerStat.player_id.in_(player_ids), PlayerStat.final_score != None)
                .order_by(PlayerStat.player_id.asc(), PlayerStat.record_date.desc(), PlayerStat.id.desc())
                .all()
            )
            latest_stats = _latest_by_player(stat_rows)

        availability_alerts = []
        for player_id, record in latest_availability.items():
            reasons = _availability_reasons(record)
            if not reasons:
                continue
            player = player_map.get(player_id)
            if not player:
                continue
            availability_alerts.append(
                {
                    "id": player_id,
                    "name": player.name,
                    "age": player.current_age,
                    "juvenile_category": player.category_year,
                    "position": deps.display_position_label(player.position),
                    "date": record.record_date,
                    "availability_pct": record.availability_pct,
                    "fatigue_pct": record.fatigue_pct,
                    "reasons": ", ".join(reasons),
                }
            )
        availability_alerts.sort(
            key=lambda item: (
                item["availability_pct"] is None,
                item["availability_pct"] if item["availability_pct"] is not None else 101,
                -(item["fatigue_pct"] or 0),
            )
        )
        availability_alerts = availability_alerts[:DASHBOARD_TOP_N]
        availability_risk_count = len(
            [
                record
                for record in latest_availability.values()
                if _availability_reasons(record)
            ]
        )

        recent_match_count = 0
        recent_report_count = 0
        if player_ids:
            recent_match_count = (
                db.query(func.count(func.distinct(PlayerMatchParticipation.player_id)))
                .join(Match, PlayerMatchParticipation.match_id == Match.id)
                .filter(
                    PlayerMatchParticipation.player_id.in_(player_ids),
                    Match.match_date >= recent_cutoff,
                    Match.match_date <= today,
                )
                .scalar()
                or 0
            )
            recent_report_count = (
                db.query(func.count(func.distinct(ScoutReport.player_id)))
                .filter(
                    ScoutReport.player_id.in_(player_ids),
                    ScoutReport.report_date >= recent_cutoff,
                    ScoutReport.report_date <= today,
                )
                .scalar()
                or 0
            )

        latest_form = []
        for player_id, stat in latest_stats.items():
            player = player_map.get(player_id)
            if not player:
                continue
            latest_form.append(
                {
                    "id": player_id,
                    "name": player.name,
                    "age": player.current_age,
                    "juvenile_category": player.category_year,
                    "position": deps.display_position_label(player.position),
                    "score": stat.final_score,
                    "date": stat.record_date,
                }
            )
        latest_form.sort(key=lambda item: item["score"] if item["score"] is not None else -1, reverse=True)
        latest_form = latest_form[:DASHBOARD_TOP_N]

        stats_in_range = (
            db.query(PlayerStat)
            .filter(
                PlayerStat.record_date >= start_date,
                PlayerStat.record_date <= end_date,
                PlayerStat.final_score != None,
            )
            .order_by(
                PlayerStat.player_id.asc(),
                PlayerStat.record_date.asc(),
                PlayerStat.id.asc(),
            )
            .all()
        )
        evolution_map: Dict[int, Dict[str, object]] = {}
        for stat in stats_in_range:
            entry = evolution_map.setdefault(
                stat.player_id,
                {
                    "first_score": stat.final_score,
                    "first_date": stat.record_date,
                    "last_score": stat.final_score,
                    "last_date": stat.record_date,
                },
            )
            if stat.record_date < entry["first_date"]:
                entry["first_date"] = stat.record_date
                entry["first_score"] = stat.final_score
            if stat.record_date >= entry["last_date"]:
                entry["last_date"] = stat.record_date
                entry["last_score"] = stat.final_score

        top_evolution = []
        for player_id, values in evolution_map.items():
            first = values["first_score"]
            last = values["last_score"]
            if first is None or last is None:
                continue
            delta = round(float(last - first), 2)
            top_evolution.append(
                {
                    "id": player_id,
                    "name": player_map[player_id].name if player_id in player_map else f"Jugador {player_id}",
                    "age": player_map[player_id].current_age if player_id in player_map else None,
                    "juvenile_category": player_map[player_id].category_year if player_id in player_map else None,
                    "delta": delta,
                    "start": values["first_date"],
                    "end": values["last_date"],
                }
            )
        top_evolution.sort(key=lambda item: item["delta"], reverse=True)
        top_evolution = top_evolution[:DASHBOARD_TOP_N]
        final_score_avg = db.query(func.avg(PlayerStat.final_score)).filter(PlayerStat.final_score != None).scalar()
        final_score_avg = round(float(final_score_avg), 2) if final_score_avg is not None else None
        db.close()

        pot_labels = list(category_counts.keys())
        pot_values = [category_counts[label] for label in pot_labels]
        if dashboard_mode == "director":
            panel_title = "Estado del plantel"
            panel_subtitle = "Vista rapida para decisiones de seguimiento, disponibilidad y forma reciente."
            primary_rows = latest_form
            primary_title = "Mejor forma reciente"
            primary_empty = "Todavia no hay valoraciones recientes para ordenar el plantel."
            primary_kind = "form"
            secondary_title = "Alertas de disponibilidad"
            secondary_rows = availability_alerts
            secondary_empty = "No hay alertas fisicas cargadas en la ultima disponibilidad registrada."
            kpi_cards = [
                {"label": "Jugadores", "value": total_players, "hint": "Base operativa"},
                {"label": "Puntaje medio", "value": final_score_avg if final_score_avg is not None else "--", "hint": "Historial 1-10"},
                {"label": "Con partido reciente", "value": recent_match_count, "hint": f"Ultimos {RECENT_ACTIVITY_DAYS} dias"},
                {"label": "Alertas fisicas", "value": availability_risk_count, "hint": "Lesion, fatiga o baja disp."},
                {"label": "Sin seguimiento", "value": stale_count, "hint": f"Sin actividad {RECENT_ACTIVITY_DAYS} dias"},
            ]
        else:
            panel_title = "Mesa de scouting"
            panel_subtitle = "Prioriza talento, seguimiento pendiente y riesgos reales de la base evaluada."
            primary_rows = top_potential[:DASHBOARD_TOP_N]
            primary_title = "Prioridad de scouting"
            primary_empty = "Carga datos para ver jugadores priorizados por potencial."
            primary_kind = "potential"
            secondary_title = "A revisar"
            secondary_rows = followup_players
            secondary_empty = "Todos los jugadores tienen actividad reciente."
            kpi_cards = [
                {"label": "Jugadores evaluados", "value": total_players, "hint": "Base operativa"},
                {"label": "Alto potencial", "value": high_potential_count, "hint": f"{high_potential_pct}% del total"},
                {"label": "Seguimiento 30 dias", "value": recent_activity_count, "hint": "Con actividad reciente"},
                {"label": "A revisar", "value": stale_count, "hint": "Sin actividad reciente"},
                {"label": "Reportes scout", "value": recent_report_count, "hint": f"Ultimos {RECENT_ACTIVITY_DAYS} dias"},
            ]

        html = render_template(
            "dashboard.html",
            dashboard_mode=dashboard_mode,
            panel_title=panel_title,
            panel_subtitle=panel_subtitle,
            kpi_cards=kpi_cards,
            positions=positions,
            pos_values=pos_values,
            pot_labels=pot_labels,
            pot_values=pot_values,
            avg_labels=avg_labels,
            avg_values=avg_values,
            total_players=total_players,
            final_score_avg=final_score_avg,
            top_potential=top_potential,
            top_by_position=top_by_position,
            top_evolution=top_evolution,
            primary_rows=primary_rows,
            primary_title=primary_title,
            primary_empty=primary_empty,
            primary_kind=primary_kind,
            secondary_title=secondary_title,
            secondary_rows=secondary_rows,
            secondary_empty=secondary_empty,
            availability_alerts=availability_alerts,
            followup_players=followup_players,
            selected_period=period,
            start_date_str=start_str,
            end_date_str=end_str,
        )
        deps.cache_set(dashboard_cache_key, html)
        return html

    return bp
