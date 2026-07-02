"""Rutas de jugadores, ficha, historial y proyeccion."""

from __future__ import annotations

import csv
import io
import json
import unicodedata
from datetime import date, datetime
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

from flask import Blueprint, Response, abort, flash, redirect, render_template, request, session, url_for
from sqlalchemy import desc, func
from sqlalchemy.orm import joinedload, load_only


def create_players_blueprint(*, deps: SimpleNamespace) -> Blueprint:
    bp = Blueprint("players", __name__)

    Session = deps.Session
    Player = deps.Player
    PlayerStat = deps.PlayerStat
    PlayerAttributeHistory = deps.PlayerAttributeHistory
    Match = deps.Match
    PlayerMatchParticipation = deps.PlayerMatchParticipation
    ScoutReport = deps.ScoutReport
    PhysicalAssessment = deps.PhysicalAssessment
    PlayerAvailability = deps.PlayerAvailability

    IMPORT_COLUMNS = [
        "Nombre",
        "DNI_ID",
        "FechaNacimiento",
        "Posicion",
        "Club",
        "Pais",
        "Ritmo",
        "Disparo",
        "Pase",
        "Regate",
        "Defensa",
        "Fisico",
        "Vision",
        "Marcaje",
        "Determinacion",
        "Tecnica",
        "AltoPotencial",
        "FotoURL",
    ]

    IMPORT_COLUMN_ALIASES = {
        "Nombre": ("nombre", "name"),
        "DNI_ID": ("dniid", "dni", "documento", "id", "dni/id"),
        "FechaNacimiento": ("fechanacimiento", "fecha nacimiento", "nacimiento", "birthdate"),
        "Posicion": ("posicion", "puesto", "position"),
        "Club": ("club", "equipo"),
        "Pais": ("pais", "country"),
        "Ritmo": ("ritmo", "pace"),
        "Disparo": ("disparo", "shooting"),
        "Pase": ("pase", "passing"),
        "Regate": ("regate", "dribbling"),
        "Defensa": ("defensa", "defending"),
        "Fisico": ("fisico", "physical"),
        "Vision": ("vision",),
        "Marcaje": ("marcaje", "tackling"),
        "Determinacion": ("determinacion", "determination"),
        "Tecnica": ("tecnica", "technique"),
        "AltoPotencial": ("altopotencial", "alto potencial", "potential", "potencial"),
        "FotoURL": ("fotourl", "foto url", "foto", "photo", "photourl"),
    }

    def _optional_strip(value: Optional[str]) -> Optional[str]:
        stripped = (value or "").strip()
        return stripped or None

    def _parse_optional_date_field(value: Optional[str], errors: List[str], label: str) -> Optional[date]:
        raw = (value or "").strip()
        if not raw:
            return None
        try:
            parsed = datetime.strptime(raw, "%Y-%m-%d").date()
        except ValueError:
            errors.append(f"{label} debe tener formato YYYY-MM-DD.")
            return None
        if parsed > date.today():
            errors.append(f"{label} no puede ser futura.")
            return None
        return parsed

    def _age_from_birth_date(value: date) -> int:
        return Player.calculate_age_from_birth_date(value)

    def _normalize_import_header(value: Optional[str]) -> str:
        raw = (value or "").strip().lower()
        normalized = unicodedata.normalize("NFKD", raw)
        ascii_text = "".join(char for char in normalized if not unicodedata.combining(char))
        return "".join(char for char in ascii_text if char.isalnum())

    def _import_value(row: Dict[str, str], header_map: Dict[str, str], column: str) -> str:
        source = header_map.get(column)
        if not source:
            return ""
        return (row.get(source) or "").strip()

    def _build_import_header_map(fieldnames: Optional[List[str]]) -> tuple[Dict[str, str], List[str]]:
        header_map: Dict[str, str] = {}
        missing: List[str] = []
        normalized_sources = {
            _normalize_import_header(fieldname): fieldname
            for fieldname in (fieldnames or [])
            if fieldname
        }
        for column in IMPORT_COLUMNS:
            candidates = (column, *IMPORT_COLUMN_ALIASES.get(column, ()))
            match = next(
                (
                    normalized_sources.get(_normalize_import_header(candidate))
                    for candidate in candidates
                    if _normalize_import_header(candidate) in normalized_sources
                ),
                None,
            )
            if match:
                header_map[column] = match
            elif column != "FotoURL":
                missing.append(column)
        return header_map, missing

    def _decode_csv_upload(upload) -> str:
        raw = upload.read()
        if not raw:
            raise ValueError("El archivo esta vacio.")
        try:
            return raw.decode("utf-8-sig")
        except UnicodeDecodeError:
            return raw.decode("cp1252")

    def _csv_rows_from_upload(upload) -> List[Dict[str, str]]:
        text = _decode_csv_upload(upload)
        sample = text[:2048]
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;")
        except csv.Error:
            dialect = csv.excel
        reader = csv.DictReader(io.StringIO(text), dialect=dialect)
        header_map, missing = _build_import_header_map(reader.fieldnames)
        if missing:
            raise ValueError("Faltan columnas: " + ", ".join(missing) + ".")
        return [
            {column: _import_value(row, header_map, column) for column in IMPORT_COLUMNS}
            for row in reader
            if any((value or "").strip() for value in row.values())
        ]

    def _validate_import_rows(rows: List[Dict[str, str]], db) -> List[Dict[str, object]]:
        current_total = db.query(func.count(Player.id)).scalar() or 0
        available_slots = max(deps.EVAL_POOL_MAX - current_total, 0)
        seen_ids = set()
        valid_count = 0
        preview_rows: List[Dict[str, object]] = []

        for idx, row in enumerate(rows, start=1):
            errors: List[str] = []
            warnings: List[str] = []
            clean: Dict[str, object] = {}

            name = row.get("Nombre", "").strip()
            national_id = deps.normalize_identifier(row.get("DNI_ID"))
            raw_birth_date = row.get("FechaNacimiento", "").strip()
            birth_date = _parse_optional_date_field(raw_birth_date, errors, "La fecha de nacimiento")
            age = _age_from_birth_date(birth_date) if birth_date else 0
            position = deps.normalize_position_choice(row.get("Posicion"))
            club = _optional_strip(row.get("Club"))
            country = _optional_strip(row.get("Pais"))
            photo_url = _optional_strip(row.get("FotoURL"))

            if not name:
                errors.append("Nombre obligatorio.")
            if not national_id:
                errors.append("DNI/ID obligatorio y solo numerico.")
            elif national_id in seen_ids:
                errors.append("DNI/ID repetido en el archivo.")
            elif db.query(Player.id).filter(Player.national_id == national_id).first():
                errors.append("DNI/ID ya registrado.")
            if birth_date is None and not raw_birth_date:
                errors.append("Fecha de nacimiento obligatoria.")
            elif birth_date is not None and not deps.is_valid_eval_age(age):
                errors.append("La edad calculada por fecha de nacimiento debe estar entre 12 y 18.")

            attr_values: Dict[str, int] = {}
            import_attr_columns = {
                "Ritmo": "pace",
                "Disparo": "shooting",
                "Pase": "passing",
                "Regate": "dribbling",
                "Defensa": "defending",
                "Fisico": "physical",
                "Vision": "vision",
                "Marcaje": "tackling",
                "Determinacion": "determination",
                "Tecnica": "technique",
            }
            for column, field in import_attr_columns.items():
                value = deps.parse_int_field(row.get(column), default=-1)
                if not deps.is_valid_attribute(value):
                    errors.append(
                        f"{column} debe estar entre {deps.ATTRIBUTE_MIN_VALUE} y {deps.ATTRIBUTE_MAX_VALUE}."
                    )
                attr_values[field] = value

            if not errors:
                valid_count += 1
                seen_ids.add(national_id)
                if valid_count > available_slots:
                    errors.append(f"Supera el cupo disponible ({available_slots}).")

            clean.update(
                {
                    "name": name,
                    "national_id": national_id,
                    "birth_date": birth_date.isoformat() if birth_date else "",
                    "age": age,
                    "position": position,
                    "club": club or "",
                    "country": country or "",
                    "photo_url": photo_url or "",
                    "potential_label": deps.parse_bool(row.get("AltoPotencial")),
                    **attr_values,
                }
            )
            preview_rows.append(
                {
                    "line": idx,
                    "data": clean,
                    "category_year": birth_date.year if birth_date else None,
                    "errors": errors,
                    "warnings": warnings,
                    "is_valid": not errors,
                }
            )
        return preview_rows

    def _parse_required_text_field(value: Optional[str], label: str, errors: List[str]) -> str:
        stripped = (value or "").strip()
        if not stripped:
            errors.append(f"{label} es obligatorio.")
        return stripped

    def _parse_int_range_field(
        value: Optional[str],
        label: str,
        errors: List[str],
        minimum: int,
        maximum: int,
        default: Optional[int] = None,
    ) -> Optional[int]:
        parsed = deps.parse_int_field(value, default)
        if parsed is None or parsed < minimum or parsed > maximum:
            errors.append(f"{label} debe estar entre {minimum} y {maximum}.")
            return parsed
        return parsed

    def _parse_stat_form(form) -> tuple[List[str], Dict[str, Any]]:
        errors: List[str] = []
        values: Dict[str, Any] = {
            "record_date": deps.parse_date_field(form.get("record_date"), errors, "La fecha del registro"),
            "matches_played": deps.validate_non_negative_int_field(
                form.get("matches_played"),
                "Partidos jugados",
                errors,
            ),
            "goals": deps.validate_non_negative_int_field(form.get("goals"), "Goles", errors),
            "assists": deps.validate_non_negative_int_field(form.get("assists"), "Asistencias", errors),
            "minutes_played": deps.validate_non_negative_int_field(
                form.get("minutes_played"),
                "Minutos jugados",
                errors,
            ),
            "yellow_cards": deps.validate_non_negative_int_field(
                form.get("yellow_cards"),
                "Tarjetas amarillas",
                errors,
            ),
            "red_cards": deps.validate_non_negative_int_field(form.get("red_cards"), "Tarjetas rojas", errors),
            "pass_accuracy": deps.validate_optional_float_range(
                form.get("pass_accuracy"),
                "Precision de pase",
                errors,
                0,
                100,
            ),
            "shot_accuracy": deps.validate_optional_float_range(
                form.get("shot_accuracy"),
                "Precision de remate",
                errors,
                0,
                100,
            ),
            "duels_won_pct": deps.validate_optional_float_range(
                form.get("duels_won_pct"),
                "Duelos ganados",
                errors,
                0,
                100,
            ),
            "final_score": deps.validate_optional_float_range(
                form.get("final_score"),
                "Valoracion final",
                errors,
                1,
                10,
            ),
            "notes": form.get("notes") or None,
        }
        return errors, values

    def _apply_stat_values(stat, values: Dict[str, Any]) -> None:
        for field, value in values.items():
            setattr(stat, field, value)
        if stat.final_score is None:
            stat.final_score = deps.calculate_stats_rating(
                {
                    "matches": stat.matches_played,
                    "goals": stat.goals,
                    "assists": stat.assists,
                    "minutes": stat.minutes_played,
                    "pass_pct": stat.pass_accuracy,
                    "shot_pct": stat.shot_accuracy,
                    "duels_pct": stat.duels_won_pct,
                }
            )

    def _parse_attribute_history_form(form) -> tuple[List[str], Dict[str, Any], bool]:
        errors: List[str] = []
        values: Dict[str, Any] = {
            "record_date": deps.parse_date_field(form.get("record_date"), errors, "La fecha del historial"),
            "notes": form.get("notes") or None,
        }
        has_any_value = False
        for field in deps.ATTRIBUTE_FIELDS:
            raw = form.get(field)
            value = deps.parse_int_field(raw, default=-1) if raw not in (None, "") else None
            if value is not None and not deps.is_valid_attribute(value):
                errors.append(
                    f"{deps.ATTRIBUTE_LABELS[field]} debe estar entre "
                    f"{deps.ATTRIBUTE_MIN_VALUE} y {deps.ATTRIBUTE_MAX_VALUE}."
                )
                value = None
            values[field] = value
            if value is not None:
                has_any_value = True
        if not has_any_value:
            errors.append("Debes cargar al menos un atributo para guardar el historial.")
        return errors, values, has_any_value

    def _apply_attribute_history_to_player(player, db) -> None:
        history = (
            db.query(PlayerAttributeHistory)
            .filter(PlayerAttributeHistory.player_id == player.id)
            .order_by(PlayerAttributeHistory.record_date.asc(), PlayerAttributeHistory.id.asc())
            .all()
        )
        values = {field: getattr(player, field) for field in deps.ATTRIBUTE_FIELDS}
        for entry in history:
            for field in deps.ATTRIBUTE_FIELDS:
                value = getattr(entry, field)
                if value is not None:
                    values[field] = value
        for field, value in values.items():
            setattr(player, field, value)

    def _parse_match_history_form(form) -> tuple[List[str], Dict[str, Any], Dict[str, Any]]:
        errors: List[str] = []
        match_values: Dict[str, Any] = {
            "match_date": deps.parse_date_field(form.get("match_date"), errors, "La fecha del partido"),
            "opponent_name": _parse_required_text_field(form.get("opponent_name"), "El rival", errors),
            "opponent_level": _parse_int_range_field(form.get("opponent_level"), "Nivel del rival", errors, 1, 5, 3),
            "tournament": _optional_strip(form.get("tournament")),
            "competition_category": _optional_strip(form.get("competition_category")),
            "venue": (form.get("venue") or "Local").strip() or "Local",
            "notes": _optional_strip(form.get("match_notes")),
        }
        participation_values: Dict[str, Any] = {
            "started": deps.parse_bool(form.get("started")),
            "position_played": deps.normalize_position_choice(form.get("position_played")),
            "minutes_played": deps.validate_non_negative_int_field(
                form.get("minutes_played"),
                "Minutos jugados",
                errors,
            ),
            "final_score": deps.validate_optional_float_range(
                form.get("final_score"),
                "Valoracion final",
                errors,
                1,
                10,
            ),
            "goals": deps.validate_non_negative_int_field(form.get("goals"), "Goles", errors),
            "assists": deps.validate_non_negative_int_field(form.get("assists"), "Asistencias", errors),
            "pass_accuracy": deps.validate_optional_float_range(
                form.get("pass_accuracy"),
                "Precision de pase",
                errors,
                0,
                100,
            ),
            "shot_accuracy": deps.validate_optional_float_range(
                form.get("shot_accuracy"),
                "Precision de remate",
                errors,
                0,
                100,
            ),
            "duels_won_pct": deps.validate_optional_float_range(
                form.get("duels_won_pct"),
                "Duelos ganados",
                errors,
                0,
                100,
            ),
            "yellow_cards": deps.validate_non_negative_int_field(
                form.get("yellow_cards"),
                "Tarjetas amarillas",
                errors,
            ),
            "red_cards": deps.validate_non_negative_int_field(form.get("red_cards"), "Tarjetas rojas", errors),
            "role_notes": _optional_strip(form.get("role_notes")),
        }
        return errors, match_values, participation_values

    def _parse_physical_assessment_form(form) -> tuple[List[str], Dict[str, Any]]:
        errors: List[str] = []
        values: Dict[str, Any] = {
            "assessment_date": deps.parse_date_field(form.get("assessment_date"), errors, "La fecha fisica"),
            "height_cm": deps.validate_optional_float_range(form.get("height_cm"), "Altura", errors, 100, 230),
            "weight_kg": deps.validate_optional_float_range(form.get("weight_kg"), "Peso", errors, 30, 130),
            "dominant_foot": _optional_strip(form.get("dominant_foot")),
            "estimated_speed": deps.validate_optional_float_range(
                form.get("estimated_speed"),
                "Velocidad estimada",
                errors,
                deps.ATTRIBUTE_MIN_VALUE,
                deps.ATTRIBUTE_MAX_VALUE,
            ),
            "endurance": deps.validate_optional_float_range(
                form.get("endurance"),
                "Resistencia",
                errors,
                deps.ATTRIBUTE_MIN_VALUE,
                deps.ATTRIBUTE_MAX_VALUE,
            ),
            "in_growth_spurt": deps.parse_bool(form.get("in_growth_spurt")),
            "notes": _optional_strip(form.get("notes")),
        }
        return errors, values

    def _parse_availability_form(form) -> tuple[List[str], Dict[str, Any]]:
        errors: List[str] = []
        values: Dict[str, Any] = {
            "record_date": deps.parse_date_field(form.get("record_date"), errors, "La fecha de disponibilidad"),
            "availability_pct": deps.validate_optional_float_range(
                form.get("availability_pct"),
                "Disponibilidad",
                errors,
                0,
                100,
            ),
            "fatigue_pct": deps.validate_optional_float_range(form.get("fatigue_pct"), "Fatiga", errors, 0, 100),
            "training_load_pct": deps.validate_optional_float_range(
                form.get("training_load_pct"),
                "Carga de entrenamiento",
                errors,
                0,
                100,
            ),
            "missed_days": deps.validate_non_negative_int_field(form.get("missed_days"), "Dias perdidos", errors),
            "injury_flag": deps.parse_bool(form.get("injury_flag")),
            "notes": _optional_strip(form.get("notes")),
        }
        return errors, values

    def _parse_scout_report_form(form) -> tuple[List[str], Dict[str, Any]]:
        errors: List[str] = []
        values: Dict[str, Any] = {
            "report_date": deps.parse_date_field(form.get("report_date"), errors, "La fecha del reporte"),
            "decision_making": _parse_int_range_field(
                form.get("decision_making"),
                "Toma de decisiones",
                errors,
                deps.ATTRIBUTE_MIN_VALUE,
                deps.ATTRIBUTE_MAX_VALUE,
            ) if form.get("decision_making") not in (None, "") else None,
            "tactical_reading": _parse_int_range_field(
                form.get("tactical_reading"),
                "Lectura tactica",
                errors,
                deps.ATTRIBUTE_MIN_VALUE,
                deps.ATTRIBUTE_MAX_VALUE,
            ) if form.get("tactical_reading") not in (None, "") else None,
            "mental_profile": _parse_int_range_field(
                form.get("mental_profile"),
                "Perfil mental",
                errors,
                deps.ATTRIBUTE_MIN_VALUE,
                deps.ATTRIBUTE_MAX_VALUE,
            ) if form.get("mental_profile") not in (None, "") else None,
            "adaptability": _parse_int_range_field(
                form.get("adaptability"),
                "Adaptabilidad",
                errors,
                deps.ATTRIBUTE_MIN_VALUE,
                deps.ATTRIBUTE_MAX_VALUE,
            ) if form.get("adaptability") not in (None, "") else None,
            "observed_projection_score": deps.validate_optional_float_range(
                form.get("observed_projection_score"),
                "Proyeccion observada",
                errors,
                1,
                10,
            ),
            "notes": _optional_strip(form.get("notes")),
        }
        return errors, values

    def _commit_history_change(db, player) -> None:
        db.commit()
        deps.refresh_player_potential(player, db)
        db.commit()
        deps.invalidate_dashboard_cache()

    @bp.route("/players")
    @deps.login_required
    def index():
        search_term = request.args.get("q")
        pos_filter = request.args.get("position")
        club_filter = request.args.get("club")
        country_filter = request.args.get("country")
        top_potential = request.args.get("top_potential")
        order_attr = request.args.get("order_attr")
        page = request.args.get("page", 1, type=int)
        per_page = deps.PLAYER_LIST_PER_PAGE
        cache_key = f"players:{deps.current_role()}:{request.query_string.decode('utf-8', errors='ignore')}"
        can_cache = not session.get("_flashes")
        if can_cache:
            cached_html = deps.cache_get(cache_key)
            if cached_html is not None:
                return cached_html
        db = Session()
        player_list_columns = [
            Player.id,
            Player.name,
            Player.national_id,
            Player.age,
            Player.birth_date,
            Player.position,
            Player.club,
            Player.country,
            Player.photo_url,
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
            Player.potential_label,
        ]
        query = db.query(Player).options(load_only(*player_list_columns))
        pos_list = [row[0] for row in db.query(Player.position).distinct().all()]
        club_list = [row[0] for row in db.query(Player.club).distinct().all() if row[0]]
        country_list = [row[0] for row in db.query(Player.country).distinct().all() if row[0]]
        if search_term:
            query = query.filter(Player.name.ilike(f"%{search_term}%"))
        if pos_filter:
            query = query.filter(Player.position == pos_filter)
        if club_filter:
            query = query.filter(Player.club == club_filter)
        if country_filter:
            query = query.filter(Player.country == country_filter)
        if order_attr and hasattr(Player, order_attr):
            query = query.order_by(desc(getattr(Player, order_attr)))
        total = query.count()
        total_pages = (total + per_page - 1) // per_page
        players = query.offset((page - 1) * per_page).limit(per_page).all()
        if top_potential:
            players = query.all()
        player_ids = [player.id for player in players]
        avg_score_map: Dict[int, Optional[float]] = {}
        if player_ids:
            player_avg_rows = (
                db.query(PlayerStat.player_id, func.avg(PlayerStat.final_score))
                .filter(PlayerStat.player_id.in_(player_ids))
                .group_by(PlayerStat.player_id)
                .all()
            )
            avg_score_map = {
                player_id: (float(avg_score) if avg_score is not None else None)
                for player_id, avg_score in player_avg_rows
            }
        projections = deps.batch_project_players(players, avg_score_map)
        player_rows = []
        for player in players:
            projection = projections.get(player.id)
            if projection:
                combined_pct = projection["combined_prob"] * 100
                row = {
                    "player": player,
                    "photo_url": player.photo_url or deps.default_player_photo_url(
                        name=player.name,
                        national_id=player.national_id,
                        fallback=str(player.id),
                    ),
                    "category": projection["category"],
                    "probability": f"{combined_pct:.1f}%",
                    "prob_value": combined_pct,
                    "fit_score": projection["fit_score"],
                    "best_position": projection["recommended_position"],
                    "best_score": projection["recommended_score"],
                }
            else:
                attr_map = deps.player_attribute_map(player)
                best_position, best_score = deps.recommend_position_from_attrs(attr_map)
                fit_score = deps.weighted_score_from_attrs(attr_map, player.position)
                row = {
                    "player": player,
                    "photo_url": player.photo_url or deps.default_player_photo_url(
                        name=player.name,
                        national_id=player.national_id,
                        fallback=str(player.id),
                    ),
                    "category": "Sin datos suficientes",
                    "probability": "--",
                    "prob_value": None,
                    "fit_score": fit_score,
                    "best_position": best_position,
                    "best_score": best_score,
                }
            player_rows.append(row)

        if top_potential:
            player_rows = [
                item for item in player_rows
                if item["prob_value"] is not None and deps.is_high_potential_probability(item["prob_value"] / 100.0)
            ]
            player_rows.sort(
                key=lambda item: (item["prob_value"] is not None, item["prob_value"] or -1.0),
                reverse=True,
            )
            total = len(player_rows)
            total_pages = max(1, (total + per_page - 1) // per_page)
            page = min(max(page, 1), total_pages)
            start = (page - 1) * per_page
            end = start + per_page
            player_rows = player_rows[start:end]
        db.close()
        html = render_template(
            "players.html",
            players=player_rows,
            search_term=search_term,
            pos_list=pos_list,
            club_list=club_list,
            country_list=country_list,
            pos_filter=pos_filter,
            club_filter=club_filter,
            country_filter=country_filter,
            top_potential=top_potential,
            order_attr=order_attr,
            page=page,
            total_pages=total_pages,
            total_results=total,
        )
        if can_cache:
            deps.cache_set(cache_key, html)
        return html

    @bp.route("/player/<int:player_id>")
    @deps.login_required
    def player_detail(player_id: int):
        db = Session()
        player = db.query(Player).filter(Player.id == player_id).first()
        if not player:
            db.close()
            abort(404)
        history_synced = False
        if deps.can_edit_player_data():
            history_synced = deps.sync_player_attribute_history(
                player,
                db,
                note="Sincronizacion automatica al abrir ficha",
            )
        if history_synced:
            db.commit()
        stats = deps.fetch_player_stats(player_id, db_session=db)
        recent_stats = list(reversed(stats[-3:])) if stats else []
        attr_history = deps.fetch_attribute_history(player_id, db_session=db)
        recent_attributes = list(reversed(attr_history[-3:])) if attr_history else []
        attribute_summary = deps.summarize_attribute_history(attr_history)
        stats_summary = deps.summarize_stats(stats)
        match_history = (
            db.query(PlayerMatchParticipation)
            .options(joinedload(PlayerMatchParticipation.match))
            .join(Match, PlayerMatchParticipation.match_id == Match.id)
            .filter(PlayerMatchParticipation.player_id == player_id)
            .order_by(Match.match_date.desc(), PlayerMatchParticipation.id.desc())
            .limit(5)
            .all()
        )
        physical_history = (
            db.query(PhysicalAssessment)
            .filter(PhysicalAssessment.player_id == player_id)
            .order_by(PhysicalAssessment.assessment_date.desc(), PhysicalAssessment.id.desc())
            .limit(5)
            .all()
        )
        availability_history = (
            db.query(PlayerAvailability)
            .filter(PlayerAvailability.player_id == player_id)
            .order_by(PlayerAvailability.record_date.desc(), PlayerAvailability.id.desc())
            .limit(5)
            .all()
        )
        scout_reports = (
            db.query(ScoutReport)
            .filter(ScoutReport.player_id == player_id)
            .order_by(ScoutReport.report_date.desc(), ScoutReport.id.desc())
            .limit(5)
            .all()
        )
        db.close()
        player_photo_url = player.photo_url or deps.default_player_photo_url(
            name=player.name,
            national_id=player.national_id,
            fallback=str(player.id),
        )
        projection = deps.compute_projection(player, stats)
        attr_map = deps.player_attribute_map(player)
        best_position, best_position_score = deps.recommend_position_from_attrs(attr_map)
        current_fit = deps.weighted_score_from_attrs(attr_map, player.position)
        current_position = deps.normalized_position(player.position)
        position_ranking = [
            {
                "position": pos,
                "score": deps.weighted_score_from_attrs(attr_map, pos),
                "is_current": current_position == pos,
            }
            for pos in deps.POSITION_OPTIONS
        ]
        position_ranking.sort(key=lambda item: item["score"], reverse=True)
        technical_attributes = [
            {"label": "Ritmo", "value": player.pace},
            {"label": "Disparo", "value": player.shooting},
            {"label": "Pase", "value": player.passing},
            {"label": "Regate", "value": player.dribbling},
            {"label": "Defensa", "value": player.defending},
            {"label": "Físico", "value": player.physical},
            {"label": "Visión", "value": player.vision},
            {"label": "Marcaje", "value": player.tackling},
            {"label": "Determinación", "value": player.determination},
            {"label": "Técnica", "value": player.technique},
        ]

        def build_trait(name: str, score: float, strengths: str, follow_up: str, improvement: str) -> dict:
            band = deps.score_band(score)
            messaging = {"Alto": strengths, "Medio": follow_up, "Bajo": improvement}
            return {
                "name": name,
                "score": round(score, 1),
                "band": band,
                "message": messaging[band],
            }

        psychological_profile = [
            build_trait(
                "Resiliencia competitiva",
                player.determination,
                "Sostiene el esfuerzo bajo presión; indicado para partidos decisivos.",
                "Trabajar rutinas de respiración y feedback constante para fortalecer su respuesta en escenarios adversos.",
                "Recomendar intervención del área psicológica y refuerzo en hábitos de disciplina diaria.",
            ),
            build_trait(
                "Visión táctica",
                (player.vision + player.passing) / 2,
                "Lee espacios y acelera cambios de juego, facilita la progresión del equipo.",
                "Incrementar análisis de vídeo para mejorar la toma de decisiones en el último tercio.",
                "Diseñar ejercicios de toma de decisiones en superioridad/inferioridad numérica.",
            ),
            build_trait(
                "Creatividad ofensiva",
                (player.technique + player.dribbling) / 2,
                "Desborde y control diferenciales; puede romper líneas defensivas.",
                "Trabajar gestos técnicos específicos a alta velocidad para trasladar virtudes al contexto profesional.",
                "Enfocar sesiones en conducción orientada y confianza en el uno contra uno.",
            ),
            build_trait(
                "Liderazgo comunicacional",
                (player.vision + player.determination + player.passing) / 3,
                "Influye en sus compañeros y ordena fases ofensivas.",
                "Definir responsabilidades puntuales dentro del equipo para ganar protagonismo progresivo.",
                "Establecer mentoría con referentes del plantel y dinámicas de comunicación en cancha.",
            ),
        ]

        development_focus = deps.compute_suggestions(player, threshold=15, top_n=3)

        return render_template(
            "player_detail.html",
            player=player,
            player_photo_url=player_photo_url,
            technical_attributes=technical_attributes,
            radar_labels=[attr["label"] for attr in technical_attributes],
            radar_values=[attr["value"] for attr in technical_attributes],
            psychological_profile=psychological_profile,
            development_focus=development_focus,
            recent_stats=recent_stats,
            stats_summary=stats_summary,
            history_payload=deps.stats_chart_payload(stats),
            attribute_history=recent_attributes,
            attribute_summary=attribute_summary,
            attribute_payload=deps.attribute_chart_payload(attr_history),
            attribute_labels=deps.ATTRIBUTE_LABELS,
            match_history=match_history,
            physical_history=physical_history,
            availability_history=availability_history,
            scout_reports=scout_reports,
            projection=projection,
            best_position=best_position,
            best_position_score=best_position_score,
            current_fit=current_fit,
            current_position=current_position,
            position_ranking=position_ranking[:3],
            position_options=deps.POSITION_OPTIONS,
        )

    @bp.route("/player/<int:player_id>/matches/add", methods=["POST"])
    @deps.roles_required(deps.ROLE_ADMIN, deps.ROLE_SCOUT)
    def add_player_match_history(player_id: int):
        deps.require_csrf()
        db = Session()
        player = db.get(Player, player_id)
        if not player:
            db.close()
            abort(404)
        errors, match_values, participation_values = _parse_match_history_form(request.form)
        if errors:
            for message in errors:
                flash(message, "danger")
            db.close()
            return redirect(url_for("player_detail", player_id=player_id) + "#historiales-jugador")
        match = Match(**match_values)
        participation = PlayerMatchParticipation(player_id=player_id, match=match, **participation_values)
        db.add(participation)
        _commit_history_change(db, player)
        db.close()
        flash("Listo: se agrego el partido al historial del jugador.", "success")
        return redirect(url_for("player_detail", player_id=player_id) + "#historiales-jugador")

    @bp.route("/player/<int:player_id>/matches/<int:participation_id>/edit", methods=["POST"])
    @deps.roles_required(deps.ROLE_ADMIN, deps.ROLE_SCOUT)
    def edit_player_match_history(player_id: int, participation_id: int):
        deps.require_csrf()
        db = Session()
        player = db.get(Player, player_id)
        participation = db.get(PlayerMatchParticipation, participation_id)
        if not player or not participation or participation.player_id != player.id:
            db.close()
            abort(404)
        errors, match_values, participation_values = _parse_match_history_form(request.form)
        if errors:
            for message in errors:
                flash(message, "danger")
            db.close()
            return redirect(url_for("player_detail", player_id=player_id) + "#historiales-jugador")
        for field, value in match_values.items():
            setattr(participation.match, field, value)
        for field, value in participation_values.items():
            setattr(participation, field, value)
        _commit_history_change(db, player)
        db.close()
        flash("Listo: se actualizo el partido del jugador.", "success")
        return redirect(url_for("player_detail", player_id=player_id) + "#historiales-jugador")

    @bp.route("/player/<int:player_id>/matches/<int:participation_id>/delete", methods=["POST"])
    @deps.roles_required(deps.ROLE_ADMIN, deps.ROLE_SCOUT)
    def delete_player_match_history(player_id: int, participation_id: int):
        deps.require_csrf()
        db = Session()
        player = db.get(Player, player_id)
        participation = db.get(PlayerMatchParticipation, participation_id)
        if not player or not participation or participation.player_id != player.id:
            db.close()
            abort(404)
        match_id = participation.match_id
        db.delete(participation)
        db.flush()
        remaining = db.query(PlayerMatchParticipation.id).filter_by(match_id=match_id).first()
        if not remaining:
            match = db.get(Match, match_id)
            if match:
                db.delete(match)
        _commit_history_change(db, player)
        db.close()
        flash("Listo: se elimino el partido del historial del jugador.", "success")
        return redirect(url_for("player_detail", player_id=player_id) + "#historiales-jugador")

    @bp.route("/player/<int:player_id>/physical/add", methods=["POST"])
    @deps.roles_required(deps.ROLE_ADMIN, deps.ROLE_SCOUT)
    def add_player_physical_assessment(player_id: int):
        deps.require_csrf()
        db = Session()
        player = db.get(Player, player_id)
        if not player:
            db.close()
            abort(404)
        errors, values = _parse_physical_assessment_form(request.form)
        if errors:
            for message in errors:
                flash(message, "danger")
            db.close()
            return redirect(url_for("player_detail", player_id=player_id) + "#historiales-jugador")
        db.add(PhysicalAssessment(player_id=player_id, **values))
        _commit_history_change(db, player)
        db.close()
        flash("Listo: se agrego la evaluacion fisica.", "success")
        return redirect(url_for("player_detail", player_id=player_id) + "#historiales-jugador")

    @bp.route("/player/<int:player_id>/physical/<int:assessment_id>/edit", methods=["POST"])
    @deps.roles_required(deps.ROLE_ADMIN, deps.ROLE_SCOUT)
    def edit_player_physical_assessment(player_id: int, assessment_id: int):
        deps.require_csrf()
        db = Session()
        player = db.get(Player, player_id)
        assessment = db.get(PhysicalAssessment, assessment_id)
        if not player or not assessment or assessment.player_id != player.id:
            db.close()
            abort(404)
        errors, values = _parse_physical_assessment_form(request.form)
        if errors:
            for message in errors:
                flash(message, "danger")
            db.close()
            return redirect(url_for("player_detail", player_id=player_id) + "#historiales-jugador")
        for field, value in values.items():
            setattr(assessment, field, value)
        _commit_history_change(db, player)
        db.close()
        flash("Listo: se actualizo la evaluacion fisica.", "success")
        return redirect(url_for("player_detail", player_id=player_id) + "#historiales-jugador")

    @bp.route("/player/<int:player_id>/physical/<int:assessment_id>/delete", methods=["POST"])
    @deps.roles_required(deps.ROLE_ADMIN, deps.ROLE_SCOUT)
    def delete_player_physical_assessment(player_id: int, assessment_id: int):
        deps.require_csrf()
        db = Session()
        player = db.get(Player, player_id)
        assessment = db.get(PhysicalAssessment, assessment_id)
        if not player or not assessment or assessment.player_id != player.id:
            db.close()
            abort(404)
        db.delete(assessment)
        _commit_history_change(db, player)
        db.close()
        flash("Listo: se elimino la evaluacion fisica.", "success")
        return redirect(url_for("player_detail", player_id=player_id) + "#historiales-jugador")

    @bp.route("/player/<int:player_id>/availability/add", methods=["POST"])
    @deps.roles_required(deps.ROLE_ADMIN, deps.ROLE_SCOUT)
    def add_player_availability(player_id: int):
        deps.require_csrf()
        db = Session()
        player = db.get(Player, player_id)
        if not player:
            db.close()
            abort(404)
        errors, values = _parse_availability_form(request.form)
        if errors:
            for message in errors:
                flash(message, "danger")
            db.close()
            return redirect(url_for("player_detail", player_id=player_id) + "#historiales-jugador")
        db.add(PlayerAvailability(player_id=player_id, **values))
        _commit_history_change(db, player)
        db.close()
        flash("Listo: se agrego la disponibilidad.", "success")
        return redirect(url_for("player_detail", player_id=player_id) + "#historiales-jugador")

    @bp.route("/player/<int:player_id>/availability/<int:availability_id>/edit", methods=["POST"])
    @deps.roles_required(deps.ROLE_ADMIN, deps.ROLE_SCOUT)
    def edit_player_availability(player_id: int, availability_id: int):
        deps.require_csrf()
        db = Session()
        player = db.get(Player, player_id)
        availability = db.get(PlayerAvailability, availability_id)
        if not player or not availability or availability.player_id != player.id:
            db.close()
            abort(404)
        errors, values = _parse_availability_form(request.form)
        if errors:
            for message in errors:
                flash(message, "danger")
            db.close()
            return redirect(url_for("player_detail", player_id=player_id) + "#historiales-jugador")
        for field, value in values.items():
            setattr(availability, field, value)
        _commit_history_change(db, player)
        db.close()
        flash("Listo: se actualizo la disponibilidad.", "success")
        return redirect(url_for("player_detail", player_id=player_id) + "#historiales-jugador")

    @bp.route("/player/<int:player_id>/availability/<int:availability_id>/delete", methods=["POST"])
    @deps.roles_required(deps.ROLE_ADMIN, deps.ROLE_SCOUT)
    def delete_player_availability(player_id: int, availability_id: int):
        deps.require_csrf()
        db = Session()
        player = db.get(Player, player_id)
        availability = db.get(PlayerAvailability, availability_id)
        if not player or not availability or availability.player_id != player.id:
            db.close()
            abort(404)
        db.delete(availability)
        _commit_history_change(db, player)
        db.close()
        flash("Listo: se elimino la disponibilidad.", "success")
        return redirect(url_for("player_detail", player_id=player_id) + "#historiales-jugador")

    @bp.route("/player/<int:player_id>/reports/add", methods=["POST"])
    @deps.roles_required(deps.ROLE_ADMIN, deps.ROLE_SCOUT)
    def add_player_scout_report(player_id: int):
        deps.require_csrf()
        db = Session()
        player = db.get(Player, player_id)
        if not player:
            db.close()
            abort(404)
        errors, values = _parse_scout_report_form(request.form)
        if errors:
            for message in errors:
                flash(message, "danger")
            db.close()
            return redirect(url_for("player_detail", player_id=player_id) + "#historiales-jugador")
        db.add(ScoutReport(player_id=player_id, **values))
        _commit_history_change(db, player)
        db.close()
        flash("Listo: se agrego el reporte scout.", "success")
        return redirect(url_for("player_detail", player_id=player_id) + "#historiales-jugador")

    @bp.route("/player/<int:player_id>/reports/<int:report_id>/edit", methods=["POST"])
    @deps.roles_required(deps.ROLE_ADMIN, deps.ROLE_SCOUT)
    def edit_player_scout_report(player_id: int, report_id: int):
        deps.require_csrf()
        db = Session()
        player = db.get(Player, player_id)
        report = db.get(ScoutReport, report_id)
        if not player or not report or report.player_id != player.id:
            db.close()
            abort(404)
        errors, values = _parse_scout_report_form(request.form)
        if errors:
            for message in errors:
                flash(message, "danger")
            db.close()
            return redirect(url_for("player_detail", player_id=player_id) + "#historiales-jugador")
        for field, value in values.items():
            setattr(report, field, value)
        _commit_history_change(db, player)
        db.close()
        flash("Listo: se actualizo el reporte scout.", "success")
        return redirect(url_for("player_detail", player_id=player_id) + "#historiales-jugador")

    @bp.route("/player/<int:player_id>/reports/<int:report_id>/delete", methods=["POST"])
    @deps.roles_required(deps.ROLE_ADMIN, deps.ROLE_SCOUT)
    def delete_player_scout_report(player_id: int, report_id: int):
        deps.require_csrf()
        db = Session()
        player = db.get(Player, player_id)
        report = db.get(ScoutReport, report_id)
        if not player or not report or report.player_id != player.id:
            db.close()
            abort(404)
        db.delete(report)
        _commit_history_change(db, player)
        db.close()
        flash("Listo: se elimino el reporte scout.", "success")
        return redirect(url_for("player_detail", player_id=player_id) + "#historiales-jugador")

    @bp.route("/player/<int:player_id>/stats", methods=["GET", "POST"])
    @deps.login_required
    def player_stats(player_id: int):
        if request.method == "POST":
            deps.require_csrf()
            if not deps.can_edit_player_data():
                abort(403)

        db = Session()
        player = db.query(Player).filter(Player.id == player_id).first()
        if not player:
            db.close()
            abort(404)

        if request.method == "POST":
            action = request.form.get("action", "add")
            if action == "recalculate":
                deps.refresh_player_potential(player, db)
                db.commit()
                deps.invalidate_dashboard_cache()
                db.close()
                flash("Listo: se actualizo la proyeccion con los ultimos datos.", "success")
                return redirect(url_for("predict_player", player_id=player_id))
            errors, values = _parse_stat_form(request.form)
            if errors:
                for message in errors:
                    flash(message, "danger")
                stats = (
                    db.query(PlayerStat)
                    .filter(PlayerStat.player_id == player_id)
                    .order_by(PlayerStat.record_date.desc(), PlayerStat.id.desc())
                    .all()
                )
                summary = deps.summarize_stats(list(reversed(stats)))
                db.close()
                return render_template(
                    "player_stats.html",
                    player=player,
                    stats=stats,
                    summary=summary,
                )

            stat = PlayerStat(player_id=player_id)
            _apply_stat_values(stat, values)
            db.add(stat)
            db.commit()
            deps.refresh_player_potential(player, db)
            db.commit()
            deps.invalidate_dashboard_cache()
            db.close()
            flash("Listo: se agrego el registro al historial del jugador.", "success")
            return redirect(url_for("player_stats", player_id=player_id))

        stats = (
            db.query(PlayerStat)
            .filter(PlayerStat.player_id == player_id)
            .order_by(PlayerStat.record_date.desc(), PlayerStat.id.desc())
            .all()
        )
        db.close()
        summary = deps.summarize_stats(list(reversed(stats)))
        return render_template(
            "player_stats.html",
            player=player,
            stats=stats,
            summary=summary,
        )

    @bp.route("/player/<int:player_id>/stats/<int:stat_id>/edit", methods=["POST"])
    @deps.roles_required(deps.ROLE_ADMIN, deps.ROLE_SCOUT)
    def edit_player_stat(player_id: int, stat_id: int):
        deps.require_csrf()

        db = Session()
        player = db.get(Player, player_id)
        stat = db.get(PlayerStat, stat_id)
        if not player or not stat or stat.player_id != player.id:
            db.close()
            abort(404)

        errors, values = _parse_stat_form(request.form)
        if errors:
            for message in errors:
                flash(message, "danger")
            db.close()
            return redirect(url_for("player_stats", player_id=player_id))

        _apply_stat_values(stat, values)
        deps.refresh_player_potential(player, db)
        db.commit()
        deps.invalidate_dashboard_cache()
        db.close()
        flash("Listo: se actualizo el registro de rendimiento.", "success")
        return redirect(url_for("player_stats", player_id=player_id))

    @bp.route("/player/<int:player_id>/stats/<int:stat_id>/delete", methods=["POST"])
    @deps.roles_required(deps.ROLE_ADMIN, deps.ROLE_SCOUT)
    def delete_player_stat(player_id: int, stat_id: int):
        deps.require_csrf()

        db = Session()
        player = db.get(Player, player_id)
        stat = db.get(PlayerStat, stat_id)
        if not player or not stat or stat.player_id != player.id:
            db.close()
            abort(404)

        db.delete(stat)
        db.flush()
        deps.refresh_player_potential(player, db)
        db.commit()
        deps.invalidate_dashboard_cache()
        db.close()
        flash("Listo: se elimino el registro de rendimiento.", "success")
        return redirect(url_for("player_stats", player_id=player_id))

    @bp.route("/player/<int:player_id>/attributes", methods=["GET", "POST"])
    @deps.login_required
    def player_attributes(player_id: int):
        if request.method == "POST":
            deps.require_csrf()
            if not deps.can_edit_player_data():
                abort(403)

        db = Session()
        player = db.query(Player).filter(Player.id == player_id).first()
        if not player:
            db.close()
            abort(404)

        if request.method == "POST":
            action = request.form.get("action", "add")
            if action == "recalculate":
                deps.refresh_player_potential(player, db)
                db.commit()
                deps.invalidate_dashboard_cache()
                db.close()
                flash("Listo: se actualizo la proyeccion con los nuevos atributos.", "success")
                return redirect(url_for("predict_player", player_id=player_id))
            errors, values, _ = _parse_attribute_history_form(request.form)
            if errors:
                for message in errors:
                    flash(message, "danger")
                history = (
                    db.query(PlayerAttributeHistory)
                    .filter(PlayerAttributeHistory.player_id == player_id)
                    .order_by(PlayerAttributeHistory.record_date.desc(), PlayerAttributeHistory.id.desc())
                    .all()
                )
                ascending_history = list(reversed(history))
                summary = deps.summarize_attribute_history(ascending_history)
                payload = deps.attribute_chart_payload(ascending_history)
                db.close()
                return render_template(
                    "player_attributes.html",
                    player=player,
                    history=history,
                    summary=summary,
                    attribute_labels=deps.ATTRIBUTE_LABELS,
                    payload=payload,
                )
            entry = PlayerAttributeHistory(player_id=player_id)
            for field, value in values.items():
                setattr(entry, field, value)
            for field in deps.ATTRIBUTE_FIELDS:
                value = values[field]
                if value is not None:
                    setattr(player, field, value)
            db.add(entry)
            deps.refresh_player_potential(player, db)
            db.commit()
            deps.invalidate_dashboard_cache()
            db.close()
            flash("Listo: se guardo el historial de atributos.", "success")
            return redirect(url_for("player_attributes", player_id=player_id))

        history = (
            db.query(PlayerAttributeHistory)
            .filter(PlayerAttributeHistory.player_id == player_id)
            .order_by(PlayerAttributeHistory.record_date.desc(), PlayerAttributeHistory.id.desc())
            .all()
        )
        ascending_history = list(reversed(history))
        summary = deps.summarize_attribute_history(ascending_history)
        payload = deps.attribute_chart_payload(ascending_history)
        db.close()
        return render_template(
            "player_attributes.html",
            player=player,
            history=history,
            summary=summary,
            attribute_labels=deps.ATTRIBUTE_LABELS,
            payload=payload,
        )

    @bp.route("/player/<int:player_id>/attributes/<int:history_id>/edit", methods=["POST"])
    @deps.roles_required(deps.ROLE_ADMIN, deps.ROLE_SCOUT)
    def edit_player_attribute_history(player_id: int, history_id: int):
        deps.require_csrf()

        db = Session()
        player = db.get(Player, player_id)
        entry = db.get(PlayerAttributeHistory, history_id)
        if not player or not entry or entry.player_id != player.id:
            db.close()
            abort(404)

        errors, values, _ = _parse_attribute_history_form(request.form)
        if errors:
            for message in errors:
                flash(message, "danger")
            db.close()
            return redirect(url_for("player_attributes", player_id=player_id))

        for field, value in values.items():
            setattr(entry, field, value)
        _apply_attribute_history_to_player(player, db)
        deps.refresh_player_potential(player, db)
        db.commit()
        deps.invalidate_dashboard_cache()
        db.close()
        flash("Listo: se actualizo el historial de atributos.", "success")
        return redirect(url_for("player_attributes", player_id=player_id))

    @bp.route("/player/<int:player_id>/attributes/<int:history_id>/delete", methods=["POST"])
    @deps.roles_required(deps.ROLE_ADMIN, deps.ROLE_SCOUT)
    def delete_player_attribute_history(player_id: int, history_id: int):
        deps.require_csrf()

        db = Session()
        player = db.get(Player, player_id)
        entry = db.get(PlayerAttributeHistory, history_id)
        if not player or not entry or entry.player_id != player.id:
            db.close()
            abort(404)

        db.delete(entry)
        db.flush()
        _apply_attribute_history_to_player(player, db)
        deps.refresh_player_potential(player, db)
        db.commit()
        deps.invalidate_dashboard_cache()
        db.close()
        flash("Listo: se elimino el historial de atributos.", "success")
        return redirect(url_for("player_attributes", player_id=player_id))

    @bp.route("/edit_player/<int:player_id>", methods=["GET", "POST"])
    @deps.roles_required(deps.ROLE_ADMIN, deps.ROLE_SCOUT)
    def edit_player(player_id):
        if request.method == "POST":
            deps.require_csrf()

        db = Session()
        player = db.get(Player, player_id)
        if not player:
            db.close()
            abort(404)
        if request.method == "POST":
            errors: List[str] = []
            name = (request.form.get("name") or "").strip()
            national_id = deps.normalize_identifier(request.form.get("national_id"))
            raw_birth_date = request.form.get("birth_date")
            birth_date = _parse_optional_date_field(
                raw_birth_date,
                errors,
                "La fecha de nacimiento",
            )
            age = _age_from_birth_date(birth_date) if birth_date else 0
            position = deps.normalize_position_choice(request.form.get("position"))
            club = (request.form.get("club") or "").strip() or None
            country = (request.form.get("country") or "").strip() or None
            photo_url = (request.form.get("photo_url") or "").strip() or None
            if not name:
                errors.append("El nombre es obligatorio.")
            if not national_id:
                errors.append("El DNI debe contener solo números.")
            else:
                repeated = (
                    db.query(Player.id)
                    .filter(Player.national_id == national_id, Player.id != player.id)
                    .first()
                )
                if repeated:
                    errors.append("El DNI ingresado pertenece a otro jugador.")
            if birth_date is None and not (raw_birth_date or "").strip():
                errors.append("La fecha de nacimiento es obligatoria.")
            elif birth_date is not None and not deps.is_valid_eval_age(age):
                errors.append("La edad calculada por fecha de nacimiento debe estar entre 12 y 18 anos.")
            attr_values: Dict[str, int] = {}
            for field in deps.ATTRIBUTE_FIELDS:
                value = deps.parse_int_field(request.form.get(field), getattr(player, field))
                if not deps.is_valid_attribute(value):
                    errors.append(
                        f"{deps.ATTRIBUTE_LABELS[field]} debe estar entre "
                        f"{deps.ATTRIBUTE_MIN_VALUE} y {deps.ATTRIBUTE_MAX_VALUE}."
                    )
                attr_values[field] = value
            if errors:
                for message in errors:
                    flash(message, "danger")
                db.close()
                return render_template(
                    "edit_player.html",
                    player=player,
                    form_data=request.form,
                    position_options=deps.POSITION_OPTIONS,
                )
            player.name = name
            player.national_id = national_id
            player.age = age
            player.birth_date = birth_date
            player.sync_age_from_birth_date()
            player.position = position
            player.club = club
            player.country = country
            player.photo_url = photo_url or deps.default_player_photo_url(
                name=name,
                national_id=national_id,
                fallback=str(player.id),
            )
            for field, value in attr_values.items():
                setattr(player, field, value)
            player.potential_label = True if request.form.get("potential_label") == "1" else False
            deps.sync_player_attribute_history(player, db, note="Actualizacion de ficha")
            deps.refresh_player_potential(player, db)
            db.commit()
            deps.invalidate_dashboard_cache()
            db.close()
            flash("Listo: se actualizo la ficha del jugador.", "success")
            return redirect(url_for("player_detail", player_id=player_id))
        db.close()
        return render_template("edit_player.html", player=player, position_options=deps.POSITION_OPTIONS)

    @bp.route("/delete_player/<int:player_id>", methods=["POST"])
    @deps.roles_required(deps.ROLE_ADMIN, deps.ROLE_SCOUT)
    def delete_player(player_id):
        deps.require_csrf()

        db = Session()
        player = db.get(Player, player_id)
        if not player:
            db.close()
            abort(404)
        db.delete(player)
        db.commit()
        deps.invalidate_dashboard_cache()
        db.close()
        flash("Listo: se elimino el jugador del seguimiento.", "success")
        return redirect(url_for("index"))

    @bp.route("/player/<int:player_id>/predict")
    @deps.login_required
    def predict_player(player_id: int):
        if deps.get_model() is None:
            return "No hay calculo disponible todavia. Actualiza los datos desde Configuracion.", 500

        db = Session()
        player = db.query(Player).filter(Player.id == player_id).first()
        if not player:
            db.close()
            abort(404)
        if deps.can_edit_player_data():
            projection = deps.refresh_player_potential(player, db)
        else:
            projection = deps.compute_projection(player, db_session=db)
        player_view = SimpleNamespace(**player.to_dict())
        player_view.photo_url = player.photo_url or deps.default_player_photo_url(
            name=player.name,
            national_id=player.national_id,
            fallback=str(player.id),
        )
        player_view.potential_label = (
            deps.is_high_potential_probability(projection["combined_prob"])
            if projection else player.potential_label
        )
        suggestions = deps.compute_suggestions(player_view)
        attr_map = deps.player_attribute_map(player)
        best_position, best_position_score = deps.recommend_position_from_attrs(attr_map)
        current_fit = deps.weighted_score_from_attrs(attr_map, player.position)

        if not projection:
            stats_summary = deps.summarize_stats([])
            prob_base = prob_combined = 0.0
            prob_calibrated_combined = None
            category = "Sin datos suficientes"
            stats_history: List[Any] = []
        else:
            prob_base = projection["base_prob"]
            prob_combined = projection["combined_prob"]
            prob_calibrated_combined = projection.get("calibrated_combined_prob")
            category = projection["category"]
            stats_summary = projection["stats_summary"]
            stats_history = projection["history"]
        history_payload = deps.stats_chart_payload(stats_history)
        attribute_payload = deps.attribute_chart_payload(deps.fetch_attribute_history(player_id))
        if deps.can_edit_player_data():
            db.commit()
            deps.invalidate_dashboard_cache()
        else:
            db.rollback()
        db.close()

        return render_template(
            "prediction.html",
            player=player_view,
            probability=f"{prob_combined*100:.1f}%",
            probability_base=f"{prob_base*100:.1f}%",
            probability_calibrated=(
                f"{float(prob_calibrated_combined)*100:.1f}%"
                if prob_calibrated_combined is not None
                else None
            ),
            probability_delta=(prob_combined - prob_base) * 100,
            category=category,
            suggestions=suggestions,
            stats_summary=stats_summary,
            history_payload=history_payload,
            attribute_payload=attribute_payload,
            attribute_labels=deps.ATTRIBUTE_LABELS,
            best_position=best_position,
            best_position_score=best_position_score,
            current_fit=current_fit,
        )

    @bp.route("/players/manage", methods=["GET", "POST"])
    @deps.roles_required(deps.ROLE_ADMIN, deps.ROLE_SCOUT)
    def manage_players():
        if request.method == "POST":
            deps.require_csrf()

        if request.method == "POST":
            mode = request.form.get("mode", "single")
            db = Session()
            created: List[str] = []
            errors: List[str] = []
            try:
                if mode == "single":
                    current_total = db.query(func.count(Player.id)).scalar() or 0
                    if current_total >= deps.EVAL_POOL_MAX:
                        errors.append(
                            f"Se alcanzo el maximo de {deps.EVAL_POOL_MAX} jugadores evaluables. "
                            "Elimina o actualiza jugadores existentes."
                        )
                    name = (request.form.get("name") or "").strip()
                    national_id = deps.normalize_identifier(request.form.get("national_id"))
                    raw_birth_date = request.form.get("birth_date")
                    birth_date = _parse_optional_date_field(
                        raw_birth_date,
                        errors,
                        "La fecha de nacimiento",
                    )
                    age = _age_from_birth_date(birth_date) if birth_date else 0
                    position = deps.normalize_position_choice(request.form.get("position"))
                    club = (request.form.get("club") or "").strip() or None
                    country = (request.form.get("country") or "").strip() or None
                    photo_url = (request.form.get("photo_url") or "").strip() or None
                    if not name:
                        errors.append("El nombre es obligatorio.")
                    if not national_id:
                        errors.append("Ingresa un DNI/ID valido (solo numeros).")
                    else:
                        repeated = (
                            db.query(Player.id)
                            .filter(Player.national_id == national_id)
                            .first()
                        )
                        if repeated:
                            errors.append("Ese DNI ya esta registrado en la base.")
                    if birth_date is None and not (raw_birth_date or "").strip():
                        errors.append("La fecha de nacimiento es obligatoria.")
                    elif birth_date is not None and not deps.is_valid_eval_age(age):
                        errors.append("La edad calculada por fecha de nacimiento debe estar entre 12 y 18 anos.")
                    attr_values: Dict[str, int] = {}
                    for field in deps.ATTRIBUTE_FIELDS:
                        value = deps.parse_int_field(request.form.get(field), 10)
                        if not deps.is_valid_attribute(value):
                            errors.append(
                                f"{deps.ATTRIBUTE_LABELS[field]} debe ubicarse entre "
                                f"{deps.ATTRIBUTE_MIN_VALUE} y {deps.ATTRIBUTE_MAX_VALUE}."
                            )
                        attr_values[field] = value
                    if not errors:
                        player = Player(
                            name=name,
                            national_id=national_id,
                            age=age,
                            birth_date=birth_date,
                            position=position,
                            club=club,
                            country=country,
                            photo_url=photo_url or deps.default_player_photo_url(name=name, national_id=national_id),
                            potential_label=deps.parse_bool(request.form.get("potential_label")),
                            **attr_values,
                        )
                        player.sync_age_from_birth_date()
                        db.add(player)
                        db.flush()
                        deps.sync_player_attribute_history(player, db, note="Alta de jugador")
                        deps.refresh_player_potential(player, db)
                        db.commit()
                        deps.invalidate_dashboard_cache()
                        created.append(player.name)
                        flash(f"Listo: se agrego a {player.name} al seguimiento.", "success")
                        return redirect(url_for("manage_players"))
                elif mode == "bulk":
                    flash("La carga masiva ahora se realiza desde la importacion CSV con plantilla.", "warning")
                    return redirect(url_for("import_players"))
                else:
                    errors.append("Modo de carga desconocido.")
            finally:
                db.close()
            if errors:
                for message in errors:
                    flash(message, "danger")
        attribute_sequence = [(field, deps.ATTRIBUTE_LABELS[field]) for field in deps.ATTRIBUTE_FIELDS]
        db_metrics = Session()
        current_total = db_metrics.query(func.count(Player.id)).scalar() or 0
        db_metrics.close()
        return render_template(
            "manage_players.html",
            attribute_labels=deps.ATTRIBUTE_LABELS,
            attribute_sequence=attribute_sequence,
            position_options=deps.POSITION_OPTIONS,
            current_total=current_total,
            max_players=deps.EVAL_POOL_MAX,
        )

    @bp.route("/players/import/template.csv")
    @deps.roles_required(deps.ROLE_ADMIN, deps.ROLE_SCOUT)
    def download_players_import_template():
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(IMPORT_COLUMNS)
        writer.writerow(
            [
                "Juan Perez",
                "40111222",
                "2010-03-21",
                "Delantero",
                "Club Local",
                "Argentina",
                "15",
                "14",
                "13",
                "16",
                "10",
                "14",
                "15",
                "12",
                "16",
                "17",
                "1",
                "",
            ]
        )
        content = "\ufeff" + output.getvalue()
        return Response(
            content,
            mimetype="text/csv; charset=utf-8",
            headers={"Content-Disposition": "attachment; filename=plantilla_jugadores_tpscouting.csv"},
        )

    @bp.route("/players/import", methods=["GET", "POST"])
    @deps.roles_required(deps.ROLE_ADMIN, deps.ROLE_SCOUT)
    def import_players():
        preview_rows: Optional[List[Dict[str, object]]] = None
        import_payload = ""
        if request.method == "POST":
            deps.require_csrf()
            mode = request.form.get("mode", "preview")
            db = Session()
            try:
                if mode == "preview":
                    upload = request.files.get("csv_file")
                    if not upload or not upload.filename:
                        flash("Selecciona un archivo CSV generado desde la plantilla.", "danger")
                    else:
                        try:
                            raw_rows = _csv_rows_from_upload(upload)
                            preview_rows = _validate_import_rows(raw_rows, db)
                            import_payload = json.dumps(raw_rows, ensure_ascii=False)
                            valid_count = sum(1 for row in preview_rows if row["is_valid"])
                            if valid_count:
                                flash(f"Previsualizacion lista: {valid_count} filas validas.", "success")
                            else:
                                flash("La previsualizacion no tiene filas validas para importar.", "warning")
                        except (UnicodeDecodeError, ValueError, csv.Error) as exc:
                            flash(f"No se pudo leer el CSV: {exc}", "danger")
                elif mode == "confirm":
                    try:
                        raw_rows = json.loads(request.form.get("import_payload") or "[]")
                    except json.JSONDecodeError:
                        raw_rows = []
                    if not raw_rows:
                        flash("No hay filas previsualizadas para importar.", "danger")
                    else:
                        preview_rows = _validate_import_rows(raw_rows, db)
                        valid_rows = [row for row in preview_rows if row["is_valid"]]
                        imported = 0
                        for row in valid_rows:
                            data = row["data"]
                            birth_date = datetime.strptime(str(data["birth_date"]), "%Y-%m-%d").date()
                            age = _age_from_birth_date(birth_date)
                            player = Player(
                                name=str(data["name"]),
                                national_id=str(data["national_id"]),
                                age=age,
                                birth_date=birth_date,
                                position=str(data["position"]),
                                club=str(data["club"]) or None,
                                country=str(data["country"]) or None,
                                photo_url=str(data["photo_url"])
                                or deps.default_player_photo_url(
                                    name=str(data["name"]),
                                    national_id=str(data["national_id"]),
                                ),
                                pace=int(data["pace"]),
                                shooting=int(data["shooting"]),
                                passing=int(data["passing"]),
                                dribbling=int(data["dribbling"]),
                                defending=int(data["defending"]),
                                physical=int(data["physical"]),
                                vision=int(data["vision"]),
                                tackling=int(data["tackling"]),
                                determination=int(data["determination"]),
                                technique=int(data["technique"]),
                                potential_label=bool(data["potential_label"]),
                            )
                            db.add(player)
                            db.flush()
                            deps.sync_player_attribute_history(player, db, note="Importacion CSV de jugador")
                            deps.refresh_player_potential(player, db)
                            imported += 1
                        if imported:
                            db.commit()
                            deps.invalidate_dashboard_cache()
                            flash(f"Listo: se importaron {imported} jugadores desde CSV.", "success")
                            return redirect(url_for("import_players"))
                        db.rollback()
                        import_payload = json.dumps(raw_rows, ensure_ascii=False)
                        flash("No se importaron jugadores porque no habia filas validas.", "warning")
                else:
                    flash("Modo de importacion desconocido.", "danger")
            finally:
                db.close()

        if preview_rows is None:
            preview_rows = []
        valid_count = sum(1 for row in preview_rows if row["is_valid"])
        invalid_count = len(preview_rows) - valid_count
        db_metrics = Session()
        current_total = db_metrics.query(func.count(Player.id)).scalar() or 0
        db_metrics.close()
        return render_template(
            "import_players.html",
            import_columns=IMPORT_COLUMNS,
            preview_rows=preview_rows,
            import_payload=import_payload,
            valid_count=valid_count,
            invalid_count=invalid_count,
            current_total=current_total,
            max_players=deps.EVAL_POOL_MAX,
        )

    return bp
