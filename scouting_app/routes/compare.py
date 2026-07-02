"""Rutas de comparacion entre jugadores."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Dict, List

from flask import Blueprint, render_template, request, session
from sqlalchemy import func
from sqlalchemy.orm import load_only


def create_compare_blueprint(*, deps: SimpleNamespace) -> Blueprint:
    bp = Blueprint("compare", __name__)

    Session = deps.Session
    Player = deps.Player
    PlayerStat = deps.PlayerStat

    def player_photo_url(player) -> str:
        return player.photo_url or deps.default_player_photo_url(
            name=player.name,
            national_id=player.national_id,
            fallback=str(player.id),
        )

    @bp.route("/compare", methods=["GET", "POST"])
    @deps.login_required
    def compare_players():
        db = Session()

        rows = (
            db.query(Player.id, Player.name, Player.position)
            .order_by(Player.name.asc())
            .limit(deps.MAX_COMPARE_PLAYERS + 1)
            .all()
        )
        truncated = len(rows) > deps.MAX_COMPARE_PLAYERS
        if truncated:
            rows = rows[: deps.MAX_COMPARE_PLAYERS]

        players = [
            SimpleNamespace(id=row[0], name=row[1], position=row[2])
            for row in rows
        ]

        selected_one = None
        selected_two = None
        comparison = None
        target_position = None

        if request.method == "POST":
            selected_one = request.form.get("player_one", type=int)
            selected_two = request.form.get("player_two", type=int)
            target_position_raw = request.form.get("target_position")
            target_position = deps.normalized_position(target_position_raw) if target_position_raw else None

            if selected_one and selected_two and selected_one != selected_two:
                selected = (
                    db.query(Player)
                    .filter(Player.id.in_([selected_one, selected_two]))
                    .all()
                )
                players_map = {player.id: player for player in selected}
                player_one = players_map.get(selected_one)
                player_two = players_map.get(selected_two)

                if player_one and player_two:
                    stats_one = deps.fetch_player_stats(player_one.id, db_session=db)
                    stats_two = deps.fetch_player_stats(player_two.id, db_session=db)

                    summary_one = deps.summarize_stats(stats_one)
                    summary_two = deps.summarize_stats(stats_two)
                    avg_score_map = {
                        player_one.id: summary_one.get("avg_final_score"),
                        player_two.id: summary_two.get("avg_final_score"),
                    }

                    projections = deps.batch_project_players([player_one, player_two], avg_score_map)
                    projection_one = projections.get(player_one.id)
                    projection_two = projections.get(player_two.id)

                    prob_one = projection_one["combined_prob"] if projection_one else 0.0
                    prob_two = projection_two["combined_prob"] if projection_two else 0.0

                    attr_rows = []
                    score_one = 0.0
                    score_two = 0.0

                    base_position = target_position or deps.normalized_position(player_one.position or player_two.position)
                    weights_map = deps.position_weights(base_position)

                    for field in deps.ATTRIBUTE_FIELDS:
                        label = deps.ATTRIBUTE_LABELS[field]
                        value_one = getattr(player_one, field)
                        value_two = getattr(player_two, field)
                        weight = weights_map.get(field, 0.0)

                        if value_one > value_two:
                            winner = 1
                            score_one += weight
                        elif value_two > value_one:
                            winner = 2
                            score_two += weight
                        else:
                            winner = 0

                        attr_rows.append(
                            {
                                "label": label,
                                "value_one": value_one,
                                "value_two": value_two,
                                "weight": weight,
                                "winner": winner,
                            }
                        )

                    avg_one = summary_one.get("avg_final_score")
                    avg_two = summary_two.get("avg_final_score")

                    attr_map_one = deps.player_attribute_map(player_one)
                    attr_map_two = deps.player_attribute_map(player_two)
                    total_one = deps.weighted_score_from_attrs(attr_map_one, base_position) + (avg_one or 0)
                    total_two = deps.weighted_score_from_attrs(attr_map_two, base_position) + (avg_two or 0)
                    best_pos_one, best_pos_one_score = deps.recommend_position_from_attrs(attr_map_one)
                    best_pos_two, best_pos_two_score = deps.recommend_position_from_attrs(attr_map_two)
                    fit_one = deps.weighted_score_from_attrs(attr_map_one, base_position)
                    fit_two = deps.weighted_score_from_attrs(attr_map_two, base_position)

                    if total_one > total_two:
                        conclusion = (
                            f"{player_one.name} presenta mejores indicadores generales "
                            f"respecto a {player_two.name}."
                        )
                    elif total_two > total_one:
                        conclusion = (
                            f"{player_two.name} presenta mejores indicadores generales "
                            f"respecto a {player_one.name}."
                        )
                    else:
                        conclusion = (
                            "Ambos jugadores presentan indicadores equivalentes "
                            "en la comparación."
                        )

                    comparison = {
                        "player_one": {
                            "id": player_one.id,
                            "name": player_one.name,
                            "position": player_one.position,
                            "age": player_one.current_age,
                            "juvenile_category": player_one.category_year,
                            "club": player_one.club,
                            "country": player_one.country,
                            "photo_url": player_photo_url(player_one),
                            "probability": prob_one,
                            "avg_score": avg_one,
                            "fit_score": fit_one,
                            "best_position": best_pos_one,
                            "best_position_score": best_pos_one_score,
                        },
                        "player_two": {
                            "id": player_two.id,
                            "name": player_two.name,
                            "position": player_two.position,
                            "age": player_two.current_age,
                            "juvenile_category": player_two.category_year,
                            "club": player_two.club,
                            "country": player_two.country,
                            "photo_url": player_photo_url(player_two),
                            "probability": prob_two,
                            "avg_score": avg_two,
                            "fit_score": fit_two,
                            "best_position": best_pos_two,
                            "best_position_score": best_pos_two_score,
                        },
                        "attributes": attr_rows,
                        "score_one": score_one,
                        "score_two": score_two,
                        "conclusion": conclusion,
                        "target_position": base_position,
                    }

        db.close()
        return render_template(
            "compare.html",
            players=players,
            selected_one=selected_one,
            selected_two=selected_two,
            comparison=comparison,
            truncated=truncated,
            max_players=deps.MAX_COMPARE_PLAYERS,
            target_position=target_position,
        )

    @bp.route("/compare/multi", methods=["GET", "POST"])
    @deps.login_required
    def compare_multi():
        cache_key = f"compare:multi:{deps.current_role()}:{request.query_string.decode('utf-8', errors='ignore')}"
        if request.method == "GET" and not session.get("_flashes"):
            cached_html = deps.cache_get(cache_key)
            if cached_html is not None:
                return cached_html
        db = Session()

        rows = (
            db.query(Player.id, Player.name, Player.position)
            .order_by(Player.name.asc())
            .limit(deps.MAX_COMPARE_PLAYERS + 1)
            .all()
        )
        truncated = len(rows) > deps.MAX_COMPARE_PLAYERS
        if truncated:
            rows = rows[: deps.MAX_COMPARE_PLAYERS]

        players = [
            SimpleNamespace(id=row[0], name=row[1], position=row[2])
            for row in rows
        ]

        selected_ids: List[int] = []
        comparison = None
        target_position = None

        position_tabs = [
            {"key": "arquero", "label": "Arqueros", "bucket": "Portero"},
            {"key": "defensa", "label": "Defensas", "bucket": "Defensa"},
            {"key": "mediocampo", "label": "Mediocampistas", "bucket": "Mediocampista"},
            {"key": "delantera", "label": "Delanteros", "bucket": "Delantero"},
        ]
        ranking_by_position: Dict[str, List[Dict[str, object]]] = {tab["key"]: [] for tab in position_tabs}

        all_players_for_ranking = (
            db.query(Player)
            .options(
                load_only(
                    Player.id,
                    Player.name,
                    Player.national_id,
                    Player.age,
                    Player.birth_date,
                    Player.position,
                    Player.club,
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
                )
            )
            .all()
        )
        ranking_avg_rows = (
            db.query(PlayerStat.player_id, func.avg(PlayerStat.final_score))
            .group_by(PlayerStat.player_id)
            .all()
        )
        ranking_avg_score_map = {
            player_id: (float(avg_score) if avg_score is not None else None)
            for player_id, avg_score in ranking_avg_rows
        }
        ranking_projections = deps.batch_project_players(all_players_for_ranking, ranking_avg_score_map)
        for player in all_players_for_ranking:
            projection = ranking_projections.get(player.id)
            if not projection:
                continue
            normalized_pos = deps.normalized_position(player.position)
            if normalized_pos == "Lateral":
                normalized_pos = "Defensa"

            tab = next((tab for tab in position_tabs if tab["bucket"] == normalized_pos), None)
            if not tab:
                continue

            attr_map = deps.player_attribute_map(player)
            fit_score = deps.weighted_score_from_attrs(attr_map, tab["bucket"])

            ranking_by_position[tab["key"]].append(
                {
                    "id": player.id,
                    "name": player.name,
                    "position": player.position,
                    "club": player.club,
                    "age": player.current_age,
                    "juvenile_category": player.category_year,
                    "photo_url": player_photo_url(player),
                    "fit_score": round(float(fit_score), 1),
                    "probability": round(float(projection["combined_prob"]) * 100, 1),
                    "category": projection["category"],
                }
            )

        talent_map_rows: List[Dict[str, object]] = []
        for tab in position_tabs:
            tab_rows = ranking_by_position[tab["key"]]
            tab_rows.sort(key=lambda item: item["probability"], reverse=True)
            ranking_by_position[tab["key"]] = tab_rows[:10]
            for row in ranking_by_position[tab["key"]]:
                talent_map_rows.append({**row, "bucket": tab["label"]})

        if talent_map_rows:
            avg_probability = round(
                sum(float(row["probability"]) for row in talent_map_rows) / len(talent_map_rows),
                1,
            )
            avg_fit_score = round(
                sum(float(row["fit_score"]) for row in talent_map_rows) / len(talent_map_rows),
                1,
            )
        else:
            avg_probability = 0.0
            avg_fit_score = 0.0

        if request.method == "POST":
            raw_ids = request.form.getlist("players")
            try:
                selected_ids = [int(player_id) for player_id in raw_ids][:10]
            except ValueError:
                selected_ids = []
            target_position_raw = request.form.get("target_position")
            target_position = deps.normalized_position(target_position_raw) if target_position_raw else None

            if selected_ids:
                selected_players = (
                    db.query(Player)
                    .filter(Player.id.in_(selected_ids))
                    .order_by(Player.name.asc())
                    .all()
                )

                if selected_players:
                    score_rows = (
                        db.query(PlayerStat.player_id, func.avg(PlayerStat.final_score))
                        .filter(PlayerStat.player_id.in_([player.id for player in selected_players]))
                        .group_by(PlayerStat.player_id)
                        .all()
                    )
                    avg_score_map = {
                        player_id: (float(avg_score) if avg_score is not None else None)
                        for player_id, avg_score in score_rows
                    }
                    selected_projections = deps.batch_project_players(selected_players, avg_score_map)

                    labels = [deps.ATTRIBUTE_LABELS[field] for field in deps.ATTRIBUTE_FIELDS]
                    datasets = []
                    colors = [
                        "#1f77b4",
                        "#ff7f0e",
                        "#2ca02c",
                        "#d62728",
                        "#9467bd",
                        "#8c564b",
                        "#e377c2",
                        "#7f7f7f",
                        "#bcbd22",
                        "#17becf",
                    ]

                    ranking: List[dict] = []
                    base_position = target_position
                    for idx, player in enumerate(selected_players):
                        values = [getattr(player, field) for field in deps.ATTRIBUTE_FIELDS]
                        color = colors[idx % len(colors)]

                        datasets.append(
                            {
                                "label": player.name,
                                "data": values,
                                "backgroundColor": color,
                                "borderColor": color,
                                "borderWidth": 1,
                            }
                        )

                        attr_map = deps.player_attribute_map(player)
                        attribute_sum = deps.weighted_score_from_attrs(attr_map, base_position or player.position)
                        avg_score = avg_score_map.get(player.id)
                        projection = selected_projections.get(player.id)
                        total = attribute_sum + (avg_score or 0)

                        attributes_map = {
                            deps.ATTRIBUTE_LABELS[field]: getattr(player, field)
                            for field in deps.ATTRIBUTE_FIELDS
                        }

                        ranking.append(
                            {
                                "name": player.name,
                                "position": player.position,
                                "age": player.current_age,
                                "juvenile_category": player.category_year,
                                "attributes_map": attributes_map,
                                "total": round(total, 2),
                                "probability": round(float(projection["combined_prob"]) * 100, 1) if projection else None,
                            }
                        )

                    ranking.sort(key=lambda item: item["total"], reverse=True)

                    score_labels = [row["name"] for row in ranking]
                    score_values = []
                    for ranking_row in ranking:
                        player = next((candidate for candidate in selected_players if candidate.name == ranking_row["name"]), None)
                        avg_val = avg_score_map.get(player.id) if player else None
                        score_values.append(avg_val or 0)

                    comparison = {
                        "chart_payload": {
                            "labels": labels,
                            "datasets": datasets,
                        },
                        "score_payload": {
                            "labels": score_labels,
                            "datasets": [
                                {
                                    "label": "Promedio historial",
                                    "data": score_values,
                                    "backgroundColor": "rgba(13, 110, 253, 0.6)",
                                    "borderColor": "#0d6efd",
                                }
                            ],
                        },
                        "ranking": ranking,
                        "target_position": base_position,
                    }

        db.close()
        html = render_template(
            "compare_multi.html",
            players=players,
            selected_ids=selected_ids,
            comparison=comparison,
            truncated=truncated,
            max_players=deps.MAX_COMPARE_PLAYERS,
            target_position=target_position,
            position_tabs=position_tabs,
            ranking_by_position=ranking_by_position,
            talent_map_rows=talent_map_rows,
            talent_map_avg_probability=avg_probability,
            talent_map_avg_fit_score=avg_fit_score,
        )
        if request.method == "GET":
            deps.cache_set(cache_key, html)
        return html

    return bp
