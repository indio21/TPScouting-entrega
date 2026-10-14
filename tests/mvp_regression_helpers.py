"""Helpers compartidos por las regresiones integrales del MVP."""

import importlib
from datetime import date

from werkzeug.security import generate_password_hash

def _availability_payload(**overrides):
    payload = {
        "record_date": "2026-04-12",
        "availability_pct": "92",
        "fatigue_pct": "18",
        "training_load_pct": "70",
        "missed_days": "0",
        "injury_flag": "",
        "notes": "Disponible",
    }
    payload.update(overrides)
    return payload

def _create_player(app_module, db, **overrides):
    payload = {
        "name": "Jugador Base",
        "national_id": "30123456",
        "age": 16,
        "birth_date": date(2010, 3, 21),
        "position": "Defensa",
        "club": "Club Base",
        "country": "Argentina",
        "photo_url": "",
        "pace": 10,
        "shooting": 9,
        "passing": 11,
        "dribbling": 10,
        "defending": 12,
        "physical": 11,
        "vision": 10,
        "tackling": 12,
        "determination": 13,
        "technique": 9,
        "potential_label": False,
    }
    payload.update(overrides)
    player = app_module.Player(**payload)
    db.add(player)
    db.commit()
    return player

def _create_user(db, User, username, password, role="scout"):
    user = User(username=username, password_hash=generate_password_hash(password), role=role)
    db.add(user)
    db.commit()
    return user

def _fit_test_preprocessor(app_module, players):
    preprocessing_module = importlib.import_module("preprocessing")
    app_module.preprocessor = preprocessing_module.build_preprocessor()
    app_module.preprocessor.fit(app_module.dataframe_from_players(players))

def _get_csrf_token(client, path="/login"):
    client.get(path)
    with client.session_transaction() as sess:
        return sess.get("csrf_token")

def _login(client, username, password):
    csrf_token = _get_csrf_token(client, "/login")
    return client.post("/login", data={"username": username, "password": password, "csrf_token": csrf_token})

def _match_history_payload(**overrides):
    payload = {
        "match_date": "2026-04-10",
        "opponent_name": "Rival Base",
        "opponent_level": "3",
        "tournament": "Liga Test",
        "competition_category": "Sub 17",
        "venue": "Local",
        "started": "1",
        "position_played": "Defensa",
        "minutes_played": "80",
        "final_score": "7.5",
        "goals": "0",
        "assists": "1",
        "pass_accuracy": "82",
        "shot_accuracy": "40",
        "duels_won_pct": "65",
        "yellow_cards": "1",
        "red_cards": "0",
        "role_notes": "Ordenado",
        "match_notes": "Partido controlado",
    }
    payload.update(overrides)
    return payload

def _physical_assessment_payload(**overrides):
    payload = {
        "assessment_date": "2026-04-11",
        "height_cm": "178",
        "weight_kg": "68",
        "dominant_foot": "Derecho",
        "estimated_speed": "14.5",
        "endurance": "15",
        "in_growth_spurt": "1",
        "notes": "Buena respuesta fisica",
    }
    payload.update(overrides)
    return payload

def _scout_report_payload(**overrides):
    payload = {
        "report_date": "2026-04-13",
        "decision_making": "15",
        "tactical_reading": "14",
        "mental_profile": "16",
        "adaptability": "13",
        "observed_projection_score": "8.2",
        "notes": "Buen perfil competitivo",
    }
    payload.update(overrides)
    return payload

def _valid_manage_player_payload(**overrides):
    payload = {
        "mode": "single",
        "name": "Nuevo Talento",
        "national_id": "40111222",
        "birth_date": "2010-03-21",
        "position": "Delantero",
        "club": "Club Nuevo",
        "country": "Argentina",
        "photo_url": "",
        "pace": "14",
        "shooting": "15",
        "passing": "10",
        "dribbling": "14",
        "defending": "6",
        "physical": "12",
        "vision": "11",
        "tackling": "5",
        "determination": "16",
        "technique": "13",
        "potential_label": "1",
    }
    payload.update(overrides)
    return payload
