import importlib
import io
import json
import logging
import sys
from datetime import date
from types import SimpleNamespace

import numpy as np
import pytest
from sqlalchemy import inspect, text
from werkzeug.security import generate_password_hash
from sqlalchemy.orm import sessionmaker


def _create_user(db, User, username, password, role="scout"):
    user = User(username=username, password_hash=generate_password_hash(password), role=role)
    db.add(user)
    db.commit()
    return user


def _get_csrf_token(client, path="/login"):
    client.get(path)
    with client.session_transaction() as sess:
        return sess.get("csrf_token")


def _login(client, username, password):
    csrf_token = _get_csrf_token(client, "/login")
    return client.post("/login", data={"username": username, "password": password, "csrf_token": csrf_token})


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


def _fit_test_preprocessor(app_module, players):
    preprocessing_module = importlib.import_module("preprocessing")
    app_module.preprocessor = preprocessing_module.build_preprocessor()
    app_module.preprocessor.fit(app_module.dataframe_from_players(players))


def test_refresh_player_potential_uses_same_high_threshold(app_module):
    dummy_player = SimpleNamespace(potential_label=False)
    original = app_module.compute_projection
    app_module.compute_projection = lambda player, db_session=None: {"combined_prob": 0.80}
    try:
        projection = app_module.refresh_player_potential(dummy_player)
    finally:
        app_module.compute_projection = original

    assert projection["combined_prob"] == 0.80
    assert dummy_player.potential_label is True


def test_potential_categories_use_requested_percentage_ranges(app_module):
    assert app_module.categorize_probability(0.59) == "Bajo potencial"
    assert app_module.categorize_probability(0.60) == "Potencial medio"
    assert app_module.categorize_probability(0.79) == "Potencial medio"
    assert app_module.categorize_probability(0.80) == "Alto potencial"
    assert app_module.is_high_potential_probability(0.79) is False
    assert app_module.is_high_potential_probability(0.80) is True


def test_potential_thresholds_ignore_invalid_env(app_module, monkeypatch):
    monkeypatch.setenv("POTENTIAL_MEDIUM_THRESHOLD", "valor-invalido")
    monkeypatch.setenv("POTENTIAL_HIGH_THRESHOLD", "2")

    assert app_module.potential_thresholds() == (0.60, 0.80)


def test_potential_thresholds_keep_medium_below_high(app_module, monkeypatch):
    monkeypatch.setenv("POTENTIAL_MEDIUM_THRESHOLD", "0.95")
    monkeypatch.setenv("POTENTIAL_HIGH_THRESHOLD", "0.80")

    assert app_module.potential_thresholds() == (0.75, 0.80)


def test_production_runtime_rejects_sqlite_without_explicit_opt_in(tmp_path, scouting_app_dir, monkeypatch):
    monkeypatch.syspath_prepend(str(scouting_app_dir))
    monkeypatch.chdir(str(scouting_app_dir))
    monkeypatch.setenv("ENV", "production")
    monkeypatch.delenv("FLASK_ENV", raising=False)
    monkeypatch.delenv("RENDER", raising=False)
    monkeypatch.delenv("ALLOW_SQLITE_IN_PRODUCTION", raising=False)
    monkeypatch.setenv("APP_SECRET_KEY", "test-secret-123")
    monkeypatch.setenv("APP_DB_URL", f"sqlite:///{(tmp_path / 'app.db').as_posix()}")
    monkeypatch.setenv("TRAINING_DB_URL", f"sqlite:///{(tmp_path / 'training.db').as_posix()}")

    module_name = "scouting_app_prod_sqlite_guard_test"
    sys.modules.pop(module_name, None)
    spec = importlib.util.spec_from_file_location(module_name, scouting_app_dir / "app.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[module_name] = mod
    try:
        with pytest.raises(RuntimeError, match="SQLite no esta permitido"):
            spec.loader.exec_module(mod)
    finally:
        sys.modules.pop(module_name, None)


def test_render_runtime_requires_secret_key(scouting_app_dir, monkeypatch):
    monkeypatch.syspath_prepend(str(scouting_app_dir))
    monkeypatch.chdir(str(scouting_app_dir))
    monkeypatch.setenv("RENDER", "true")
    monkeypatch.delenv("APP_SECRET_KEY", raising=False)
    monkeypatch.setenv("APP_DB_URL", "postgresql://user:pass@localhost/app_db")
    monkeypatch.setenv("TRAINING_DB_URL", "postgresql://user:pass@localhost/training_db")

    module_name = "scouting_app_render_secret_guard_test"
    sys.modules.pop(module_name, None)
    spec = importlib.util.spec_from_file_location(module_name, scouting_app_dir / "app.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[module_name] = mod
    try:
        with pytest.raises(RuntimeError, match="APP_SECRET_KEY must be set"):
            spec.loader.exec_module(mod)
    finally:
        sys.modules.pop(module_name, None)


def test_bootstrap_admin_rejects_weak_password(app_module, db, monkeypatch):
    monkeypatch.setenv("ADMIN_USERNAME", "weak_bootstrap_admin")
    monkeypatch.setenv("ADMIN_PASSWORD", "abcdef")

    app_module.init_admin_user()

    assert db.query(app_module.User).filter_by(username="weak_bootstrap_admin").count() == 0


def test_pipeline_file_lock_blocks_parallel_acquire(tmp_path, scouting_app_dir, monkeypatch):
    monkeypatch.syspath_prepend(str(scouting_app_dir))
    from services.locks import PipelineFileLock

    lock_path = tmp_path / "pipeline.lock"
    first_lock = PipelineFileLock(lock_path)
    second_lock = PipelineFileLock(lock_path)

    assert first_lock.acquire(blocking=False) is True
    try:
        assert second_lock.acquire(blocking=False) is False
    finally:
        first_lock.release()

    assert second_lock.acquire(blocking=False) is True
    second_lock.release()


def test_settings_pipeline_exception_returns_message_and_releases_lock(scouting_app_dir, monkeypatch):
    monkeypatch.syspath_prepend(str(scouting_app_dir))
    settings_module = importlib.import_module("routes.settings")
    captured = {}

    def fake_render_template(template_name, **kwargs):
        captured.update(kwargs)
        return "ok"

    monkeypatch.setattr(settings_module, "render_template", fake_render_template)

    from flask import Flask

    class FakeLock:
        def __init__(self):
            self.released = False

        def acquire(self, blocking=True):
            return True

        def release(self):
            self.released = True

    class FakeSession:
        def close(self):
            pass

    def roles_required(_role):
        def decorator(fn):
            return fn

        return decorator

    def failing_pipeline(**_kwargs):
        raise RuntimeError("fallo controlado")

    fake_lock = FakeLock()
    deps = SimpleNamespace(
        ROLE_ADMIN="administrador",
        roles_required=roles_required,
        require_csrf=lambda: None,
        pipeline_lock=fake_lock,
        update_database_pipeline=failing_pipeline,
        EVAL_POOL_MAX=100,
        SYNC_SHORTLIST_ENABLED=False,
        Session=lambda: FakeSession(),
        compute_operational_data_quality=lambda _db: {},
        cleanup_operational_data=lambda _db: {},
        invalidate_dashboard_cache=lambda: None,
    )
    app = Flask(__name__)
    app.secret_key = "test-secret"
    app.register_blueprint(settings_module.create_settings_blueprint(deps=deps))

    response = app.test_client().post("/settings", data={"action": "update_database"})

    assert response.status_code == 200
    assert fake_lock.released is True
    assert captured["status_messages"][0] == "No se pudo completar la actualizacion: fallo controlado"


def test_app_module_fixture_uses_stable_module_name(app_module):
    assert app_module.__name__ == "scouting_app_app_test"


def test_dashboard_cache_respects_max_entries(app_module, monkeypatch):
    app_module._CACHE.clear()
    monkeypatch.setattr(app_module, "_CACHE_MAX_ENTRIES", 2)
    monkeypatch.setattr(app_module, "_CACHE_TTL_SECONDS", 60)

    app_module._cache_set("dashboard:first", "first")
    app_module._cache_set("dashboard:second", "second")
    app_module._cache_set("dashboard:third", "third")

    assert len(app_module._CACHE) == 2
    assert "dashboard:first" not in app_module._CACHE
    assert app_module._cache_get("dashboard:second") == "second"
    assert app_module._cache_get("dashboard:third") == "third"


def test_calculate_stats_rating_uses_named_rating_limits(app_module):
    rating = app_module.calculate_stats_rating(
        {
            "matches": 10,
            "goals": 8,
            "assists": 7,
            "minutes": 180,
            "pass_pct": 100.0,
            "shot_pct": 100.0,
            "duels_pct": 100.0,
        }
    )

    assert rating == app_module.MAX_STATS_RATING


def test_batch_projection_uses_raw_probability_as_primary_score(app_module, monkeypatch):
    import torch

    player = SimpleNamespace(
        id=1,
        position="Defensa",
        pace=10,
        shooting=9,
        passing=11,
        dribbling=10,
        defending=12,
        physical=11,
        vision=10,
        tackling=12,
        determination=13,
        technique=9,
    )

    class DummyModel:
        def __call__(self, _tensor):
            return torch.tensor([[0.0]], dtype=torch.float32)

    class DummyCalibrator:
        def predict(self, raw):
            return np.full(len(raw), 0.90, dtype=np.float32)

    app_module.model = DummyModel()
    app_module.preprocessor = object()
    app_module.probability_calibrator = DummyCalibrator()
    monkeypatch.setattr(
        app_module,
        "players_to_model_tensor",
        lambda *args, **kwargs: torch.zeros((1, 1), dtype=torch.float32),
    )

    projection = app_module.batch_project_players(
        [player],
        stats_feature_map={1: {}},
        attribute_feature_map={1: {}},
        match_feature_map={1: {}},
        scout_report_feature_map={1: {}},
        physical_feature_map={1: {}},
        availability_feature_map={1: {}},
    )[1]

    assert round(float(projection["base_prob"]), 4) == 0.5
    assert projection["base_prob_source"] == "raw_pytorch_sigmoid"
    assert round(float(projection["calibrated_base_prob"]), 4) == 0.9
    assert projection["combined_prob"] < projection["calibrated_combined_prob"]


def test_register_rejects_weak_password(client, app_module, db):
    _create_user(db, app_module.User, "adminuser", "admin1234", role="administrador")
    _login(client, "adminuser", "admin1234")
    csrf_token = _get_csrf_token(client, "/register")

    response = client.post(
        "/register",
        data={
            "username": "nuevo",
            "password": "abc",
            "role": "scout",
            "csrf_token": csrf_token,
        },
        follow_redirects=True,
    )

    assert response.status_code == 200
    assert "al menos 8 caracteres" in response.get_data(as_text=True)


def test_director_cannot_access_manage_players(client, app_module, db):
    _create_user(db, app_module.User, "director1", "director123", role="director")
    _login(client, "director1", "director123")

    response = client.get("/players/manage")

    assert response.status_code == 403


def test_scout_can_access_manage_players(client, app_module, db):
    _create_user(db, app_module.User, "scout1", "scout1234", role="scout")
    _login(client, "scout1", "scout1234")

    response = client.get("/players/manage")

    assert response.status_code == 200


def test_director_cannot_post_player_stats(client, app_module, db):
    director = _create_user(db, app_module.User, "director2", "director123", role="director")
    player = app_module.Player(
        name="Jugador Test",
        national_id="30111222",
        age=16,
        position="Defensa",
        club="Club Test",
        country="Argentina",
        photo_url="",
        pace=10,
        shooting=10,
        passing=10,
        dribbling=10,
        defending=10,
        physical=10,
        vision=10,
        tackling=10,
        determination=10,
        technique=10,
        potential_label=False,
    )
    db.add(player)
    db.commit()

    _login(client, director.username, "director123")
    csrf_token = _get_csrf_token(client, f"/player/{player.id}/stats")
    response = client.post(
        f"/player/{player.id}/stats",
        data={
            "csrf_token": csrf_token,
            "action": "add",
            "record_date": "2026-04-15",
            "matches_played": 1,
            "minutes_played": 90,
        },
    )

    assert response.status_code == 403


def test_login_rate_limit_blocks_after_repeated_failures(client, app_module, db):
    _create_user(db, app_module.User, "blocked_user", "blocked123", role="scout")

    for _ in range(5):
        csrf_token = _get_csrf_token(client, "/login")
        response = client.post(
            "/login",
            data={"username": "blocked_user", "password": "mal", "csrf_token": csrf_token},
        )
        assert response.status_code == 200

    csrf_token = _get_csrf_token(client, "/login")
    blocked = client.post(
        "/login",
        data={"username": "blocked_user", "password": "mal", "csrf_token": csrf_token},
    )

    assert blocked.status_code == 429
    assert "bloquearon temporalmente" in blocked.get_data(as_text=True)


def test_timestamps_are_populated_on_new_records(app_module, db):
    player = app_module.Player(
        name="Jugador Audit",
        national_id="44555666",
        age=17,
        position="Delantero",
        club="Club Audit",
        country="Argentina",
        photo_url="",
        pace=12,
        shooting=13,
        passing=11,
        dribbling=14,
        defending=8,
        physical=12,
        vision=11,
        tackling=7,
        determination=15,
        technique=13,
        potential_label=True,
    )
    db.add(player)
    db.commit()
    db.refresh(player)

    assert player.created_at is not None
    assert player.updated_at is not None


def test_cleanup_operational_data_removes_legacy_players_without_national_id(app_module, db):
    invalid_player = app_module.Player(
        name="Legacy Sin DNI",
        national_id=None,
        age=16,
        position="Defensa",
        club="Club Demo",
        country="Argentina",
        photo_url="",
        pace=10,
        shooting=9,
        passing=11,
        dribbling=10,
        defending=12,
        physical=11,
        vision=10,
        tackling=12,
        determination=13,
        technique=9,
        potential_label=False,
    )
    valid_player = app_module.Player(
        name="Jugador Valido",
        national_id="55444333",
        age=15,
        position="Mediocampista",
        club="Club Demo",
        country="Argentina",
        photo_url="",
        pace=12,
        shooting=10,
        passing=13,
        dribbling=12,
        defending=9,
        physical=11,
        vision=13,
        tackling=8,
        determination=14,
        technique=12,
        potential_label=True,
    )
    db.add_all([invalid_player, valid_player])
    db.commit()

    summary = app_module.cleanup_operational_data(db)
    db.commit()

    assert summary["removed_invalid_players"] == 1
    assert summary["missing_national_id"] == 0
    assert db.query(app_module.Player).filter_by(name="Legacy Sin DNI").count() == 0
    assert db.query(app_module.Player).filter_by(name="Jugador Valido").count() == 1


def test_health_reports_operational_data_quality(client, app_module, db):
    player = app_module.Player(
        name="Legacy Health",
        national_id=None,
        age=16,
        position="Delantero",
        club="Club Test",
        country="Argentina",
        photo_url="",
        pace=11,
        shooting=12,
        passing=9,
        dribbling=13,
        defending=7,
        physical=11,
        vision=10,
        tackling=6,
        determination=14,
        technique=12,
        potential_label=False,
    )
    db.add(player)
    db.commit()

    response = client.get("/health")
    payload = response.get_json()

    assert response.status_code == 200
    assert payload["status"] == "ok"
    assert payload["database"] == "ok"
    assert payload["data_quality"]["missing_national_id"] == 1


def test_register_creates_user_with_valid_role(client, app_module, db):
    _create_user(db, app_module.User, "adminok", "admin1234", role="administrador")
    _login(client, "adminok", "admin1234")
    csrf_token = _get_csrf_token(client, "/register")

    response = client.post(
        "/register",
        data={
            "username": "director_ok",
            "password": "Director123",
            "role": "director",
            "csrf_token": csrf_token,
        },
        follow_redirects=False,
    )

    assert response.status_code in (301, 302)
    created = db.query(app_module.User).filter_by(username="director_ok").first()
    assert created is not None
    assert created.role == "director"


def test_register_requires_csrf_token(client, app_module, db):
    _create_user(db, app_module.User, "admincsrf", "admin1234", role="administrador")
    _login(client, "admincsrf", "admin1234")

    response = client.post(
        "/register",
        data={
            "username": "sin_csrf",
            "password": "Password123",
            "role": "scout",
        },
    )

    assert response.status_code == 400


def test_mutating_post_routes_reject_missing_csrf(client, app_module, db):
    _create_user(db, app_module.User, "admin_no_csrf", "admin1234", role="administrador")
    player = _create_player(app_module, db, name="CSRF Base", national_id="46660111")
    coach = app_module.Coach(name="Coach CSRF", role="Entrenador", age=40, club="Club CSRF", country="Argentina")
    director = app_module.Director(
        name="Dir CSRF",
        position="Director deportivo",
        age=50,
        club="Club CSRF",
        country="Argentina",
    )
    db.add_all([coach, director])
    stat = app_module.PlayerStat(
        player_id=player.id,
        record_date=date(2026, 4, 1),
        matches_played=1,
        minutes_played=90,
    )
    attribute_entry = app_module.PlayerAttributeHistory(
        player_id=player.id,
        record_date=date(2026, 4, 1),
        pace=12,
    )
    match = app_module.Match(
        match_date=date(2026, 4, 2),
        opponent_name="Rival CSRF",
        opponent_level=3,
        tournament="Liga CSRF",
        competition_category="Sub 17",
        venue="Local",
    )
    participation = app_module.PlayerMatchParticipation(
        player_id=player.id,
        match=match,
        position_played="Defensa",
        minutes_played=60,
    )
    physical = app_module.PhysicalAssessment(
        player_id=player.id,
        assessment_date=date(2026, 4, 3),
        height_cm=178,
    )
    availability = app_module.PlayerAvailability(
        player_id=player.id,
        record_date=date(2026, 4, 4),
        availability_pct=90,
        missed_days=0,
    )
    report = app_module.ScoutReport(
        player_id=player.id,
        report_date=date(2026, 4, 5),
        decision_making=14,
    )
    db.add_all([stat, attribute_entry, participation, physical, availability, report])
    db.commit()
    _login(client, "admin_no_csrf", "admin1234")

    valid_player_payload = _valid_manage_player_payload(
        name="Sin Token",
        national_id="46660112",
    )
    edit_payload = _valid_manage_player_payload(
        name="Editado Sin Token",
        national_id=player.national_id,
        birth_date=player.birth_date.strftime("%Y-%m-%d") if player.birth_date else "2010-03-21",
        position=player.position,
        potential_label="0",
    )
    routes = [
        ("/login", {"username": "admin_no_csrf", "password": "admin1234"}),
        ("/logout", {}),
        ("/register", {"username": "sin_token", "password": "Password123", "role": "scout"}),
        ("/players/manage", valid_player_payload),
        (
            f"/player/{player.id}/stats",
            {
                "action": "add",
                "record_date": "2026-04-15",
                "matches_played": "1",
                "minutes_played": "90",
            },
        ),
        (f"/player/{player.id}/attributes", {"action": "add", "record_date": "2026-04-15", "pace": "12"}),
        (f"/player/{player.id}/matches/add", _match_history_payload()),
        (f"/player/{player.id}/physical/add", _physical_assessment_payload()),
        (f"/player/{player.id}/availability/add", _availability_payload()),
        (f"/player/{player.id}/reports/add", _scout_report_payload()),
        (
            f"/player/{player.id}/stats/{stat.id}/edit",
            {"record_date": "2026-04-16", "matches_played": "2", "minutes_played": "80"},
        ),
        (f"/player/{player.id}/stats/{stat.id}/delete", {}),
        (
            f"/player/{player.id}/attributes/{attribute_entry.id}/edit",
            {"record_date": "2026-04-16", "pace": "14"},
        ),
        (f"/player/{player.id}/attributes/{attribute_entry.id}/delete", {}),
        (f"/player/{player.id}/matches/{participation.id}/edit", _match_history_payload(opponent_name="Edit CSRF")),
        (f"/player/{player.id}/matches/{participation.id}/delete", {}),
        (f"/player/{player.id}/physical/{physical.id}/edit", _physical_assessment_payload(height_cm="180")),
        (f"/player/{player.id}/physical/{physical.id}/delete", {}),
        (f"/player/{player.id}/availability/{availability.id}/edit", _availability_payload(availability_pct="80")),
        (f"/player/{player.id}/availability/{availability.id}/delete", {}),
        (f"/player/{player.id}/reports/{report.id}/edit", _scout_report_payload(decision_making="16")),
        (f"/player/{player.id}/reports/{report.id}/delete", {}),
        (f"/edit_player/{player.id}", edit_payload),
        (f"/delete_player/{player.id}", {}),
        ("/coaches/new", {"name": "Coach Nuevo", "role": "Ayudante", "age": "35"}),
        (f"/coaches/edit/{coach.id}", {"name": "Coach Editado", "role": "Ayudante", "age": "36"}),
        (f"/coaches/delete/{coach.id}", {}),
        ("/directors/new", {"name": "Director Nuevo", "position": "Presidente", "age": "55"}),
        (f"/directors/edit/{director.id}", {"name": "Director Editado", "position": "Presidente", "age": "56"}),
        (f"/directors/delete/{director.id}", {}),
        ("/settings", {"action": "cleanup_demo_data"}),
    ]

    for path, data in routes:
        response = client.post(path, data=data, follow_redirects=False)
        assert response.status_code == 400, path

    db.expire_all()
    assert db.query(app_module.User).filter_by(username="sin_token").count() == 0
    assert db.query(app_module.Player).filter_by(national_id="46660112").count() == 0
    assert db.query(app_module.Player).filter_by(id=player.id).count() == 1
    assert db.query(app_module.PlayerStat).filter_by(player_id=player.id).count() == 1
    assert db.query(app_module.PlayerAttributeHistory).filter_by(player_id=player.id).count() == 1
    assert db.query(app_module.PlayerMatchParticipation).filter_by(player_id=player.id).count() == 1
    assert db.query(app_module.PhysicalAssessment).filter_by(player_id=player.id).count() == 1
    assert db.query(app_module.PlayerAvailability).filter_by(player_id=player.id).count() == 1
    assert db.query(app_module.ScoutReport).filter_by(player_id=player.id).count() == 1
    db.refresh(player)
    db.refresh(coach)
    db.refresh(director)
    assert player.name == "CSRF Base"
    assert coach.name == "Coach CSRF"
    assert director.name == "Dir CSRF"


def test_manage_players_creates_valid_player(client, app_module, db):
    _create_user(db, app_module.User, "scout_create", "scout1234", role="scout")
    _login(client, "scout_create", "scout1234")
    csrf_token = _get_csrf_token(client, "/players/manage")

    response = client.post(
        "/players/manage",
        data=_valid_manage_player_payload(csrf_token=csrf_token),
        follow_redirects=False,
    )

    assert response.status_code in (301, 302)
    created = db.query(app_module.Player).filter_by(national_id="40111222").first()
    assert created is not None
    assert created.name == "Nuevo Talento"
    assert created.birth_date == date(2010, 3, 21)
    assert created.category_year == 2010
    assert created.age == app_module.Player.calculate_age_from_birth_date(created.birth_date)


def test_manage_players_rejects_invalid_age(client, app_module, db):
    _create_user(db, app_module.User, "scout_age", "scout1234", role="scout")
    _login(client, "scout_age", "scout1234")
    csrf_token = _get_csrf_token(client, "/players/manage")

    response = client.post(
        "/players/manage",
        data=_valid_manage_player_payload(
            national_id="40111223",
            birth_date="2007-03-21",
            csrf_token=csrf_token,
        ),
        follow_redirects=True,
    )

    assert response.status_code == 200
    assert "edad calculada por fecha de nacimiento" in response.get_data(as_text=True).lower()
    assert db.query(app_module.Player).filter_by(national_id="40111223").count() == 0


def test_manage_players_rejects_missing_required_fields(client, app_module, db):
    _create_user(db, app_module.User, "scout_required", "scout1234", role="scout")
    _login(client, "scout_required", "scout1234")
    csrf_token = _get_csrf_token(client, "/players/manage")

    response = client.post(
        "/players/manage",
        data=_valid_manage_player_payload(
            name="",
            national_id="",
            birth_date="",
            csrf_token=csrf_token,
        ),
        follow_redirects=True,
    )
    body = response.get_data(as_text=True)

    assert response.status_code == 200
    assert "El nombre es obligatorio" in body
    assert "DNI" in body
    assert "fecha de nacimiento" in body.lower()
    assert db.query(app_module.Player).count() == 0


def test_manage_players_rejects_duplicate_national_id(client, app_module, db):
    _create_user(db, app_module.User, "scout_dup", "scout1234", role="scout")
    _create_player(app_module, db, name="Existente", national_id="41122334")
    _login(client, "scout_dup", "scout1234")
    csrf_token = _get_csrf_token(client, "/players/manage")

    response = client.post(
        "/players/manage",
        data={
            "mode": "single",
            "name": "Duplicado",
            "national_id": "41122334",
            "birth_date": "2010-03-21",
            "position": "Defensa",
            "club": "Club Dup",
            "country": "Argentina",
            "pace": "10",
            "shooting": "9",
            "passing": "10",
            "dribbling": "9",
            "defending": "12",
            "physical": "11",
            "vision": "10",
            "tackling": "12",
            "determination": "13",
            "technique": "9",
            "csrf_token": csrf_token,
        },
        follow_redirects=True,
    )

    assert response.status_code == 200
    assert "ya esta registrado" in response.get_data(as_text=True)
    assert db.query(app_module.Player).filter_by(name="Duplicado").count() == 0


def test_manage_players_rejects_attribute_below_one(client, app_module, db):
    _create_user(db, app_module.User, "scout_attr_low", "scout1234", role="scout")
    _login(client, "scout_attr_low", "scout1234")
    csrf_token = _get_csrf_token(client, "/players/manage")

    response = client.post(
        "/players/manage",
        data=_valid_manage_player_payload(
            national_id="40111224",
            pace="0",
            csrf_token=csrf_token,
        ),
        follow_redirects=True,
    )

    assert response.status_code == 200
    assert "Ritmo debe ubicarse entre 1 y 20" in response.get_data(as_text=True)
    assert db.query(app_module.Player).filter_by(national_id="40111224").count() == 0


def test_import_players_preview_and_confirm_csv(client, app_module, db):
    _create_user(db, app_module.User, "scout_import", "scout1234", role="scout")
    _login(client, "scout_import", "scout1234")
    csrf_token = _get_csrf_token(client, "/players/import")
    csv_text = (
        "Nombre,DNI_ID,FechaNacimiento,Posicion,Club,Pais,Ritmo,Disparo,Pase,Regate,"
        "Defensa,Fisico,Vision,Marcaje,Determinacion,Tecnica,AltoPotencial,FotoURL\n"
        "CSV Juvenil,50111222,2010-03-21,Delantero,Club CSV,Argentina,15,14,13,16,"
        "10,14,15,12,16,17,1,\n"
    )

    preview_response = client.post(
        "/players/import",
        data={
            "csrf_token": csrf_token,
            "mode": "preview",
            "csv_file": (io.BytesIO(csv_text.encode("utf-8")), "jugadores.csv"),
        },
        content_type="multipart/form-data",
    )
    preview_body = preview_response.get_data(as_text=True)

    assert preview_response.status_code == 200
    assert "CSV Juvenil" in preview_body
    assert "Cat. 2010" in preview_body
    assert "Importar 1 filas validas" in preview_body

    csrf_token = _get_csrf_token(client, "/players/import")
    payload = [
        {
            "Nombre": "CSV Juvenil",
            "DNI_ID": "50111222",
            "FechaNacimiento": "2010-03-21",
            "Posicion": "Delantero",
            "Club": "Club CSV",
            "Pais": "Argentina",
            "Ritmo": "15",
            "Disparo": "14",
            "Pase": "13",
            "Regate": "16",
            "Defensa": "10",
            "Fisico": "14",
            "Vision": "15",
            "Marcaje": "12",
            "Determinacion": "16",
            "Tecnica": "17",
            "AltoPotencial": "1",
            "FotoURL": "",
        }
    ]
    confirm_response = client.post(
        "/players/import",
        data={
            "csrf_token": csrf_token,
            "mode": "confirm",
            "import_payload": json.dumps(payload),
        },
        follow_redirects=False,
    )

    assert confirm_response.status_code in (301, 302)
    created = db.query(app_module.Player).filter_by(national_id="50111222").first()
    assert created is not None
    assert created.name == "CSV Juvenil"
    assert created.birth_date == date(2010, 3, 21)
    assert created.category_year == 2010
    assert created.age == app_module.Player.calculate_age_from_birth_date(created.birth_date)


def test_edit_player_updates_core_fields(client, app_module, db):
    _create_user(db, app_module.User, "scout_edit", "scout1234", role="scout")
    player = _create_player(app_module, db, name="Editar Base", national_id="42233445")
    _login(client, "scout_edit", "scout1234")
    csrf_token = _get_csrf_token(client, f"/edit_player/{player.id}")

    response = client.post(
        f"/edit_player/{player.id}",
        data={
            "name": "Editar Final",
            "national_id": "42233446",
            "birth_date": "2009-03-21",
            "position": "Mediocampista",
            "club": "Club Editado",
            "country": "Uruguay",
            "photo_url": "",
            "pace": "13",
            "shooting": "10",
            "passing": "15",
            "dribbling": "14",
            "defending": "11",
            "physical": "12",
            "vision": "15",
            "tackling": "9",
            "determination": "16",
            "technique": "14",
            "potential_label": "1",
            "csrf_token": csrf_token,
        },
        follow_redirects=False,
    )

    assert response.status_code in (301, 302)
    db.refresh(player)
    assert player.name == "Editar Final"
    assert player.national_id == "42233446"
    assert player.birth_date == date(2009, 3, 21)
    assert player.age == app_module.Player.calculate_age_from_birth_date(player.birth_date)
    assert player.position == "Mediocampista"


def test_delete_player_removes_record(client, app_module, db):
    _create_user(db, app_module.User, "scout_delete", "scout1234", role="scout")
    player = _create_player(app_module, db, name="Eliminar Base", national_id="43344556")
    _login(client, "scout_delete", "scout1234")
    csrf_token = _get_csrf_token(client, f"/player/{player.id}")

    response = client.post(
        f"/delete_player/{player.id}",
        data={"csrf_token": csrf_token},
        follow_redirects=False,
    )

    assert response.status_code in (301, 302)
    assert db.query(app_module.Player).filter_by(id=player.id).count() == 0


def test_predict_player_without_loaded_model_returns_controlled_error(client, app_module, db):
    _create_user(db, app_module.User, "scout_no_model", "scout1234", role="scout")
    player = _create_player(app_module, db, name="Sin Modelo", national_id="44455666")
    app_module.model = None
    _login(client, "scout_no_model", "scout1234")

    response = client.get(f"/player/{player.id}/predict")

    assert response.status_code == 500
    assert "No hay calculo disponible todavia" in response.get_data(as_text=True)


def test_player_stats_reject_invalid_percentages(client, app_module, db):
    _create_user(db, app_module.User, "scout_stats", "scout1234", role="scout")
    player = _create_player(app_module, db, name="Stats Base", national_id="44455667")
    _login(client, "scout_stats", "scout1234")
    csrf_token = _get_csrf_token(client, f"/player/{player.id}/stats")

    response = client.post(
        f"/player/{player.id}/stats",
        data={
            "csrf_token": csrf_token,
            "action": "add",
            "record_date": "2026-04-15",
            "matches_played": "1",
            "minutes_played": "90",
            "pass_accuracy": "150",
        },
        follow_redirects=True,
    )

    assert response.status_code == 200
    assert "Precision de pase debe estar entre 0 y 100" in response.get_data(as_text=True)
    assert db.query(app_module.PlayerStat).filter_by(player_id=player.id).count() == 0


def test_player_attributes_reject_values_out_of_range(client, app_module, db):
    _create_user(db, app_module.User, "scout_attrs", "scout1234", role="scout")
    player = _create_player(app_module, db, name="Attrs Base", national_id="45566778")
    _login(client, "scout_attrs", "scout1234")
    csrf_token = _get_csrf_token(client, f"/player/{player.id}/attributes")

    response = client.post(
        f"/player/{player.id}/attributes",
        data={
            "csrf_token": csrf_token,
            "action": "add",
            "record_date": "2026-04-15",
            "pace": "25",
        },
        follow_redirects=True,
    )

    assert response.status_code == 200
    assert "Ritmo debe estar entre 1 y 20" in response.get_data(as_text=True)
    assert db.query(app_module.PlayerAttributeHistory).filter_by(player_id=player.id).count() == 0


def test_edit_player_stat_updates_record(client, app_module, db):
    _create_user(db, app_module.User, "scout_stat_edit", "scout1234", role="scout")
    player = _create_player(app_module, db, name="Stats Edit", national_id="45566779")
    stat = app_module.PlayerStat(
        player_id=player.id,
        record_date=date(2026, 4, 1),
        matches_played=1,
        goals=0,
        assists=0,
        minutes_played=70,
        yellow_cards=0,
        red_cards=0,
        pass_accuracy=70,
        shot_accuracy=30,
        duels_won_pct=50,
        final_score=6.0,
        notes="Inicial",
    )
    db.add(stat)
    db.commit()
    _login(client, "scout_stat_edit", "scout1234")
    csrf_token = _get_csrf_token(client, f"/player/{player.id}/stats")

    response = client.post(
        f"/player/{player.id}/stats/{stat.id}/edit",
        data={
            "csrf_token": csrf_token,
            "record_date": "2026-04-20",
            "matches_played": "2",
            "goals": "1",
            "assists": "1",
            "minutes_played": "160",
            "yellow_cards": "1",
            "red_cards": "0",
            "pass_accuracy": "82.5",
            "shot_accuracy": "45",
            "duels_won_pct": "61",
            "final_score": "8.1",
            "notes": "Editado",
        },
        follow_redirects=False,
    )

    assert response.status_code in (301, 302)
    db.refresh(stat)
    assert stat.record_date == date(2026, 4, 20)
    assert stat.matches_played == 2
    assert stat.goals == 1
    assert stat.final_score == 8.1
    assert stat.notes == "Editado"


def test_delete_player_stat_removes_record(client, app_module, db):
    _create_user(db, app_module.User, "scout_stat_delete", "scout1234", role="scout")
    player = _create_player(app_module, db, name="Stats Delete", national_id="45566780")
    stat = app_module.PlayerStat(
        player_id=player.id,
        record_date=date(2026, 4, 1),
        matches_played=1,
        goals=0,
        assists=0,
        minutes_played=70,
        yellow_cards=0,
        red_cards=0,
    )
    db.add(stat)
    db.commit()
    _login(client, "scout_stat_delete", "scout1234")
    csrf_token = _get_csrf_token(client, f"/player/{player.id}/stats")

    response = client.post(
        f"/player/{player.id}/stats/{stat.id}/delete",
        data={"csrf_token": csrf_token},
        follow_redirects=False,
    )

    assert response.status_code in (301, 302)
    assert db.query(app_module.PlayerStat).filter_by(id=stat.id).count() == 0


def test_edit_player_attribute_history_updates_record_and_player(client, app_module, db):
    _create_user(db, app_module.User, "scout_attr_edit", "scout1234", role="scout")
    player = _create_player(app_module, db, name="Attr Edit", national_id="45566781", pace=10)
    entry = app_module.PlayerAttributeHistory(
        player_id=player.id,
        record_date=date(2026, 4, 1),
        pace=10,
        notes="Inicial",
    )
    db.add(entry)
    db.commit()
    _login(client, "scout_attr_edit", "scout1234")
    csrf_token = _get_csrf_token(client, f"/player/{player.id}/attributes")

    response = client.post(
        f"/player/{player.id}/attributes/{entry.id}/edit",
        data={
            "csrf_token": csrf_token,
            "record_date": "2026-04-21",
            "pace": "16",
            "shooting": "",
            "passing": "",
            "dribbling": "",
            "defending": "",
            "physical": "",
            "vision": "",
            "tackling": "",
            "determination": "",
            "technique": "",
            "notes": "Mejoro ritmo",
        },
        follow_redirects=False,
    )

    assert response.status_code in (301, 302)
    db.refresh(entry)
    db.refresh(player)
    assert entry.record_date == date(2026, 4, 21)
    assert entry.pace == 16
    assert entry.notes == "Mejoro ritmo"
    assert player.pace == 16


def test_delete_player_attribute_history_resyncs_player(client, app_module, db):
    _create_user(db, app_module.User, "scout_attr_delete", "scout1234", role="scout")
    player = _create_player(app_module, db, name="Attr Delete", national_id="45566782", pace=18)
    older = app_module.PlayerAttributeHistory(
        player_id=player.id,
        record_date=date(2026, 3, 1),
        pace=11,
    )
    latest = app_module.PlayerAttributeHistory(
        player_id=player.id,
        record_date=date(2026, 4, 1),
        pace=18,
    )
    db.add_all([older, latest])
    db.commit()
    _login(client, "scout_attr_delete", "scout1234")
    csrf_token = _get_csrf_token(client, f"/player/{player.id}/attributes")

    response = client.post(
        f"/player/{player.id}/attributes/{latest.id}/delete",
        data={"csrf_token": csrf_token},
        follow_redirects=False,
    )

    assert response.status_code in (301, 302)
    assert db.query(app_module.PlayerAttributeHistory).filter_by(id=latest.id).count() == 0
    db.refresh(player)
    assert player.pace == 11


def test_match_history_modal_routes_add_edit_delete(client, app_module, db):
    _create_user(db, app_module.User, "scout_match_history", "scout1234", role="scout")
    player = _create_player(app_module, db, name="Match History", national_id="45566784")
    _login(client, "scout_match_history", "scout1234")
    csrf_token = _get_csrf_token(client, f"/player/{player.id}")

    response = client.post(
        f"/player/{player.id}/matches/add",
        data=_match_history_payload(csrf_token=csrf_token),
        follow_redirects=False,
    )

    assert response.status_code in (301, 302)
    db.expire_all()
    participation = db.query(app_module.PlayerMatchParticipation).filter_by(player_id=player.id).one()
    match_id = participation.match_id
    assert participation.minutes_played == 80
    assert participation.match.opponent_name == "Rival Base"

    response = client.post(
        f"/player/{player.id}/matches/{participation.id}/edit",
        data=_match_history_payload(
            csrf_token=csrf_token,
            opponent_name="Rival Editado",
            minutes_played="90",
            final_score="8.0",
        ),
        follow_redirects=False,
    )

    assert response.status_code in (301, 302)
    db.expire_all()
    participation = db.get(app_module.PlayerMatchParticipation, participation.id)
    assert participation.minutes_played == 90
    assert participation.final_score == 8.0
    assert participation.match.opponent_name == "Rival Editado"

    participation_id = participation.id
    response = client.post(
        f"/player/{player.id}/matches/{participation_id}/delete",
        data={"csrf_token": csrf_token},
        follow_redirects=False,
    )

    assert response.status_code in (301, 302)
    db.expire_all()
    assert db.query(app_module.PlayerMatchParticipation).filter_by(id=participation_id).count() == 0
    assert db.query(app_module.Match).filter_by(id=match_id).count() == 0


def test_physical_history_modal_routes_add_edit_delete(client, app_module, db):
    _create_user(db, app_module.User, "scout_physical_history", "scout1234", role="scout")
    player = _create_player(app_module, db, name="Physical History", national_id="45566785")
    _login(client, "scout_physical_history", "scout1234")
    csrf_token = _get_csrf_token(client, f"/player/{player.id}")

    response = client.post(
        f"/player/{player.id}/physical/add",
        data=_physical_assessment_payload(csrf_token=csrf_token),
        follow_redirects=False,
    )

    assert response.status_code in (301, 302)
    db.expire_all()
    assessment = db.query(app_module.PhysicalAssessment).filter_by(player_id=player.id).one()
    assert assessment.height_cm == 178
    assert assessment.in_growth_spurt is True

    response = client.post(
        f"/player/{player.id}/physical/{assessment.id}/edit",
        data=_physical_assessment_payload(
            csrf_token=csrf_token,
            height_cm="180",
            in_growth_spurt="",
            notes="Ajuste fisico",
        ),
        follow_redirects=False,
    )

    assert response.status_code in (301, 302)
    db.expire_all()
    assessment = db.get(app_module.PhysicalAssessment, assessment.id)
    assert assessment.height_cm == 180
    assert assessment.in_growth_spurt is False
    assert assessment.notes == "Ajuste fisico"

    assessment_id = assessment.id
    response = client.post(
        f"/player/{player.id}/physical/{assessment_id}/delete",
        data={"csrf_token": csrf_token},
        follow_redirects=False,
    )

    assert response.status_code in (301, 302)
    db.expire_all()
    assert db.query(app_module.PhysicalAssessment).filter_by(id=assessment_id).count() == 0


def test_availability_history_modal_routes_add_edit_delete(client, app_module, db):
    _create_user(db, app_module.User, "scout_availability_history", "scout1234", role="scout")
    player = _create_player(app_module, db, name="Availability History", national_id="45566786")
    _login(client, "scout_availability_history", "scout1234")
    csrf_token = _get_csrf_token(client, f"/player/{player.id}")

    response = client.post(
        f"/player/{player.id}/availability/add",
        data=_availability_payload(csrf_token=csrf_token),
        follow_redirects=False,
    )

    assert response.status_code in (301, 302)
    db.expire_all()
    availability = db.query(app_module.PlayerAvailability).filter_by(player_id=player.id).one()
    assert availability.availability_pct == 92
    assert availability.injury_flag is False

    response = client.post(
        f"/player/{player.id}/availability/{availability.id}/edit",
        data=_availability_payload(
            csrf_token=csrf_token,
            availability_pct="75",
            injury_flag="1",
            missed_days="2",
        ),
        follow_redirects=False,
    )

    assert response.status_code in (301, 302)
    db.expire_all()
    availability = db.get(app_module.PlayerAvailability, availability.id)
    assert availability.availability_pct == 75
    assert availability.injury_flag is True
    assert availability.missed_days == 2

    availability_id = availability.id
    response = client.post(
        f"/player/{player.id}/availability/{availability_id}/delete",
        data={"csrf_token": csrf_token},
        follow_redirects=False,
    )

    assert response.status_code in (301, 302)
    db.expire_all()
    assert db.query(app_module.PlayerAvailability).filter_by(id=availability_id).count() == 0


def test_scout_report_modal_routes_add_edit_delete(client, app_module, db):
    _create_user(db, app_module.User, "scout_report_history", "scout1234", role="scout")
    player = _create_player(app_module, db, name="Scout Report History", national_id="45566787")
    _login(client, "scout_report_history", "scout1234")
    csrf_token = _get_csrf_token(client, f"/player/{player.id}")

    response = client.post(
        f"/player/{player.id}/reports/add",
        data=_scout_report_payload(csrf_token=csrf_token),
        follow_redirects=False,
    )

    assert response.status_code in (301, 302)
    db.expire_all()
    report = db.query(app_module.ScoutReport).filter_by(player_id=player.id).one()
    assert report.decision_making == 15
    assert report.observed_projection_score == 8.2

    response = client.post(
        f"/player/{player.id}/reports/{report.id}/edit",
        data=_scout_report_payload(
            csrf_token=csrf_token,
            decision_making="17",
            observed_projection_score="8.8",
            notes="Mejor lectura",
        ),
        follow_redirects=False,
    )

    assert response.status_code in (301, 302)
    db.expire_all()
    report = db.get(app_module.ScoutReport, report.id)
    assert report.decision_making == 17
    assert report.observed_projection_score == 8.8
    assert report.notes == "Mejor lectura"

    report_id = report.id
    response = client.post(
        f"/player/{player.id}/reports/{report_id}/delete",
        data={"csrf_token": csrf_token},
        follow_redirects=False,
    )

    assert response.status_code in (301, 302)
    db.expire_all()
    assert db.query(app_module.ScoutReport).filter_by(id=report_id).count() == 0


def test_player_detail_renders_complementary_history_modals(client, app_module, db):
    _create_user(db, app_module.User, "scout_detail_modals", "scout1234", role="scout")
    player = _create_player(app_module, db, name="Detail Modals", national_id="45566788")
    match = app_module.Match(
        match_date=date(2026, 4, 20),
        opponent_name="Rival Modales",
        opponent_level=3,
        venue="Local",
    )
    participation = app_module.PlayerMatchParticipation(
        player_id=player.id,
        match=match,
        position_played="Defensa",
        minutes_played=75,
    )
    physical = app_module.PhysicalAssessment(
        player_id=player.id,
        assessment_date=date(2026, 4, 21),
        height_cm=179,
    )
    availability = app_module.PlayerAvailability(
        player_id=player.id,
        record_date=date(2026, 4, 22),
        availability_pct=91,
        missed_days=0,
    )
    report = app_module.ScoutReport(
        player_id=player.id,
        report_date=date(2026, 4, 23),
        decision_making=15,
    )
    db.add_all([participation, physical, availability, report])
    db.commit()

    _login(client, "scout_detail_modals", "scout1234")
    response = client.get(f"/player/{player.id}")
    body = response.get_data(as_text=True)

    assert response.status_code == 200
    assert f"editMatchModal{participation.id}" in body
    assert f"editPhysicalModal{physical.id}" in body
    assert f"editAvailabilityModal{availability.id}" in body
    assert f"editScoutReportModal{report.id}" in body


def test_director_cannot_modify_history_records(client, app_module, db):
    director = _create_user(db, app_module.User, "director_history", "director123", role="director")
    player = _create_player(app_module, db, name="Director History", national_id="45566783")
    stat = app_module.PlayerStat(
        player_id=player.id,
        record_date=date(2026, 4, 1),
        matches_played=1,
        goals=0,
        assists=0,
        minutes_played=70,
        yellow_cards=0,
        red_cards=0,
    )
    entry = app_module.PlayerAttributeHistory(
        player_id=player.id,
        record_date=date(2026, 4, 1),
        pace=12,
    )
    physical = app_module.PhysicalAssessment(
        player_id=player.id,
        assessment_date=date(2026, 4, 1),
        height_cm=177,
    )
    db.add_all([stat, entry, physical])
    db.commit()
    _login(client, director.username, "director123")
    csrf_token = _get_csrf_token(client, f"/player/{player.id}/stats")

    response = client.post(
        f"/player/{player.id}/stats/{stat.id}/delete",
        data={"csrf_token": csrf_token},
        follow_redirects=False,
    )
    attr_response = client.post(
        f"/player/{player.id}/attributes/{entry.id}/delete",
        data={"csrf_token": csrf_token},
        follow_redirects=False,
    )
    physical_response = client.post(
        f"/player/{player.id}/physical/{physical.id}/delete",
        data={"csrf_token": csrf_token},
        follow_redirects=False,
    )

    assert response.status_code == 403
    assert attr_response.status_code == 403
    assert physical_response.status_code == 403
    assert db.query(app_module.PlayerStat).filter_by(id=stat.id).count() == 1
    assert db.query(app_module.PlayerAttributeHistory).filter_by(id=entry.id).count() == 1
    assert db.query(app_module.PhysicalAssessment).filter_by(id=physical.id).count() == 1


def test_admin_can_create_and_delete_coach(client, app_module, db):
    _create_user(db, app_module.User, "admin_staff", "admin1234", role="administrador")
    _login(client, "admin_staff", "admin1234")

    csrf_token = _get_csrf_token(client, "/coaches/new")
    create_response = client.post(
        "/coaches/new",
        data={
            "csrf_token": csrf_token,
            "name": "Profe Uno",
            "role": "Entrenador",
            "age": "40",
            "club": "Club Staff",
            "country": "Argentina",
        },
        follow_redirects=False,
    )

    assert create_response.status_code in (301, 302)
    coach = db.query(app_module.Coach).filter_by(name="Profe Uno").first()
    assert coach is not None

    csrf_token = _get_csrf_token(client, "/coaches")
    delete_response = client.post(
        f"/coaches/delete/{coach.id}",
        data={"csrf_token": csrf_token},
        follow_redirects=False,
    )

    assert delete_response.status_code in (301, 302)
    assert db.query(app_module.Coach).filter_by(id=coach.id).count() == 0


def test_prepare_input_matches_batch_transformation(app_module, db):
    player = _create_player(app_module, db, name="Tensor Uno", national_id="46677889")
    second_player = _create_player(app_module, db, name="Tensor Dos", national_id="47788990", position="Delantero")
    _fit_test_preprocessor(app_module, [player, second_player])

    single_tensor = app_module.prepare_input(player).detach().cpu().numpy()
    batch_tensor = app_module.players_to_model_tensor([player, second_player]).detach().cpu().numpy()

    assert single_tensor.shape[1] == batch_tensor.shape[1]
    assert np.allclose(single_tensor[0], batch_tensor[0], equal_nan=True)


def test_prepare_input_includes_historical_features(app_module, db):
    player = _create_player(app_module, db, name="Hist Uno", national_id="49900112")
    _fit_test_preprocessor(app_module, [player])
    match_one = app_module.Match(
        match_date=date(2026, 3, 15),
        opponent_name="Racing Juvenil",
        opponent_level=4,
        tournament="Liga Juvenil Regional",
        competition_category="Sub-17",
        venue="Visitante",
    )
    match_two = app_module.Match(
        match_date=date(2026, 4, 10),
        opponent_name="Belgrano Inferiores",
        opponent_level=2,
        tournament="Copa Proyeccion",
        competition_category="Sub-17",
        venue="Local",
    )
    db.add_all(
        [
            app_module.PlayerStat(
                player_id=player.id,
                record_date=date(2026, 4, 1),
                matches_played=1,
                minutes_played=90,
                pass_accuracy=72.0,
                final_score=6.5,
            ),
            app_module.PlayerStat(
                player_id=player.id,
                record_date=date(2026, 4, 8),
                matches_played=1,
                minutes_played=90,
                pass_accuracy=78.0,
                final_score=7.0,
            ),
            app_module.PlayerAttributeHistory(
                player_id=player.id,
                record_date=date(2025, 3, 1),
                pace=8,
                shooting=7,
                passing=9,
                dribbling=8,
                defending=10,
                physical=9,
                vision=8,
                tackling=10,
                determination=11,
                technique=8,
            ),
            app_module.PlayerAttributeHistory(
                player_id=player.id,
                record_date=date(2025, 10, 1),
                pace=9,
                shooting=8,
                passing=10,
                dribbling=9,
                defending=11,
                physical=10,
                vision=9,
                tackling=11,
                determination=12,
                technique=9,
            ),
            app_module.PlayerAttributeHistory(
                player_id=player.id,
                record_date=date(2026, 1, 1),
                pace=9,
                shooting=8,
                passing=10,
                dribbling=9,
                defending=11,
                physical=10,
                vision=9,
                tackling=11,
                determination=12,
                technique=9,
            ),
            app_module.PlayerAttributeHistory(
                player_id=player.id,
                record_date=date(2026, 3, 1),
                pace=10,
                shooting=8,
                passing=10,
                dribbling=9,
                defending=11,
                physical=10,
                vision=9,
                tackling=11,
                determination=12,
                technique=9,
            ),
            match_one,
            match_two,
            app_module.PlayerMatchParticipation(
                player_id=player.id,
                match=match_one,
                started=True,
                position_played="Defensa",
                minutes_played=90,
                final_score=6.7,
                pass_accuracy=73.0,
                shot_accuracy=35.0,
                duels_won_pct=68.0,
            ),
            app_module.PlayerMatchParticipation(
                player_id=player.id,
                match=match_two,
                started=False,
                position_played="Lateral",
                minutes_played=28,
                final_score=7.4,
                pass_accuracy=79.0,
                shot_accuracy=42.0,
                duels_won_pct=71.0,
            ),
            app_module.ScoutReport(
                player_id=player.id,
                report_date=date(2026, 2, 20),
                decision_making=11,
                tactical_reading=12,
                mental_profile=13,
                adaptability=10,
                observed_projection_score=6.6,
                notes="Reporte uno",
            ),
            app_module.ScoutReport(
                player_id=player.id,
                report_date=date(2026, 4, 12),
                decision_making=12,
                tactical_reading=13,
                mental_profile=14,
                adaptability=11,
                observed_projection_score=7.2,
                notes="Reporte dos",
            ),
            app_module.PhysicalAssessment(
                player_id=player.id,
                assessment_date=date(2026, 2, 20),
                height_cm=171.5,
                weight_kg=64.0,
                dominant_foot="Izquierda",
                estimated_speed=13.8,
                endurance=12.4,
                in_growth_spurt=True,
            ),
            app_module.PhysicalAssessment(
                player_id=player.id,
                assessment_date=date(2026, 4, 12),
                height_cm=173.0,
                weight_kg=65.2,
                dominant_foot="Izquierda",
                estimated_speed=14.3,
                endurance=13.1,
                in_growth_spurt=False,
            ),
            app_module.PlayerAvailability(
                player_id=player.id,
                record_date=date(2026, 2, 20),
                availability_pct=76.0,
                fatigue_pct=34.0,
                training_load_pct=71.0,
                missed_days=2,
                injury_flag=False,
            ),
            app_module.PlayerAvailability(
                player_id=player.id,
                record_date=date(2026, 4, 12),
                availability_pct=84.0,
                fatigue_pct=26.0,
                training_load_pct=68.0,
                missed_days=0,
                injury_flag=False,
            ),
        ]
    )
    db.commit()

    stats_feature_map = app_module.fetch_player_stat_feature_map([player.id])
    attr_feature_map = app_module.fetch_player_attribute_feature_map([player])
    match_feature_map = app_module.fetch_player_match_feature_map([player])
    scout_feature_map = app_module.fetch_player_scout_report_feature_map([player.id])
    physical_feature_map = app_module.fetch_player_physical_feature_map([player.id])
    availability_feature_map = app_module.fetch_player_availability_feature_map([player.id])
    assert stats_feature_map[player.id]["stats_entry_count"] == 2
    assert round(float(stats_feature_map[player.id]["avg_final_score_hist"]), 2) == 6.75
    assert round(float(stats_feature_map[player.id]["avg_pass_accuracy_hist"]), 2) == 75.0
    assert round(float(stats_feature_map[player.id]["latest_final_score_hist"]), 2) == 7.0
    assert attr_feature_map[player.id]["attr_history_entry_count"] == 4
    assert float(attr_feature_map[player.id]["attr_weighted_improvement_90d"]) > 0
    assert float(attr_feature_map[player.id]["attr_current_vs_recent_gap"]) > 0
    assert match_feature_map[player.id]["match_entry_count"] == 2
    assert round(float(match_feature_map[player.id]["match_avg_minutes"]), 2) == 59.0
    assert round(float(match_feature_map[player.id]["match_natural_position_rate"]), 2) == 0.5
    assert scout_feature_map[player.id]["scout_report_count"] == 2
    assert round(float(scout_feature_map[player.id]["scout_latest_projection_score"]), 2) == 7.2
    assert physical_feature_map[player.id]["phys_assessment_count"] == 2
    assert round(float(physical_feature_map[player.id]["phys_recent_height_cm"]), 1) == 173.0
    assert round(float(availability_feature_map[player.id]["avail_recent_pct"]), 2) == 84.0
    assert round(float(availability_feature_map[player.id]["avail_injury_rate"]), 2) == 0.0

    single_tensor = app_module.prepare_input(player).detach().cpu().numpy()
    batch_tensor = app_module.players_to_model_tensor(
        [player],
        stats_feature_map=stats_feature_map,
        attribute_feature_map=attr_feature_map,
        match_feature_map=match_feature_map,
        scout_report_feature_map=scout_feature_map,
        physical_feature_map=physical_feature_map,
        availability_feature_map=availability_feature_map,
    ).detach().cpu().numpy()
    assert np.allclose(single_tensor[0], batch_tensor[0], equal_nan=True)


def test_training_main_persists_preprocessor_artifact(tmp_path, scouting_app_dir, monkeypatch):
    monkeypatch.syspath_prepend(str(scouting_app_dir))
    monkeypatch.chdir(str(scouting_app_dir))

    train_module = importlib.import_module("train_model")
    generate_module = importlib.import_module("generate_data")

    db_path = tmp_path / "train_pipeline.db"
    db_url = f"sqlite:///{db_path.as_posix()}"
    generate_module.main(180, db_url, seed=42, min_age=12, max_age=18)

    model_path = tmp_path / "model.pt"
    preprocessor_path = tmp_path / "preprocessor.joblib"
    calibrator_path = tmp_path / "probability_calibrator.joblib"
    metadata_path = tmp_path / "training_metadata.json"
    splits_path = tmp_path / "training_splits.json"
    train_module.main(
        db_url,
        str(model_path),
        str(preprocessor_path),
        str(calibrator_path),
        str(metadata_path),
        epochs=4,
        lr=1e-3,
        patience=2,
    )

    assert model_path.exists()
    assert preprocessor_path.exists()
    assert metadata_path.exists()
    assert splits_path.exists()

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    splits = json.loads(splits_path.read_text(encoding="utf-8"))
    assert metadata["dataset_summary"]["age_range_filtered"]["min"] >= 12
    assert metadata["dataset_summary"]["age_range_filtered"]["max"] <= 18
    assert metadata["dataset"]["validation_size"] > 0
    assert 0.0 <= metadata["pytorch"]["selected_threshold"] <= 1.0
    assert metadata["dataset_summary"]["target_column"] == "temporal_target_label"
    assert metadata["model"]["checkpoint_version"] == train_module.MODEL_CHECKPOINT_VERSION
    assert metadata["model"]["input_dim"] > 0
    assert metadata["artifacts"]["splits_path"] == str(splits_path.resolve())
    checkpoint = train_module.load_model_checkpoint(str(model_path))
    assert checkpoint["input_dim"] == metadata["model"]["input_dim"]
    assert checkpoint["seed"] == metadata["seed"]
    assert checkpoint["is_legacy_state_dict"] is False
    assert len(splits["train_player_ids"]) > 0
    assert len(splits["validation_player_ids"]) > 0
    assert len(splits["test_player_ids"]) > 0


def test_classification_metrics_reports_unavailable_metric_warnings(scouting_app_dir, monkeypatch):
    monkeypatch.syspath_prepend(str(scouting_app_dir))
    monkeypatch.chdir(str(scouting_app_dir))

    train_module = importlib.import_module("train_model")

    metrics = train_module.classification_metrics(
        np.zeros(4, dtype=np.float32),
        np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32),
        0.5,
    )

    assert metrics["roc_auc"] == ""
    assert metrics["pr_auc"] == ""
    assert any("ROC-AUC no calculado" in warning for warning in metrics["warnings"])
    assert any("PR-AUC no calculado" in warning for warning in metrics["warnings"])


def test_classification_metrics_logs_unavailable_metric_warnings(scouting_app_dir, monkeypatch, caplog):
    monkeypatch.syspath_prepend(str(scouting_app_dir))
    monkeypatch.chdir(str(scouting_app_dir))

    train_module = importlib.import_module("train_model")

    with caplog.at_level(logging.WARNING, logger="train_model"):
        train_module.classification_metrics(
            np.zeros(4, dtype=np.float32),
            np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32),
            0.5,
        )

    assert "ROC-AUC no calculado" in caplog.text
    assert "PR-AUC no calculado" in caplog.text


def test_train_loader_shuffle_is_seeded(scouting_app_dir, monkeypatch):
    monkeypatch.syspath_prepend(str(scouting_app_dir))
    monkeypatch.chdir(str(scouting_app_dir))

    train_module = importlib.import_module("train_model")
    x_values = np.arange(24, dtype=np.float32).reshape(12, 2)
    y_values = np.array([0, 1] * 6, dtype=np.float32)

    loader_one = train_module.make_train_loader(x_values, y_values, batch_size=3, strategy="shuffle")
    loader_two = train_module.make_train_loader(x_values, y_values, batch_size=3, strategy="shuffle")
    order_one = [batch_x[:, 0].tolist() for batch_x, _ in loader_one]
    order_two = [batch_x[:, 0].tolist() for batch_x, _ in loader_two]

    assert order_one == order_two


def test_temporal_training_dataframe_cache_reuses_cached_artifact(tmp_path, scouting_app_dir, monkeypatch):
    monkeypatch.syspath_prepend(str(scouting_app_dir))
    monkeypatch.chdir(str(scouting_app_dir))

    preprocessing_module = importlib.import_module("preprocessing")
    generate_module = importlib.import_module("generate_data")
    db_utils_module = importlib.import_module("db_utils")

    db_path = tmp_path / "temporal_cache.db"
    db_url = f"sqlite:///{db_path.as_posix()}"
    cache_path = tmp_path / "temporal_training_dataframe.joblib"
    generate_module.main(40, db_url, seed=42, min_age=12, max_age=18)

    normalized_db_url = db_utils_module.normalize_db_url(db_url, base_dir=str(scouting_app_dir))
    engine = db_utils_module.create_app_engine(normalized_db_url)
    first_df = preprocessing_module.temporal_training_dataframe_from_engine(
        engine,
        use_cache=True,
        cache_path=str(cache_path),
    )
    assert cache_path.exists()

    def fail_if_rebuilt(*args, **kwargs):
        raise AssertionError("No deberia reconstruirse el dataframe temporal cuando el cache sigue valido.")

    monkeypatch.setattr(preprocessing_module, "build_temporal_training_dataframe", fail_if_rebuilt)
    second_df = preprocessing_module.temporal_training_dataframe_from_engine(
        engine,
        use_cache=True,
        cache_path=str(cache_path),
    )
    assert second_df.equals(first_df)


def test_temporal_training_dataframe_cache_invalidates_when_db_changes(tmp_path, scouting_app_dir, monkeypatch):
    monkeypatch.syspath_prepend(str(scouting_app_dir))
    monkeypatch.chdir(str(scouting_app_dir))

    preprocessing_module = importlib.import_module("preprocessing")
    generate_module = importlib.import_module("generate_data")
    db_utils_module = importlib.import_module("db_utils")

    db_path = tmp_path / "temporal_cache_refresh.db"
    db_url = f"sqlite:///{db_path.as_posix()}"
    cache_path = tmp_path / "temporal_training_dataframe.joblib"

    generate_module.main(30, db_url, seed=42, min_age=12, max_age=18, reset_existing=True)
    normalized_db_url = db_utils_module.normalize_db_url(db_url, base_dir=str(scouting_app_dir))
    engine = db_utils_module.create_app_engine(normalized_db_url)
    first_df = preprocessing_module.temporal_training_dataframe_from_engine(
        engine,
        use_cache=True,
        cache_path=str(cache_path),
    )
    assert len(first_df) == 30

    generate_module.main(45, db_url, seed=42, min_age=12, max_age=18, reset_existing=True)
    engine = db_utils_module.create_app_engine(normalized_db_url)
    second_df = preprocessing_module.temporal_training_dataframe_from_engine(
        engine,
        use_cache=True,
        cache_path=str(cache_path),
    )
    assert len(second_df) == 45


def test_evaluate_saved_model_uses_persisted_splits_and_rejects_missing_players(tmp_path, scouting_app_dir, monkeypatch):
    monkeypatch.syspath_prepend(str(scouting_app_dir))
    monkeypatch.chdir(str(scouting_app_dir))

    train_module = importlib.import_module("train_model")
    generate_module = importlib.import_module("generate_data")
    evaluate_module = importlib.import_module("evaluate_saved_model")

    db_path = tmp_path / "evaluate_pipeline.db"
    db_url = f"sqlite:///{db_path.as_posix()}"
    generate_module.main(140, db_url, seed=42, min_age=12, max_age=18)

    model_path = tmp_path / "model.pt"
    preprocessor_path = tmp_path / "preprocessor.joblib"
    calibrator_path = tmp_path / "probability_calibrator.joblib"
    metadata_path = tmp_path / "training_metadata.json"
    splits_path = tmp_path / "training_splits.json"

    train_module.main(
        db_url,
        str(model_path),
        str(preprocessor_path),
        str(calibrator_path),
        str(metadata_path),
        epochs=3,
        lr=1e-3,
        patience=2,
        splits_out=str(splits_path),
    )

    results = evaluate_module.evaluate_saved_model(
        db_url=db_url,
        model_path=str(model_path),
        preprocessor_path=str(preprocessor_path),
        splits_path=str(splits_path),
        metadata_path=str(metadata_path),
        calibrator_path=str(calibrator_path),
        use_cache=True,
    )
    assert results["splits"]["train_size"] > 0
    assert results["splits"]["validation_size"] > 0
    assert results["splits"]["test_size"] > 0
    assert results["pytorch"]["raw_test"]["pr_auc"] >= 0.0
    assert results["baseline_logistic"]["test"]["pr_auc"] >= 0.0

    broken_splits_path = tmp_path / "broken_training_splits.json"
    broken_splits = json.loads(splits_path.read_text(encoding="utf-8"))
    broken_splits["test_player_ids"].append(99999999)
    broken_splits_path.write_text(json.dumps(broken_splits, indent=2, ensure_ascii=False), encoding="utf-8")

    try:
        evaluate_module.evaluate_saved_model(
            db_url=db_url,
            model_path=str(model_path),
            preprocessor_path=str(preprocessor_path),
            splits_path=str(broken_splits_path),
            metadata_path=str(metadata_path),
            calibrator_path=str(calibrator_path),
            use_cache=True,
        )
    except ValueError as exc:
        assert "faltan 1 player_id" in str(exc)
    else:
        raise AssertionError("La evaluacion deberia fallar si los splits no coinciden con el dataframe actual.")


def test_sync_shortlist_replace_copies_rich_demo_data(tmp_path, scouting_app_dir, monkeypatch):
    monkeypatch.syspath_prepend(str(scouting_app_dir))
    monkeypatch.chdir(str(scouting_app_dir))

    generate_module = importlib.import_module("generate_data")
    sync_module = importlib.import_module("sync_shortlist")
    models_module = importlib.import_module("models")
    db_utils_module = importlib.import_module("db_utils")

    src_path = tmp_path / "training_source.db"
    dst_path = tmp_path / "operational_demo.db"
    src_url = f"sqlite:///{src_path.as_posix()}"
    dst_url = f"sqlite:///{dst_path.as_posix()}"

    generate_module.main(40, src_url, seed=42, min_age=12, max_age=18, reset_existing=True)
    src_engine = db_utils_module.create_app_engine(
        db_utils_module.normalize_db_url(src_url, base_dir=str(scouting_app_dir))
    )
    with src_engine.begin() as conn:
        conn.execute(text("UPDATE players SET birth_date = NULL"))

    dst_engine = db_utils_module.create_app_engine(
        db_utils_module.normalize_db_url(dst_url, base_dir=str(scouting_app_dir))
    )
    models_module.Base.metadata.create_all(dst_engine)
    DstSession = sessionmaker(bind=dst_engine)
    session = DstSession()
    try:
        session.add(
            models_module.User(
                username="admin",
                password_hash="hash",
                role="administrador",
            )
        )
        session.commit()
    finally:
        session.close()

    summary = sync_module.sync_shortlist(src_url, dst_url, limit=12, min_age=12, max_age=18, replace=True)

    assert summary["inserted"] == 12
    assert summary["total_operational"] == 12
    assert summary["limit"] == 12

    session = DstSession()
    try:
        assert session.query(models_module.User).count() == 1
        assert session.query(models_module.Player).count() == 12
        for player in session.query(models_module.Player).all():
            assert player.birth_date is not None
            assert 12 <= player.current_age <= 18
            assert player.category_year == player.birth_date.year
        assert session.query(models_module.PlayerStat).count() > 0
        assert session.query(models_module.PlayerAttributeHistory).count() > 0
        assert session.query(models_module.Match).count() > 0
        assert session.query(models_module.PlayerMatchParticipation).count() > 0
        assert session.query(models_module.ScoutReport).count() > 0
        assert session.query(models_module.PhysicalAssessment).count() > 0
        assert session.query(models_module.PlayerAvailability).count() > 0
    finally:
        session.close()


def test_sync_shortlist_rejects_invalid_limits(scouting_app_dir, monkeypatch):
    monkeypatch.syspath_prepend(str(scouting_app_dir))
    monkeypatch.chdir(str(scouting_app_dir))

    sync_module = importlib.import_module("sync_shortlist")

    with pytest.raises(ValueError, match="limite"):
        sync_module.validate_sync_params(0, 12, 18)
    with pytest.raises(ValueError, match="edad minima"):
        sync_module.validate_sync_params(10, 9, 18)
    with pytest.raises(ValueError, match="edad maxima"):
        sync_module.validate_sync_params(10, 18, 12)


def test_ensure_player_columns_migrates_physical_and_availability_tables(tmp_path, scouting_app_dir, monkeypatch):
    monkeypatch.syspath_prepend(str(scouting_app_dir))
    monkeypatch.chdir(str(scouting_app_dir))

    db_utils_module = importlib.import_module("db_utils")

    db_path = tmp_path / "legacy_longitudinal_tables.db"
    db_url = f"sqlite:///{db_path.as_posix()}"
    engine = db_utils_module.create_app_engine(db_url)
    with engine.begin() as conn:
        conn.execute(
            text(
                "CREATE TABLE physical_assessments ("
                "id INTEGER PRIMARY KEY, player_id INTEGER NOT NULL, assessment_date DATE NOT NULL)"
            )
        )
        conn.execute(
            text(
                "CREATE TABLE player_availability ("
                "id INTEGER PRIMARY KEY, player_id INTEGER NOT NULL, record_date DATE NOT NULL)"
            )
        )
        conn.execute(
            text(
                "INSERT INTO physical_assessments (id, player_id, assessment_date) "
                "VALUES (1, 10, '2026-01-01')"
            )
        )
        conn.execute(
            text(
                "INSERT INTO player_availability (id, player_id, record_date) "
                "VALUES (1, 10, '2026-01-01')"
            )
        )

    added_columns = db_utils_module.ensure_player_columns(engine)

    assert added_columns == 4
    inspector = inspect(engine)
    for table_name in ("physical_assessments", "player_availability"):
        columns = {column["name"] for column in inspector.get_columns(table_name)}
        assert {"created_at", "updated_at"}.issubset(columns)
        with engine.connect() as conn:
            row = conn.execute(
                text(f"SELECT created_at, updated_at FROM {table_name} WHERE id = 1")
            ).one()
        assert row.created_at is not None
        assert row.updated_at is not None


def test_ensure_player_columns_clamps_attribute_scale_to_one_twenty(tmp_path, scouting_app_dir, monkeypatch):
    monkeypatch.syspath_prepend(str(scouting_app_dir))
    monkeypatch.chdir(str(scouting_app_dir))

    db_utils_module = importlib.import_module("db_utils")

    db_path = tmp_path / "legacy_attribute_scale.db"
    db_url = f"sqlite:///{db_path.as_posix()}"
    engine = db_utils_module.create_app_engine(db_url)
    with engine.begin() as conn:
        conn.execute(
            text(
                "CREATE TABLE players ("
                "id INTEGER PRIMARY KEY, "
                "pace INTEGER, shooting INTEGER, passing INTEGER, dribbling INTEGER, defending INTEGER, "
                "physical INTEGER, vision INTEGER, tackling INTEGER, determination INTEGER, technique INTEGER)"
            )
        )
        conn.execute(
            text(
                "CREATE TABLE scout_reports ("
                "id INTEGER PRIMARY KEY, decision_making INTEGER, tactical_reading INTEGER, "
                "mental_profile INTEGER, adaptability INTEGER)"
            )
        )
        conn.execute(
            text(
                "CREATE TABLE physical_assessments ("
                "id INTEGER PRIMARY KEY, estimated_speed FLOAT, endurance FLOAT)"
            )
        )
        conn.execute(
            text(
                "INSERT INTO players "
                "(id, pace, shooting, passing, dribbling, defending, physical, vision, tackling, determination, technique) "
                "VALUES (1, 0, 21, 10, 10, 10, 10, 10, 10, 10, 10)"
            )
        )
        conn.execute(text("INSERT INTO scout_reports VALUES (1, 0, 21, NULL, 12)"))
        conn.execute(text("INSERT INTO physical_assessments VALUES (1, 0.0, 22.5)"))

    db_utils_module.ensure_player_columns(engine)

    with engine.connect() as conn:
        player = conn.execute(text("SELECT pace, shooting FROM players WHERE id = 1")).one()
        report = conn.execute(
            text("SELECT decision_making, tactical_reading, mental_profile FROM scout_reports WHERE id = 1")
        ).one()
        physical = conn.execute(text("SELECT estimated_speed, endurance FROM physical_assessments WHERE id = 1")).one()

    assert player.pace == 1
    assert player.shooting == 20
    assert report.decision_making == 1
    assert report.tactical_reading == 20
    assert report.mental_profile is None
    assert physical.estimated_speed == 1
    assert physical.endurance == 20


def test_ensure_player_columns_adds_birth_date_to_legacy_players_table(tmp_path, scouting_app_dir, monkeypatch):
    monkeypatch.syspath_prepend(str(scouting_app_dir))
    monkeypatch.chdir(str(scouting_app_dir))

    db_utils_module = importlib.import_module("db_utils")

    db_path = tmp_path / "legacy_players_birth_date.db"
    db_url = f"sqlite:///{db_path.as_posix()}"
    engine = db_utils_module.create_app_engine(db_url)
    with engine.begin() as conn:
        conn.execute(
            text(
                "CREATE TABLE players ("
                "id INTEGER PRIMARY KEY, "
                "name TEXT NOT NULL, "
                "national_id TEXT, "
                "age INTEGER NOT NULL, "
                "position TEXT NOT NULL, "
                "photo_url TEXT)"
            )
        )

    added_columns = db_utils_module.ensure_player_columns(engine)

    assert added_columns == 3
    inspector = inspect(engine)
    columns = {column["name"] for column in inspector.get_columns("players")}
    assert {"birth_date", "created_at", "updated_at"}.issubset(columns)


def test_training_dataframe_merges_historical_features(tmp_path, scouting_app_dir, monkeypatch):
    monkeypatch.syspath_prepend(str(scouting_app_dir))
    monkeypatch.chdir(str(scouting_app_dir))

    preprocessing_module = importlib.import_module("preprocessing")
    models_module = importlib.import_module("models")
    db_utils_module = importlib.import_module("db_utils")

    db_path = tmp_path / "train_stats_features.db"
    db_url = f"sqlite:///{db_path.as_posix()}"
    normalized_db_url = db_utils_module.normalize_db_url(db_url, base_dir=str(scouting_app_dir))
    engine = db_utils_module.create_app_engine(normalized_db_url)
    models_module.Base.metadata.create_all(engine)
    db_utils_module.ensure_player_columns(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    try:
        player = models_module.Player(
            name="Train Stats",
            national_id="59999100",
            age=16,
            position="Mediocampista",
            club="Club Train",
            country="Argentina",
            photo_url="",
            pace=12,
            shooting=10,
            passing=14,
            dribbling=13,
            defending=10,
            physical=11,
            vision=14,
            tackling=9,
            determination=15,
            technique=13,
            potential_label=True,
        )
        session.add(player)
        session.flush()
        session.add_all(
            [
                models_module.PlayerStat(
                    player_id=player.id,
                    record_date=date(2026, 3, 1),
                    matches_played=1,
                    minutes_played=90,
                    pass_accuracy=70.0,
                    final_score=6.0,
                ),
                models_module.PlayerStat(
                    player_id=player.id,
                    record_date=date(2026, 4, 1),
                    matches_played=1,
                    minutes_played=90,
                    pass_accuracy=80.0,
                    final_score=8.0,
                ),
                models_module.PlayerAttributeHistory(
                    player_id=player.id,
                    record_date=date(2025, 3, 1),
                    pace=9,
                    shooting=7,
                    passing=10,
                    dribbling=10,
                    defending=7,
                    physical=8,
                    vision=10,
                    tackling=6,
                    determination=11,
                    technique=10,
                ),
                models_module.PlayerAttributeHistory(
                    player_id=player.id,
                    record_date=date(2025, 10, 1),
                    pace=10,
                    shooting=8,
                    passing=11,
                    dribbling=11,
                    defending=8,
                    physical=9,
                    vision=11,
                    tackling=7,
                    determination=12,
                    technique=11,
                ),
                models_module.PlayerAttributeHistory(
                    player_id=player.id,
                    record_date=date(2026, 1, 1),
                    pace=11,
                    shooting=9,
                    passing=12,
                    dribbling=12,
                    defending=9,
                    physical=10,
                    vision=12,
                    tackling=8,
                    determination=13,
                    technique=12,
                ),
                models_module.PlayerAttributeHistory(
                    player_id=player.id,
                    record_date=date(2026, 3, 1),
                    pace=11,
                    shooting=9,
                    passing=12,
                    dribbling=12,
                    defending=9,
                    physical=10,
                    vision=12,
                    tackling=8,
                    determination=13,
                    technique=12,
                ),
                models_module.Match(
                    match_date=date(2026, 3, 18),
                    opponent_name="Lanus Proyeccion",
                    opponent_level=4,
                    tournament="Liga Juvenil Regional",
                    competition_category="Sub-17",
                    venue="Visitante",
                ),
                models_module.Match(
                    match_date=date(2026, 4, 12),
                    opponent_name="Velez Desarrollo",
                    opponent_level=2,
                    tournament="Copa Proyeccion",
                    competition_category="Sub-17",
                    venue="Local",
                ),
            ]
        )
        session.flush()
        created_matches = session.query(models_module.Match).order_by(models_module.Match.id.asc()).all()
        session.add_all(
            [
                models_module.PlayerMatchParticipation(
                    player_id=player.id,
                    match_id=created_matches[0].id,
                    started=True,
                    position_played="Mediocampista",
                    minutes_played=90,
                    final_score=6.4,
                    pass_accuracy=72.0,
                    shot_accuracy=46.0,
                    duels_won_pct=57.0,
                ),
                models_module.PlayerMatchParticipation(
                    player_id=player.id,
                    match_id=created_matches[1].id,
                    started=False,
                    position_played="Lateral",
                    minutes_played=25,
                    final_score=7.1,
                    pass_accuracy=78.0,
                    shot_accuracy=49.0,
                    duels_won_pct=60.0,
                ),
                models_module.ScoutReport(
                    player_id=player.id,
                    report_date=date(2026, 3, 20),
                    decision_making=13,
                    tactical_reading=14,
                    mental_profile=15,
                    adaptability=12,
                    observed_projection_score=7.3,
                ),
                models_module.ScoutReport(
                    player_id=player.id,
                    report_date=date(2026, 4, 15),
                    decision_making=14,
                    tactical_reading=15,
                    mental_profile=16,
                    adaptability=13,
                    observed_projection_score=7.9,
                ),
                models_module.PhysicalAssessment(
                    player_id=player.id,
                    assessment_date=date(2026, 3, 20),
                    height_cm=172.4,
                    weight_kg=64.1,
                    dominant_foot="Derecha",
                    estimated_speed=13.2,
                    endurance=12.7,
                    in_growth_spurt=True,
                ),
                models_module.PhysicalAssessment(
                    player_id=player.id,
                    assessment_date=date(2026, 4, 18),
                    height_cm=173.2,
                    weight_kg=65.0,
                    dominant_foot="Derecha",
                    estimated_speed=13.8,
                    endurance=13.3,
                    in_growth_spurt=False,
                ),
                models_module.PlayerAvailability(
                    player_id=player.id,
                    record_date=date(2026, 3, 20),
                    availability_pct=74.0,
                    fatigue_pct=36.0,
                    training_load_pct=72.0,
                    missed_days=2,
                    injury_flag=False,
                ),
                models_module.PlayerAvailability(
                    player_id=player.id,
                    record_date=date(2026, 4, 18),
                    availability_pct=86.0,
                    fatigue_pct=24.0,
                    training_load_pct=69.0,
                    missed_days=0,
                    injury_flag=False,
                ),
            ]
        )
        session.commit()
    finally:
        session.close()

    df = preprocessing_module.training_dataframe_from_engine(engine)
    row = df.iloc[0]

    assert int(row["stats_entry_count"]) == 2
    assert round(float(row["avg_final_score_hist"]), 2) == 7.0
    assert round(float(row["avg_pass_accuracy_hist"]), 2) == 75.0
    assert round(float(row["latest_final_score_hist"]), 2) == 8.0
    assert int(row["attr_history_entry_count"]) == 4
    assert float(row["attr_avg_improvement_90d"]) >= 0
    assert float(row["attr_avg_improvement_180d"]) > 0
    assert float(row["attr_avg_improvement_365d"]) > 0
    assert float(row["attr_weighted_improvement_90d"]) >= 0
    assert float(row["attr_current_vs_recent_gap"]) > 0
    assert int(row["match_entry_count"]) == 2
    assert round(float(row["match_natural_position_rate"]), 2) == 0.5
    assert int(row["scout_report_count"]) == 2
    assert round(float(row["scout_latest_projection_score"]), 2) == 7.9
    assert int(row["phys_assessment_count"]) == 2
    assert round(float(row["phys_recent_bmi"]), 2) > 20.0
    assert round(float(row["avail_recent_pct"]), 2) == 86.0
    assert float(row["avail_availability_trend"]) > 0


def test_temporal_training_dataframe_uses_future_progression_target(tmp_path, scouting_app_dir, monkeypatch):
    monkeypatch.syspath_prepend(str(scouting_app_dir))
    monkeypatch.chdir(str(scouting_app_dir))

    preprocessing_module = importlib.import_module("preprocessing")
    models_module = importlib.import_module("models")
    db_utils_module = importlib.import_module("db_utils")

    db_path = tmp_path / "temporal_target.db"
    db_url = f"sqlite:///{db_path.as_posix()}"
    normalized_db_url = db_utils_module.normalize_db_url(db_url, base_dir=str(scouting_app_dir))
    engine = db_utils_module.create_app_engine(normalized_db_url)
    models_module.Base.metadata.create_all(engine)
    db_utils_module.ensure_player_columns(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    try:
        player = models_module.Player(
            name="Temporal Prospect",
            national_id="70000111",
            age=16,
            position="Mediocampista",
            club="Club Temporal",
            country="Argentina",
            photo_url="",
            pace=14,
            shooting=11,
            passing=15,
            dribbling=14,
            defending=10,
            physical=12,
            vision=15,
            tackling=9,
            determination=16,
            technique=14,
            potential_label=True,
        )
        session.add(player)
        session.flush()
        session.add_all(
            [
                models_module.PlayerAttributeHistory(
                    player_id=player.id,
                    record_date=date(2025, 8, 1),
                    pace=10,
                    shooting=8,
                    passing=11,
                    dribbling=10,
                    defending=8,
                    physical=9,
                    vision=11,
                    tackling=7,
                    determination=12,
                    technique=10,
                ),
                models_module.PlayerAttributeHistory(
                    player_id=player.id,
                    record_date=date(2025, 10, 1),
                    pace=11,
                    shooting=8,
                    passing=12,
                    dribbling=11,
                    defending=8,
                    physical=9,
                    vision=12,
                    tackling=7,
                    determination=13,
                    technique=11,
                ),
                models_module.PlayerAttributeHistory(
                    player_id=player.id,
                    record_date=date(2026, 1, 1),
                    pace=12,
                    shooting=9,
                    passing=13,
                    dribbling=12,
                    defending=9,
                    physical=10,
                    vision=13,
                    tackling=8,
                    determination=14,
                    technique=12,
                ),
                models_module.PlayerAttributeHistory(
                    player_id=player.id,
                    record_date=date(2026, 3, 1),
                    pace=13,
                    shooting=10,
                    passing=14,
                    dribbling=13,
                    defending=9,
                    physical=11,
                    vision=14,
                    tackling=8,
                    determination=15,
                    technique=13,
                ),
                models_module.PlayerStat(
                    player_id=player.id,
                    record_date=date(2025, 8, 1),
                    matches_played=2,
                    minutes_played=110,
                    pass_accuracy=68.0,
                    final_score=5.9,
                ),
                models_module.PlayerStat(
                    player_id=player.id,
                    record_date=date(2025, 10, 1),
                    matches_played=2,
                    minutes_played=125,
                    pass_accuracy=71.0,
                    final_score=6.3,
                ),
                models_module.PlayerStat(
                    player_id=player.id,
                    record_date=date(2026, 1, 1),
                    matches_played=2,
                    minutes_played=140,
                    pass_accuracy=76.0,
                    final_score=7.0,
                ),
                models_module.PlayerStat(
                    player_id=player.id,
                    record_date=date(2026, 3, 1),
                    matches_played=3,
                    minutes_played=180,
                    pass_accuracy=80.0,
                    final_score=7.6,
                ),
                models_module.Match(
                    match_date=date(2026, 3, 10),
                    opponent_name="Rival Exigente",
                    opponent_level=4,
                    tournament="Liga Juvenil",
                    competition_category="Sub-17",
                    venue="Visitante",
                ),
                models_module.Match(
                    match_date=date(2026, 3, 24),
                    opponent_name="Rival Regional",
                    opponent_level=3,
                    tournament="Liga Juvenil",
                    competition_category="Sub-17",
                    venue="Local",
                ),
            ]
        )
        session.flush()
        future_matches = session.query(models_module.Match).order_by(models_module.Match.match_date.asc()).all()
        session.add_all(
            [
                models_module.PlayerMatchParticipation(
                    player_id=player.id,
                    match_id=future_matches[0].id,
                    started=True,
                    position_played="Mediocampista",
                    minutes_played=88,
                    final_score=7.7,
                    pass_accuracy=81.0,
                    shot_accuracy=42.0,
                    duels_won_pct=63.0,
                ),
                models_module.PlayerMatchParticipation(
                    player_id=player.id,
                    match_id=future_matches[1].id,
                    started=True,
                    position_played="Mediocampista",
                    minutes_played=76,
                    final_score=7.4,
                    pass_accuracy=79.0,
                    shot_accuracy=38.0,
                    duels_won_pct=61.0,
                ),
                models_module.PlayerAvailability(
                    player_id=player.id,
                    record_date=date(2025, 8, 1),
                    availability_pct=72.0,
                    fatigue_pct=34.0,
                    training_load_pct=68.0,
                    missed_days=1,
                    injury_flag=False,
                ),
                models_module.PlayerAvailability(
                    player_id=player.id,
                    record_date=date(2025, 10, 1),
                    availability_pct=76.0,
                    fatigue_pct=31.0,
                    training_load_pct=71.0,
                    missed_days=1,
                    injury_flag=False,
                ),
                models_module.PlayerAvailability(
                    player_id=player.id,
                    record_date=date(2026, 1, 1),
                    availability_pct=82.0,
                    fatigue_pct=28.0,
                    training_load_pct=73.0,
                    missed_days=0,
                    injury_flag=False,
                ),
                models_module.PlayerAvailability(
                    player_id=player.id,
                    record_date=date(2026, 3, 1),
                    availability_pct=88.0,
                    fatigue_pct=22.0,
                    training_load_pct=69.0,
                    missed_days=0,
                    injury_flag=False,
                ),
                models_module.ScoutReport(
                    player_id=player.id,
                    report_date=date(2025, 10, 5),
                    decision_making=13,
                    tactical_reading=13,
                    mental_profile=14,
                    adaptability=12,
                    observed_projection_score=6.7,
                    notes="Seguimiento observado",
                ),
                models_module.ScoutReport(
                    player_id=player.id,
                    report_date=date(2026, 3, 12),
                    decision_making=16,
                    tactical_reading=15,
                    mental_profile=16,
                    adaptability=14,
                    observed_projection_score=8.1,
                    notes="Seguimiento futuro",
                ),
            ]
        )
        session.commit()
    finally:
        session.close()

    temporal_df = preprocessing_module.temporal_training_dataframe_from_engine(engine)

    assert len(temporal_df) == 1
    row = temporal_df.iloc[0]
    assert bool(row["temporal_target_label"]) is True
    assert float(row["weighted_score_growth"]) > 0
    assert float(row["future_final_score"]) > float(row["observed_final_score"])
    assert float(row["attr_history_entry_count"]) >= 2
    assert float(row["future_availability_pct"]) >= 80.0


def test_generate_data_creates_match_context_and_scout_reports(tmp_path, scouting_app_dir, monkeypatch):
    monkeypatch.syspath_prepend(str(scouting_app_dir))
    monkeypatch.chdir(str(scouting_app_dir))

    generate_module = importlib.import_module("generate_data")
    models_module = importlib.import_module("models")
    db_utils_module = importlib.import_module("db_utils")

    db_path = tmp_path / "synthetic_context.db"
    db_url = f"sqlite:///{db_path.as_posix()}"
    generate_module.main(12, db_url, seed=42, min_age=12, max_age=18)

    normalized_db_url = db_utils_module.normalize_db_url(db_url, base_dir=str(scouting_app_dir))
    engine = db_utils_module.create_app_engine(normalized_db_url)
    Session = sessionmaker(bind=engine)
    session = Session()
    try:
        assert session.query(models_module.Player).count() == 12
        assert session.query(models_module.PlayerAttributeHistory).count() > 0
        assert session.query(models_module.PhysicalAssessment).count() > 0
        assert session.query(models_module.PlayerAvailability).count() > 0
        assert session.query(models_module.Match).count() > 0
        assert session.query(models_module.PlayerMatchParticipation).count() > 0
        assert session.query(models_module.PlayerStat).count() > 0
        assert session.query(models_module.ScoutReport).count() > 0
    finally:
        session.close()


def test_load_data_filters_training_range(tmp_path, scouting_app_dir, monkeypatch):
    monkeypatch.syspath_prepend(str(scouting_app_dir))
    monkeypatch.chdir(str(scouting_app_dir))

    train_module = importlib.import_module("train_model")
    models_module = importlib.import_module("models")
    db_utils_module = importlib.import_module("db_utils")

    db_path = tmp_path / "train_filter.db"
    db_url = f"sqlite:///{db_path.as_posix()}"
    normalized_db_url = db_utils_module.normalize_db_url(db_url, base_dir=str(scouting_app_dir))
    engine = db_utils_module.create_app_engine(normalized_db_url)
    models_module.Base.metadata.create_all(engine)
    db_utils_module.ensure_player_columns(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    try:
        session.add_all(
            [
                models_module.Player(
                    name="Juvenil",
                    national_id="59999001",
                    age=17,
                    position="Defensa",
                    club="Club Train",
                    country="Argentina",
                    photo_url="",
                    pace=10,
                    shooting=8,
                    passing=11,
                    dribbling=9,
                    defending=14,
                    physical=12,
                    vision=10,
                    tackling=13,
                    determination=12,
                    technique=9,
                    potential_label=False,
                ),
                models_module.Player(
                    name="Mayor",
                    national_id="59999002",
                    age=21,
                    position="Delantero",
                    club="Club Train",
                    country="Argentina",
                    photo_url="",
                    pace=16,
                    shooting=17,
                    passing=10,
                    dribbling=15,
                    defending=6,
                    physical=12,
                    vision=10,
                    tackling=5,
                    determination=16,
                    technique=14,
                    potential_label=True,
                ),
            ]
        )
        session.flush()
        session.add_all(
            [
                models_module.PlayerAttributeHistory(
                    player_id=1,
                    record_date=date(2025, 8, 1),
                    pace=8,
                    shooting=7,
                    passing=9,
                    dribbling=8,
                    defending=11,
                    physical=10,
                    vision=8,
                    tackling=11,
                    determination=10,
                    technique=8,
                ),
                models_module.PlayerAttributeHistory(
                    player_id=1,
                    record_date=date(2025, 11, 1),
                    pace=9,
                    shooting=7,
                    passing=10,
                    dribbling=8,
                    defending=12,
                    physical=11,
                    vision=9,
                    tackling=12,
                    determination=11,
                    technique=8,
                ),
                models_module.PlayerAttributeHistory(
                    player_id=1,
                    record_date=date(2026, 2, 1),
                    pace=10,
                    shooting=8,
                    passing=11,
                    dribbling=9,
                    defending=13,
                    physical=12,
                    vision=10,
                    tackling=13,
                    determination=12,
                    technique=9,
                ),
                models_module.PlayerAttributeHistory(
                    player_id=1,
                    record_date=date(2026, 4, 1),
                    pace=10,
                    shooting=8,
                    passing=11,
                    dribbling=9,
                    defending=14,
                    physical=12,
                    vision=10,
                    tackling=13,
                    determination=12,
                    technique=9,
                ),
                models_module.PlayerStat(
                    player_id=1,
                    record_date=date(2025, 8, 1),
                    matches_played=1,
                    minutes_played=80,
                    pass_accuracy=65.0,
                    final_score=5.8,
                ),
                models_module.PlayerStat(
                    player_id=1,
                    record_date=date(2025, 11, 1),
                    matches_played=1,
                    minutes_played=90,
                    pass_accuracy=69.0,
                    final_score=6.2,
                ),
                models_module.PlayerStat(
                    player_id=1,
                    record_date=date(2026, 2, 1),
                    matches_played=2,
                    minutes_played=150,
                    pass_accuracy=73.0,
                    final_score=6.9,
                ),
                models_module.PlayerStat(
                    player_id=1,
                    record_date=date(2026, 4, 1),
                    matches_played=2,
                    minutes_played=165,
                    pass_accuracy=75.0,
                    final_score=7.1,
                ),
                models_module.PlayerAttributeHistory(
                    player_id=2,
                    record_date=date(2025, 8, 1),
                    pace=13,
                    shooting=14,
                    passing=8,
                    dribbling=13,
                    defending=5,
                    physical=10,
                    vision=8,
                    tackling=4,
                    determination=14,
                    technique=12,
                ),
                models_module.PlayerAttributeHistory(
                    player_id=2,
                    record_date=date(2025, 11, 1),
                    pace=14,
                    shooting=15,
                    passing=8,
                    dribbling=14,
                    defending=5,
                    physical=11,
                    vision=8,
                    tackling=4,
                    determination=15,
                    technique=13,
                ),
                models_module.PlayerAttributeHistory(
                    player_id=2,
                    record_date=date(2026, 2, 1),
                    pace=15,
                    shooting=16,
                    passing=9,
                    dribbling=15,
                    defending=6,
                    physical=11,
                    vision=9,
                    tackling=4,
                    determination=15,
                    technique=13,
                ),
                models_module.PlayerAttributeHistory(
                    player_id=2,
                    record_date=date(2026, 4, 1),
                    pace=16,
                    shooting=17,
                    passing=10,
                    dribbling=15,
                    defending=6,
                    physical=12,
                    vision=10,
                    tackling=5,
                    determination=16,
                    technique=14,
                ),
                models_module.PlayerStat(
                    player_id=2,
                    record_date=date(2025, 8, 1),
                    matches_played=1,
                    minutes_played=75,
                    pass_accuracy=58.0,
                    final_score=5.7,
                ),
                models_module.PlayerStat(
                    player_id=2,
                    record_date=date(2025, 11, 1),
                    matches_played=1,
                    minutes_played=85,
                    pass_accuracy=60.0,
                    final_score=6.0,
                ),
                models_module.PlayerStat(
                    player_id=2,
                    record_date=date(2026, 2, 1),
                    matches_played=2,
                    minutes_played=145,
                    pass_accuracy=64.0,
                    final_score=6.5,
                ),
                models_module.PlayerStat(
                    player_id=2,
                    record_date=date(2026, 4, 1),
                    matches_played=2,
                    minutes_played=160,
                    pass_accuracy=66.0,
                    final_score=6.8,
                ),
            ]
        )
        session.commit()
    finally:
        session.close()

    df, y, summary = train_module.load_data(db_url)

    assert len(df) == 1
    assert float(df.iloc[0]["age"]) <= 17.0
    assert len(y) == 1
    assert summary["raw_rows"] == 2
    assert summary["filtered_rows"] == 1
    assert summary["target_column"] == "temporal_target_label"


def test_label_probability_favors_younger_and_position_fit(scouting_app_dir, monkeypatch):
    monkeypatch.syspath_prepend(str(scouting_app_dir))
    monkeypatch.chdir(str(scouting_app_dir))
    generate_module = importlib.import_module("generate_data")

    attrs = {
        "pace": 14,
        "shooting": 8,
        "passing": 12,
        "dribbling": 13,
        "defending": 15,
        "physical": 12,
        "vision": 11,
        "tackling": 14,
        "determination": 16,
        "technique": 12,
    }

    younger_prob, _ = generate_module.label_probability("Defensa", 13, attrs)
    older_prob, _ = generate_module.label_probability("Defensa", 18, attrs)
    aligned_prob, _ = generate_module.label_probability("Defensa", 15, attrs)
    misaligned_prob, _ = generate_module.label_probability("Delantero", 15, attrs)

    assert younger_prob > older_prob
    assert aligned_prob > misaligned_prob
