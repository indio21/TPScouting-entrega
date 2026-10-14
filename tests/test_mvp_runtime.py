"""Regresiones del MVP: runtime."""

import importlib
import sqlite3
import subprocess
import sys
from types import SimpleNamespace
import numpy as np
import pytest

from mvp_regression_helpers import (
    _create_player,
)


def test_professor_demo_script_creates_portable_verified_dataset(tmp_path, scouting_app_dir):
    repo_root = scouting_app_dir.parent
    script_path = repo_root / "scripts" / "iniciar_demo.py"
    db_path = tmp_path / "professor_demo.db"
    command = [
        sys.executable,
        str(script_path),
        "--players",
        "60",
        "--db-path",
        str(db_path),
        "--solo-preparar",
    ]

    first_run = subprocess.run(command, cwd=repo_root, capture_output=True, text=True, check=False)
    assert first_run.returncode == 0, first_run.stdout + first_run.stderr
    assert "Jugadores disponibles: 60" in first_run.stdout

    with sqlite3.connect(db_path) as connection:
        total, missing_birth_date, min_age, max_age = connection.execute(
            """
            SELECT COUNT(*),
                   SUM(CASE WHEN birth_date IS NULL THEN 1 ELSE 0 END),
                   MIN(age),
                   MAX(age)
            FROM players
            """
        ).fetchone()
        admin_count = connection.execute(
            "SELECT COUNT(*) FROM users WHERE username = ? AND role = ?",
            ("profesor_demo", "administrador"),
        ).fetchone()[0]
        related_counts = [
            connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in (
                "player_stats",
                "player_attribute_history",
                "player_match_participations",
                "scout_reports",
                "physical_assessments",
                "player_availability",
            )
        ]

    assert total == 60
    assert missing_birth_date == 0
    assert 12 <= min_age <= max_age <= 18
    assert admin_count == 1
    assert all(count > 0 for count in related_counts)

    second_run = subprocess.run(command, cwd=repo_root, capture_output=True, text=True, check=False)
    assert second_run.returncode == 0, second_run.stdout + second_run.stderr
    with sqlite3.connect(db_path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM players").fetchone()[0] == 60

def test_default_player_photo_url_uses_local_silhouette(app_module):
    assert app_module.default_player_photo_url(name="Jugador Sin Foto") == "/static/img/player-silhouette.svg"

def test_player_photo_url_policy_accepts_only_https_or_own_static_paths(app_module):
    accepted = ["", "/static/img/player-silhouette.svg", "https://images.example/player.jpg"]
    rejected = [
        "http://images.example/player.jpg",
        "//images.example/player.jpg",
        "javascript:alert(1)",
        "data:image/svg+xml,<svg></svg>",
        "/uploads/player.jpg",
        "/static/../secret.txt",
        "/static/%252e%252e/secret.txt",
        r"/static\evil.svg",
        "https://user:password@images.example/player.jpg",
    ]

    assert all(app_module.is_valid_player_photo_url(value) for value in accepted)
    assert not any(app_module.is_valid_player_photo_url(value) for value in rejected)

def test_photo_backfill_replaces_legacy_dicebear_avatar(app_module, db):
    player = _create_player(
        app_module,
        db,
        national_id="30123457",
        photo_url="https://api.dicebear.com/9.x/adventurer/svg?seed=legacy",
    )

    assert app_module.backfill_player_photo_urls(db) == 1
    db.commit()
    db.refresh(player)

    assert player.photo_url == "/static/img/player-silhouette.svg"

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

def test_settings_does_not_run_pipeline_in_production_request(scouting_app_dir, monkeypatch):
    monkeypatch.syspath_prepend(str(scouting_app_dir))
    settings_module = importlib.import_module("routes.settings")
    from flask import Flask

    captured = {}
    calls = []

    monkeypatch.setattr(
        settings_module,
        "render_template",
        lambda _template_name, **kwargs: captured.update(kwargs) or "ok",
    )

    class FakeSession:
        def close(self):
            pass

    def roles_required(_role):
        return lambda fn: fn

    deps = SimpleNamespace(
        ROLE_ADMIN="administrador",
        roles_required=roles_required,
        require_csrf=lambda: None,
        PRODUCTION_MODE=True,
        pipeline_lock=SimpleNamespace(acquire=lambda **_kwargs: calls.append("lock")),
        update_database_pipeline=lambda **_kwargs: calls.append("pipeline"),
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
    assert calls == []
    assert "desactivados en produccion" in captured["status_messages"][0]

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
