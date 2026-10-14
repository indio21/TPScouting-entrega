"""Regresiones del MVP: ml data."""

import importlib
import json
import logging
from datetime import date
import numpy as np
import pytest
from sqlalchemy import inspect, text
from sqlalchemy.orm import sessionmaker

from mvp_regression_helpers import (
    _create_player,
    _fit_test_preprocessor,
)


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
