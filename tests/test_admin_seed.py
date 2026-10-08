import importlib

from sqlalchemy.orm import sessionmaker


def test_create_admin_is_idempotent_and_uses_temporary_database(
    tmp_path, scouting_app_dir, monkeypatch
):
    monkeypatch.syspath_prepend(str(scouting_app_dir))
    module = importlib.import_module("create_admin")
    db_path = tmp_path / "admin.db"
    monkeypatch.setenv("APP_DB_URL", f"sqlite:///{db_path.as_posix()}")
    monkeypatch.setenv("ADMIN_USERNAME", "admin_prueba")
    monkeypatch.setenv("ADMIN_PASSWORD", "Clave1234")

    assert module.main() == 0
    assert module.main() == 0

    engine = module.create_app_engine(f"sqlite:///{db_path.as_posix()}")
    session = sessionmaker(bind=engine)()
    try:
        users = session.query(module.User).filter_by(username="admin_prueba").all()
        assert len(users) == 1
        assert users[0].role == "administrador"
    finally:
        session.close()
        engine.dispose()


def test_create_admin_rejects_missing_credentials_without_creating_database(
    tmp_path, scouting_app_dir, monkeypatch
):
    monkeypatch.syspath_prepend(str(scouting_app_dir))
    module = importlib.import_module("create_admin")
    db_path = tmp_path / "missing.db"
    monkeypatch.setenv("APP_DB_URL", f"sqlite:///{db_path.as_posix()}")
    monkeypatch.delenv("ADMIN_USERNAME", raising=False)
    monkeypatch.delenv("ADMIN_PASSWORD", raising=False)

    assert module.main() == 2
    assert not db_path.exists()


def test_seed_demo_disabled_has_no_side_effects(tmp_path, scouting_app_dir, monkeypatch):
    monkeypatch.syspath_prepend(str(scouting_app_dir))
    module = importlib.import_module("seed_demo_data")
    db_path = tmp_path / "disabled.db"
    monkeypatch.setenv("APP_DB_URL", f"sqlite:///{db_path.as_posix()}")
    monkeypatch.setenv("DEMO_SEED_ON_STARTUP", "0")

    assert module.main() == 0
    assert not db_path.exists()


def test_seed_demo_uses_requested_count_and_seed(tmp_path, scouting_app_dir, monkeypatch):
    monkeypatch.syspath_prepend(str(scouting_app_dir))
    module = importlib.import_module("seed_demo_data")
    db_path = tmp_path / "seed.db"
    calls = []
    monkeypatch.setenv("APP_DB_URL", f"sqlite:///{db_path.as_posix()}")
    monkeypatch.setenv("DEMO_SEED_ON_STARTUP", "1")
    monkeypatch.setenv("DEMO_SEED_PLAYERS", "7")
    monkeypatch.setenv("DEMO_SEED", "123")
    monkeypatch.setattr(module, "generate_demo_data", lambda *args, **kwargs: calls.append((args, kwargs)))

    assert module.main() == 0
    assert len(calls) == 1
    args, kwargs = calls[0]
    assert args[:2] == (7, f"sqlite:///{db_path.as_posix()}")
    assert kwargs["seed"] == 123
    assert kwargs["reset_existing"] is False
