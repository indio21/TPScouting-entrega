import os
import sys
import importlib.util
from pathlib import Path

import pytest

@pytest.fixture(scope="session")
def scouting_app_dir() -> Path:
    here = Path(__file__).resolve().parent
    return (here.parent / "scouting_app").resolve()

@pytest.fixture()
def app_module(tmp_path, scouting_app_dir, monkeypatch):
    db_path = tmp_path / "test_players.db"
    training_db_path = tmp_path / "test_training.db"

    app_db_url = f"sqlite:///{db_path.as_posix()}"
    training_db_url = f"sqlite:///{training_db_path.as_posix()}"

    monkeypatch.setenv("APP_SECRET_KEY", "test-secret")
    monkeypatch.setenv("APP_DB_URL", app_db_url)
    monkeypatch.setenv("TRAINING_DB_URL", training_db_url)

    monkeypatch.syspath_prepend(str(scouting_app_dir))
    monkeypatch.chdir(str(scouting_app_dir))

    app_py = scouting_app_dir / "app.py"
    module_name = "scouting_app_app_test"
    sys.modules.pop(module_name, None)
    spec = importlib.util.spec_from_file_location(module_name, app_py)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod

@pytest.fixture()
def client(app_module):
    app = app_module.app
    app.config.update({"TESTING": True})
    return app.test_client()

@pytest.fixture()
def db(app_module):
    return app_module.Session()
