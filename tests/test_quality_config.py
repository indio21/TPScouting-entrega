from datetime import datetime, timezone


def test_env_int_uses_default_and_minimum(app_module, monkeypatch):
    monkeypatch.setenv("QUALITY_INT", "invalido")
    assert app_module.env_int("QUALITY_INT", 12, minimum=2) == 12
    monkeypatch.setenv("QUALITY_INT", "-5")
    assert app_module.env_int("QUALITY_INT", 12, minimum=2) == 2
    monkeypatch.setenv("QUALITY_INT", "25")
    assert app_module.env_int("QUALITY_INT", 12, minimum=2) == 25


def test_timestamp_defaults_remain_naive_utc_compatible(app_module):
    value = app_module.Player.created_at.default.arg(None)
    utc_now = datetime.now(timezone.utc).replace(tzinfo=None)

    assert value.tzinfo is None
    assert abs((utc_now - value).total_seconds()) < 5
