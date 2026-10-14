"""Regresiones del MVP: auth security."""

from datetime import date

from mvp_regression_helpers import (
    _availability_payload,
    _create_player,
    _create_user,
    _get_csrf_token,
    _login,
    _match_history_payload,
    _physical_assessment_payload,
    _scout_report_payload,
    _valid_manage_player_payload,
)


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

def test_login_rate_limit_cannot_be_evaded_with_rotating_forwarded_headers(client, app_module, db):
    _create_user(db, app_module.User, "rotating_headers", "blocked123", role="scout")

    for attempt in range(5):
        csrf_token = _get_csrf_token(client, "/login")
        response = client.post(
            "/login",
            data={"username": "rotating_headers", "password": "mal", "csrf_token": csrf_token},
            headers={"X-Forwarded-For": f"198.51.100.{attempt + 1}"},
            environ_overrides={"REMOTE_ADDR": f"203.0.113.{attempt + 1}"},
        )
        assert response.status_code == 200

    csrf_token = _get_csrf_token(client, "/login")
    blocked = client.post(
        "/login",
        data={"username": "rotating_headers", "password": "mal", "csrf_token": csrf_token},
        headers={"X-Forwarded-For": "192.0.2.200"},
        environ_overrides={"REMOTE_ADDR": "192.0.2.201"},
    )

    assert blocked.status_code == 429

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

def test_public_health_returns_only_minimal_status(client, app_module, db):
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
    assert payload == {"status": "ok"}

def test_public_health_hides_internal_exception_details(client, app_module, monkeypatch):
    def fail_connect():
        raise RuntimeError("postgresql://secret-user:secret-password@private-host/database")

    monkeypatch.setattr(app_module.engine, "connect", fail_connect)
    response = client.get("/health")

    assert response.status_code == 500
    assert response.get_json() == {"status": "error"}
    assert "secret-password" not in response.get_data(as_text=True)

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
