from datetime import date

from werkzeug.security import generate_password_hash
from flask import url_for

def _create_user(db, User, username, password, role="scout"):
    u = User(username=username, password_hash=generate_password_hash(password), role=role)
    db.add(u)
    db.commit()
    return u

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

def _get_csrf_token(client, path="/login"):
    client.get(path)
    with client.session_transaction() as sess:
        return sess.get("csrf_token")

def _login(client, username, password):
    csrf_token = _get_csrf_token(client, "/login")
    return client.post("/login", data={"username": username, "password": password, "csrf_token": csrf_token})


def _logout(client):
    csrf_token = _get_csrf_token(client, "/login")
    return client.post("/logout", data={"csrf_token": csrf_token})

def test_players_list_ok_after_login(client, app_module, db):
    _create_user(db, app_module.User, "u", "p", role="scout")
    _create_player(app_module, db, name="Juvenil Listado", age=16)
    _login(client, "u", "p")
    resp = client.get("/players")
    body = resp.get_data(as_text=True)
    assert resp.status_code == 200
    assert "Juvenil Listado" in body
    assert "16 anos" in body
    assert "Cat. 2010" in body

def test_dashboard_ok_after_login(client, app_module, db):
    _create_user(db, app_module.User, "u2", "p2", role="scout")
    _login(client, "u2", "p2")
    resp = client.get("/dashboard")
    assert resp.status_code == 200


def test_dashboard_copy_changes_by_role(client, app_module, db):
    _create_user(db, app_module.User, "admin_dash", "admin1234", role="administrador")
    _create_user(db, app_module.User, "director_dash", "director123", role="director")
    _create_player(app_module, db, name="Juvenil Dashboard", age=16)

    _login(client, "admin_dash", "admin1234")
    admin_resp = client.get("/dashboard?period=custom&start=2026-01-01&end=2026-01-31")
    admin_body = admin_resp.get_data(as_text=True)
    assert admin_resp.status_code == 200
    assert "Mesa de scouting" in admin_body
    assert "Estado del plantel" not in admin_body
    assert "Juvenil Dashboard" in admin_body
    assert "16 anos" in admin_body
    assert "Cat. 2010" in admin_body

    _logout(client)
    _login(client, "director_dash", "director123")
    director_resp = client.get("/dashboard?period=custom&start=2026-01-01&end=2026-01-31")
    director_body = director_resp.get_data(as_text=True)
    assert director_resp.status_code == 200
    assert "Estado del plantel" in director_body
    assert "Mesa de scouting" not in director_body

def test_settings_requires_admin_after_hotfix(client, app_module, db):
    _create_user(db, app_module.User, "u3", "p3", role="scout")
    _login(client, "u3", "p3")
    resp = client.get("/settings")
    assert resp.status_code == 403


def test_login_page_renders_polished_layout(client):
    resp = client.get("/login")
    body = resp.get_data(as_text=True)

    assert resp.status_code == 200
    assert "auth-hero" in body
    assert "Nombre de usuario" in body
    assert "csrf_token" in body


def test_staff_blueprint_keeps_legacy_endpoint_names(app_module):
    with app_module.app.test_request_context():
        assert url_for("list_coaches") == "/coaches"
        assert url_for("new_coach") == "/coaches/new"
        assert url_for("edit_coach", coach_id=7) == "/coaches/edit/7"
        assert url_for("delete_coach", coach_id=7) == "/coaches/delete/7"
        assert url_for("list_directors") == "/directors"
        assert url_for("new_director") == "/directors/new"
        assert url_for("edit_director", director_id=8) == "/directors/edit/8"
        assert url_for("delete_director", director_id=8) == "/directors/delete/8"


def test_players_blueprint_keeps_legacy_endpoint_names(app_module):
    with app_module.app.test_request_context():
        assert url_for("index") == "/players"
        assert url_for("manage_players") == "/players/manage"
        assert url_for("import_players") == "/players/import"
        assert url_for("download_players_import_template") == "/players/import/template.csv"
        assert url_for("player_detail", player_id=9) == "/player/9"
        assert url_for("add_player_match_history", player_id=9) == "/player/9/matches/add"
        assert (
            url_for("edit_player_match_history", player_id=9, participation_id=2)
            == "/player/9/matches/2/edit"
        )
        assert (
            url_for("delete_player_match_history", player_id=9, participation_id=2)
            == "/player/9/matches/2/delete"
        )
        assert url_for("add_player_physical_assessment", player_id=9) == "/player/9/physical/add"
        assert (
            url_for("edit_player_physical_assessment", player_id=9, assessment_id=5)
            == "/player/9/physical/5/edit"
        )
        assert (
            url_for("delete_player_physical_assessment", player_id=9, assessment_id=5)
            == "/player/9/physical/5/delete"
        )
        assert url_for("add_player_availability", player_id=9) == "/player/9/availability/add"
        assert (
            url_for("edit_player_availability", player_id=9, availability_id=6)
            == "/player/9/availability/6/edit"
        )
        assert (
            url_for("delete_player_availability", player_id=9, availability_id=6)
            == "/player/9/availability/6/delete"
        )
        assert url_for("add_player_scout_report", player_id=9) == "/player/9/reports/add"
        assert (
            url_for("edit_player_scout_report", player_id=9, report_id=7)
            == "/player/9/reports/7/edit"
        )
        assert (
            url_for("delete_player_scout_report", player_id=9, report_id=7)
            == "/player/9/reports/7/delete"
        )
        assert url_for("player_stats", player_id=9) == "/player/9/stats"
        assert url_for("edit_player_stat", player_id=9, stat_id=3) == "/player/9/stats/3/edit"
        assert url_for("delete_player_stat", player_id=9, stat_id=3) == "/player/9/stats/3/delete"
        assert url_for("player_attributes", player_id=9) == "/player/9/attributes"
        assert (
            url_for("edit_player_attribute_history", player_id=9, history_id=4)
            == "/player/9/attributes/4/edit"
        )
        assert (
            url_for("delete_player_attribute_history", player_id=9, history_id=4)
            == "/player/9/attributes/4/delete"
        )
        assert url_for("predict_player", player_id=9) == "/player/9/predict"
        assert url_for("edit_player", player_id=9) == "/edit_player/9"
        assert url_for("delete_player", player_id=9) == "/delete_player/9"


def test_compare_and_settings_blueprints_keep_legacy_endpoint_names(app_module):
    with app_module.app.test_request_context():
        assert url_for("dashboard") == "/dashboard"
        assert url_for("compare_players") == "/compare"
        assert url_for("compare_multi") == "/compare/multi"
        assert url_for("settings") == "/settings"


def test_compare_pages_ok_after_login(client, app_module, db):
    _create_user(db, app_module.User, "u4", "p4", role="scout")
    _login(client, "u4", "p4")
    resp_compare = client.get("/compare")
    resp_compare_multi = client.get("/compare/multi")
    assert resp_compare.status_code == 200
    assert resp_compare_multi.status_code == 200
