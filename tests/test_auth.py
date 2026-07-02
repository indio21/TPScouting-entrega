from werkzeug.security import generate_password_hash
from flask import url_for

def _create_user(db, User, username, password, role="scout"):
    u = User(username=username, password_hash=generate_password_hash(password), role=role)
    db.add(u)
    db.commit()
    return u

def _get_csrf_token(client, path="/login"):
    client.get(path)
    with client.session_transaction() as sess:
        return sess.get("csrf_token")

def test_landing_public_ok(client):
    resp = client.get("/")
    assert resp.status_code == 200


def test_security_headers_are_present(client):
    resp = client.get("/")

    assert resp.headers["X-Content-Type-Options"] == "nosniff"
    assert resp.headers["X-Frame-Options"] == "DENY"
    assert resp.headers["Referrer-Policy"] == "same-origin"


def test_auth_blueprint_keeps_legacy_endpoint_names(app_module):
    with app_module.app.test_request_context():
        assert url_for("login") == "/login"
        assert url_for("logout") == "/logout"
        assert url_for("register") == "/register"

def test_protected_requires_login_redirects_to_login(client):
    resp = client.get("/players")
    assert resp.status_code in (301, 302)
    assert "/login" in resp.headers.get("Location", "")


def test_core_protected_pages_require_login(client):
    protected_paths = [
        "/players",
        "/dashboard",
        "/compare",
        "/compare/multi",
        "/settings",
        "/players/manage",
        "/players/import",
        "/player/1/predict",
    ]

    for path in protected_paths:
        resp = client.get(path, follow_redirects=False)
        assert resp.status_code in (301, 302), path
        assert "/login" in resp.headers.get("Location", ""), path


def test_login_success_sets_session_and_redirects(client, app_module, db):
    _create_user(db, app_module.User, "user1", "pass1", role="scout")
    csrf_token = _get_csrf_token(client, "/login")
    resp = client.post(
        "/login",
        data={"username": "user1", "password": "pass1", "csrf_token": csrf_token},
        follow_redirects=False,
    )
    assert resp.status_code in (301, 302)
    assert "/players" in resp.headers.get("Location", "")

def test_logout_clears_session_and_redirects_to_landing(client, app_module, db):
    _create_user(db, app_module.User, "user2", "pass2", role="scout")
    csrf_token = _get_csrf_token(client, "/login")
    client.post("/login", data={"username": "user2", "password": "pass2", "csrf_token": csrf_token})
    resp = client.post("/logout", data={"csrf_token": csrf_token}, follow_redirects=False)
    assert resp.status_code in (301, 302)
    assert resp.headers.get("Location", "").endswith("/")


def test_logout_rejects_get_and_missing_csrf(client, app_module, db):
    _create_user(db, app_module.User, "user_logout", "pass2", role="scout")
    csrf_token = _get_csrf_token(client, "/login")
    client.post("/login", data={"username": "user_logout", "password": "pass2", "csrf_token": csrf_token})

    get_resp = client.get("/logout", follow_redirects=False)
    assert get_resp.status_code == 405

    post_resp = client.post("/logout", data={}, follow_redirects=False)
    assert post_resp.status_code == 400


def test_register_requires_admin_role(client, app_module, db):
    _create_user(db, app_module.User, "user3", "pass3", role="scout")
    csrf_token = _get_csrf_token(client, "/login")
    client.post("/login", data={"username": "user3", "password": "pass3", "csrf_token": csrf_token})
    resp = client.get("/register")
    assert resp.status_code == 403
