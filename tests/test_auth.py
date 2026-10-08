import re

import pytest
from werkzeug.security import generate_password_hash
from flask import url_for
from werkzeug.exceptions import BadRequest

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
    policy = resp.headers["Content-Security-Policy"]
    assert "default-src 'self'" in policy
    assert "object-src 'none'" in policy
    assert "frame-ancestors 'none'" in policy
    nonce_match = re.search(r"'nonce-([^']+)'", policy)
    assert nonce_match is not None
    assert f'nonce="{nonce_match.group(1)}"' in resp.get_data(as_text=True)


def test_hsts_is_limited_to_production(client, app_module, monkeypatch):
    assert "Strict-Transport-Security" not in client.get("/").headers

    monkeypatch.setattr(app_module, "_is_prod_runtime", True)
    response = client.get("/")
    assert response.headers["Strict-Transport-Security"] == "max-age=31536000; includeSubDomains"


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


def test_login_accepts_only_safe_internal_next_targets(client, app_module, db):
    _create_user(db, app_module.User, "safe_next", "pass1", role="scout")

    csrf_token = _get_csrf_token(client, "/login")
    valid = client.post(
        "/login?next=/dashboard?period=30",
        data={"username": "safe_next", "password": "pass1", "csrf_token": csrf_token},
        follow_redirects=False,
    )
    assert valid.headers["Location"].endswith("/dashboard?period=30")

    invalid_targets = [
        "https://evil.example/",
        "//evil.example/",
        "///evil.example/",
        r"/\evil.example/",
        "javascript:alert(1)",
        "dashboard",
        "/%2f%2fevil.example/",
        "/%5cevil.example/",
        "/safe\r\nLocation: https://evil.example/",
        "http://[invalid",
    ]
    for target in invalid_targets:
        csrf_token = _get_csrf_token(client, "/login")
        response = client.post(
            "/login",
            query_string={"next": target},
            data={"username": "safe_next", "password": "pass1", "csrf_token": csrf_token},
            follow_redirects=False,
        )
        assert response.headers["Location"].endswith("/players"), target


def test_login_clears_previous_session_state(client, app_module, db):
    _create_user(db, app_module.User, "clean_session", "pass1", role="scout")
    csrf_token = _get_csrf_token(client, "/login")
    with client.session_transaction() as sess:
        sess["untrusted_marker"] = "must-disappear"

    response = client.post(
        "/login",
        data={"username": "clean_session", "password": "pass1", "csrf_token": csrf_token},
        follow_redirects=False,
    )

    assert response.status_code == 302
    with client.session_transaction() as sess:
        assert sess["username"] == "clean_session"
        assert "untrusted_marker" not in sess
        assert "csrf_token" not in sess

def test_logout_clears_session_and_redirects_to_landing(client, app_module, db):
    _create_user(db, app_module.User, "user2", "pass2", role="scout")
    csrf_token = _get_csrf_token(client, "/login")
    client.post("/login", data={"username": "user2", "password": "pass2", "csrf_token": csrf_token})
    csrf_token = _get_csrf_token(client, "/players")
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


def test_deleted_user_session_is_invalidated(client, app_module, db):
    user = _create_user(db, app_module.User, "deleted_user", "pass1", role="scout")
    csrf_token = _get_csrf_token(client, "/login")
    client.post(
        "/login",
        data={"username": "deleted_user", "password": "pass1", "csrf_token": csrf_token},
    )

    db.delete(user)
    db.commit()
    response = client.get("/players", follow_redirects=False)

    assert response.status_code == 302
    assert "/login" in response.headers["Location"]
    with client.session_transaction() as sess:
        assert "user_id" not in sess
        assert "role" not in sess


def test_role_change_is_applied_on_next_protected_request(client, app_module, db):
    user = _create_user(db, app_module.User, "changed_role", "pass1", role="administrador")
    csrf_token = _get_csrf_token(client, "/login")
    client.post(
        "/login",
        data={"username": "changed_role", "password": "pass1", "csrf_token": csrf_token},
    )

    user.role = "scout"
    db.commit()
    response = client.get("/register", follow_redirects=False)

    assert response.status_code == 403
    with client.session_transaction() as sess:
        assert sess["role"] == "scout"


def test_invalid_database_role_invalidates_session(client, app_module, db):
    user = _create_user(db, app_module.User, "invalid_role", "pass1", role="scout")
    csrf_token = _get_csrf_token(client, "/login")
    client.post(
        "/login",
        data={"username": "invalid_role", "password": "pass1", "csrf_token": csrf_token},
    )

    user.role = "rol_inexistente"
    db.commit()
    response = client.get("/players", follow_redirects=False)

    assert response.status_code == 302
    assert "/login" in response.headers["Location"]


def test_csrf_uses_constant_time_comparison(app_module, monkeypatch):
    comparisons = []

    def recording_compare_digest(received, expected):
        comparisons.append((received, expected))
        return received == expected

    monkeypatch.setattr(
        app_module.service_require_csrf.__globals__["secrets"],
        "compare_digest",
        recording_compare_digest,
    )
    app_module.service_require_csrf(
        {"csrf_token": "known-token"},
        {},
        {"csrf_token": "known-token"},
    )
    assert comparisons == [("known-token", "known-token")]

    with app_module.app.test_request_context():
        with pytest.raises(BadRequest):
            app_module.service_require_csrf(
                {"csrf_token": 123},
                {},
                {"csrf_token": "known-token"},
            )
