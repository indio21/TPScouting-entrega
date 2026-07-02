"""Rutas de autenticacion y registro de usuarios."""

from __future__ import annotations

from typing import Callable, Optional

from flask import Blueprint, redirect, render_template, request, session, url_for
from werkzeug.security import check_password_hash, generate_password_hash


def create_auth_blueprint(
    *,
    Session,
    User,
    require_csrf: Callable[[], None],
    is_login_rate_limited: Callable[[Optional[str]], bool],
    clear_failed_logins: Callable[[Optional[str]], None],
    register_failed_login: Callable[[Optional[str]], None],
    normalize_role: Callable[[Optional[str]], str],
    roles_required: Callable[..., Callable],
    is_strong_password: Callable[[str], bool],
    role_admin: str,
    role_scout: str,
    role_director: str,
) -> Blueprint:
    bp = Blueprint("auth", __name__)

    @bp.route("/login", methods=["GET", "POST"])
    def login():
        if request.method == "POST":
            require_csrf()

        if request.method == "POST":
            username = request.form.get("username")
            password = request.form.get("password")
            if is_login_rate_limited(username):
                return (
                    render_template(
                        "login.html",
                        error=(
                            "Se bloquearon temporalmente los intentos de acceso. "
                            "Espera unos minutos antes de reintentar."
                        ),
                    ),
                    429,
                )
            db = Session()
            user = db.query(User).filter(User.username == username).first()
            db.close()
            if user and check_password_hash(user.password_hash, password):
                clear_failed_logins(username)
                session["user_id"] = user.id
                session["username"] = user.username
                session["role"] = normalize_role(user.role)
                return redirect(request.args.get("next") or url_for("index"))
            register_failed_login(username)
            return render_template("login.html", error="Usuario o contrasena invalidos")
        return render_template("login.html")

    @bp.route("/logout", methods=["POST"])
    def logout():
        require_csrf()
        session.clear()
        return redirect(url_for("landing"))

    @bp.route("/register", methods=["GET", "POST"])
    @roles_required(role_admin)
    def register():
        if request.method == "POST":
            require_csrf()
        if request.method == "POST":
            username = request.form.get("username")
            password = request.form.get("password")
            role = normalize_role(request.form.get("role"))
            if not username or not password or not role:
                return render_template("register.html", error="Todos los campos son obligatorios")
            if not is_strong_password(password):
                return render_template(
                    "register.html",
                    error="La contrasena debe tener al menos 8 caracteres e incluir letras y numeros",
                )
            if role not in {role_admin, role_scout, role_director}:
                return render_template("register.html", error="Rol invalido")
            db = Session()
            if db.query(User).filter(User.username == username).first():
                db.close()
                return render_template("register.html", error="El usuario ya existe")
            user = User(
                username=username,
                password_hash=generate_password_hash(password),
                role=role,
            )
            db.add(user)
            db.commit()
            db.close()
            return redirect(url_for("index"))
        return render_template("register.html")

    return bp
