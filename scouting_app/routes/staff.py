"""Rutas de cuerpo tecnico y dirigentes."""

from __future__ import annotations

from typing import Callable

from flask import Blueprint, abort, redirect, render_template, request, url_for


def create_staff_blueprint(
    *,
    Session,
    Coach,
    Director,
    require_csrf: Callable[[], None],
    login_required: Callable,
    roles_required: Callable[..., Callable],
    role_admin: str,
) -> Blueprint:
    bp = Blueprint("staff", __name__)

    @bp.route("/coaches")
    @login_required
    def list_coaches():
        db = Session()
        coaches = db.query(Coach).all()
        db.close()
        return render_template("coaches.html", coaches=coaches)

    @bp.route("/coaches/new", methods=["GET", "POST"])
    @roles_required(role_admin)
    def new_coach():
        if request.method == "POST":
            require_csrf()

        if request.method == "POST":
            db = Session()
            coach = Coach(
                name=request.form["name"],
                role=request.form["role"],
                age=request.form.get("age"),
                club=request.form.get("club"),
                country=request.form.get("country"),
            )
            db.add(coach)
            db.commit()
            db.close()
            return redirect(url_for("list_coaches"))
        return render_template("coach_form.html", coach=None)

    @bp.route("/coaches/edit/<int:coach_id>", methods=["GET", "POST"])
    @roles_required(role_admin)
    def edit_coach(coach_id):
        if request.method == "POST":
            require_csrf()

        db = Session()
        coach = db.get(Coach, coach_id)
        if not coach:
            db.close()
            abort(404)
        if request.method == "POST":
            coach.name = request.form["name"]
            coach.role = request.form["role"]
            coach.age = request.form.get("age")
            coach.club = request.form.get("club")
            coach.country = request.form.get("country")
            db.commit()
            db.close()
            return redirect(url_for("list_coaches"))
        db.close()
        return render_template("coach_form.html", coach=coach)

    @bp.route("/coaches/delete/<int:coach_id>", methods=["POST"])
    @roles_required(role_admin)
    def delete_coach(coach_id):
        require_csrf()

        db = Session()
        coach = db.get(Coach, coach_id)
        if not coach:
            db.close()
            abort(404)
        db.delete(coach)
        db.commit()
        db.close()
        return redirect(url_for("list_coaches"))

    @bp.route("/directors")
    @login_required
    def list_directors():
        db = Session()
        directors = db.query(Director).all()
        db.close()
        return render_template("directors.html", directors=directors)

    @bp.route("/directors/new", methods=["GET", "POST"])
    @roles_required(role_admin)
    def new_director():
        if request.method == "POST":
            require_csrf()

        if request.method == "POST":
            db = Session()
            director = Director(
                name=request.form["name"],
                position=request.form["position"],
                age=request.form.get("age"),
                club=request.form.get("club"),
                country=request.form.get("country"),
            )
            db.add(director)
            db.commit()
            db.close()
            return redirect(url_for("list_directors"))
        return render_template("director_form.html", director=None)

    @bp.route("/directors/edit/<int:director_id>", methods=["GET", "POST"])
    @roles_required(role_admin)
    def edit_director(director_id):
        if request.method == "POST":
            require_csrf()

        db = Session()
        director = db.get(Director, director_id)
        if not director:
            db.close()
            abort(404)
        if request.method == "POST":
            director.name = request.form["name"]
            director.position = request.form["position"]
            director.age = request.form.get("age")
            director.club = request.form.get("club")
            director.country = request.form.get("country")
            db.commit()
            db.close()
            return redirect(url_for("list_directors"))
        db.close()
        return render_template("director_form.html", director=director)

    @bp.route("/directors/delete/<int:director_id>", methods=["POST"])
    @roles_required(role_admin)
    def delete_director(director_id):
        require_csrf()

        db = Session()
        director = db.get(Director, director_id)
        if not director:
            db.close()
            abort(404)
        db.delete(director)
        db.commit()
        db.close()
        return redirect(url_for("list_directors"))

    return bp
