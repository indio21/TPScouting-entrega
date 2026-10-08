from datetime import date

from werkzeug.security import generate_password_hash


def _create_user(db, app_module, username, role):
    user = app_module.User(
        username=username,
        password_hash=generate_password_hash("Clave1234"),
        role=role,
    )
    db.add(user)
    db.commit()


def _login(client, username):
    client.get("/login")
    with client.session_transaction() as session:
        token = session["csrf_token"]
    return client.post(
        "/login",
        data={"username": username, "password": "Clave1234", "csrf_token": token},
    )


def _csrf(client, path):
    client.get(path)
    with client.session_transaction() as session:
        return session["csrf_token"]


def _player(app_module, name, national_id, **attributes):
    values = {field: 10 for field in app_module.ATTRIBUTE_FIELDS}
    values.update(attributes)
    return app_module.Player(
        name=name,
        national_id=national_id,
        age=16,
        birth_date=date(2010, 3, 21),
        position="Delantero",
        club="Club Prueba",
        country="Argentina",
        photo_url="",
        potential_label=False,
        **values,
    )


def test_compare_two_players_builds_a_real_conclusion(client, app_module, db):
    _create_user(db, app_module, "scout_compare", "scout")
    first = _player(app_module, "Jugador Superior", "41000001", shooting=19, technique=19)
    second = _player(app_module, "Jugador Base", "41000002", shooting=4, technique=4)
    db.add_all([first, second])
    db.commit()
    _login(client, "scout_compare")

    response = client.post(
        "/compare",
        data={"player_one": first.id, "player_two": second.id, "target_position": "Delantero"},
    )
    body = response.get_data(as_text=True)

    assert response.status_code == 200
    assert "Jugador Superior" in body
    assert "Jugador Base" in body
    assert "mejores indicadores generales" in body


def test_compare_rejects_invalid_multi_ids_without_server_error(client, app_module, db):
    _create_user(db, app_module, "scout_multi", "scout")
    _login(client, "scout_multi")

    response = client.post("/compare/multi", data={"players": ["invalido"]})

    assert response.status_code == 200


def test_admin_director_crud_and_missing_record(client, app_module, db):
    _create_user(db, app_module, "admin_staff", "administrador")
    _login(client, "admin_staff")
    token = _csrf(client, "/directors/new")

    created = client.post(
        "/directors/new",
        data={
            "name": "Directora Prueba",
            "position": "Directora deportiva",
            "age": "45",
            "club": "Club Prueba",
            "country": "Argentina",
            "csrf_token": token,
        },
    )
    assert created.status_code == 302
    director = db.query(app_module.Director).filter_by(name="Directora Prueba").one()
    director_id = director.id

    token = _csrf(client, f"/directors/edit/{director_id}")
    edited = client.post(
        f"/directors/edit/{director_id}",
        data={
            "name": "Directora Editada",
            "position": "Presidenta",
            "age": "46",
            "club": "Club Prueba",
            "country": "Argentina",
            "csrf_token": token,
        },
    )
    assert edited.status_code == 302
    db.expire_all()
    assert db.get(app_module.Director, director_id).name == "Directora Editada"

    token = _csrf(client, "/directors")
    deleted = client.post(f"/directors/delete/{director_id}", data={"csrf_token": token})
    assert deleted.status_code == 302
    db.expire_all()
    assert db.get(app_module.Director, director_id) is None
    assert client.get("/directors/edit/999999").status_code == 404


def test_scout_cannot_create_staff_record(client, app_module, db):
    _create_user(db, app_module, "scout_staff", "scout")
    _login(client, "scout_staff")
    token = _csrf(client, "/coaches")

    response = client.post(
        "/coaches/new",
        data={"name": "No Crear", "role": "DT", "csrf_token": token},
    )

    assert response.status_code == 403
    assert db.query(app_module.Coach).filter_by(name="No Crear").count() == 0
