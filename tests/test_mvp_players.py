"""Regresiones del MVP: players."""

import io
import json
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


def test_manage_players_creates_valid_player(client, app_module, db):
    _create_user(db, app_module.User, "scout_create", "scout1234", role="scout")
    _login(client, "scout_create", "scout1234")
    csrf_token = _get_csrf_token(client, "/players/manage")

    response = client.post(
        "/players/manage",
        data=_valid_manage_player_payload(csrf_token=csrf_token),
        follow_redirects=False,
    )

    assert response.status_code in (301, 302)
    created = db.query(app_module.Player).filter_by(national_id="40111222").first()
    assert created is not None
    assert created.name == "Nuevo Talento"
    assert created.birth_date == date(2010, 3, 21)
    assert created.category_year == 2010
    assert created.age == app_module.Player.calculate_age_from_birth_date(created.birth_date)

def test_manage_players_rejects_invalid_age(client, app_module, db):
    _create_user(db, app_module.User, "scout_age", "scout1234", role="scout")
    _login(client, "scout_age", "scout1234")
    csrf_token = _get_csrf_token(client, "/players/manage")

    response = client.post(
        "/players/manage",
        data=_valid_manage_player_payload(
            national_id="40111223",
            birth_date="2007-03-21",
            csrf_token=csrf_token,
        ),
        follow_redirects=True,
    )

    assert response.status_code == 200
    assert "edad calculada por fecha de nacimiento" in response.get_data(as_text=True).lower()
    assert db.query(app_module.Player).filter_by(national_id="40111223").count() == 0

def test_manage_players_rejects_unsafe_photo_url(client, app_module, db):
    _create_user(db, app_module.User, "scout_photo_create", "scout1234", role="scout")
    _login(client, "scout_photo_create", "scout1234")
    csrf_token = _get_csrf_token(client, "/players/manage")

    response = client.post(
        "/players/manage",
        data=_valid_manage_player_payload(
            national_id="40111229",
            photo_url="javascript:alert(1)",
            csrf_token=csrf_token,
        ),
        follow_redirects=True,
    )

    assert response.status_code == 200
    assert "La foto debe usar HTTPS" in response.get_data(as_text=True)
    assert db.query(app_module.Player).filter_by(national_id="40111229").count() == 0

def test_manage_players_rejects_missing_required_fields(client, app_module, db):
    _create_user(db, app_module.User, "scout_required", "scout1234", role="scout")
    _login(client, "scout_required", "scout1234")
    csrf_token = _get_csrf_token(client, "/players/manage")

    response = client.post(
        "/players/manage",
        data=_valid_manage_player_payload(
            name="",
            national_id="",
            birth_date="",
            csrf_token=csrf_token,
        ),
        follow_redirects=True,
    )
    body = response.get_data(as_text=True)

    assert response.status_code == 200
    assert "El nombre es obligatorio" in body
    assert "DNI" in body
    assert "fecha de nacimiento" in body.lower()
    assert db.query(app_module.Player).count() == 0

def test_manage_players_rejects_duplicate_national_id(client, app_module, db):
    _create_user(db, app_module.User, "scout_dup", "scout1234", role="scout")
    _create_player(app_module, db, name="Existente", national_id="41122334")
    _login(client, "scout_dup", "scout1234")
    csrf_token = _get_csrf_token(client, "/players/manage")

    response = client.post(
        "/players/manage",
        data={
            "mode": "single",
            "name": "Duplicado",
            "national_id": "41122334",
            "birth_date": "2010-03-21",
            "position": "Defensa",
            "club": "Club Dup",
            "country": "Argentina",
            "pace": "10",
            "shooting": "9",
            "passing": "10",
            "dribbling": "9",
            "defending": "12",
            "physical": "11",
            "vision": "10",
            "tackling": "12",
            "determination": "13",
            "technique": "9",
            "csrf_token": csrf_token,
        },
        follow_redirects=True,
    )

    assert response.status_code == 200
    assert "ya esta registrado" in response.get_data(as_text=True)
    assert db.query(app_module.Player).filter_by(name="Duplicado").count() == 0

def test_manage_players_rejects_attribute_below_one(client, app_module, db):
    _create_user(db, app_module.User, "scout_attr_low", "scout1234", role="scout")
    _login(client, "scout_attr_low", "scout1234")
    csrf_token = _get_csrf_token(client, "/players/manage")

    response = client.post(
        "/players/manage",
        data=_valid_manage_player_payload(
            national_id="40111224",
            pace="0",
            csrf_token=csrf_token,
        ),
        follow_redirects=True,
    )

    assert response.status_code == 200
    assert "Ritmo debe ubicarse entre 1 y 20" in response.get_data(as_text=True)
    assert db.query(app_module.Player).filter_by(national_id="40111224").count() == 0

def test_import_players_preview_and_confirm_csv(client, app_module, db):
    _create_user(db, app_module.User, "scout_import", "scout1234", role="scout")
    _login(client, "scout_import", "scout1234")
    csrf_token = _get_csrf_token(client, "/players/import")
    csv_text = (
        "Nombre,DNI_ID,FechaNacimiento,Posicion,Club,Pais,Ritmo,Disparo,Pase,Regate,"
        "Defensa,Fisico,Vision,Marcaje,Determinacion,Tecnica,AltoPotencial,FotoURL\n"
        "CSV Juvenil,50111222,2010-03-21,Delantero,Club CSV,Argentina,15,14,13,16,"
        "10,14,15,12,16,17,1,\n"
    )

    preview_response = client.post(
        "/players/import",
        data={
            "csrf_token": csrf_token,
            "mode": "preview",
            "csv_file": (io.BytesIO(csv_text.encode("utf-8")), "jugadores.csv"),
        },
        content_type="multipart/form-data",
    )
    preview_body = preview_response.get_data(as_text=True)

    assert preview_response.status_code == 200
    assert "CSV Juvenil" in preview_body
    assert "Cat. 2010" in preview_body
    assert "Importar 1 filas validas" in preview_body

    csrf_token = _get_csrf_token(client, "/players/import")
    payload = [
        {
            "Nombre": "CSV Juvenil",
            "DNI_ID": "50111222",
            "FechaNacimiento": "2010-03-21",
            "Posicion": "Delantero",
            "Club": "Club CSV",
            "Pais": "Argentina",
            "Ritmo": "15",
            "Disparo": "14",
            "Pase": "13",
            "Regate": "16",
            "Defensa": "10",
            "Fisico": "14",
            "Vision": "15",
            "Marcaje": "12",
            "Determinacion": "16",
            "Tecnica": "17",
            "AltoPotencial": "1",
            "FotoURL": "",
        }
    ]
    confirm_response = client.post(
        "/players/import",
        data={
            "csrf_token": csrf_token,
            "mode": "confirm",
            "import_payload": json.dumps(payload),
        },
        follow_redirects=False,
    )

    assert confirm_response.status_code in (301, 302)
    created = db.query(app_module.Player).filter_by(national_id="50111222").first()
    assert created is not None
    assert created.name == "CSV Juvenil"
    assert created.birth_date == date(2010, 3, 21)
    assert created.category_year == 2010
    assert created.age == app_module.Player.calculate_age_from_birth_date(created.birth_date)

def test_import_players_rejects_unsafe_photo_url(client, app_module, db):
    _create_user(db, app_module.User, "scout_import_photo", "scout1234", role="scout")
    _login(client, "scout_import_photo", "scout1234")
    csrf_token = _get_csrf_token(client, "/players/import")
    csv_text = (
        "Nombre,DNI_ID,FechaNacimiento,Posicion,Club,Pais,Ritmo,Disparo,Pase,Regate,"
        "Defensa,Fisico,Vision,Marcaje,Determinacion,Tecnica,AltoPotencial,FotoURL\n"
        "CSV Foto,50111229,2010-03-21,Delantero,Club CSV,Argentina,15,14,13,16,"
        "10,14,15,12,16,17,1,javascript:alert(1)\n"
    )

    response = client.post(
        "/players/import",
        data={
            "csrf_token": csrf_token,
            "mode": "preview",
            "csv_file": (io.BytesIO(csv_text.encode("utf-8")), "jugadores.csv"),
        },
        content_type="multipart/form-data",
    )

    assert response.status_code == 200
    assert "FotoURL debe ser una URL HTTPS" in response.get_data(as_text=True)
    assert db.query(app_module.Player).filter_by(national_id="50111229").count() == 0

def test_edit_player_updates_core_fields(client, app_module, db):
    _create_user(db, app_module.User, "scout_edit", "scout1234", role="scout")
    player = _create_player(app_module, db, name="Editar Base", national_id="42233445")
    _login(client, "scout_edit", "scout1234")
    csrf_token = _get_csrf_token(client, f"/edit_player/{player.id}")

    response = client.post(
        f"/edit_player/{player.id}",
        data={
            "name": "Editar Final",
            "national_id": "42233446",
            "birth_date": "2009-03-21",
            "position": "Mediocampista",
            "club": "Club Editado",
            "country": "Uruguay",
            "photo_url": "",
            "pace": "13",
            "shooting": "10",
            "passing": "15",
            "dribbling": "14",
            "defending": "11",
            "physical": "12",
            "vision": "15",
            "tackling": "9",
            "determination": "16",
            "technique": "14",
            "potential_label": "1",
            "csrf_token": csrf_token,
        },
        follow_redirects=False,
    )

    assert response.status_code in (301, 302)
    db.refresh(player)
    assert player.name == "Editar Final"
    assert player.national_id == "42233446"
    assert player.birth_date == date(2009, 3, 21)
    assert player.age == app_module.Player.calculate_age_from_birth_date(player.birth_date)
    assert player.position == "Mediocampista"

def test_edit_player_rejects_unsafe_photo_url_without_mutating_player(client, app_module, db):
    _create_user(db, app_module.User, "scout_photo_edit", "scout1234", role="scout")
    player = _create_player(
        app_module,
        db,
        name="Foto Original",
        national_id="42233449",
        photo_url="/static/img/player-silhouette.svg",
    )
    _login(client, "scout_photo_edit", "scout1234")
    csrf_token = _get_csrf_token(client, f"/edit_player/{player.id}")
    payload = _valid_manage_player_payload(
        name="Foto Alterada",
        national_id="42233449",
        photo_url="data:image/svg+xml,<svg></svg>",
        csrf_token=csrf_token,
    )

    response = client.post(f"/edit_player/{player.id}", data=payload, follow_redirects=True)

    assert response.status_code == 200
    assert "La foto debe usar HTTPS" in response.get_data(as_text=True)
    db.refresh(player)
    assert player.name == "Foto Original"
    assert player.photo_url == "/static/img/player-silhouette.svg"

def test_delete_player_removes_record(client, app_module, db):
    _create_user(db, app_module.User, "scout_delete", "scout1234", role="scout")
    player = _create_player(app_module, db, name="Eliminar Base", national_id="43344556")
    _login(client, "scout_delete", "scout1234")
    csrf_token = _get_csrf_token(client, f"/player/{player.id}")

    response = client.post(
        f"/delete_player/{player.id}",
        data={"csrf_token": csrf_token},
        follow_redirects=False,
    )

    assert response.status_code in (301, 302)
    assert db.query(app_module.Player).filter_by(id=player.id).count() == 0

def test_predict_player_without_loaded_model_returns_controlled_error(client, app_module, db):
    _create_user(db, app_module.User, "scout_no_model", "scout1234", role="scout")
    player = _create_player(app_module, db, name="Sin Modelo", national_id="44455666")
    app_module.model = None
    _login(client, "scout_no_model", "scout1234")

    response = client.get(f"/player/{player.id}/predict")

    assert response.status_code == 500
    assert "No hay calculo disponible todavia" in response.get_data(as_text=True)

def test_predict_player_renders_current_age_and_combined_adjustment(client, app_module, db):
    _create_user(db, app_module.User, "scout_prediction_view", "scout1234", role="scout")
    player = _create_player(app_module, db, name="Edad Visible", national_id="44455668")
    _login(client, "scout_prediction_view", "scout1234")

    response = client.get(f"/player/{player.id}/predict")

    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert f"{player.current_age} anos" in html
    assert "Ajuste combinado" in html
    assert "Impacto de historial y ajuste posicional." in html

def test_player_stats_reject_invalid_percentages(client, app_module, db):
    _create_user(db, app_module.User, "scout_stats", "scout1234", role="scout")
    player = _create_player(app_module, db, name="Stats Base", national_id="44455667")
    _login(client, "scout_stats", "scout1234")
    csrf_token = _get_csrf_token(client, f"/player/{player.id}/stats")

    response = client.post(
        f"/player/{player.id}/stats",
        data={
            "csrf_token": csrf_token,
            "action": "add",
            "record_date": "2026-04-15",
            "matches_played": "1",
            "minutes_played": "90",
            "pass_accuracy": "150",
        },
        follow_redirects=True,
    )

    assert response.status_code == 200
    assert "Precision de pase debe estar entre 0 y 100" in response.get_data(as_text=True)
    assert db.query(app_module.PlayerStat).filter_by(player_id=player.id).count() == 0

def test_player_attributes_reject_values_out_of_range(client, app_module, db):
    _create_user(db, app_module.User, "scout_attrs", "scout1234", role="scout")
    player = _create_player(app_module, db, name="Attrs Base", national_id="45566778")
    _login(client, "scout_attrs", "scout1234")
    csrf_token = _get_csrf_token(client, f"/player/{player.id}/attributes")

    response = client.post(
        f"/player/{player.id}/attributes",
        data={
            "csrf_token": csrf_token,
            "action": "add",
            "record_date": "2026-04-15",
            "pace": "25",
        },
        follow_redirects=True,
    )

    assert response.status_code == 200
    assert "Ritmo debe estar entre 1 y 20" in response.get_data(as_text=True)
    assert db.query(app_module.PlayerAttributeHistory).filter_by(player_id=player.id).count() == 0

def test_edit_player_stat_updates_record(client, app_module, db):
    _create_user(db, app_module.User, "scout_stat_edit", "scout1234", role="scout")
    player = _create_player(app_module, db, name="Stats Edit", national_id="45566779")
    stat = app_module.PlayerStat(
        player_id=player.id,
        record_date=date(2026, 4, 1),
        matches_played=1,
        goals=0,
        assists=0,
        minutes_played=70,
        yellow_cards=0,
        red_cards=0,
        pass_accuracy=70,
        shot_accuracy=30,
        duels_won_pct=50,
        final_score=6.0,
        notes="Inicial",
    )
    db.add(stat)
    db.commit()
    _login(client, "scout_stat_edit", "scout1234")
    csrf_token = _get_csrf_token(client, f"/player/{player.id}/stats")

    response = client.post(
        f"/player/{player.id}/stats/{stat.id}/edit",
        data={
            "csrf_token": csrf_token,
            "record_date": "2026-04-20",
            "matches_played": "2",
            "goals": "1",
            "assists": "1",
            "minutes_played": "160",
            "yellow_cards": "1",
            "red_cards": "0",
            "pass_accuracy": "82.5",
            "shot_accuracy": "45",
            "duels_won_pct": "61",
            "final_score": "8.1",
            "notes": "Editado",
        },
        follow_redirects=False,
    )

    assert response.status_code in (301, 302)
    db.refresh(stat)
    assert stat.record_date == date(2026, 4, 20)
    assert stat.matches_played == 2
    assert stat.goals == 1
    assert stat.final_score == 8.1
    assert stat.notes == "Editado"

def test_delete_player_stat_removes_record(client, app_module, db):
    _create_user(db, app_module.User, "scout_stat_delete", "scout1234", role="scout")
    player = _create_player(app_module, db, name="Stats Delete", national_id="45566780")
    stat = app_module.PlayerStat(
        player_id=player.id,
        record_date=date(2026, 4, 1),
        matches_played=1,
        goals=0,
        assists=0,
        minutes_played=70,
        yellow_cards=0,
        red_cards=0,
    )
    db.add(stat)
    db.commit()
    _login(client, "scout_stat_delete", "scout1234")
    csrf_token = _get_csrf_token(client, f"/player/{player.id}/stats")

    response = client.post(
        f"/player/{player.id}/stats/{stat.id}/delete",
        data={"csrf_token": csrf_token},
        follow_redirects=False,
    )

    assert response.status_code in (301, 302)
    assert db.query(app_module.PlayerStat).filter_by(id=stat.id).count() == 0

def test_edit_player_attribute_history_updates_record_and_player(client, app_module, db):
    _create_user(db, app_module.User, "scout_attr_edit", "scout1234", role="scout")
    player = _create_player(app_module, db, name="Attr Edit", national_id="45566781", pace=10)
    entry = app_module.PlayerAttributeHistory(
        player_id=player.id,
        record_date=date(2026, 4, 1),
        pace=10,
        notes="Inicial",
    )
    db.add(entry)
    db.commit()
    _login(client, "scout_attr_edit", "scout1234")
    csrf_token = _get_csrf_token(client, f"/player/{player.id}/attributes")

    response = client.post(
        f"/player/{player.id}/attributes/{entry.id}/edit",
        data={
            "csrf_token": csrf_token,
            "record_date": "2026-04-21",
            "pace": "16",
            "shooting": "",
            "passing": "",
            "dribbling": "",
            "defending": "",
            "physical": "",
            "vision": "",
            "tackling": "",
            "determination": "",
            "technique": "",
            "notes": "Mejoro ritmo",
        },
        follow_redirects=False,
    )

    assert response.status_code in (301, 302)
    db.refresh(entry)
    db.refresh(player)
    assert entry.record_date == date(2026, 4, 21)
    assert entry.pace == 16
    assert entry.notes == "Mejoro ritmo"
    assert player.pace == 16

def test_delete_player_attribute_history_resyncs_player(client, app_module, db):
    _create_user(db, app_module.User, "scout_attr_delete", "scout1234", role="scout")
    player = _create_player(app_module, db, name="Attr Delete", national_id="45566782", pace=18)
    older = app_module.PlayerAttributeHistory(
        player_id=player.id,
        record_date=date(2026, 3, 1),
        pace=11,
    )
    latest = app_module.PlayerAttributeHistory(
        player_id=player.id,
        record_date=date(2026, 4, 1),
        pace=18,
    )
    db.add_all([older, latest])
    db.commit()
    _login(client, "scout_attr_delete", "scout1234")
    csrf_token = _get_csrf_token(client, f"/player/{player.id}/attributes")

    response = client.post(
        f"/player/{player.id}/attributes/{latest.id}/delete",
        data={"csrf_token": csrf_token},
        follow_redirects=False,
    )

    assert response.status_code in (301, 302)
    assert db.query(app_module.PlayerAttributeHistory).filter_by(id=latest.id).count() == 0
    db.refresh(player)
    assert player.pace == 11

def test_match_history_modal_routes_add_edit_delete(client, app_module, db):
    _create_user(db, app_module.User, "scout_match_history", "scout1234", role="scout")
    player = _create_player(app_module, db, name="Match History", national_id="45566784")
    _login(client, "scout_match_history", "scout1234")
    csrf_token = _get_csrf_token(client, f"/player/{player.id}")

    response = client.post(
        f"/player/{player.id}/matches/add",
        data=_match_history_payload(csrf_token=csrf_token),
        follow_redirects=False,
    )

    assert response.status_code in (301, 302)
    db.expire_all()
    participation = db.query(app_module.PlayerMatchParticipation).filter_by(player_id=player.id).one()
    match_id = participation.match_id
    assert participation.minutes_played == 80
    assert participation.match.opponent_name == "Rival Base"

    response = client.post(
        f"/player/{player.id}/matches/{participation.id}/edit",
        data=_match_history_payload(
            csrf_token=csrf_token,
            opponent_name="Rival Editado",
            minutes_played="90",
            final_score="8.0",
        ),
        follow_redirects=False,
    )

    assert response.status_code in (301, 302)
    db.expire_all()
    participation = db.get(app_module.PlayerMatchParticipation, participation.id)
    assert participation.minutes_played == 90
    assert participation.final_score == 8.0
    assert participation.match.opponent_name == "Rival Editado"

    participation_id = participation.id
    response = client.post(
        f"/player/{player.id}/matches/{participation_id}/delete",
        data={"csrf_token": csrf_token},
        follow_redirects=False,
    )

    assert response.status_code in (301, 302)
    db.expire_all()
    assert db.query(app_module.PlayerMatchParticipation).filter_by(id=participation_id).count() == 0
    assert db.query(app_module.Match).filter_by(id=match_id).count() == 0

def test_physical_history_modal_routes_add_edit_delete(client, app_module, db):
    _create_user(db, app_module.User, "scout_physical_history", "scout1234", role="scout")
    player = _create_player(app_module, db, name="Physical History", national_id="45566785")
    _login(client, "scout_physical_history", "scout1234")
    csrf_token = _get_csrf_token(client, f"/player/{player.id}")

    response = client.post(
        f"/player/{player.id}/physical/add",
        data=_physical_assessment_payload(csrf_token=csrf_token),
        follow_redirects=False,
    )

    assert response.status_code in (301, 302)
    db.expire_all()
    assessment = db.query(app_module.PhysicalAssessment).filter_by(player_id=player.id).one()
    assert assessment.height_cm == 178
    assert assessment.in_growth_spurt is True

    response = client.post(
        f"/player/{player.id}/physical/{assessment.id}/edit",
        data=_physical_assessment_payload(
            csrf_token=csrf_token,
            height_cm="180",
            in_growth_spurt="",
            notes="Ajuste fisico",
        ),
        follow_redirects=False,
    )

    assert response.status_code in (301, 302)
    db.expire_all()
    assessment = db.get(app_module.PhysicalAssessment, assessment.id)
    assert assessment.height_cm == 180
    assert assessment.in_growth_spurt is False
    assert assessment.notes == "Ajuste fisico"

    assessment_id = assessment.id
    response = client.post(
        f"/player/{player.id}/physical/{assessment_id}/delete",
        data={"csrf_token": csrf_token},
        follow_redirects=False,
    )

    assert response.status_code in (301, 302)
    db.expire_all()
    assert db.query(app_module.PhysicalAssessment).filter_by(id=assessment_id).count() == 0

def test_availability_history_modal_routes_add_edit_delete(client, app_module, db):
    _create_user(db, app_module.User, "scout_availability_history", "scout1234", role="scout")
    player = _create_player(app_module, db, name="Availability History", national_id="45566786")
    _login(client, "scout_availability_history", "scout1234")
    csrf_token = _get_csrf_token(client, f"/player/{player.id}")

    response = client.post(
        f"/player/{player.id}/availability/add",
        data=_availability_payload(csrf_token=csrf_token),
        follow_redirects=False,
    )

    assert response.status_code in (301, 302)
    db.expire_all()
    availability = db.query(app_module.PlayerAvailability).filter_by(player_id=player.id).one()
    assert availability.availability_pct == 92
    assert availability.injury_flag is False

    response = client.post(
        f"/player/{player.id}/availability/{availability.id}/edit",
        data=_availability_payload(
            csrf_token=csrf_token,
            availability_pct="75",
            injury_flag="1",
            missed_days="2",
        ),
        follow_redirects=False,
    )

    assert response.status_code in (301, 302)
    db.expire_all()
    availability = db.get(app_module.PlayerAvailability, availability.id)
    assert availability.availability_pct == 75
    assert availability.injury_flag is True
    assert availability.missed_days == 2

    availability_id = availability.id
    response = client.post(
        f"/player/{player.id}/availability/{availability_id}/delete",
        data={"csrf_token": csrf_token},
        follow_redirects=False,
    )

    assert response.status_code in (301, 302)
    db.expire_all()
    assert db.query(app_module.PlayerAvailability).filter_by(id=availability_id).count() == 0

def test_scout_report_modal_routes_add_edit_delete(client, app_module, db):
    _create_user(db, app_module.User, "scout_report_history", "scout1234", role="scout")
    player = _create_player(app_module, db, name="Scout Report History", national_id="45566787")
    _login(client, "scout_report_history", "scout1234")
    csrf_token = _get_csrf_token(client, f"/player/{player.id}")

    response = client.post(
        f"/player/{player.id}/reports/add",
        data=_scout_report_payload(csrf_token=csrf_token),
        follow_redirects=False,
    )

    assert response.status_code in (301, 302)
    db.expire_all()
    report = db.query(app_module.ScoutReport).filter_by(player_id=player.id).one()
    assert report.decision_making == 15
    assert report.observed_projection_score == 8.2

    response = client.post(
        f"/player/{player.id}/reports/{report.id}/edit",
        data=_scout_report_payload(
            csrf_token=csrf_token,
            decision_making="17",
            observed_projection_score="8.8",
            notes="Mejor lectura",
        ),
        follow_redirects=False,
    )

    assert response.status_code in (301, 302)
    db.expire_all()
    report = db.get(app_module.ScoutReport, report.id)
    assert report.decision_making == 17
    assert report.observed_projection_score == 8.8
    assert report.notes == "Mejor lectura"

    report_id = report.id
    response = client.post(
        f"/player/{player.id}/reports/{report_id}/delete",
        data={"csrf_token": csrf_token},
        follow_redirects=False,
    )

    assert response.status_code in (301, 302)
    db.expire_all()
    assert db.query(app_module.ScoutReport).filter_by(id=report_id).count() == 0

def test_player_detail_renders_complementary_history_modals(client, app_module, db):
    _create_user(db, app_module.User, "scout_detail_modals", "scout1234", role="scout")
    player = _create_player(app_module, db, name="Detail Modals", national_id="45566788")
    match = app_module.Match(
        match_date=date(2026, 4, 20),
        opponent_name="Rival Modales",
        opponent_level=3,
        venue="Local",
    )
    participation = app_module.PlayerMatchParticipation(
        player_id=player.id,
        match=match,
        position_played="Defensa",
        minutes_played=75,
    )
    physical = app_module.PhysicalAssessment(
        player_id=player.id,
        assessment_date=date(2026, 4, 21),
        height_cm=179,
    )
    availability = app_module.PlayerAvailability(
        player_id=player.id,
        record_date=date(2026, 4, 22),
        availability_pct=91,
        missed_days=0,
    )
    report = app_module.ScoutReport(
        player_id=player.id,
        report_date=date(2026, 4, 23),
        decision_making=15,
    )
    db.add_all([participation, physical, availability, report])
    db.commit()

    _login(client, "scout_detail_modals", "scout1234")
    response = client.get(f"/player/{player.id}")
    body = response.get_data(as_text=True)

    assert response.status_code == 200
    assert f"editMatchModal{participation.id}" in body
    assert f"editPhysicalModal{physical.id}" in body
    assert f"editAvailabilityModal{availability.id}" in body
    assert f"editScoutReportModal{report.id}" in body

def test_director_cannot_modify_history_records(client, app_module, db):
    director = _create_user(db, app_module.User, "director_history", "director123", role="director")
    player = _create_player(app_module, db, name="Director History", national_id="45566783")
    stat = app_module.PlayerStat(
        player_id=player.id,
        record_date=date(2026, 4, 1),
        matches_played=1,
        goals=0,
        assists=0,
        minutes_played=70,
        yellow_cards=0,
        red_cards=0,
    )
    entry = app_module.PlayerAttributeHistory(
        player_id=player.id,
        record_date=date(2026, 4, 1),
        pace=12,
    )
    physical = app_module.PhysicalAssessment(
        player_id=player.id,
        assessment_date=date(2026, 4, 1),
        height_cm=177,
    )
    db.add_all([stat, entry, physical])
    db.commit()
    _login(client, director.username, "director123")
    csrf_token = _get_csrf_token(client, f"/player/{player.id}/stats")

    response = client.post(
        f"/player/{player.id}/stats/{stat.id}/delete",
        data={"csrf_token": csrf_token},
        follow_redirects=False,
    )
    attr_response = client.post(
        f"/player/{player.id}/attributes/{entry.id}/delete",
        data={"csrf_token": csrf_token},
        follow_redirects=False,
    )
    physical_response = client.post(
        f"/player/{player.id}/physical/{physical.id}/delete",
        data={"csrf_token": csrf_token},
        follow_redirects=False,
    )

    assert response.status_code == 403
    assert attr_response.status_code == 403
    assert physical_response.status_code == 403
    assert db.query(app_module.PlayerStat).filter_by(id=stat.id).count() == 1
    assert db.query(app_module.PlayerAttributeHistory).filter_by(id=entry.id).count() == 1
    assert db.query(app_module.PhysicalAssessment).filter_by(id=physical.id).count() == 1

def test_admin_can_create_and_delete_coach(client, app_module, db):
    _create_user(db, app_module.User, "admin_staff", "admin1234", role="administrador")
    _login(client, "admin_staff", "admin1234")

    csrf_token = _get_csrf_token(client, "/coaches/new")
    create_response = client.post(
        "/coaches/new",
        data={
            "csrf_token": csrf_token,
            "name": "Profe Uno",
            "role": "Entrenador",
            "age": "40",
            "club": "Club Staff",
            "country": "Argentina",
        },
        follow_redirects=False,
    )

    assert create_response.status_code in (301, 302)
    coach = db.query(app_module.Coach).filter_by(name="Profe Uno").first()
    assert coach is not None

    csrf_token = _get_csrf_token(client, "/coaches")
    delete_response = client.post(
        f"/coaches/delete/{coach.id}",
        data={"csrf_token": csrf_token},
        follow_redirects=False,
    )

    assert delete_response.status_code in (301, 302)
    assert db.query(app_module.Coach).filter_by(id=coach.id).count() == 0
