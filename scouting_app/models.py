"""Modelos de SQLAlchemy para la aplicación de scouting.

Definimos tablas para almacenar los datos de los jugadores, sus habilidades,
estadísticas y predicciones. La estructura es sencilla para permitir una
creación rápida de prototipos; en un entorno real se recomienda
normalizar aún más los datos y añadir restricciones de integridad.
"""

from datetime import date, datetime

from sqlalchemy import Column, Integer, String, Float, Boolean, Date, DateTime, ForeignKey, Text
from sqlalchemy.orm import declarative_base, relationship

# Declarative base de SQLAlchemy
Base = declarative_base()


class TimestampMixin:
    created_at = Column(DateTime, nullable=True, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=True, default=datetime.utcnow, onupdate=datetime.utcnow)


def calculate_age_from_birth_date(value, today=None):
    if value is None:
        return None
    today = today or date.today()
    birthday_passed = (today.month, today.day) >= (value.month, value.day)
    return today.year - value.year - (0 if birthday_passed else 1)


class Player(TimestampMixin, Base):
    """Tabla que representa a un jugador.

    Incluye datos personales básicos y una serie de atributos técnicos
    y mentales valorados en una escala de 1 a 20.
    """

    __tablename__ = "players"

    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    national_id = Column(String, unique=True, nullable=True)
    age = Column(Integer, nullable=False)
    birth_date = Column(Date, nullable=True)
    position = Column(String, nullable=False)
    club = Column(String, nullable=True)
    country = Column(String, nullable=True)
    photo_url = Column(String, nullable=True)

    # Atributos técnicos y físicos (valorados 0–20)
    pace = Column(Integer, nullable=False)
    shooting = Column(Integer, nullable=False)
    passing = Column(Integer, nullable=False)
    dribbling = Column(Integer, nullable=False)
    defending = Column(Integer, nullable=False)
    physical = Column(Integer, nullable=False)
    vision = Column(Integer, nullable=False)
    tackling = Column(Integer, nullable=False)
    determination = Column(Integer, nullable=False)
    technique = Column(Integer, nullable=False)

    # Potencial real/histórico (etiqueta).  En datos generados es aleatoria.
    potential_label = Column(Boolean, nullable=False)

    calculate_age_from_birth_date = staticmethod(calculate_age_from_birth_date)

    @property
    def category_year(self):
        return self.birth_date.year if self.birth_date else None

    @property
    def current_age(self):
        calculated = calculate_age_from_birth_date(self.birth_date)
        return calculated if calculated is not None else self.age

    def sync_age_from_birth_date(self):
        calculated = calculate_age_from_birth_date(self.birth_date)
        if calculated is not None:
            self.age = calculated
        return self.age

    stats = relationship("PlayerStat", back_populates="player", cascade="all, delete-orphan")
    attribute_history = relationship("PlayerAttributeHistory", back_populates="player", cascade="all, delete-orphan")
    match_participations = relationship(
        "PlayerMatchParticipation",
        back_populates="player",
        cascade="all, delete-orphan",
    )
    scout_reports = relationship("ScoutReport", back_populates="player", cascade="all, delete-orphan")
    physical_assessments = relationship(
        "PhysicalAssessment",
        back_populates="player",
        cascade="all, delete-orphan",
    )
    availability_records = relationship(
        "PlayerAvailability",
        back_populates="player",
        cascade="all, delete-orphan",
    )

    def to_dict(self) -> dict:
        """Convierte el modelo a un diccionario útil para JSON o plantillas."""
        return {
            "id": self.id,
            "name": self.name,
            "national_id": self.national_id,
            "age": self.current_age,
            "birth_date": self.birth_date.isoformat() if self.birth_date else None,
            "category_year": self.category_year,
            "position": self.position,
            "club": self.club,
            "country": self.country,
            "photo_url": self.photo_url,
            "pace": self.pace,
            "shooting": self.shooting,
            "passing": self.passing,
            "dribbling": self.dribbling,
            "defending": self.defending,
            "physical": self.physical,
            "vision": self.vision,
            "tackling": self.tackling,
            "determination": self.determination,
            "technique": self.technique,
            "potential_label": self.potential_label,
        }

# ---------------------------------------------------------------------------
# Nuevos modelos para soporte ABM (Alta, Baja, Modificación) de cuerpos
# técnicos y dirigentes.  En un sistema real estos modelos podrían tener más
# atributos (correo electrónico, teléfono, etc.), pero para el manual de
# usuario definimos campos esenciales.

class Coach(TimestampMixin, Base):
    """Representa a un miembro del cuerpo técnico.

    Incluye información básica como nombre, rol (entrenador de porteros,
    preparador físico, etc.), edad, club y país.  El ID actúa como
    clave primaria.
    """

    __tablename__ = "coaches"
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    role = Column(String, nullable=False)  # rol dentro del cuerpo técnico
    age = Column(Integer, nullable=True)
    club = Column(String, nullable=True)
    country = Column(String, nullable=True)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "role": self.role,
            "age": self.age,
            "club": self.club,
            "country": self.country,
        }


class Director(TimestampMixin, Base):
    """Representa a un dirigente o miembro de la directiva del club.

    A diferencia de los entrenadores, aquí usamos un campo cargo (por
    ejemplo presidente, vicepresidente, director deportivo).  También
    incluimos edad, club y país.
    """

    __tablename__ = "directors"
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    position = Column(String, nullable=False)  # cargo en la directiva
    age = Column(Integer, nullable=True)
    club = Column(String, nullable=True)
    country = Column(String, nullable=True)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "position": self.position,
            "age": self.age,
            "club": self.club,
            "country": self.country,
        }


# ---------------------------------------------------------------------------
# Modelo de usuario para autenticación

class User(TimestampMixin, Base):
    """Representa un usuario del sistema.

    Contiene un nombre de usuario único y un hash de contraseña.  En esta
    aplicación se utiliza para manejar la autenticación básica.  Para
    implementaciones más robustas se recomienda un sistema como Flask‑Login y
    almacenaje de más metadatos (roles, fechas de creación, etc.).
    """

    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    username = Column(String, unique=True, nullable=False)
    password_hash = Column(String, nullable=False)
    role = Column(String, nullable=False, default="scout")  # roles: administrador, scout/ojeador, director

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "username": self.username,
        }


class PlayerStat(TimestampMixin, Base):
    """Evolucion historica de rendimiento y observaciones."""

    __tablename__ = "player_stats"

    id = Column(Integer, primary_key=True)
    player_id = Column(Integer, ForeignKey("players.id"), nullable=False)
    record_date = Column(Date, nullable=False)
    matches_played = Column(Integer, nullable=False, default=0)
    goals = Column(Integer, nullable=False, default=0)
    assists = Column(Integer, nullable=False, default=0)
    minutes_played = Column(Integer, nullable=False, default=0)
    yellow_cards = Column(Integer, nullable=False, default=0)
    red_cards = Column(Integer, nullable=False, default=0)
    pass_accuracy = Column(Float, nullable=True)  # porcentaje (0-100)
    shot_accuracy = Column(Float, nullable=True)  # porcentaje (0-100)
    duels_won_pct = Column(Float, nullable=True)  # porcentaje (0-100)
    final_score = Column(Float, nullable=True)    # valoración 1-10
    notes = Column(Text, nullable=True)

    player = relationship("Player", back_populates="stats")

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "player_id": self.player_id,
            "record_date": self.record_date.isoformat(),
            "matches_played": self.matches_played,
            "goals": self.goals,
            "assists": self.assists,
            "minutes_played": self.minutes_played,
            "yellow_cards": self.yellow_cards,
            "red_cards": self.red_cards,
            "pass_accuracy": self.pass_accuracy,
            "shot_accuracy": self.shot_accuracy,
            "duels_won_pct": self.duels_won_pct,
            "final_score": self.final_score,
            "notes": self.notes,
        }


class PlayerAttributeHistory(TimestampMixin, Base):
    """Historial de atributos técnicos/mentales para seguir progresión."""

    __tablename__ = "player_attribute_history"

    id = Column(Integer, primary_key=True)
    player_id = Column(Integer, ForeignKey("players.id"), nullable=False)
    record_date = Column(Date, nullable=False)
    pace = Column(Integer, nullable=True)
    shooting = Column(Integer, nullable=True)
    passing = Column(Integer, nullable=True)
    dribbling = Column(Integer, nullable=True)
    defending = Column(Integer, nullable=True)
    physical = Column(Integer, nullable=True)
    vision = Column(Integer, nullable=True)
    tackling = Column(Integer, nullable=True)
    determination = Column(Integer, nullable=True)
    technique = Column(Integer, nullable=True)
    notes = Column(Text, nullable=True)

    player = relationship("Player", back_populates="attribute_history")

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "player_id": self.player_id,
            "record_date": self.record_date.isoformat(),
            "pace": self.pace,
            "shooting": self.shooting,
            "passing": self.passing,
            "dribbling": self.dribbling,
            "defending": self.defending,
            "physical": self.physical,
            "vision": self.vision,
            "tackling": self.tackling,
            "determination": self.determination,
            "technique": self.technique,
            "notes": self.notes,
        }


class Match(TimestampMixin, Base):
    """Contexto minimo del partido para enriquecer rendimiento y prediccion."""

    __tablename__ = "matches"

    id = Column(Integer, primary_key=True)
    match_date = Column(Date, nullable=False)
    opponent_name = Column(String, nullable=False)
    opponent_level = Column(Integer, nullable=False, default=3)  # escala 1-5
    tournament = Column(String, nullable=True)
    competition_category = Column(String, nullable=True)
    venue = Column(String, nullable=False, default="Local")
    notes = Column(Text, nullable=True)

    participations = relationship(
        "PlayerMatchParticipation",
        back_populates="match",
        cascade="all, delete-orphan",
    )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "match_date": self.match_date.isoformat(),
            "opponent_name": self.opponent_name,
            "opponent_level": self.opponent_level,
            "tournament": self.tournament,
            "competition_category": self.competition_category,
            "venue": self.venue,
            "notes": self.notes,
        }


class PlayerMatchParticipation(TimestampMixin, Base):
    """Participacion puntual del jugador en un partido."""

    __tablename__ = "player_match_participations"

    id = Column(Integer, primary_key=True)
    player_id = Column(Integer, ForeignKey("players.id"), nullable=False)
    match_id = Column(Integer, ForeignKey("matches.id"), nullable=False)
    started = Column(Boolean, nullable=False, default=False)
    position_played = Column(String, nullable=False)
    minutes_played = Column(Integer, nullable=False, default=0)
    final_score = Column(Float, nullable=True)  # valoracion 1-10
    goals = Column(Integer, nullable=False, default=0)
    assists = Column(Integer, nullable=False, default=0)
    pass_accuracy = Column(Float, nullable=True)
    shot_accuracy = Column(Float, nullable=True)
    duels_won_pct = Column(Float, nullable=True)
    yellow_cards = Column(Integer, nullable=False, default=0)
    red_cards = Column(Integer, nullable=False, default=0)
    role_notes = Column(Text, nullable=True)

    player = relationship("Player", back_populates="match_participations")
    match = relationship("Match", back_populates="participations")

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "player_id": self.player_id,
            "match_id": self.match_id,
            "started": self.started,
            "position_played": self.position_played,
            "minutes_played": self.minutes_played,
            "final_score": self.final_score,
            "goals": self.goals,
            "assists": self.assists,
            "pass_accuracy": self.pass_accuracy,
            "shot_accuracy": self.shot_accuracy,
            "duels_won_pct": self.duels_won_pct,
            "yellow_cards": self.yellow_cards,
            "red_cards": self.red_cards,
            "role_notes": self.role_notes,
        }


class ScoutReport(TimestampMixin, Base):
    """Observacion cualitativa sintetica o manual del scout."""

    __tablename__ = "scout_reports"

    id = Column(Integer, primary_key=True)
    player_id = Column(Integer, ForeignKey("players.id"), nullable=False)
    report_date = Column(Date, nullable=False)
    decision_making = Column(Integer, nullable=True)  # escala 1-20
    tactical_reading = Column(Integer, nullable=True)  # escala 1-20
    mental_profile = Column(Integer, nullable=True)  # escala 1-20
    adaptability = Column(Integer, nullable=True)  # escala 1-20
    observed_projection_score = Column(Float, nullable=True)  # valoracion 1-10
    notes = Column(Text, nullable=True)

    player = relationship("Player", back_populates="scout_reports")

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "player_id": self.player_id,
            "report_date": self.report_date.isoformat(),
            "decision_making": self.decision_making,
            "tactical_reading": self.tactical_reading,
            "mental_profile": self.mental_profile,
            "adaptability": self.adaptability,
            "observed_projection_score": self.observed_projection_score,
            "notes": self.notes,
        }


class PhysicalAssessment(TimestampMixin, Base):
    """Evaluacion fisica puntual del jugador."""

    __tablename__ = "physical_assessments"

    id = Column(Integer, primary_key=True)
    player_id = Column(Integer, ForeignKey("players.id"), nullable=False)
    assessment_date = Column(Date, nullable=False)
    height_cm = Column(Float, nullable=True)
    weight_kg = Column(Float, nullable=True)
    dominant_foot = Column(String, nullable=True)
    estimated_speed = Column(Float, nullable=True)  # escala 1-20
    endurance = Column(Float, nullable=True)  # escala 1-20
    in_growth_spurt = Column(Boolean, nullable=False, default=False)
    notes = Column(Text, nullable=True)

    player = relationship("Player", back_populates="physical_assessments")

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "player_id": self.player_id,
            "assessment_date": self.assessment_date.isoformat(),
            "height_cm": self.height_cm,
            "weight_kg": self.weight_kg,
            "dominant_foot": self.dominant_foot,
            "estimated_speed": self.estimated_speed,
            "endurance": self.endurance,
            "in_growth_spurt": self.in_growth_spurt,
            "notes": self.notes,
        }


class PlayerAvailability(TimestampMixin, Base):
    """Disponibilidad longitudinal del jugador para entrenar y competir."""

    __tablename__ = "player_availability"

    id = Column(Integer, primary_key=True)
    player_id = Column(Integer, ForeignKey("players.id"), nullable=False)
    record_date = Column(Date, nullable=False)
    availability_pct = Column(Float, nullable=True)  # 0-100
    fatigue_pct = Column(Float, nullable=True)  # 0-100
    training_load_pct = Column(Float, nullable=True)  # 0-100
    missed_days = Column(Integer, nullable=False, default=0)
    injury_flag = Column(Boolean, nullable=False, default=False)
    notes = Column(Text, nullable=True)

    player = relationship("Player", back_populates="availability_records")

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "player_id": self.player_id,
            "record_date": self.record_date.isoformat(),
            "availability_pct": self.availability_pct,
            "fatigue_pct": self.fatigue_pct,
            "training_load_pct": self.training_load_pct,
            "missed_days": self.missed_days,
            "injury_flag": self.injury_flag,
            "notes": self.notes,
        }
