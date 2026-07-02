"""Generador de datos sinteticos para la base de entrenamiento.

Este script crea jugadores juveniles, sintetiza su trayectoria tecnica y,
a partir de esa evolucion, deriva su historial de rendimiento.
"""

from __future__ import annotations

import argparse
from datetime import date, timedelta
import math
import os
import random
from typing import Dict, List

from sqlalchemy.orm import sessionmaker

from db_utils import create_app_engine, ensure_player_columns, normalize_db_url
from models import (
    Base,
    Match,
    PhysicalAssessment,
    Player,
    PlayerAttributeHistory,
    PlayerAvailability,
    PlayerMatchParticipation,
    PlayerStat,
    ScoutReport,
)
from player_logic import (
    ATTRIBUTE_FIELDS,
    ATTRIBUTE_MAX_VALUE,
    ATTRIBUTE_MIN_VALUE,
    EVAL_MAX_AGE,
    EVAL_MIN_AGE,
    default_player_photo_url,
    normalized_position,
    position_weights,
    recommend_position_from_attrs,
    weighted_score_from_attrs,
)

used_identifiers = set()
DEFAULT_SEED = int(os.environ.get("SEED", "42"))
OPPONENT_NAMES = [
    "Racing Juvenil",
    "Talleres Formativo",
    "Belgrano Inferiores",
    "Instituto Juvenil",
    "Newell's Academy",
    "Lanus Proyeccion",
    "Banfield Formativo",
    "Velez Desarrollo",
]
TOURNAMENTS = [
    "Liga Juvenil Regional",
    "Torneo Formativo Metropolitano",
    "Copa Proyeccion",
    "Liga de Desarrollo",
]
COMPETITION_CATEGORIES = [
    "Sub-13",
    "Sub-15",
    "Sub-17",
    "Reserva juvenil",
]
DEVELOPMENT_ARCHETYPES = [
    "steady_builder",
    "late_bloomer",
    "early_burst",
    "setback_rebound",
    "volatile_creator",
    "consistent_ceiling",
    "fatigue_limited",
]
DEVELOPMENT_ARCHETYPE_WEIGHTS = [0.24, 0.18, 0.13, 0.14, 0.13, 0.11, 0.07]
DOMINANT_FEET = ["Derecha", "Izquierda", "Ambos"]


def next_identifier() -> str:
    while True:
        value = random.randint(10000000, 59999999)
        if value not in used_identifiers:
            used_identifiers.add(value)
            return str(value)


def load_existing_identifiers(session) -> None:
    existing_ids = session.query(Player.national_id).filter(Player.national_id.isnot(None)).all()
    used_identifiers.update(
        int(national_id)
        for (national_id,) in existing_ids
        if national_id is not None and str(national_id).isdigit()
    )


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def player_attr_map(player: Player) -> Dict[str, int]:
    return {field: int(getattr(player, field) or ATTRIBUTE_MIN_VALUE) for field in ATTRIBUTE_FIELDS}


def age_potential_bonus(age: int) -> float:
    return round((EVAL_MAX_AGE - age) * 0.18, 2)


def mental_bonus(position: str, attrs: Dict[str, int]) -> float:
    determination = float(attrs["determination"])
    technique = float(attrs["technique"])
    vision = float(attrs["vision"])
    if position == "Mediocampista":
        return 0.35 * vision + 0.30 * technique + 0.20 * determination
    if position == "Delantero":
        return 0.20 * vision + 0.35 * technique + 0.25 * determination
    if position == "Portero":
        return 0.15 * vision + 0.20 * technique + 0.30 * determination
    return 0.18 * vision + 0.22 * technique + 0.28 * determination


def label_probability(position: str, age: int, attrs: Dict[str, int]) -> tuple[float, float]:
    weighted_score = weighted_score_from_attrs(attrs, position)
    recommended_position, recommended_score = recommend_position_from_attrs(attrs)
    alignment_bonus = 0.55 if recommended_position == position else -0.20
    mental_component = mental_bonus(position, attrs) / 20.0
    youth_bonus = age_potential_bonus(age)
    controlled_noise = random.gauss(0.0, 0.35)

    latent_score = (
        weighted_score
        + alignment_bonus
        + youth_bonus
        + mental_component
        + controlled_noise
        + (recommended_score - weighted_score) * 0.15
    )
    probability = 1.0 / (1.0 + math.exp(-(latent_score - 13.65) / 0.90))
    return max(0.01, min(probability, 0.99)), weighted_score


def generate_player(min_age: int = EVAL_MIN_AGE, max_age: int = EVAL_MAX_AGE) -> Player:
    """Genera un jugador con atributos aleatorios y etiqueta sintetica."""
    names = [
        "Juan", "Pedro", "Carlos", "Mateo", "Lucas", "Gabriel",
        "Nicolas", "Diego", "Federico", "Martin", "Rodrigo", "Franco",
        "Facundo", "Santiago", "Tomas", "Julian", "Pablo", "Ignacio",
        "Marcelo", "Hernan",
    ]
    surnames = [
        "Garcia", "Lopez", "Martinez", "Rodriguez", "Gonzalez", "Perez",
        "Sanchez", "Romero", "Ferreira", "Suarez", "Herrera", "Ramirez",
        "Flores", "Torres", "Luna", "Alvarez", "Rojas", "Bautista",
        "Cordoba", "Vega",
    ]
    positions = ["Portero", "Defensa", "Lateral", "Mediocampista", "Delantero"]
    clubs = [
        "Club A", "Club B", "Club C", "Club D", "Club E", "Club F",
        "Academia Juvenil", "Escuela de Futbol",
    ]
    countries = ["Argentina", "Brasil", "Uruguay", "Chile", "Paraguay"]

    name = f"{random.choice(names)} {random.choice(surnames)}"
    age = random.randint(min_age, max_age)
    birth_date = date(date.today().year - age, 1, 1)
    position = normalized_position(random.choice(positions))
    club = random.choice(clubs)
    country = random.choice(countries)

    attrs = {field: random.randint(ATTRIBUTE_MIN_VALUE, ATTRIBUTE_MAX_VALUE) for field in ATTRIBUTE_FIELDS}
    probability, _weighted_score = label_probability(position, age, attrs)
    potential = random.random() < probability

    return Player(
        name=name,
        national_id=next_identifier(),
        age=age,
        birth_date=birth_date,
        position=position,
        club=club,
        country=country,
        photo_url=default_player_photo_url(name=name),
        pace=attrs["pace"],
        shooting=attrs["shooting"],
        passing=attrs["passing"],
        dribbling=attrs["dribbling"],
        defending=attrs["defending"],
        physical=attrs["physical"],
        vision=attrs["vision"],
        tackling=attrs["tackling"],
        determination=attrs["determination"],
        technique=attrs["technique"],
        potential_label=bool(potential),
    )


def build_development_profile(player: Player) -> Dict[str, float]:
    attrs = player_attr_map(player)
    age = int(player.age or EVAL_MAX_AGE)
    development_archetype = random.choices(
        DEVELOPMENT_ARCHETYPES,
        weights=DEVELOPMENT_ARCHETYPE_WEIGHTS,
        k=1,
    )[0]
    dominant_foot = random.choices(DOMINANT_FEET, weights=[0.68, 0.18, 0.14], k=1)[0]
    resilience = clamp(
        0.45 + attrs["determination"] / 26.0 + attrs["physical"] / 90.0 + random.uniform(-0.12, 0.12),
        0.45,
        1.45,
    )
    tactical_discipline = clamp(
        0.40 + attrs["vision"] / 38.0 + attrs["passing"] / 80.0 + attrs["defending"] / 95.0 + random.uniform(-0.10, 0.10),
        0.35,
        1.40,
    )
    adaptability_bias = clamp(
        0.35 + attrs["technique"] / 44.0 + attrs["vision"] / 70.0 + attrs["pace"] / 110.0 + random.uniform(-0.10, 0.10),
        0.30,
        1.35,
    )
    professionalism = clamp(
        0.42 + attrs["determination"] / 34.0 + attrs["technique"] / 75.0 + random.uniform(-0.10, 0.10),
        0.35,
        1.35,
    )
    growth_factor = clamp(
        0.55
        + (EVAL_MAX_AGE - age) * 0.12
        + attrs["determination"] / 24.0
        + attrs["technique"] / 42.0
        + attrs["vision"] / 48.0,
        0.65,
        2.8,
    )
    volatility = clamp(
        0.10 + (20 - attrs["determination"]) / 110.0 + random.uniform(0.0, 0.12),
        0.08,
        0.42,
    )
    availability = clamp(
        0.66 + attrs["physical"] / 55.0 + attrs["determination"] / 95.0 + random.uniform(-0.06, 0.06),
        0.55,
        0.97,
    )
    body_maturity = clamp(
        0.50 + (age - EVAL_MIN_AGE) * 0.08 + attrs["physical"] / 65.0 + random.uniform(-0.10, 0.10),
        0.45,
        1.35,
    )
    injury_proneness = clamp(
        0.18
        + (1.0 - availability) * 0.55
        + max(0.0, 0.95 - resilience) * 0.20
        + random.uniform(-0.06, 0.06),
        0.06,
        0.72,
    )
    recovery_quality = clamp(
        0.42 + resilience * 0.34 + professionalism * 0.18 + random.uniform(-0.05, 0.05),
        0.35,
        1.20,
    )
    pressure_resistance = clamp(
        0.38
        + attrs["determination"] / 34.0
        + attrs["vision"] / 90.0
        + resilience * 0.24
        + random.uniform(-0.08, 0.08),
        0.35,
        1.45,
    )
    confidence_baseline = clamp(
        0.40
        + attrs["technique"] / 58.0
        + attrs["determination"] / 95.0
        + professionalism * 0.18
        + random.uniform(-0.08, 0.08),
        0.35,
        1.35,
    )
    coachability = clamp(
        0.40
        + tactical_discipline * 0.26
        + professionalism * 0.22
        + attrs["vision"] / 120.0
        + random.uniform(-0.07, 0.07),
        0.35,
        1.35,
    )
    fatigue_sensitivity = clamp(
        0.16
        + injury_proneness * 0.42
        + max(0.0, 0.98 - recovery_quality) * 0.16
        + random.uniform(-0.04, 0.04),
        0.08,
        0.75,
    )
    adaptation_speed = clamp(
        0.42
        + adaptability_bias * 0.26
        + coachability * 0.18
        + random.uniform(-0.06, 0.06),
        0.35,
        1.30,
    )
    plateau_risk = clamp(
        0.18
        + max(0.0, 0.98 - professionalism) * 0.30
        + (0.10 if development_archetype == "early_burst" else 0.0)
        + random.uniform(-0.04, 0.05),
        0.08,
        0.68,
    )
    trajectory_strength = clamp(
        0.72
        + resilience * 0.12
        + professionalism * 0.16
        + coachability * 0.10
        + random.uniform(-0.08, 0.10),
        0.55,
        1.35,
    )
    adversity_timing = clamp(random.uniform(0.34, 0.62), 0.28, 0.70)
    adversity_severity = clamp(
        0.18
        + injury_proneness * 0.26
        + fatigue_sensitivity * 0.20
        + random.uniform(-0.05, 0.07),
        0.06,
        0.62,
    )
    if development_archetype == "consistent_ceiling":
        growth_factor *= 0.84
        volatility = clamp(volatility * 0.68, 0.05, 0.34)
        plateau_risk = clamp(plateau_risk + 0.07, 0.08, 0.72)
        trajectory_strength = clamp(trajectory_strength * 0.86, 0.50, 1.15)
    elif development_archetype == "fatigue_limited":
        availability = clamp(availability - 0.05, 0.48, 0.94)
        fatigue_sensitivity = clamp(fatigue_sensitivity + 0.13, 0.12, 0.86)
        injury_proneness = clamp(injury_proneness + 0.06, 0.08, 0.78)
        adversity_severity = clamp(adversity_severity + 0.12, 0.12, 0.72)
    elif development_archetype == "late_bloomer":
        trajectory_strength = clamp(trajectory_strength + 0.12, 0.60, 1.45)
        adversity_timing = clamp(random.uniform(0.54, 0.74), 0.48, 0.80)
    elif development_archetype == "early_burst":
        plateau_risk = clamp(plateau_risk + 0.08, 0.10, 0.78)
        adversity_timing = clamp(random.uniform(0.58, 0.78), 0.50, 0.84)
    elif development_archetype == "setback_rebound":
        adversity_severity = clamp(adversity_severity + 0.10, 0.12, 0.74)
    elif development_archetype == "volatile_creator":
        volatility = clamp(volatility * 1.18, 0.10, 0.52)
    baseline_height_cm = clamp(
        144.0 + age * 2.6 + attrs["physical"] * 0.9 + random.uniform(-8.0, 8.0),
        145.0,
        196.0,
    )
    height_ceiling_cm = clamp(
        baseline_height_cm + random.uniform(1.5, 8.5) * max(0.4, 1.05 - body_maturity * 0.25),
        baseline_height_cm + 1.0,
        200.0,
    )
    baseline_weight_kg = clamp(
        37.0 + age * 1.65 + attrs["physical"] * 0.75 + random.uniform(-4.5, 4.5),
        38.0,
        94.0,
    )
    snapshot_count = random.randint(6, 12)
    slump_month = random.randint(2, snapshot_count - 1) if snapshot_count >= 4 and random.random() < 0.42 else None
    return {
        "growth_factor": growth_factor * ((0.88 + professionalism * 0.10) * (0.92 + resilience * 0.08)),
        "volatility": volatility,
        "availability": availability,
        "snapshot_count": snapshot_count,
        "slump_month": slump_month or 0,
        "form_bias": random.gauss(0.0, 0.18),
        "resilience": resilience,
        "tactical_discipline": tactical_discipline,
        "adaptability_bias": adaptability_bias,
        "professionalism": professionalism,
        "development_archetype": development_archetype,
        "dominant_foot": dominant_foot,
        "body_maturity": body_maturity,
        "injury_proneness": injury_proneness,
        "recovery_quality": recovery_quality,
        "pressure_resistance": pressure_resistance,
        "confidence_baseline": confidence_baseline,
        "coachability": coachability,
        "fatigue_sensitivity": fatigue_sensitivity,
        "adaptation_speed": adaptation_speed,
        "plateau_risk": plateau_risk,
        "trajectory_strength": trajectory_strength,
        "adversity_timing": adversity_timing,
        "adversity_severity": adversity_severity,
        "baseline_height_cm": baseline_height_cm,
        "height_ceiling_cm": height_ceiling_cm,
        "baseline_weight_kg": baseline_weight_kg,
    }


def total_growth_by_attribute(player: Player, profile: Dict[str, float]) -> Dict[str, float]:
    weights = position_weights(player.position)
    current_attrs = player_attr_map(player)
    growth_map: Dict[str, float] = {}
    for field in ATTRIBUTE_FIELDS:
        current_value = float(current_attrs[field])
        ceiling_room = max(0.10, (20.0 - current_value) / 20.0)
        weighted_factor = 0.55 + (weights[field] * 2.0)
        trait_multiplier = 1.0
        if field in ("determination", "technique"):
            trait_multiplier *= 0.92 + profile["professionalism"] * 0.10
        if field in ("vision", "passing"):
            trait_multiplier *= 0.90 + profile["tactical_discipline"] * 0.12
        if field in ("pace", "dribbling"):
            trait_multiplier *= 0.90 + profile["adaptability_bias"] * 0.10
        if field in ("physical", "tackling", "defending"):
            trait_multiplier *= 0.90 + profile["resilience"] * 0.10
        random_adjustment = random.uniform(0.85, 1.25)
        growth = profile["growth_factor"] * weighted_factor * ceiling_room * trait_multiplier * random_adjustment
        growth_map[field] = clamp(growth, 0.0, 5.5)
    return growth_map


def avg(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def smoothstep(value: float, lower: float, upper: float) -> float:
    if upper <= lower:
        return 1.0 if value >= upper else 0.0
    x = clamp((value - lower) / (upper - lower), 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def trajectory_state(progress_fraction: float, profile: Dict[str, float]) -> Dict[str, float]:
    """Estado latente del mes: modela trayectoria juvenil sin mirar el target."""
    progress_fraction = clamp(progress_fraction, 0.0, 1.0)
    archetype = str(profile.get("development_archetype", "steady_builder"))
    strength = float(profile.get("trajectory_strength", 1.0))
    severity = float(profile.get("adversity_severity", 0.20))
    timing = float(profile.get("adversity_timing", 0.50))
    resilience = float(profile.get("resilience", 0.85))
    plateau_risk = float(profile.get("plateau_risk", 0.20))
    fatigue_sensitivity = float(profile.get("fatigue_sensitivity", 0.24))

    state = {
        "growth_boost": 0.0,
        "performance_shift": 0.0,
        "availability_shift": 0.0,
        "fatigue_shift": 0.0,
        "confidence_shift": 0.0,
        "trust_shift": 0.0,
        "scout_shift": 0.0,
        "volatility_multiplier": 1.0,
    }

    if archetype == "late_bloomer":
        surge = smoothstep(progress_fraction, 0.50, 0.92)
        early_penalty = 1.0 - surge
        state["growth_boost"] = (-0.10 * early_penalty + 0.34 * surge) * strength
        state["performance_shift"] = -0.10 * early_penalty + 0.26 * surge
        state["confidence_shift"] = -0.05 * early_penalty + 0.15 * surge
        state["trust_shift"] = 0.14 * surge
        state["scout_shift"] = 0.20 * surge
        state["volatility_multiplier"] = 0.96 + 0.12 * surge
    elif archetype == "early_burst":
        early_gain = 1.0 - smoothstep(progress_fraction, 0.30, 0.62)
        plateau = smoothstep(progress_fraction, 0.58, 0.94)
        state["growth_boost"] = 0.27 * early_gain * strength - 0.22 * plateau * plateau_risk
        state["performance_shift"] = 0.18 * early_gain - 0.20 * plateau * plateau_risk
        state["availability_shift"] = -5.0 * plateau * plateau_risk
        state["fatigue_shift"] = 8.0 * plateau * plateau_risk
        state["trust_shift"] = -0.10 * plateau * plateau_risk
        state["scout_shift"] = -0.14 * plateau * plateau_risk
        state["volatility_multiplier"] = 0.92 + 0.24 * plateau
    elif archetype == "setback_rebound":
        setback = math.exp(-((progress_fraction - timing) ** 2) / (2.0 * 0.11**2))
        rebound = smoothstep(progress_fraction, timing + 0.08, 0.95)
        rebound_quality = clamp(0.55 + resilience * 0.35, 0.55, 1.15)
        state["growth_boost"] = -0.34 * severity * setback + 0.30 * rebound * rebound_quality
        state["performance_shift"] = -0.34 * severity * setback + 0.28 * rebound * rebound_quality
        state["availability_shift"] = -12.0 * severity * setback + 5.0 * rebound
        state["fatigue_shift"] = 17.0 * severity * setback - 5.0 * rebound
        state["confidence_shift"] = -0.18 * severity * setback + 0.18 * rebound
        state["trust_shift"] = -0.15 * severity * setback + 0.14 * rebound
        state["scout_shift"] = -0.22 * severity * setback + 0.24 * rebound
        state["volatility_multiplier"] = 1.0 + 0.34 * setback
    elif archetype == "volatile_creator":
        oscillation = math.sin(progress_fraction * math.pi * 3.0 + strength)
        positive_wave = max(0.0, oscillation)
        negative_wave = max(0.0, -oscillation)
        state["growth_boost"] = 0.12 * oscillation
        state["performance_shift"] = 0.22 * oscillation
        state["availability_shift"] = -2.0 * negative_wave
        state["fatigue_shift"] = 4.0 * negative_wave
        state["confidence_shift"] = 0.11 * oscillation
        state["trust_shift"] = 0.08 * oscillation
        state["scout_shift"] = 0.12 * positive_wave - 0.08 * negative_wave
        state["volatility_multiplier"] = 1.22
    elif archetype == "consistent_ceiling":
        late_plateau = smoothstep(progress_fraction, 0.64, 0.96)
        state["growth_boost"] = 0.08 * progress_fraction - 0.10 * late_plateau
        state["performance_shift"] = 0.05 * progress_fraction - 0.05 * late_plateau
        state["availability_shift"] = 3.0
        state["fatigue_shift"] = -4.0
        state["confidence_shift"] = 0.04
        state["trust_shift"] = 0.07
        state["scout_shift"] = 0.04
        state["volatility_multiplier"] = 0.68
    elif archetype == "fatigue_limited":
        load_phase = smoothstep(progress_fraction, 0.45, 0.95)
        fatigue_drag = load_phase * fatigue_sensitivity
        state["growth_boost"] = 0.08 * progress_fraction - 0.22 * fatigue_drag
        state["performance_shift"] = 0.08 * progress_fraction - 0.30 * fatigue_drag
        state["availability_shift"] = -10.0 * fatigue_drag
        state["fatigue_shift"] = 20.0 * fatigue_drag
        state["confidence_shift"] = -0.10 * fatigue_drag
        state["trust_shift"] = -0.10 * fatigue_drag
        state["scout_shift"] = -0.13 * fatigue_drag
        state["volatility_multiplier"] = 0.98 + 0.28 * load_phase
    else:
        state["growth_boost"] = (progress_fraction - 0.45) * 0.12 * strength
        state["performance_shift"] = (progress_fraction - 0.45) * 0.08
        state["availability_shift"] = 1.5 * progress_fraction
        state["fatigue_shift"] = -2.0 * progress_fraction
        state["confidence_shift"] = 0.05 * progress_fraction
        state["trust_shift"] = 0.05 * progress_fraction
        state["scout_shift"] = 0.06 * progress_fraction
        state["volatility_multiplier"] = 0.88

    state["volatility_multiplier"] = clamp(state["volatility_multiplier"], 0.55, 1.55)
    return state


def trajectory_curve_multiplier(progress_fraction: float, archetype: str) -> float:
    progress_fraction = clamp(progress_fraction, 0.0, 1.0)
    if archetype == "late_bloomer":
        return 0.70 + (progress_fraction**1.8) * 0.95
    if archetype == "early_burst":
        return 0.85 + ((1.0 - progress_fraction) ** 1.6) * 0.70
    if archetype == "setback_rebound":
        return 0.78 + math.sin(progress_fraction * math.pi) * 0.55
    if archetype == "volatile_creator":
        return 0.82 + math.sin(progress_fraction * math.pi * 2.2) * 0.22
    if archetype == "consistent_ceiling":
        return 0.78 + progress_fraction * 0.20
    if archetype == "fatigue_limited":
        return 0.92 + progress_fraction * 0.16 - smoothstep(progress_fraction, 0.58, 0.95) * 0.24
    return 0.88 + progress_fraction * 0.35


def physical_growth_curve(progress_fraction: float, body_maturity: float, archetype: str) -> float:
    base_curve = trajectory_curve_multiplier(progress_fraction, archetype)
    return clamp(base_curve * (0.82 + body_maturity * 0.22), 0.55, 1.65)


def adjacent_positions(position: str) -> List[str]:
    mapping = {
        "Portero": ["Portero"],
        "Defensa": ["Defensa", "Lateral"],
        "Lateral": ["Lateral", "Defensa", "Mediocampista"],
        "Mediocampista": ["Mediocampista", "Lateral", "Delantero"],
        "Delantero": ["Delantero", "Mediocampista"],
    }
    return mapping.get(position, [position])


def position_synergy_score(position: str, attrs: Dict[str, float]) -> float:
    if position == "Mediocampista":
        raw = (
            attrs["passing"] * attrs["vision"]
            + attrs["vision"] * attrs["technique"]
            + attrs["passing"] * attrs["determination"]
        ) / (20.0 * 20.0 * 3.0)
    elif position == "Delantero":
        raw = (
            attrs["shooting"] * attrs["technique"]
            + attrs["pace"] * attrs["dribbling"]
            + attrs["shooting"] * attrs["determination"]
        ) / (20.0 * 20.0 * 3.0)
    elif position == "Defensa":
        raw = (
            attrs["defending"] * attrs["tackling"]
            + attrs["physical"] * attrs["determination"]
            + attrs["defending"] * attrs["vision"]
        ) / (20.0 * 20.0 * 3.0)
    elif position == "Lateral":
        raw = (
            attrs["pace"] * attrs["dribbling"]
            + attrs["passing"] * attrs["vision"]
            + attrs["physical"] * attrs["determination"]
        ) / (20.0 * 20.0 * 3.0)
    else:
        raw = (
            attrs["vision"] * attrs["determination"]
            + attrs["physical"] * attrs["tackling"]
            + attrs["passing"] * attrs["technique"]
        ) / (20.0 * 20.0 * 3.0)
    return clamp(raw, 0.0, 1.15)


def synthetic_attribute_history(
    player: Player,
    profile: Dict[str, float],
) -> List[PlayerAttributeHistory]:
    current_attrs = player_attr_map(player)
    growth_map = total_growth_by_attribute(player, profile)
    snapshot_count = int(profile["snapshot_count"])
    slump_month = int(profile["slump_month"])
    history: List[PlayerAttributeHistory] = []
    today = date.today()

    for months_back in range(snapshot_count, 0, -1):
        remaining_fraction = months_back / float(snapshot_count + 1)
        progress_fraction = (snapshot_count - months_back + 1) / float(snapshot_count + 1)
        trajectory = trajectory_state(progress_fraction, profile)
        archetype_curve = trajectory_curve_multiplier(progress_fraction, str(profile["development_archetype"]))
        archetype_curve = clamp(archetype_curve * (1.0 + trajectory["growth_boost"]), 0.35, 1.90)
        wave = math.sin(progress_fraction * math.pi)
        values: Dict[str, int] = {}
        for field in ATTRIBUTE_FIELDS:
            current_value = float(current_attrs[field])
            weight = position_weights(player.position)[field]
            progression_component = growth_map[field] * remaining_fraction * archetype_curve
            curve_component = growth_map[field] * 0.10 * wave * (0.85 + profile["adaptability_bias"] * 0.08)
            noise_scale = (
                profile["volatility"]
                * trajectory["volatility_multiplier"]
                * (0.30 + weight)
                * (1.12 if profile["development_archetype"] == "volatile_creator" else 0.95)
            )
            noise = random.gauss(0.0, noise_scale)
            slump = 0.0
            if slump_month and months_back == slump_month:
                slump = random.uniform(-0.9, -0.3) * (0.30 + weight) * (1.18 - profile["resilience"] * 0.14)
                if profile["development_archetype"] == "setback_rebound":
                    slump *= 1.25
            growth_spurt = 0.0
            if player.age <= 14 and months_back in (1, 2) and random.random() < 0.35:
                growth_spurt = random.uniform(0.10, 0.35) * (0.25 + weight) * profile["professionalism"]
                if profile["development_archetype"] == "late_bloomer":
                    growth_spurt *= 1.18
            if profile["development_archetype"] == "early_burst" and progress_fraction <= 0.30:
                growth_spurt += random.uniform(0.05, 0.18) * (0.20 + weight)
            trajectory_level = trajectory["growth_boost"] * growth_map[field] * (0.08 + weight * 0.10)
            historical_value = current_value - progression_component + curve_component + trajectory_level + noise + slump
            values[field] = int(
                round(clamp(historical_value + growth_spurt, float(ATTRIBUTE_MIN_VALUE), current_value))
            )

        history.append(
            PlayerAttributeHistory(
                record_date=today - timedelta(days=30 * months_back),
                notes="Dato sintetico de trayectoria tecnica",
                **values,
            )
        )
    return history


def synthetic_physical_assessments(
    player: Player,
    attribute_history: List[PlayerAttributeHistory],
    profile: Dict[str, float],
) -> List[PhysicalAssessment]:
    if not attribute_history:
        return []

    assessments: List[PhysicalAssessment] = []
    total_points = max(1, len(attribute_history))
    previous_height = float(profile["baseline_height_cm"])
    previous_weight = float(profile["baseline_weight_kg"])

    for index, entry in enumerate(attribute_history):
        progress_fraction = (index + 1) / float(total_points)
        curve = physical_growth_curve(progress_fraction, float(profile["body_maturity"]), str(profile["development_archetype"]))
        attrs = {field: float(getattr(entry, field) or ATTRIBUTE_MIN_VALUE) for field in ATTRIBUTE_FIELDS}
        growth_spurt = bool(
            player.age <= 15
            and curve >= 1.08
            and progress_fraction >= 0.55
            and random.random() < 0.42
        )
        height_room = max(0.5, float(profile["height_ceiling_cm"]) - previous_height)
        height_increment = clamp(
            (height_room / max(1.0, total_points - index + 0.5)) * curve + random.uniform(-0.2, 0.45),
            0.0,
            2.4 if growth_spurt else 1.2,
        )
        height_cm = clamp(previous_height + height_increment, 145.0, float(profile["height_ceiling_cm"]))
        weight_trend = 0.35 + attrs["physical"] * 0.08 + float(profile["body_maturity"]) * 0.45
        if growth_spurt:
            weight_trend *= 0.82
        weight_kg = clamp(
            previous_weight + random.uniform(0.1, 0.9) * curve + weight_trend * 0.08,
            38.0,
            95.0,
        )
        estimated_speed = clamp(
            attrs["pace"] * 0.72
            + attrs["physical"] * 0.18
            + float(profile["adaptability_bias"]) * 1.4
            - (0.25 if growth_spurt else 0.0)
            + random.gauss(0.0, 0.45),
            3.0,
            20.0,
        )
        endurance = clamp(
            attrs["physical"] * 0.62
            + attrs["determination"] * 0.20
            + float(profile["resilience"]) * 1.6
            + float(profile["professionalism"]) * 0.8
            + random.gauss(0.0, 0.45),
            3.0,
            20.0,
        )
        assessments.append(
            PhysicalAssessment(
                player_id=player.id,
                assessment_date=entry.record_date,
                height_cm=round(height_cm, 1),
                weight_kg=round(weight_kg, 1),
                dominant_foot=str(profile["dominant_foot"]),
                estimated_speed=round(estimated_speed, 2),
                endurance=round(endurance, 2),
                in_growth_spurt=growth_spurt,
                notes="Dato sintetico de evaluacion fisica longitudinal",
            )
        )
        previous_height = height_cm
        previous_weight = weight_kg

    return assessments


def synthetic_availability_history(
    player: Player,
    attribute_history: List[PlayerAttributeHistory],
    physical_assessments: List[PhysicalAssessment],
    profile: Dict[str, float],
) -> List[PlayerAvailability]:
    if not attribute_history:
        return []

    physical_map = {assessment.assessment_date: assessment for assessment in physical_assessments}
    availability_rows: List[PlayerAvailability] = []
    previous_fatigue = 24.0
    previous_availability = float(profile["availability"]) * 100.0
    total_points = max(1, len(attribute_history))

    for index, entry in enumerate(attribute_history):
        progress_fraction = (index + 1) / float(total_points)
        trajectory = trajectory_state(progress_fraction, profile)
        attrs = {field: float(getattr(entry, field) or ATTRIBUTE_MIN_VALUE) for field in ATTRIBUTE_FIELDS}
        physical = physical_map.get(entry.record_date)
        growth_spurt = bool(physical.in_growth_spurt) if physical is not None else False
        load_base = 56.0 + attrs["determination"] * 1.1 + float(profile["professionalism"]) * 9.0
        load_base += trajectory["performance_shift"] * 5.0
        if profile["development_archetype"] == "early_burst":
            load_base += 4.5
            if progress_fraction >= 0.60:
                load_base += float(profile["plateau_risk"]) * 6.0
        if profile["development_archetype"] == "setback_rebound" and 0.35 <= progress_fraction <= 0.65:
            load_base -= 6.0
        training_load_pct = clamp(load_base + random.gauss(0.0, 6.0), 38.0, 96.0)
        fatigue_pct = clamp(
            previous_fatigue * 0.40
            + training_load_pct * 0.36
            + (18.0 if growth_spurt else 0.0)
            + float(profile["injury_proneness"]) * 18.0
            + float(profile["fatigue_sensitivity"]) * 12.0
            - float(profile["recovery_quality"]) * 12.0
            + trajectory["fatigue_shift"]
            + random.gauss(0.0, 5.0),
            8.0,
            92.0,
        )
        injury_risk = clamp(
            float(profile["injury_proneness"]) * 0.55
            + fatigue_pct / 180.0
            + (0.12 if growth_spurt else 0.0)
            - float(profile["resilience"]) * 0.08
            + random.uniform(-0.05, 0.05),
            0.02,
            0.72,
        )
        injured = random.random() < injury_risk * (0.32 if index == total_points - 1 else 0.24)
        missed_days = int(round(clamp(random.gauss(0.0, 1.5) + (6 if injured else 0) + max(0.0, fatigue_pct - 70.0) / 8.0, 0.0, 18.0)))
        availability_pct = clamp(
            previous_availability * 0.28
            + float(profile["availability"]) * 52.0
            + attrs["determination"] * 0.9
            + float(profile["recovery_quality"]) * 10.0
            + (4.0 if profile["development_archetype"] == "late_bloomer" and progress_fraction >= 0.60 else 0.0)
            - fatigue_pct * 0.34
            - missed_days * 2.4
            - (7.0 if injured else 0.0)
            - float(profile["plateau_risk"]) * max(0.0, progress_fraction - 0.55) * 12.0
            + trajectory["availability_shift"]
            + random.gauss(0.0, 4.0),
            35.0,
            98.0,
        )
        availability_rows.append(
            PlayerAvailability(
                player_id=player.id,
                record_date=entry.record_date,
                availability_pct=round(availability_pct, 2),
                fatigue_pct=round(fatigue_pct, 2),
                training_load_pct=round(training_load_pct, 2),
                missed_days=missed_days,
                injury_flag=injured,
                notes="Dato sintetico de disponibilidad mensual",
            )
        )
        previous_fatigue = fatigue_pct
        previous_availability = availability_pct

    return availability_rows


def synthetic_matches_and_stats(
    player: Player,
    attribute_history: List[PlayerAttributeHistory],
    physical_assessments: List[PhysicalAssessment],
    availability_history: List[PlayerAvailability],
    profile: Dict[str, float],
) -> tuple[List[Match], List[PlayerMatchParticipation], List[PlayerStat]]:
    if not attribute_history:
        return [], [], []

    matches: List[Match] = []
    participations: List[PlayerMatchParticipation] = []
    stats: List[PlayerStat] = []
    previous_weighted_score = None
    confidence_state = clamp(float(profile["confidence_baseline"]), 0.35, 1.45)
    coach_trust_state = clamp(
        0.45 + float(profile["coachability"]) * 0.34 + float(profile["professionalism"]) * 0.22,
        0.35,
        1.50,
    )
    recent_role_fit = 0.82
    previous_month_avg_score = 6.10
    physical_map = {assessment.assessment_date: assessment for assessment in physical_assessments}
    availability_map = {row.record_date: row for row in availability_history}

    for entry_index, entry in enumerate(attribute_history):
        progress_fraction = (entry_index + 1) / float(max(1, len(attribute_history)))
        trajectory = trajectory_state(progress_fraction, profile)
        attrs = {field: float(getattr(entry, field) or ATTRIBUTE_MIN_VALUE) for field in ATTRIBUTE_FIELDS}
        weighted_score = weighted_score_from_attrs(attrs, player.position)
        avg_attr_score = sum(attrs.values()) / len(attrs)
        synergy_score = position_synergy_score(player.position, attrs)
        decision_profile = clamp(
            (
                attrs["vision"] * 0.38
                + attrs["technique"] * 0.28
                + attrs["determination"] * 0.20
                + attrs["passing"] * 0.14
            )
            / 20.0,
            0.0,
            1.25,
        )
        momentum = 0.0 if previous_weighted_score is None else weighted_score - previous_weighted_score
        previous_weighted_score = weighted_score
        physical = physical_map.get(entry.record_date)
        availability = availability_map.get(entry.record_date)
        availability_pct = float(availability.availability_pct) if availability and availability.availability_pct is not None else float(profile["availability"]) * 100.0
        fatigue_pct = float(availability.fatigue_pct) if availability and availability.fatigue_pct is not None else 28.0
        training_load_pct = float(availability.training_load_pct) if availability and availability.training_load_pct is not None else 62.0
        missed_days = int(availability.missed_days) if availability is not None else 0
        is_injured = bool(availability.injury_flag) if availability is not None else False
        speed_score = float(physical.estimated_speed) if physical and physical.estimated_speed is not None else attrs["pace"]
        endurance_score = float(physical.endurance) if physical and physical.endurance is not None else attrs["physical"]
        growth_spurt = bool(physical.in_growth_spurt) if physical is not None else False

        if is_injured or random.random() > (availability_pct / 100.0):
            confidence_state = clamp(confidence_state - 0.05, 0.35, 1.45)
            coach_trust_state = clamp(coach_trust_state - 0.04, 0.35, 1.50)
            continue

        monthly_match_count = max(1, min(3, random.randint(1, 3) - (1 if missed_days >= 6 else 0)))
        monthly_participations: List[PlayerMatchParticipation] = []
        monthly_role_fits: List[float] = []
        alternative_positions = [value for value in adjacent_positions(player.position) if value != player.position]

        for match_idx in range(monthly_match_count):
            opponent_level = random.choices([1, 2, 3, 4, 5], weights=[0.10, 0.24, 0.32, 0.22, 0.12], k=1)[0]
            venue = "Local" if random.random() < 0.56 else "Visitante"
            pressure = (opponent_level - 3) * 0.22 + (-0.10 if venue == "Visitante" else 0.06)
            pressure -= (profile["resilience"] - 0.85) * 0.18
            pressure -= (profile["tactical_discipline"] - 0.80) * 0.10
            pressure_response = clamp(
                0.34
                + float(profile["pressure_resistance"]) * 0.38
                + confidence_state * 0.18
                + coach_trust_state * 0.14
                + decision_profile * 0.20
                + trajectory["performance_shift"] * 0.18
                - float(profile["fatigue_sensitivity"]) * max(0.0, fatigue_pct - 48.0) / 70.0
                - (0.06 if growth_spurt else 0.0),
                0.15,
                1.65,
            )
            starter_probability = clamp(
                0.18
                + weighted_score / 36.0
                + momentum / 7.0
                + synergy_score * 0.16
                + confidence_state * 0.08
                + coach_trust_state * 0.12
                + decision_profile * 0.08
                + trajectory["trust_shift"] * 0.16
                + (availability_pct / 100.0) * 0.12
                + (profile["professionalism"] - 0.85) * 0.05
                - fatigue_pct / 300.0
                - max(0, opponent_level - 3) * 0.05,
                0.10,
                0.93,
            )
            is_starter = random.random() < starter_probability
            if alternative_positions and random.random() > (0.82 - profile["adaptability_bias"] * 0.12):
                position_played = random.choice(alternative_positions)
            else:
                position_played = player.position
            role_fit = 1.0 if position_played == player.position else clamp(
                0.56
                + float(profile["adaptation_speed"]) * 0.26
                + recent_role_fit * 0.10
                - max(0.0, opponent_level - 3) * 0.04,
                0.35,
                1.05,
            )
            monthly_role_fits.append(role_fit)

            minutes_base = random.randint(62, 90) if is_starter else random.randint(12, 38)
            fatigue_penalty = max(
                0.0,
                (1.0 - (availability_pct / 100.0)) * random.uniform(0.0, 1.5) * (1.12 - profile["resilience"] * 0.10)
                + fatigue_pct / 65.0
                + float(profile["fatigue_sensitivity"]) * 0.55
                + (0.45 if growth_spurt else 0.0),
            )
            minutes_played = int(round(clamp(minutes_base - fatigue_penalty * 8 - missed_days * 0.8, 5, 90)))
            pass_accuracy = clamp(
                (attrs["passing"] * 0.34 + attrs["vision"] * 0.28 + attrs["technique"] * 0.20) * 5
                + synergy_score * 8.0
                + pressure_response * max(0.0, opponent_level - 3.0) * 1.8
                + momentum * 3.0
                - pressure * 4.5
                + (profile["tactical_discipline"] - 0.85) * 6.0
                + (role_fit - 0.75) * 8.0
                - max(0.0, training_load_pct - 78.0) * 0.10
                - fatigue_pct * 0.06
                + random.gauss(0.0, 5.0),
                35.0,
                96.0,
            )
            shot_accuracy = clamp(
                (attrs["shooting"] * 0.45 + attrs["technique"] * 0.24 + attrs["vision"] * 0.12) * 5
                + synergy_score * 9.0
                + pressure_response * max(0.0, opponent_level - 3.0) * 1.4
                + momentum * 2.4
                - pressure * 3.4
                + (profile["professionalism"] - 0.85) * 4.0
                + (confidence_state - 0.80) * 4.8
                + (speed_score - attrs["pace"]) * 0.9
                - fatigue_pct * 0.05
                + random.gauss(0.0, 6.5),
                18.0,
                92.0,
            )
            duels_won_pct = clamp(
                (attrs["defending"] * 0.34 + attrs["physical"] * 0.28 + attrs["tackling"] * 0.18) * 5
                + synergy_score * 8.0
                + pressure_response * max(0.0, opponent_level - 3.0) * 1.8
                + momentum * 2.8
                - pressure * 4.0
                + (profile["resilience"] - 0.85) * 5.5
                + (role_fit - 0.75) * 6.0
                + (endurance_score - attrs["physical"]) * 0.75
                - fatigue_pct * 0.04
                + random.gauss(0.0, 6.0),
                25.0,
                95.0,
            )
            challenge_bonus = max(0.0, opponent_level - 3.0) * (pressure_response - 0.78) * 0.34
            final_score = clamp(
                3.75
                + (weighted_score / 20.0) * 3.8
                + synergy_score * 1.30
                + decision_profile * 0.65
                + momentum * 0.55
                + challenge_bonus
                - pressure
                + profile["form_bias"] * 0.85
                + (0.18 if is_starter else -0.12)
                + (previous_month_avg_score - 6.15) * 0.18
                + (coach_trust_state - 0.82) * 0.30
                + (confidence_state - 0.80) * 0.36
                + (role_fit - 0.75) * 0.44
                + (profile["adaptability_bias"] - 0.80) * (0.12 if position_played != player.position else 0.05)
                + (availability_pct - 78.0) * 0.008
                + trajectory["performance_shift"] * 0.45
                + trajectory["confidence_shift"] * 0.24
                + trajectory["trust_shift"] * 0.20
                - fatigue_pct * (0.010 + float(profile["fatigue_sensitivity"]) * 0.006)
                - float(profile["plateau_risk"]) * max(0.0, progress_fraction - 0.62) * 0.60
                - (0.10 if growth_spurt else 0.0)
                + random.gauss(0.0, 0.45),
                1.0,
                10.0,
            )

            scoring_chance = max(0.0, (final_score - 5.5) / 5.0)
            goals = 0
            assists = 0
            if player.position == "Delantero":
                goals = 1 if random.random() < scoring_chance * 0.65 else 0
                assists = 1 if random.random() < scoring_chance * 0.25 else 0
            elif player.position == "Mediocampista":
                goals = 1 if random.random() < scoring_chance * 0.25 else 0
                assists = 1 if random.random() < scoring_chance * 0.45 else 0
            elif player.position == "Lateral":
                assists = 1 if random.random() < scoring_chance * 0.30 else 0

            match = Match(
                match_date=entry.record_date + timedelta(days=min(27, match_idx * 7 + random.randint(0, 5))),
                opponent_name=random.choice(OPPONENT_NAMES),
                opponent_level=opponent_level,
                tournament=random.choice(TOURNAMENTS),
                competition_category=random.choice(COMPETITION_CATEGORIES),
                venue=venue,
                notes="Dato sintetico de contexto de partido",
            )
            participation = PlayerMatchParticipation(
                player_id=player.id,
                match=match,
                started=is_starter,
                position_played=position_played,
                minutes_played=minutes_played,
                final_score=round(final_score, 2),
                goals=goals,
                assists=assists,
                pass_accuracy=round(pass_accuracy, 2),
                shot_accuracy=round(shot_accuracy, 2),
                duels_won_pct=round(duels_won_pct, 2),
                yellow_cards=random.randint(0, 1 if minutes_played < 45 else 2),
                red_cards=1 if random.random() < 0.012 else 0,
                role_notes="Dato sintetico de participacion por partido",
            )
            matches.append(match)
            participations.append(participation)
            monthly_participations.append(participation)
            confidence_state = clamp(
                confidence_state * 0.92 + (final_score - 6.35) * 0.020 + challenge_bonus * 0.03,
                0.35,
                1.45,
            )
            coach_trust_state = clamp(
                coach_trust_state * 0.94
                + (0.030 if is_starter else -0.010)
                + (final_score - 6.40) * 0.016
                + (role_fit - 0.75) * 0.030,
                0.35,
                1.50,
            )

        if not monthly_participations:
            continue

        monthly_avg_score = avg([float(item.final_score or 0.0) for item in monthly_participations])
        confidence_state = clamp(
            confidence_state * 0.72
            + float(profile["confidence_baseline"]) * 0.18
            + (monthly_avg_score - 6.30) * 0.11
            + momentum * 0.08
            + trajectory["confidence_shift"] * 0.20
            - (0.05 if missed_days >= 6 else 0.0)
            + (0.04 if profile["development_archetype"] == "late_bloomer" and progress_fraction >= 0.58 else 0.0),
            0.35,
            1.45,
        )
        coach_trust_state = clamp(
            coach_trust_state * 0.70
            + float(profile["coachability"]) * 0.16
            + (monthly_avg_score - 6.25) * 0.10
            + (availability_pct - 75.0) / 250.0
            + trajectory["trust_shift"] * 0.18
            - float(profile["plateau_risk"]) * max(0.0, progress_fraction - 0.65) * 0.08,
            0.35,
            1.50,
        )
        recent_role_fit = clamp(avg(monthly_role_fits), 0.35, 1.05) if monthly_role_fits else recent_role_fit
        previous_month_avg_score = monthly_avg_score

        stats.append(
            PlayerStat(
                player_id=player.id,
                record_date=entry.record_date,
                matches_played=len(monthly_participations),
                goals=sum(item.goals for item in monthly_participations),
                assists=sum(item.assists for item in monthly_participations),
                minutes_played=sum(item.minutes_played for item in monthly_participations),
                yellow_cards=sum(item.yellow_cards for item in monthly_participations),
                red_cards=sum(item.red_cards for item in monthly_participations),
                pass_accuracy=round(avg([float(item.pass_accuracy or 0.0) for item in monthly_participations]), 2),
                shot_accuracy=round(avg([float(item.shot_accuracy or 0.0) for item in monthly_participations]), 2),
                duels_won_pct=round(avg([float(item.duels_won_pct or 0.0) for item in monthly_participations]), 2),
                final_score=round(avg([float(item.final_score or 0.0) for item in monthly_participations]), 2),
                notes="Dato sintetico agregado desde participacion por partido",
            )
        )

    return matches, participations, stats


def synthetic_scout_reports(
    player: Player,
    attribute_history: List[PlayerAttributeHistory],
    stats: List[PlayerStat],
    physical_assessments: List[PhysicalAssessment],
    availability_history: List[PlayerAvailability],
    profile: Dict[str, float],
) -> List[ScoutReport]:
    if not attribute_history:
        return []

    report_stride = random.randint(2, 3)
    stats_by_date = {stat.record_date: stat for stat in stats}
    physical_by_date = {assessment.assessment_date: assessment for assessment in physical_assessments}
    availability_by_date = {row.record_date: row for row in availability_history}
    reports: List[ScoutReport] = []
    previous_weighted_score = None

    for idx, entry in enumerate(attribute_history):
        if idx != len(attribute_history) - 1 and ((idx + 1) % report_stride != 0):
            continue

        progress_fraction = (idx + 1) / float(max(1, len(attribute_history)))
        trajectory = trajectory_state(progress_fraction, profile)
        attrs = {field: float(getattr(entry, field) or ATTRIBUTE_MIN_VALUE) for field in ATTRIBUTE_FIELDS}
        weighted_score = weighted_score_from_attrs(attrs, player.position)
        recent_stat = stats_by_date.get(entry.record_date)
        physical = physical_by_date.get(entry.record_date)
        availability = availability_by_date.get(entry.record_date)
        recent_final_score = float(recent_stat.final_score) if recent_stat and recent_stat.final_score is not None else 6.0
        availability_pct = float(availability.availability_pct) if availability and availability.availability_pct is not None else float(profile["availability"]) * 100.0
        fatigue_pct = float(availability.fatigue_pct) if availability and availability.fatigue_pct is not None else 24.0
        endurance_score = float(physical.endurance) if physical and physical.endurance is not None else attrs["physical"]
        trend = 0.0 if previous_weighted_score is None else weighted_score - previous_weighted_score
        previous_weighted_score = weighted_score

        if player.position in ("Defensa", "Lateral"):
            tactical_base = attrs["defending"] * 0.28 + attrs["tackling"] * 0.28 + attrs["vision"] * 0.20
        elif player.position == "Mediocampista":
            tactical_base = attrs["vision"] * 0.30 + attrs["passing"] * 0.24 + attrs["technique"] * 0.20
        elif player.position == "Delantero":
            tactical_base = attrs["vision"] * 0.24 + attrs["shooting"] * 0.24 + attrs["technique"] * 0.20
        else:
            tactical_base = attrs["vision"] * 0.20 + attrs["physical"] * 0.18 + attrs["technique"] * 0.18

        decision_making = int(
            round(
                clamp(
                    attrs["vision"] * 0.34
                    + attrs["technique"] * 0.26
                    + attrs["determination"] * 0.22
                    + recent_final_score * 0.45
                    + random.gauss(0.0, 1.0),
                    float(ATTRIBUTE_MIN_VALUE),
                    float(ATTRIBUTE_MAX_VALUE),
                )
            )
        )
        tactical_reading = int(
            round(
                clamp(
                    tactical_base + recent_final_score * 0.40 + random.gauss(0.0, 1.1),
                    float(ATTRIBUTE_MIN_VALUE),
                    float(ATTRIBUTE_MAX_VALUE),
                )
            )
        )
        mental_profile = int(
            round(
                clamp(
                    attrs["determination"] * 0.62
                    + attrs["physical"] * 0.16
                    + (availability_pct / 100.0) * 4.0
                    + profile["resilience"] * 2.2
                    + profile["pressure_resistance"] * 1.6
                    + endurance_score * 0.10
                    + recent_final_score * 0.32
                    + trajectory["confidence_shift"] * 1.1
                    - profile["plateau_risk"] * 2.0
                    - fatigue_pct * 0.03
                    + random.gauss(0.0, 1.0),
                    float(ATTRIBUTE_MIN_VALUE),
                    float(ATTRIBUTE_MAX_VALUE),
                )
            )
        )
        adaptability = int(
            round(
                clamp(
                    attrs["technique"] * 0.28
                    + attrs["vision"] * 0.24
                    + attrs["pace"] * 0.14
                    + attrs["physical"] * 0.14
                    + profile["adaptability_bias"] * 2.4
                    + profile["coachability"] * 1.5
                    + recent_final_score * 0.28
                    + random.gauss(0.0, 1.1),
                    float(ATTRIBUTE_MIN_VALUE),
                    float(ATTRIBUTE_MAX_VALUE),
                )
            )
        )
        observed_projection_score = clamp(
            2.75
            + (weighted_score / 20.0) * 5.2
            + age_potential_bonus(player.age) * 0.30
            + trend * 0.75
            + (recent_final_score - 6.0) * 0.35
            + (profile["professionalism"] - 0.85) * 0.45
            + (profile["pressure_resistance"] - 0.85) * 0.35
            + (profile["coachability"] - 0.85) * 0.28
            + trajectory["scout_shift"]
            + trajectory["performance_shift"] * 0.18
            - max(0.0, fatigue_pct - 65.0) * 0.01
            + random.gauss(0.0, 0.35),
            1.0,
            10.0,
        )

        trend_label = "progreso sostenido" if trend >= 0.15 else ("estancado" if trend <= -0.10 else "evolucion gradual")
        reports.append(
            ScoutReport(
                player_id=player.id,
                report_date=entry.record_date,
                decision_making=decision_making,
                tactical_reading=tactical_reading,
                mental_profile=mental_profile,
                adaptability=adaptability,
                observed_projection_score=round(observed_projection_score, 2),
                notes=f"Reporte sintetico: {trend_label} con contexto de rendimiento reciente.",
            )
        )

    return reports


def main(
    num_players: int,
    db_url: str,
    seed: int = DEFAULT_SEED,
    min_age: int = EVAL_MIN_AGE,
    max_age: int = EVAL_MAX_AGE,
    reset_existing: bool = False,
) -> None:
    """Genera `num_players` jugadores y sus historiales coherentes."""
    random.seed(seed)
    used_identifiers.clear()
    normalized_db_url = normalize_db_url(db_url, base_dir=os.path.dirname(os.path.abspath(__file__)))
    engine = create_app_engine(normalized_db_url)
    if reset_existing:
        Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    ensure_player_columns(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    load_existing_identifiers(session)

    batch_size = 500
    created = 0
    while created < num_players:
        remaining = num_players - created
        current_batch = min(batch_size, remaining)
        players_batch = [generate_player(min_age=min_age, max_age=max_age) for _ in range(current_batch)]
        session.add_all(players_batch)
        session.flush()

        attribute_history_batch: List[PlayerAttributeHistory] = []
        physical_assessment_batch: List[PhysicalAssessment] = []
        availability_batch: List[PlayerAvailability] = []
        matches_batch: List[Match] = []
        participations_batch: List[PlayerMatchParticipation] = []
        stats_batch: List[PlayerStat] = []
        scout_reports_batch: List[ScoutReport] = []
        for player in players_batch:
            profile = build_development_profile(player)
            attribute_history = synthetic_attribute_history(player, profile)
            for entry in attribute_history:
                entry.player_id = player.id
            attribute_history_batch.extend(attribute_history)

            physical_assessments = synthetic_physical_assessments(player, attribute_history, profile)
            availability_rows = synthetic_availability_history(player, attribute_history, physical_assessments, profile)
            physical_assessment_batch.extend(physical_assessments)
            availability_batch.extend(availability_rows)

            generated_matches, generated_participations, generated_stats = synthetic_matches_and_stats(
                player,
                attribute_history,
                physical_assessments,
                availability_rows,
                profile,
            )
            generated_reports = synthetic_scout_reports(
                player,
                attribute_history,
                generated_stats,
                physical_assessments,
                availability_rows,
                profile,
            )

            matches_batch.extend(generated_matches)
            participations_batch.extend(generated_participations)
            stats_batch.extend(generated_stats)
            scout_reports_batch.extend(generated_reports)

        if attribute_history_batch:
            session.add_all(attribute_history_batch)
        if physical_assessment_batch:
            session.add_all(physical_assessment_batch)
        if availability_batch:
            session.add_all(availability_batch)
        if matches_batch:
            session.add_all(matches_batch)
        if participations_batch:
            session.add_all(participations_batch)
        if stats_batch:
            session.add_all(stats_batch)
        if scout_reports_batch:
            session.add_all(scout_reports_batch)
        session.commit()
        created += current_batch

    print(
        f"Se generaron {num_players} jugadores en la base de datos: {normalized_db_url} "
        f"(seed={seed}, edades={min_age}-{max_age}, reset={reset_existing})"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Genera jugadores aleatorios")
    parser.add_argument("--num-players", type=int, default=1000, help="Numero de jugadores a crear")
    parser.add_argument("--db-url", type=str, default="sqlite:///players_training.db", help="URL de la base de datos")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Semilla para reproduccion de datos")
    parser.add_argument("--min-age", type=int, default=EVAL_MIN_AGE, help="Edad minima de generacion")
    parser.add_argument("--max-age", type=int, default=EVAL_MAX_AGE, help="Edad maxima de generacion")
    parser.add_argument("--reset", action="store_true", help="Recrea la base de entrenamiento antes de generar")
    args = parser.parse_args()
    main(
        args.num_players,
        args.db_url,
        seed=args.seed,
        min_age=args.min_age,
        max_age=args.max_age,
        reset_existing=args.reset,
    )
