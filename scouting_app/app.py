import os
import logging
import time
import secrets
import sys
import subprocess
from typing import Any, List, Tuple, Callable, Optional, Dict
from datetime import datetime, date, timedelta
from statistics import mean
from types import SimpleNamespace
from flask import Flask, render_template, redirect, url_for, request, session, flash, abort, jsonify
from sqlalchemy import func, select
from sqlalchemy.orm import Session as SQLAlchemySession, sessionmaker, load_only
import numpy as np
import pandas as pd
import torch
from models import (
    Base,
    Match,
    PhysicalAssessment,
    Player,
    Coach,
    Director,
    User,
    PlayerAvailability,
    PlayerStat,
    PlayerAttributeHistory,
    PlayerMatchParticipation,
    ScoutReport,
)
from werkzeug.security import generate_password_hash
from functools import wraps
from db_utils import normalize_db_url, create_app_engine, ensure_player_columns, is_sqlite_url
from ml.runtime import (
    apply_probability_calibrator,
    load_model_state,
    load_probability_calibrator,
    load_runtime_artifacts,
)
from preprocessing import (
    aggregate_attribute_history_dataframe,
    aggregate_availability_dataframe,
    aggregate_match_participation_dataframe,
    aggregate_physical_assessment_dataframe,
    aggregate_scout_report_dataframe,
    aggregate_stats_dataframe,
    availability_feature_defaults,
    attribute_history_feature_defaults,
    dataframe_from_players,
    match_feature_defaults,
    physical_feature_defaults,
    player_base_dataframe_from_players,
    scout_report_feature_defaults,
    stats_feature_defaults,
    transform_features,
)
from player_logic import (
    ATTRIBUTE_FIELDS,
    ATTRIBUTE_LABELS,
    ATTRIBUTE_MAX_VALUE,
    ATTRIBUTE_MIN_VALUE,
    POSITION_CHOICES,
    normalized_position,
    position_weights,
    recommend_position_from_attrs,
    weighted_score_from_attrs,
    is_valid_attribute,
    is_valid_eval_age,
    default_player_photo_url,
)
from services.cache import TTLCache
from services.locks import PipelineFileLock
from services.operational_data import (
    backfill_player_photo_urls,
    cleanup_operational_data as _cleanup_operational_data,
    compute_operational_data_quality as _compute_operational_data_quality,
    sync_attribute_history_baseline,
    sync_player_attribute_history,
    trim_operational_player_pool as _trim_operational_player_pool,
)
from services.security import (
    LoginRateLimiter,
    client_ip_from_request,
    csrf_token as service_csrf_token,
    require_csrf as service_require_csrf,
)
from routes import register_legacy_endpoint_aliases
from routes.auth import create_auth_blueprint
from routes.compare import create_compare_blueprint
from routes.dashboard import create_dashboard_blueprint
from routes.players import create_players_blueprint
from routes.settings import create_settings_blueprint
from routes.staff import create_staff_blueprint

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)

# Límite de payload para requests (mitiga abusos y errores por uploads grandes)
# Default: 2MB. Ajustable por env var `MAX_CONTENT_LENGTH`.
app.config["MAX_CONTENT_LENGTH"] = int(os.environ.get("MAX_CONTENT_LENGTH", str(2 * 1024 * 1024)))


# --- Observabilidad mínima ---
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
try:
    app.logger.setLevel(LOG_LEVEL)
except Exception:
    app.logger.setLevel(logging.INFO)

# Logger root para librerías (opcional)
logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO))


def env_flag(name: str, default: str = "0") -> bool:
    return (os.environ.get(name, default).strip().lower() in {"1", "true", "yes", "y", "si", "s", "on"})


def is_production_runtime() -> bool:
    env = (os.environ.get("FLASK_ENV") or os.environ.get("ENV") or "").strip().lower()
    return env in {"production", "prod"} or env_flag("RENDER")


# --- Cache in-memory con TTL (MVP) ---
DEFAULT_CACHE_TTL_SECONDS = 60
DEFAULT_CACHE_MAX_ENTRIES = 128
DEFAULT_PLAYER_LIST_PER_PAGE = 50
DEFAULT_MAX_COMPARE_PLAYERS = 2000
DEFAULT_EVAL_POOL_MAX = 100
TRAINING_DATASET_DEFAULT_PLAYERS = 20000
PIPELINE_LOCK_MIN_STALE_SECONDS = 60
PIPELINE_LOCK_DEFAULT_STALE_SECONDS = 12 * 60 * 60

MATCH_MINUTES_DENOMINATOR = 90.0
MAX_MINUTES_FACTOR = 1.5
MAX_GOALS_RATED = 3
MAX_ASSISTS_RATED = 3
MAX_MATCHES_RATED = 3
BASE_STATS_RATING = 1.5
GOAL_RATING_WEIGHT = 1.5
ASSIST_RATING_WEIGHT = 1.2
PASS_ACCURACY_RATING_WEIGHT = 2.0
SHOT_ACCURACY_RATING_WEIGHT = 1.5
DUEL_ACCURACY_RATING_WEIGHT = 1.3
MATCH_CONSISTENCY_WEIGHT = 0.5
PERCENT_DENOMINATOR = 100.0
MIN_STATS_RATING = 1.0
MAX_STATS_RATING = 10.0
DEFAULT_SUGGESTION_THRESHOLD = 14
DEFAULT_SUGGESTION_COUNT = 3
SCORE_BAND_HIGH_THRESHOLD = 15
SCORE_BAND_MEDIUM_THRESHOLD = 10

try:
    _CACHE_TTL_SECONDS = max(1, int(os.environ.get("CACHE_TTL_SECONDS", str(DEFAULT_CACHE_TTL_SECONDS))))
except ValueError:
    _CACHE_TTL_SECONDS = DEFAULT_CACHE_TTL_SECONDS
try:
    _CACHE_MAX_ENTRIES = max(1, int(os.environ.get("CACHE_MAX_ENTRIES", str(DEFAULT_CACHE_MAX_ENTRIES))))
except ValueError:
    _CACHE_MAX_ENTRIES = DEFAULT_CACHE_MAX_ENTRIES
try:
    PLAYER_LIST_PER_PAGE = max(1, int(os.environ.get("PLAYER_LIST_PER_PAGE", str(DEFAULT_PLAYER_LIST_PER_PAGE))))
except ValueError:
    PLAYER_LIST_PER_PAGE = DEFAULT_PLAYER_LIST_PER_PAGE
try:
    MAX_COMPARE_PLAYERS = max(1, int(os.environ.get("MAX_COMPARE_PLAYERS", str(DEFAULT_MAX_COMPARE_PLAYERS))))
except ValueError:
    MAX_COMPARE_PLAYERS = DEFAULT_MAX_COMPARE_PLAYERS
_LOGIN_RATE_LIMIT_WINDOW_SECONDS = int(os.environ.get("LOGIN_RATE_LIMIT_WINDOW_SECONDS", "900"))
_LOGIN_RATE_LIMIT_MAX_ATTEMPTS = int(os.environ.get("LOGIN_RATE_LIMIT_MAX_ATTEMPTS", "5"))
_DASHBOARD_CACHE = TTLCache(_CACHE_TTL_SECONDS, _CACHE_MAX_ENTRIES)
_CACHE = _DASHBOARD_CACHE.store
_LOGIN_RATE_LIMITER = LoginRateLimiter(_LOGIN_RATE_LIMIT_WINDOW_SECONDS, _LOGIN_RATE_LIMIT_MAX_ATTEMPTS)
_LOGIN_ATTEMPTS = _LOGIN_RATE_LIMITER.attempts


# --- Guardrails pipeline (evita doble ejecucion concurrente) ---
try:
    _PIPELINE_LOCK_STALE_SECONDS = max(
        PIPELINE_LOCK_MIN_STALE_SECONDS,
        int(os.environ.get("PIPELINE_LOCK_STALE_SECONDS", str(PIPELINE_LOCK_DEFAULT_STALE_SECONDS))),
    )
except ValueError:
    _PIPELINE_LOCK_STALE_SECONDS = PIPELINE_LOCK_DEFAULT_STALE_SECONDS
_PIPELINE_LOCK = PipelineFileLock(
    os.environ.get("PIPELINE_LOCK_PATH", os.path.join(BASE_DIR, ".pipeline_update.lock")),
    stale_seconds=_PIPELINE_LOCK_STALE_SECONDS,
)


def _cache_get(key: str) -> Optional[Any]:
    return _DASHBOARD_CACHE.get(key)


def _cache_prune_expired(now_ts: Optional[float] = None) -> None:
    _DASHBOARD_CACHE.prune_expired(now_ts)


def _cache_set(key: str, value: Any) -> None:
    _DASHBOARD_CACHE.ttl_seconds = max(1, int(_CACHE_TTL_SECONDS))
    _DASHBOARD_CACHE.max_entries = max(1, int(_CACHE_MAX_ENTRIES))
    _DASHBOARD_CACHE.set(key, value)


def _cache_invalidate_prefix(prefix: str) -> None:
    _DASHBOARD_CACHE.invalidate_prefix(prefix)


def invalidate_dashboard_cache() -> None:
    _cache_invalidate_prefix("dashboard:")
    _cache_invalidate_prefix("players:")
    _cache_invalidate_prefix("compare:")


def client_ip() -> str:
    return client_ip_from_request(request)


def login_attempt_key(username: Optional[str]) -> str:
    return _LOGIN_RATE_LIMITER.key(username, client_ip())


def _prune_login_attempts(now_ts: Optional[float] = None) -> None:
    _LOGIN_RATE_LIMITER.prune(now_ts)


def is_login_rate_limited(username: Optional[str]) -> bool:
    return _LOGIN_RATE_LIMITER.is_limited(username, client_ip())


def register_failed_login(username: Optional[str]) -> None:
    _LOGIN_RATE_LIMITER.register_failure(username, client_ip())


def clear_failed_logins(username: Optional[str]) -> None:
    _LOGIN_RATE_LIMITER.clear(username, client_ip())


# --- Secret key obligatoria en producción ---
_is_prod_runtime = is_production_runtime()
_secret = os.environ.get("APP_SECRET_KEY", "")
if _is_prod_runtime and (not _secret or _secret == "reemplazar-esta-clave"):
    raise RuntimeError("APP_SECRET_KEY must be set in production")
if _secret and _secret != "reemplazar-esta-clave":
    app.secret_key = _secret
else:
    app.secret_key = secrets.token_urlsafe(32)
    if not _is_prod_runtime:
        app.logger.warning("APP_SECRET_KEY no configurada. Se genero una clave efimera para esta ejecucion.")


# --- Cookies de sesión (hardening mínimo) ---
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
if _is_prod_runtime:
    app.config["SESSION_COOKIE_SECURE"] = True


@app.after_request
def add_security_headers(response):
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("Referrer-Policy", "same-origin")
    return response


# --- CSRF mínimo (session token) ---
def _csrf_token() -> str:
    return service_csrf_token(session)

@app.context_processor
def inject_csrf_token() -> Dict[str, Callable[[], str]]:
    return {"csrf_token": _csrf_token()}


def _require_csrf() -> None:
    # Solo para endpoints críticos (llamar manualmente en POST)
    service_require_csrf(request.form, request.headers, session)

APP_DB_URL = normalize_db_url(
    os.environ.get("APP_DB_URL", "sqlite:///players_updated_v2.db"),
    base_dir=BASE_DIR,
)
if APP_DB_URL.rsplit("/", 1)[-1] == "players.db":
    app.logger.warning("APP_DB_URL apunta a players.db (legacy). Se recomienda players_updated_v2.db.")

TRAINING_DB_URL = normalize_db_url(
    os.environ.get("TRAINING_DB_URL", "sqlite:///players_training.db"),
    base_dir=BASE_DIR,
)
if _is_prod_runtime and not env_flag("ALLOW_SQLITE_IN_PRODUCTION") and (
    is_sqlite_url(APP_DB_URL) or is_sqlite_url(TRAINING_DB_URL)
):
    raise RuntimeError(
        "SQLite no esta permitido en produccion sin ALLOW_SQLITE_IN_PRODUCTION=1. "
        "Configurar APP_DB_URL y TRAINING_DB_URL con PostgreSQL para evitar perdida de datos."
    )
try:
    EVAL_POOL_MAX = max(1, int(os.environ.get("EVAL_POOL_MAX", str(DEFAULT_EVAL_POOL_MAX))))
except ValueError:
    EVAL_POOL_MAX = DEFAULT_EVAL_POOL_MAX

SYNC_SHORTLIST_ENABLED = env_flag("SYNC_SHORTLIST_ENABLED")

ENFORCE_EVAL_POOL_LIMIT = env_flag("ENFORCE_EVAL_POOL_LIMIT", "1")

engine = create_app_engine(APP_DB_URL)
Session = sessionmaker(bind=engine, expire_on_commit=False)
Base.metadata.create_all(engine)
ensure_player_columns(engine)


def trim_operational_player_pool(db_session: SQLAlchemySession, max_players: int = EVAL_POOL_MAX) -> int:
    return _trim_operational_player_pool(db_session, max_players)


def compute_operational_data_quality(db_session: SQLAlchemySession) -> Dict[str, int]:
    return _compute_operational_data_quality(db_session, EVAL_POOL_MAX)


def cleanup_operational_data(db_session: SQLAlchemySession) -> Dict[str, int]:
    return _cleanup_operational_data(
        db_session,
        eval_pool_max=EVAL_POOL_MAX,
        enforce_eval_pool_limit=ENFORCE_EVAL_POOL_LIMIT,
        sync_history=sync_attribute_history_baseline,
    )


def enforce_operational_pool_limit_on_startup() -> None:
    if not ENFORCE_EVAL_POOL_LIMIT:
        return
    db_session = Session()
    try:
        removed = trim_operational_player_pool(db_session, EVAL_POOL_MAX)
        if removed:
            db_session.commit()
            app.logger.warning(
                "Base operativa recortada a %s jugadores (eliminados: %s).",
                EVAL_POOL_MAX,
                removed,
            )
        photo_updates = backfill_player_photo_urls(db_session)
        if photo_updates:
            db_session.commit()
            app.logger.info("Fotos de jugadores completadas: %s", photo_updates)
        history_updates = sync_attribute_history_baseline(db_session)
        if history_updates:
            db_session.commit()
            app.logger.info("Historial tecnico sincronizado desde ficha actual: %s jugadores", history_updates)
    except Exception:
        db_session.rollback()
        app.logger.exception("No se pudo aplicar el limite de jugadores operativos al iniciar.")
    finally:
        db_session.close()


def run_subprocess(command: List[str], description: str) -> Tuple[bool, str]:
    """Ejecuta un comando externo y devuelve (exito, mensaje)."""
    try:
        result = subprocess.run(
            command,
            cwd=BASE_DIR,
            capture_output=True,
            text=True,
            check=False,
        )
        output = result.stdout.strip()
        error = result.stderr.strip()
        log = f"[{description}] {'OK' if result.returncode == 0 else 'ERROR'}"
        if output:
            log += f"\n{output}"
        if error:
            log += f"\n{error}"
        return result.returncode == 0, log
    except Exception as exc:
        return False, f"[{description}] ERROR: {exc}"


def ensure_training_dataset(min_players: int = 1) -> Tuple[bool, List[str]]:
    """Verifica que la BD de entrenamiento tenga datos; genera si esta vacia."""
    logs: List[str] = []
    training_engine = create_app_engine(TRAINING_DB_URL)
    TrainingSession = sessionmaker(bind=training_engine)
    Base.metadata.create_all(training_engine)
    ensure_player_columns(training_engine)
    training_session = TrainingSession()
    try:
        count = training_session.query(func.count(Player.id)).scalar() or 0
    finally:
        training_session.close()
    logs.append(f"Jugadores disponibles en base de referencia: {count}")
    if count >= min_players:
        return True, logs
    cmd = [
        sys.executable,
        "generate_data.py",
        "--num-players",
        str(TRAINING_DATASET_DEFAULT_PLAYERS),
        "--db-url",
        TRAINING_DB_URL,
    ]
    success, log = run_subprocess(cmd, "Preparar base de referencia")
    logs.append(log)
    return success, logs


def update_database_pipeline(limit: int = EVAL_POOL_MAX, sync_shortlist: bool = SYNC_SHORTLIST_ENABLED) -> Tuple[bool, List[str]]:
    """Ejecuta entrenamiento del modelo y sincronizacion opcional para la base operativa."""
    overall_logs: List[str] = []
    ok, logs = ensure_training_dataset()
    overall_logs.extend(logs)
    if not ok:
        return False, overall_logs
    train_cmd = [
        sys.executable,
        "train_model.py",
        "--db-url",
        TRAINING_DB_URL,
        "--model-out",
        MODEL_PATH,
        "--preprocessor-out",
        PREPROCESSOR_PATH,
        "--calibrator-out",
        CALIBRATOR_PATH,
        "--metadata-out",
        TRAINING_METADATA_PATH,
        "--splits-out",
        TRAINING_SPLITS_PATH,
        "--epochs",
        "30",
    ]
    ok, train_log = run_subprocess(train_cmd, "Actualizacion de puntajes")
    overall_logs.append(train_log)
    if not ok:
        return False, overall_logs

    if sync_shortlist:
        ensure_player_columns(engine)
        sync_cmd = [
            sys.executable,
            "sync_shortlist.py",
            "--src-db",
            TRAINING_DB_URL,
            "--dst-db",
            APP_DB_URL,
            "--limit",
            str(limit),
        ]
        ok, sync_log = run_subprocess(sync_cmd, "Sincronizacion de base operativa")
        overall_logs.append(sync_log)
        if not ok:
            return False, overall_logs
    else:
        overall_logs.append("Sincronizacion de base operativa omitida.")

    # Guardrail final: la base operativa no debe superar EVAL_POOL_MAX.
    db_session = Session()
    try:
        removed = trim_operational_player_pool(db_session, EVAL_POOL_MAX)
        if removed:
            db_session.commit()
            overall_logs.append(f"Base operativa recortada a {EVAL_POOL_MAX} jugadores (eliminados {removed}).")
        else:
            db_session.rollback()
            overall_logs.append(f"Base operativa dentro del limite ({EVAL_POOL_MAX} jugadores).")
    except Exception as exc:
        db_session.rollback()
        overall_logs.append(f"No se pudo aplicar el recorte de base operativa: {exc}")
        return False, overall_logs
    finally:
        db_session.close()

    try:
        global model, preprocessor, probability_calibrator
        model, preprocessor, probability_calibrator = load_runtime_artifacts(
            MODEL_PATH,
            PREPROCESSOR_PATH,
            CALIBRATOR_PATH,
            allow_retrain=False,
            update_callback=update_database_pipeline,
        )
        invalidate_dashboard_cache()
        overall_logs.append("Modelo, preprocesador y calibrador recargados en memoria; cache del dashboard invalidado.")
    except Exception as exc:
        overall_logs.append(f"No se pudieron recargar los artefactos del modelo despues del entrenamiento: {exc}")
        return False, overall_logs

    return True, overall_logs

# ----------------------------------------------------
# Usuario administrador inicial (opcional en desarrollo)
def init_admin_user() -> None:
    username = (os.environ.get("ADMIN_USERNAME") or "admin").strip()
    password = (os.environ.get("ADMIN_PASSWORD") or "").strip()
    allow_default = env_flag("ALLOW_DEFAULT_ADMIN")
    is_prod = is_production_runtime()

    if not password:
        # En produccion no crear admin por defecto.
        if is_prod:
            app.logger.warning("ADMIN_PASSWORD no configurado. No se crea usuario admin inicial en produccion.")
            return
        # En desarrollo, solo permitir admin por defecto si se habilita explicito.
        if not allow_default:
            app.logger.info(
                "ADMIN_PASSWORD no configurado. Se omite bootstrap de admin. "
                "Setear ADMIN_PASSWORD o ALLOW_DEFAULT_ADMIN=true para desarrollo local."
            )
            return
        password = "admin"

    if len(username) < 3 or len(password) < 6:
        app.logger.warning("Credenciales de bootstrap invalidas. No se crea usuario admin inicial.")
        return
    if not is_strong_password(password):
        app.logger.warning(
            "La contraseña de bootstrap no cumple el minimo de seguridad (8+ caracteres, letra y numero)."
        )
        return

    db_session = Session()
    existing = db_session.query(User).filter(User.username == username).first()
    if not existing:
        user = User(username=username,
                    password_hash=generate_password_hash(password),
                    role="administrador")
        db_session.add(user)
        db_session.commit()
    db_session.close()

# ----------------------------------------------------
# Landing
@app.route("/")
def landing():
    db = Session()
    total_players = db.query(func.count(Player.id)).scalar() or 0
    age_values = [
        player.current_age
        for player in db.query(Player).options(load_only(Player.age, Player.birth_date)).all()
        if player.current_age is not None
    ]
    avg_age = sum(age_values) / len(age_values) if age_values else None
    countries = db.query(Player.country).distinct().count()
    positions = db.query(Player.position).distinct().count()
    db.close()

    metrics = {
        "total_players": int(total_players),
        "avg_age": float(avg_age) if avg_age else None,
        "countries": countries,
        "positions": positions,
    }

    if session.get("user_id"):
        call_to_action_url = url_for("index")
        call_to_action_label = "Ir al panel"
    else:
        call_to_action_url = url_for("login")
        call_to_action_label = "Iniciar sesión"

    return render_template(
        "landing.html",
        metrics=metrics,
        call_to_action_url=call_to_action_url,
        call_to_action_label=call_to_action_label,
    )

# ----------------------------------------------------

@app.context_processor
def navbar_url_helpers() -> Dict[str, Callable[..., str]]:
    def first_url(*endpoints: str, **values: Any) -> str:
        """Devuelve la primera URL válida entre una lista de endpoints.
        Si ninguno existe, devuelve '#'.
        """
        for ep in endpoints:
            try:
                return url_for(ep, **values)
            except Exception:
                continue
        return "#"
    return dict(first_url=first_url)

@app.context_processor
def auth_flags() -> Dict[str, object]:
    role = current_role()
    return {
        "is_authenticated": bool(session.get("user_id")),
        "current_username": session.get("username", "admin"),
        "current_role": role,
        "current_role_label": role_display_label(role),
        "is_admin": role == "administrador",
        "can_edit_player_data": can_edit_player_data(),
        "can_manage_players": can_manage_players(),
        "can_manage_staff": can_manage_staff(),
        "can_manage_users": can_manage_users(),
    }


def display_position_label(value: Optional[str]) -> str:
    normalized = normalized_position(value)
    return "Arquero" if normalized == "Portero" else normalized


def is_strong_password(password: str) -> bool:
    if len(password or "") < 8:
        return False
    has_letter = any(char.isalpha() for char in password)
    has_digit = any(char.isdigit() for char in password)
    return has_letter and has_digit


init_admin_user()


ROLE_ADMIN = "administrador"
ROLE_SCOUT = "scout"
ROLE_DIRECTOR = "director"
ROLE_ALIASES = {
    "admin": ROLE_ADMIN,
    "administrador": ROLE_ADMIN,
    "scout": ROLE_SCOUT,
    "ojeador": ROLE_SCOUT,
    "director": ROLE_DIRECTOR,
}


def normalize_role(role: Optional[str]) -> str:
    raw = (role or "").strip().lower()
    return ROLE_ALIASES.get(raw, ROLE_SCOUT)


def current_role() -> str:
    return normalize_role(session.get("role"))


def role_display_label(role: Optional[str]) -> str:
    normalized = normalize_role(role)
    return {
        ROLE_ADMIN: "Administrador",
        ROLE_SCOUT: "Scout",
        ROLE_DIRECTOR: "Director",
    }.get(normalized, "Scout")


def has_any_role(*roles: str) -> bool:
    expected = {normalize_role(role) for role in roles}
    return current_role() in expected


def can_edit_player_data() -> bool:
    return bool(session.get("user_id")) and has_any_role(ROLE_ADMIN, ROLE_SCOUT)


def can_manage_players() -> bool:
    return can_edit_player_data()


def can_manage_staff() -> bool:
    return bool(session.get("user_id")) and has_any_role(ROLE_ADMIN)


def can_manage_users() -> bool:
    return bool(session.get("user_id")) and has_any_role(ROLE_ADMIN)


@app.context_processor
def position_labels() -> Dict[str, Callable[[Optional[str]], str]]:
    return {"display_position": display_position_label}

# Decorador de login
def login_required(view_func: Callable) -> Callable:
    @wraps(view_func)
    def wrapper(*args, **kwargs):
        if not session.get('user_id'):
            return redirect(url_for('login', next=request.url))
        return view_func(*args, **kwargs)
    return wrapper


def roles_required(*roles: str) -> Callable:
    normalized_roles = {normalize_role(role) for role in roles}

    def decorator(view_func: Callable) -> Callable:
        @wraps(view_func)
        def wrapper(*args, **kwargs):
            if not session.get('user_id'):
                return redirect(url_for('login', next=request.url))
            if current_role() not in normalized_roles:
                abort(403)
            return view_func(*args, **kwargs)

        return wrapper

    return decorator


auth_blueprint = create_auth_blueprint(
    Session=Session,
    User=User,
    require_csrf=_require_csrf,
    is_login_rate_limited=is_login_rate_limited,
    clear_failed_logins=clear_failed_logins,
    register_failed_login=register_failed_login,
    normalize_role=normalize_role,
    roles_required=roles_required,
    is_strong_password=is_strong_password,
    role_admin=ROLE_ADMIN,
    role_scout=ROLE_SCOUT,
    role_director=ROLE_DIRECTOR,
)
app.register_blueprint(auth_blueprint)
register_legacy_endpoint_aliases(
    app,
    "auth",
    (
        ("login", "/login", ("GET", "POST")),
        ("logout", "/logout", ("POST",)),
        ("register", "/register", ("GET", "POST")),
    ),
)

staff_blueprint = create_staff_blueprint(
    Session=Session,
    Coach=Coach,
    Director=Director,
    require_csrf=_require_csrf,
    login_required=login_required,
    roles_required=roles_required,
    role_admin=ROLE_ADMIN,
)
app.register_blueprint(staff_blueprint)
register_legacy_endpoint_aliases(
    app,
    "staff",
    (
        ("list_coaches", "/coaches", ("GET",)),
        ("new_coach", "/coaches/new", ("GET", "POST")),
        ("edit_coach", "/coaches/edit/<int:coach_id>", ("GET", "POST")),
        ("delete_coach", "/coaches/delete/<int:coach_id>", ("POST",)),
        ("list_directors", "/directors", ("GET",)),
        ("new_director", "/directors/new", ("GET", "POST")),
        ("edit_director", "/directors/edit/<int:director_id>", ("GET", "POST")),
        ("delete_director", "/directors/delete/<int:director_id>", ("POST",)),
    ),
)

MODEL_PATH = os.path.join(BASE_DIR, "model.pt")
PREPROCESSOR_PATH = os.path.join(BASE_DIR, "preprocessor.joblib")
CALIBRATOR_PATH = os.path.join(BASE_DIR, "probability_calibrator.joblib")
TRAINING_METADATA_PATH = os.path.join(BASE_DIR, "training_metadata.json")
TRAINING_SPLITS_PATH = os.path.join(BASE_DIR, "training_splits.json")
AUTO_TRAIN_ON_STARTUP = (os.environ.get("AUTO_TRAIN_ON_STARTUP") or "").strip().lower() in {
    "1", "true", "yes", "y", "si", "s", "on"
}
try:
    model, preprocessor, probability_calibrator = load_runtime_artifacts(
        MODEL_PATH,
        PREPROCESSOR_PATH,
        CALIBRATOR_PATH,
        allow_retrain=AUTO_TRAIN_ON_STARTUP,
        update_callback=update_database_pipeline,
    )
except (FileNotFoundError, RuntimeError):
    model = None
    preprocessor = None
    probability_calibrator = None
    print(
        "Advertencia: modelo o preprocesador no encontrados. "
        "Ejecute la corrida oficial del MVP o habilite AUTO_TRAIN_ON_STARTUP=true."
    )

def legacy_health_endpoint():
    """Healthcheck básico: app viva + conectividad DB."""
    try:
        # Validación mínima de conectividad
        with engine.connect() as conn:
            conn.exec_driver_sql("SELECT 1")
        return jsonify({"status": "ok"}), 200
    except Exception as e:
        app.logger.exception("Healthcheck failed")
        return jsonify({"status": "error", "detail": str(e)}), 500

@app.route("/health")
def health():
    """Healthcheck basico: app viva + conectividad DB + calidad operativa."""
    db = Session()
    try:
        with engine.connect() as conn:
            conn.exec_driver_sql("SELECT 1")
        data_quality = compute_operational_data_quality(db)
        return jsonify(
            {
                "status": "ok",
                "database": "ok",
                "data_quality": data_quality,
                "limits": {
                    "eval_pool_max": EVAL_POOL_MAX,
                },
            }
        ), 200
    except Exception as e:
        app.logger.exception("Healthcheck failed")
        return jsonify({"status": "error", "detail": str(e)}), 500
    finally:
        db.close()


def prepare_input(player: Player) -> torch.Tensor:
    return players_to_model_tensor([player])

def compute_suggestions(
    player: Player,
    threshold: int = DEFAULT_SUGGESTION_THRESHOLD,
    top_n: int = DEFAULT_SUGGESTION_COUNT,
) -> List[Tuple[str, int]]:
    attrs = {
        "Ritmo (pace)": player.pace,
        "Disparo (shooting)": player.shooting,
        "Pase (passing)": player.passing,
        "Regate (dribbling)": player.dribbling,
        "Defensa (defending)": player.defending,
        "Físico (physical)": player.physical,
        "Visión (vision)": player.vision,
        "Marcaje (tackling)": player.tackling,
        "Determinación (determination)": player.determination,
        "Técnica (technique)": player.technique,
    }
    gaps = {name: threshold - value for name, value in attrs.items() if value < threshold}
    return sorted(gaps.items(), key=lambda item: item[1], reverse=True)[:top_n]


def score_band(score: float) -> str:
    if score >= SCORE_BAND_HIGH_THRESHOLD:
        return "Alto"
    if score >= SCORE_BAND_MEDIUM_THRESHOLD:
        return "Medio"
    return "Bajo"

POSITION_OPTIONS = POSITION_CHOICES


def player_attribute_map(player: Player) -> Dict[str, int]:
    return {field: getattr(player, field) for field in ATTRIBUTE_FIELDS}


def fetch_player_stat_feature_map(player_ids: List[int]) -> Dict[int, Dict[str, Optional[float]]]:
    if not player_ids:
        return {}
    query = (
        select(
            PlayerStat.player_id,
            PlayerStat.record_date,
            PlayerStat.pass_accuracy,
            PlayerStat.final_score,
        )
        .where(PlayerStat.player_id.in_(player_ids))
    )
    with engine.connect() as connection:
        stats_df = pd.read_sql(query, connection)
    aggregated_df = aggregate_stats_dataframe(stats_df)
    if aggregated_df.empty:
        return {}
    feature_map: Dict[int, Dict[str, Optional[float]]] = {}
    for row in aggregated_df.to_dict(orient="records"):
        player_id = int(row["player_id"])
        feature_map[player_id] = stats_feature_defaults(row)
    return feature_map


def fetch_player_attribute_feature_map(players: List[Player]) -> Dict[int, Dict[str, Optional[float]]]:
    player_ids = [player.id for player in players if player.id]
    if not player_ids:
        return {}
    query = (
        select(
            PlayerAttributeHistory.player_id,
            PlayerAttributeHistory.record_date,
            PlayerAttributeHistory.pace,
            PlayerAttributeHistory.shooting,
            PlayerAttributeHistory.passing,
            PlayerAttributeHistory.dribbling,
            PlayerAttributeHistory.defending,
            PlayerAttributeHistory.physical,
            PlayerAttributeHistory.vision,
            PlayerAttributeHistory.tackling,
            PlayerAttributeHistory.determination,
            PlayerAttributeHistory.technique,
        )
        .where(PlayerAttributeHistory.player_id.in_(player_ids))
    )
    with engine.connect() as connection:
        history_df = pd.read_sql(query, connection)
    players_df = player_base_dataframe_from_players(players)
    aggregated_df = aggregate_attribute_history_dataframe(players_df, history_df)
    if aggregated_df.empty:
        return {}
    feature_map: Dict[int, Dict[str, Optional[float]]] = {}
    for row in aggregated_df.to_dict(orient="records"):
        player_id = int(row["player_id"])
        feature_map[player_id] = attribute_history_feature_defaults(row)
    return feature_map


def fetch_player_match_feature_map(players: List[Player]) -> Dict[int, Dict[str, Optional[float]]]:
    player_ids = [player.id for player in players if player.id]
    if not player_ids:
        return {}
    query = (
        select(
            PlayerMatchParticipation.player_id,
            Match.match_date,
            Match.opponent_level,
            PlayerMatchParticipation.started,
            PlayerMatchParticipation.position_played,
            PlayerMatchParticipation.minutes_played,
            PlayerMatchParticipation.final_score,
        )
        .join(Match, PlayerMatchParticipation.match_id == Match.id)
        .where(PlayerMatchParticipation.player_id.in_(player_ids))
    )
    with engine.connect() as connection:
        participation_df = pd.read_sql(query, connection)
    players_df = player_base_dataframe_from_players(players)
    aggregated_df = aggregate_match_participation_dataframe(players_df, participation_df)
    if aggregated_df.empty:
        return {}
    feature_map: Dict[int, Dict[str, Optional[float]]] = {}
    for row in aggregated_df.to_dict(orient="records"):
        player_id = int(row["player_id"])
        feature_map[player_id] = match_feature_defaults(row)
    return feature_map


def fetch_player_scout_report_feature_map(player_ids: List[int]) -> Dict[int, Dict[str, Optional[float]]]:
    if not player_ids:
        return {}
    query = (
        select(
            ScoutReport.player_id,
            ScoutReport.report_date,
            ScoutReport.decision_making,
            ScoutReport.tactical_reading,
            ScoutReport.mental_profile,
            ScoutReport.adaptability,
            ScoutReport.observed_projection_score,
        )
        .where(ScoutReport.player_id.in_(player_ids))
    )
    with engine.connect() as connection:
        scout_report_df = pd.read_sql(query, connection)
    aggregated_df = aggregate_scout_report_dataframe(scout_report_df)
    if aggregated_df.empty:
        return {}
    feature_map: Dict[int, Dict[str, Optional[float]]] = {}
    for row in aggregated_df.to_dict(orient="records"):
        player_id = int(row["player_id"])
        feature_map[player_id] = scout_report_feature_defaults(row)
    return feature_map


def fetch_player_physical_feature_map(player_ids: List[int]) -> Dict[int, Dict[str, Optional[float]]]:
    if not player_ids:
        return {}
    query = (
        select(
            PhysicalAssessment.player_id,
            PhysicalAssessment.assessment_date,
            PhysicalAssessment.height_cm,
            PhysicalAssessment.weight_kg,
            PhysicalAssessment.dominant_foot,
            PhysicalAssessment.estimated_speed,
            PhysicalAssessment.endurance,
            PhysicalAssessment.in_growth_spurt,
        )
        .where(PhysicalAssessment.player_id.in_(player_ids))
    )
    with engine.connect() as connection:
        physical_df = pd.read_sql(query, connection)
    aggregated_df = aggregate_physical_assessment_dataframe(physical_df)
    if aggregated_df.empty:
        return {}
    feature_map: Dict[int, Dict[str, Optional[float]]] = {}
    for row in aggregated_df.to_dict(orient="records"):
        player_id = int(row["player_id"])
        feature_map[player_id] = physical_feature_defaults(row)
    return feature_map


def fetch_player_availability_feature_map(player_ids: List[int]) -> Dict[int, Dict[str, Optional[float]]]:
    if not player_ids:
        return {}
    query = (
        select(
            PlayerAvailability.player_id,
            PlayerAvailability.record_date,
            PlayerAvailability.availability_pct,
            PlayerAvailability.fatigue_pct,
            PlayerAvailability.training_load_pct,
            PlayerAvailability.missed_days,
            PlayerAvailability.injury_flag,
        )
        .where(PlayerAvailability.player_id.in_(player_ids))
    )
    with engine.connect() as connection:
        availability_df = pd.read_sql(query, connection)
    aggregated_df = aggregate_availability_dataframe(availability_df)
    if aggregated_df.empty:
        return {}
    feature_map: Dict[int, Dict[str, Optional[float]]] = {}
    for row in aggregated_df.to_dict(orient="records"):
        player_id = int(row["player_id"])
        feature_map[player_id] = availability_feature_defaults(row)
    return feature_map


def players_to_model_tensor(
    players: List[Player],
    stats_feature_map: Optional[Dict[int, Dict[str, Optional[float]]]] = None,
    attribute_feature_map: Optional[Dict[int, Dict[str, Optional[float]]]] = None,
    match_feature_map: Optional[Dict[int, Dict[str, Optional[float]]]] = None,
    scout_report_feature_map: Optional[Dict[int, Dict[str, Optional[float]]]] = None,
    physical_feature_map: Optional[Dict[int, Dict[str, Optional[float]]]] = None,
    availability_feature_map: Optional[Dict[int, Dict[str, Optional[float]]]] = None,
) -> torch.Tensor:
    if preprocessor is None:
        raise RuntimeError("No hay preprocesador cargado para inferencia.")
    player_ids = [player.id for player in players if player.id]
    stats_feature_map = stats_feature_map or fetch_player_stat_feature_map(player_ids)
    attribute_feature_map = attribute_feature_map or fetch_player_attribute_feature_map(players)
    match_feature_map = match_feature_map or fetch_player_match_feature_map(players)
    scout_report_feature_map = scout_report_feature_map or fetch_player_scout_report_feature_map(player_ids)
    physical_feature_map = physical_feature_map or fetch_player_physical_feature_map(player_ids)
    availability_feature_map = availability_feature_map or fetch_player_availability_feature_map(player_ids)
    features_df = dataframe_from_players(
        players,
        stats_feature_map=stats_feature_map,
        attribute_feature_map=attribute_feature_map,
        match_feature_map=match_feature_map,
        scout_report_feature_map=scout_report_feature_map,
        physical_feature_map=physical_feature_map,
        availability_feature_map=availability_feature_map,
    )
    features_array = transform_features(features_df, preprocessor)
    return torch.tensor(features_array, dtype=torch.float32)


def batch_project_players(
    players: List[Player],
    avg_score_map: Optional[Dict[int, Optional[float]]] = None,
    stats_feature_map: Optional[Dict[int, Dict[str, Optional[float]]]] = None,
    attribute_feature_map: Optional[Dict[int, Dict[str, Optional[float]]]] = None,
    match_feature_map: Optional[Dict[int, Dict[str, Optional[float]]]] = None,
    scout_report_feature_map: Optional[Dict[int, Dict[str, Optional[float]]]] = None,
    physical_feature_map: Optional[Dict[int, Dict[str, Optional[float]]]] = None,
    availability_feature_map: Optional[Dict[int, Dict[str, Optional[float]]]] = None,
) -> Dict[int, Dict[str, object]]:
    if not players or model is None or preprocessor is None:
        return {}

    avg_score_map = avg_score_map or {}
    player_ids = [player.id for player in players if player.id]
    stats_feature_map = stats_feature_map or fetch_player_stat_feature_map(player_ids)
    attribute_feature_map = attribute_feature_map or fetch_player_attribute_feature_map(players)
    match_feature_map = match_feature_map or fetch_player_match_feature_map(players)
    scout_report_feature_map = scout_report_feature_map or fetch_player_scout_report_feature_map(player_ids)
    physical_feature_map = physical_feature_map or fetch_player_physical_feature_map(player_ids)
    availability_feature_map = availability_feature_map or fetch_player_availability_feature_map(player_ids)
    with torch.no_grad():
        probs_tensor = torch.sigmoid(
            model(
                players_to_model_tensor(
                    players,
                    stats_feature_map=stats_feature_map,
                    attribute_feature_map=attribute_feature_map,
                    match_feature_map=match_feature_map,
                    scout_report_feature_map=scout_report_feature_map,
                    physical_feature_map=physical_feature_map,
                    availability_feature_map=availability_feature_map,
                )
            )
        )
    raw_base_probs = np.asarray(probs_tensor.detach().cpu().numpy()).reshape(-1)
    calibrated_base_probs = (
        apply_probability_calibrator(probability_calibrator, raw_base_probs.tolist())
        if probability_calibrator is not None
        else None
    )

    projections: Dict[int, Dict[str, object]] = {}
    for idx, player in enumerate(players):
        base_prob = float(raw_base_probs[idx])
        attr_map = player_attribute_map(player)
        best_position, best_score = recommend_position_from_attrs(attr_map)
        fit_score = weighted_score_from_attrs(attr_map, player.position)
        player_stats_features = stats_feature_map.get(player.id, {})
        avg_final_score = avg_score_map.get(player.id)
        if avg_final_score is None:
            avg_final_score = player_stats_features.get("avg_final_score_hist")
        stats_summary = {"avg_final_score": avg_final_score}
        combined = combine_probability(base_prob, stats_summary, fit_score=fit_score)
        calibrated_base_prob = (
            float(calibrated_base_probs[idx])
            if calibrated_base_probs is not None
            else None
        )
        calibrated_combined = (
            combine_probability(calibrated_base_prob, stats_summary, fit_score=fit_score)
            if calibrated_base_prob is not None
            else None
        )
        projections[player.id] = {
            "base_prob": base_prob,
            "base_prob_source": "raw_pytorch_sigmoid",
            "calibrated_base_prob": calibrated_base_prob,
            "calibrated_combined_prob": calibrated_combined,
            "combined_prob": combined,
            "category": categorize_probability(combined),
            "stats_summary": stats_summary,
            "fit_score": fit_score,
            "recommended_position": best_position,
            "recommended_score": best_score,
        }
    return projections


def player_fit_score(player: Player, target_position: Optional[str] = None) -> float:
    return weighted_score_from_attrs(player_attribute_map(player), target_position)


def normalize_identifier(raw_value: Optional[str]) -> Optional[str]:
    if not raw_value:
        return None
    digits = "".join(ch for ch in raw_value if ch.isdigit())
    if len(digits) < 6:
        return None
    return digits



def fetch_player_stats(player_id: int, db_session: Optional[SQLAlchemySession] = None) -> List[PlayerStat]:
    close_session = False
    if db_session is None:
        db_session = Session()
        close_session = True
    stats = (db_session.query(PlayerStat)
             .filter(PlayerStat.player_id == player_id)
             .order_by(PlayerStat.record_date.asc(), PlayerStat.id.asc())
             .all())
    if close_session:
        db_session.close()
    return stats


def summarize_stats(stats: List[PlayerStat]) -> Dict[str, Optional[float]]:
    if not stats:
        return {
            "entries": 0,
            "total_matches": 0,
            "total_goals": 0,
            "total_assists": 0,
            "total_minutes": 0,
            "avg_pass_accuracy": None,
            "avg_shot_accuracy": None,
            "avg_duels": None,
            "avg_final_score": None,
            "latest_final_score": None,
            "latest_date": None,
        }

    def avg(values: List[Optional[float]]) -> Optional[float]:
        filtered = [v for v in values if v is not None]
        return round(mean(filtered), 2) if filtered else None

    return {
        "entries": len(stats),
        "total_matches": sum(s.matches_played for s in stats),
        "total_goals": sum(s.goals for s in stats),
        "total_assists": sum(s.assists for s in stats),
        "total_minutes": sum(s.minutes_played for s in stats),
        "avg_pass_accuracy": avg([s.pass_accuracy for s in stats]),
        "avg_shot_accuracy": avg([s.shot_accuracy for s in stats]),
        "avg_duels": avg([s.duels_won_pct for s in stats]),
        "avg_final_score": avg([s.final_score for s in stats if s.final_score is not None]),
        "latest_final_score": stats[-1].final_score,
        "latest_date": stats[-1].record_date.isoformat(),
    }


def fetch_attribute_history(
    player_id: int,
    db_session: Optional[SQLAlchemySession] = None,
) -> List[PlayerAttributeHistory]:
    close_session = False
    if db_session is None:
        db_session = Session()
        close_session = True
    history = (db_session.query(PlayerAttributeHistory)
               .filter(PlayerAttributeHistory.player_id == player_id)
               .order_by(PlayerAttributeHistory.record_date.asc(), PlayerAttributeHistory.id.asc())
               .all())
    if close_session:
        db_session.close()
    return history


enforce_operational_pool_limit_on_startup()


def summarize_attribute_history(history: List[PlayerAttributeHistory]) -> Dict[str, Optional[int]]:
    if not history:
        summary: Dict[str, Optional[int]] = {field: None for field in ATTRIBUTE_FIELDS}
        summary["entries"] = 0
        summary["latest_date"] = None
        return summary
    latest = history[-1]
    summary = {field: getattr(latest, field) for field in ATTRIBUTE_FIELDS}
    summary["entries"] = len(history)
    summary["latest_date"] = latest.record_date.isoformat()
    return summary


def combine_probability(base_prob: float, stats_summary: Dict[str, Optional[float]], fit_score: Optional[float] = None) -> float:
    """Combina la probabilidad del modelo con señales simples del historial y del fit del jugador.

    - base_prob: salida del modelo (0..1)
    - avg_final_score: promedio histórico (1..10) si existe
    - fit_score: puntaje ponderado por posición (0..20) si existe
    """
    avg_score = stats_summary.get("avg_final_score")
    rating_weight = None if avg_score is None else min(max(float(avg_score) / 10.0, 0.0), 1.0)

    fit_weight = None
    if fit_score is not None:
        try:
            fit_weight = min(max(float(fit_score) / 20.0, 0.0), 1.0)
        except Exception:
            fit_weight = None

    # Pesos (tuneables vía env vars)
    w_model = float(os.environ.get("POT_W_MODEL", "0.35"))
    w_rating = float(os.environ.get("POT_W_RATING", "0.35"))
    w_fit = float(os.environ.get("POT_W_FIT", "0.30"))

    # Si no hay rating o fit, re-normalizamos para no castigar por falta de datos
    components = [(w_model, base_prob)]
    if rating_weight is not None:
        components.append((w_rating, rating_weight))
    if fit_weight is not None:
        components.append((w_fit, fit_weight))

    weight_sum = sum(w for w, _ in components) or 1.0
    combined = sum(w * v for w, v in components) / weight_sum

    return max(0.0, min(combined, 0.99))
def stats_chart_payload(stats: List[PlayerStat]) -> Dict[str, List]:
    labels = []
    final_scores = []
    pass_pct = []
    shot_pct = []
    duel_pct = []
    for entry in stats:
        labels.append(entry.record_date.strftime("%Y-%m-%d"))
        final_scores.append(entry.final_score if entry.final_score is not None else None)
        pass_pct.append(entry.pass_accuracy if entry.pass_accuracy is not None else None)
        shot_pct.append(entry.shot_accuracy if entry.shot_accuracy is not None else None)
        duel_pct.append(entry.duels_won_pct if entry.duels_won_pct is not None else None)
    return {
        "labels": labels,
        "final_scores": final_scores,
        "pass_pct": pass_pct,
        "shot_pct": shot_pct,
        "duel_pct": duel_pct,
    }


def calculate_stats_rating(metrics: Dict[str, Optional[float]]) -> float:
    matches = metrics.get("matches", 0) or 0
    goals = metrics.get("goals", 0) or 0
    assists = metrics.get("assists", 0) or 0
    minutes = metrics.get("minutes", 0) or 0
    pass_pct = metrics.get("pass_pct") or 0.0
    shot_pct = metrics.get("shot_pct") or 0.0
    duels_pct = metrics.get("duels_pct") or 0.0

    minutes_factor = min(minutes / MATCH_MINUTES_DENOMINATOR, MAX_MINUTES_FACTOR)
    scoring_factor = (
        min(goals, MAX_GOALS_RATED) * GOAL_RATING_WEIGHT
        + min(assists, MAX_ASSISTS_RATED) * ASSIST_RATING_WEIGHT
    )
    accuracy_factor = (
        (pass_pct / PERCENT_DENOMINATOR) * PASS_ACCURACY_RATING_WEIGHT
        + (shot_pct / PERCENT_DENOMINATOR) * SHOT_ACCURACY_RATING_WEIGHT
        + (duels_pct / PERCENT_DENOMINATOR) * DUEL_ACCURACY_RATING_WEIGHT
    )
    consistency_factor = min(matches, MAX_MATCHES_RATED) * MATCH_CONSISTENCY_WEIGHT

    raw_score = BASE_STATS_RATING + minutes_factor + scoring_factor + accuracy_factor + consistency_factor
    return round(max(MIN_STATS_RATING, min(MAX_STATS_RATING, raw_score)), 2)


def attribute_chart_payload(history: List[PlayerAttributeHistory]) -> Dict[str, List[Optional[int]]]:
    labels: List[str] = []
    series: Dict[str, List[Optional[int]]] = {field: [] for field in ATTRIBUTE_FIELDS}
    for entry in history:
        labels.append(entry.record_date.strftime("%Y-%m-%d"))
        for field in ATTRIBUTE_FIELDS:
            series[field].append(getattr(entry, field))
    return {"labels": labels, "series": series}


DEFAULT_POTENTIAL_MEDIUM_THRESHOLD = 0.60
DEFAULT_POTENTIAL_HIGH_THRESHOLD = 0.80


def probability_env(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError:
        app.logger.warning("%s invalido (%r). Se usa %.2f.", name, raw, default)
        return default
    if not 0.0 <= value <= 1.0:
        app.logger.warning("%s fuera de rango (%r). Se usa %.2f.", name, raw, default)
        return default
    return value


def potential_thresholds() -> Tuple[float, float]:
    high = probability_env("POTENTIAL_HIGH_THRESHOLD", DEFAULT_POTENTIAL_HIGH_THRESHOLD)
    medium = probability_env("POTENTIAL_MEDIUM_THRESHOLD", DEFAULT_POTENTIAL_MEDIUM_THRESHOLD)
    if medium >= high:
        app.logger.warning(
            "POTENTIAL_MEDIUM_THRESHOLD debe ser menor que POTENTIAL_HIGH_THRESHOLD. "
            "Se ajusta medium a %.2f.",
            max(0.0, high - 0.05),
        )
        medium = max(0.0, high - 0.05)
    return medium, high


def is_high_potential_probability(probability: float) -> bool:
    _, high = potential_thresholds()
    return probability >= high


def categorize_probability(probability: float) -> str:
    medium, high = potential_thresholds()

    if probability >= high:
        return "Alto potencial"
    if probability >= medium:
        return "Potencial medio"
    return "Bajo potencial"
def compute_projection(
    player: Player,
    stats: Optional[List[PlayerStat]] = None,
    db_session: Optional[SQLAlchemySession] = None,
) -> Optional[Dict[str, object]]:
    if model is None:
        return None
    if stats is None:
        stats_list = fetch_player_stats(player.id, db_session=db_session)
    else:
        stats_list = stats
    summary = summarize_stats(stats_list)
    avg_score = summary.get("avg_final_score")
    stats_feature_map = {
        player.id: stats_feature_defaults(
            {
                "stats_entry_count": summary.get("entries"),
                "avg_final_score_hist": avg_score,
                "avg_pass_accuracy_hist": summary.get("avg_pass_accuracy"),
                "latest_final_score_hist": summary.get("latest_final_score"),
            }
        )
    }
    attribute_feature_map = fetch_player_attribute_feature_map([player])
    match_feature_map = fetch_player_match_feature_map([player])
    scout_report_feature_map = fetch_player_scout_report_feature_map([player.id])
    projection = batch_project_players(
        [player],
        {player.id: avg_score},
        stats_feature_map=stats_feature_map,
        attribute_feature_map=attribute_feature_map,
        match_feature_map=match_feature_map,
        scout_report_feature_map=scout_report_feature_map,
    ).get(player.id)
    if not projection:
        return None
    projection["history"] = stats_list
    return projection


def refresh_player_potential(
    player: Player,
    db_session: Optional[SQLAlchemySession] = None,
) -> Optional[Dict[str, object]]:
    projection = compute_projection(player, db_session=db_session)
    if projection:
        player.potential_label = is_high_potential_probability(projection["combined_prob"])
    return projection

# ----------------------------------------------------
# DASHBOARD
dashboard_blueprint = create_dashboard_blueprint(
    deps=SimpleNamespace(
        Session=Session,
        Player=Player,
        PlayerStat=PlayerStat,
        PlayerAttributeHistory=PlayerAttributeHistory,
        Match=Match,
        PlayerMatchParticipation=PlayerMatchParticipation,
        ScoutReport=ScoutReport,
        PhysicalAssessment=PhysicalAssessment,
        PlayerAvailability=PlayerAvailability,
        login_required=login_required,
        current_role=current_role,
        ROLE_DIRECTOR=ROLE_DIRECTOR,
        cache_get=_cache_get,
        cache_set=_cache_set,
        display_position_label=display_position_label,
        ATTRIBUTE_FIELDS=ATTRIBUTE_FIELDS,
        ATTRIBUTE_LABELS=ATTRIBUTE_LABELS,
        batch_project_players=batch_project_players,
        get_model=lambda: model,
    )
)
app.register_blueprint(dashboard_blueprint)
register_legacy_endpoint_aliases(
    app,
    "dashboard",
    (
        ("dashboard", "/dashboard", ("GET",)),
    ),
)

# ----------------------------------------------------
# COMPARADORES Y SETTINGS
compare_blueprint = create_compare_blueprint(
    deps=SimpleNamespace(
        Session=Session,
        Player=Player,
        PlayerStat=PlayerStat,
        login_required=login_required,
        MAX_COMPARE_PLAYERS=MAX_COMPARE_PLAYERS,
        current_role=current_role,
        cache_get=_cache_get,
        cache_set=_cache_set,
        normalized_position=normalized_position,
        fetch_player_stats=fetch_player_stats,
        summarize_stats=summarize_stats,
        batch_project_players=batch_project_players,
        position_weights=position_weights,
        ATTRIBUTE_FIELDS=ATTRIBUTE_FIELDS,
        ATTRIBUTE_LABELS=ATTRIBUTE_LABELS,
        player_attribute_map=player_attribute_map,
        weighted_score_from_attrs=weighted_score_from_attrs,
        recommend_position_from_attrs=recommend_position_from_attrs,
        default_player_photo_url=default_player_photo_url,
    )
)
app.register_blueprint(compare_blueprint)
register_legacy_endpoint_aliases(
    app,
    "compare",
    (
        ("compare_players", "/compare", ("GET", "POST")),
        ("compare_multi", "/compare/multi", ("GET", "POST")),
    ),
)

settings_blueprint = create_settings_blueprint(
    deps=SimpleNamespace(
        roles_required=roles_required,
        ROLE_ADMIN=ROLE_ADMIN,
        require_csrf=_require_csrf,
        pipeline_lock=_PIPELINE_LOCK,
        update_database_pipeline=update_database_pipeline,
        EVAL_POOL_MAX=EVAL_POOL_MAX,
        SYNC_SHORTLIST_ENABLED=SYNC_SHORTLIST_ENABLED,
        Session=Session,
        cleanup_operational_data=cleanup_operational_data,
        invalidate_dashboard_cache=invalidate_dashboard_cache,
        compute_operational_data_quality=compute_operational_data_quality,
    )
)
app.register_blueprint(settings_blueprint)
register_legacy_endpoint_aliases(
    app,
    "settings",
    (
        ("settings", "/settings", ("GET", "POST")),
    ),
)

def parse_bool(value: Optional[str]) -> bool:
    if value is None:
        return False
    value = value.strip().lower()
    return value in {"1", "true", "t", "yes", "y", "si", "sí", "alto"}


def parse_int_field(value: Optional[str], default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def parse_date_field(value: Optional[str], errors: List[str], label: str = "La fecha") -> date:
    if not value:
        return date.today()
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError:
        errors.append(f"{label} debe tener formato YYYY-MM-DD.")
        return date.today()


def validate_non_negative_int_field(
    value: Optional[str],
    label: str,
    errors: List[str],
    allow_blank: bool = True,
) -> int:
    if value in (None, ""):
        if allow_blank:
            return 0
        errors.append(f"{label} es obligatorio.")
        return 0
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        errors.append(f"{label} debe ser un numero entero.")
        return 0
    if parsed < 0:
        errors.append(f"{label} no puede ser negativo.")
    return parsed


def validate_optional_float_range(
    value: Optional[str],
    label: str,
    errors: List[str],
    min_value: float,
    max_value: float,
) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        errors.append(f"{label} debe ser un numero valido.")
        return None
    if parsed < min_value or parsed > max_value:
        errors.append(f"{label} debe estar entre {min_value:g} y {max_value:g}.")
        return None
    return parsed


def normalize_position_choice(value: Optional[str]) -> str:
    return normalized_position(value)


players_blueprint = create_players_blueprint(
    deps=SimpleNamespace(
        Session=Session,
        Player=Player,
        PlayerStat=PlayerStat,
        PlayerAttributeHistory=PlayerAttributeHistory,
        Match=Match,
        PlayerMatchParticipation=PlayerMatchParticipation,
        ScoutReport=ScoutReport,
        PhysicalAssessment=PhysicalAssessment,
        PlayerAvailability=PlayerAvailability,
        login_required=login_required,
        roles_required=roles_required,
        require_csrf=_require_csrf,
        can_edit_player_data=can_edit_player_data,
        current_role=current_role,
        cache_get=_cache_get,
        cache_set=_cache_set,
        PLAYER_LIST_PER_PAGE=PLAYER_LIST_PER_PAGE,
        EVAL_POOL_MAX=EVAL_POOL_MAX,
        POSITION_OPTIONS=POSITION_OPTIONS,
        ROLE_ADMIN=ROLE_ADMIN,
        ROLE_SCOUT=ROLE_SCOUT,
        ATTRIBUTE_FIELDS=ATTRIBUTE_FIELDS,
        ATTRIBUTE_LABELS=ATTRIBUTE_LABELS,
        ATTRIBUTE_MIN_VALUE=ATTRIBUTE_MIN_VALUE,
        ATTRIBUTE_MAX_VALUE=ATTRIBUTE_MAX_VALUE,
        batch_project_players=batch_project_players,
        default_player_photo_url=default_player_photo_url,
        player_attribute_map=player_attribute_map,
        recommend_position_from_attrs=recommend_position_from_attrs,
        weighted_score_from_attrs=weighted_score_from_attrs,
        is_high_potential_probability=is_high_potential_probability,
        sync_player_attribute_history=sync_player_attribute_history,
        fetch_player_stats=fetch_player_stats,
        fetch_attribute_history=fetch_attribute_history,
        summarize_attribute_history=summarize_attribute_history,
        summarize_stats=summarize_stats,
        stats_chart_payload=stats_chart_payload,
        attribute_chart_payload=attribute_chart_payload,
        compute_projection=compute_projection,
        score_band=score_band,
        compute_suggestions=compute_suggestions,
        normalized_position=normalized_position,
        parse_date_field=parse_date_field,
        validate_non_negative_int_field=validate_non_negative_int_field,
        validate_optional_float_range=validate_optional_float_range,
        calculate_stats_rating=calculate_stats_rating,
        refresh_player_potential=refresh_player_potential,
        invalidate_dashboard_cache=invalidate_dashboard_cache,
        parse_int_field=parse_int_field,
        is_valid_attribute=is_valid_attribute,
        normalize_identifier=normalize_identifier,
        is_valid_eval_age=is_valid_eval_age,
        normalize_position_choice=normalize_position_choice,
        parse_bool=parse_bool,
        get_model=lambda: model,
    )
)
app.register_blueprint(players_blueprint)
register_legacy_endpoint_aliases(
    app,
    "players",
    (
        ("index", "/players", ("GET",)),
        ("player_detail", "/player/<int:player_id>", ("GET",)),
        ("add_player_match_history", "/player/<int:player_id>/matches/add", ("POST",)),
        (
            "edit_player_match_history",
            "/player/<int:player_id>/matches/<int:participation_id>/edit",
            ("POST",),
        ),
        (
            "delete_player_match_history",
            "/player/<int:player_id>/matches/<int:participation_id>/delete",
            ("POST",),
        ),
        ("add_player_physical_assessment", "/player/<int:player_id>/physical/add", ("POST",)),
        (
            "edit_player_physical_assessment",
            "/player/<int:player_id>/physical/<int:assessment_id>/edit",
            ("POST",),
        ),
        (
            "delete_player_physical_assessment",
            "/player/<int:player_id>/physical/<int:assessment_id>/delete",
            ("POST",),
        ),
        ("add_player_availability", "/player/<int:player_id>/availability/add", ("POST",)),
        (
            "edit_player_availability",
            "/player/<int:player_id>/availability/<int:availability_id>/edit",
            ("POST",),
        ),
        (
            "delete_player_availability",
            "/player/<int:player_id>/availability/<int:availability_id>/delete",
            ("POST",),
        ),
        ("add_player_scout_report", "/player/<int:player_id>/reports/add", ("POST",)),
        (
            "edit_player_scout_report",
            "/player/<int:player_id>/reports/<int:report_id>/edit",
            ("POST",),
        ),
        (
            "delete_player_scout_report",
            "/player/<int:player_id>/reports/<int:report_id>/delete",
            ("POST",),
        ),
        ("player_stats", "/player/<int:player_id>/stats", ("GET", "POST")),
        ("edit_player_stat", "/player/<int:player_id>/stats/<int:stat_id>/edit", ("POST",)),
        ("delete_player_stat", "/player/<int:player_id>/stats/<int:stat_id>/delete", ("POST",)),
        ("player_attributes", "/player/<int:player_id>/attributes", ("GET", "POST")),
        (
            "edit_player_attribute_history",
            "/player/<int:player_id>/attributes/<int:history_id>/edit",
            ("POST",),
        ),
        (
            "delete_player_attribute_history",
            "/player/<int:player_id>/attributes/<int:history_id>/delete",
            ("POST",),
        ),
        ("edit_player", "/edit_player/<int:player_id>", ("GET", "POST")),
        ("delete_player", "/delete_player/<int:player_id>", ("POST",)),
        ("predict_player", "/player/<int:player_id>/predict", ("GET",)),
        ("manage_players", "/players/manage", ("GET", "POST")),
        ("import_players", "/players/import", ("GET", "POST")),
        ("download_players_import_template", "/players/import/template.csv", ("GET",)),
    ),
)

# --- Error handlers (observabilidad mínima) ---

@app.errorhandler(400)
def handle_400(e):
    app.logger.warning("400 Bad Request - path=%s user_id=%s", request.path, session.get("user_id"))
    return render_template("error.html", code=400, message="Solicitud inválida"), 400

@app.errorhandler(413)
def handle_413(e):
    app.logger.warning("413 Payload Too Large - path=%s user_id=%s", request.path, session.get("user_id"))
    return render_template("error.html", code=413, message="Request demasiado grande"), 413



@app.errorhandler(403)
def handle_403(e):
    app.logger.warning("403 Forbidden - path=%s user_id=%s role=%s", request.path, session.get("user_id"), session.get("role"))
    return render_template("error.html", code=403, message="Acceso denegado"), 403

@app.errorhandler(404)
def handle_404(e):
    app.logger.info("404 Not Found - path=%s user_id=%s", request.path, session.get("user_id"))
    return render_template("error.html", code=404, message="Recurso no encontrado"), 404

@app.errorhandler(500)
def handle_500(e):
    app.logger.exception("500 Internal Server Error - path=%s user_id=%s role=%s", request.path, session.get("user_id"), session.get("role"))
    return render_template("error.html", code=500, message="Error interno del servidor"), 500
if __name__ == "__main__":
    app.run(debug=True)
