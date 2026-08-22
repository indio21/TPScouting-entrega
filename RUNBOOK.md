# Runbook TPScouting

Guia corta de operacion para la entrega del MVP.

## Variables principales

- `APP_SECRET_KEY`: obligatoria en produccion.
- `APP_DB_URL`: base operativa. En local puede ser `sqlite:///players_updated_v2.db`.
- `TRAINING_DB_URL`: base de entrenamiento. En local puede ser `sqlite:///players_training.db`.
- `ADMIN_USERNAME` y `ADMIN_PASSWORD`: credenciales iniciales para crear un usuario administrador.
- `DEMO_SEED_ON_STARTUP`: si vale `1`, permite generar datos demo con `seed_demo_data.py`.
- `DEMO_SEED_PLAYERS`: cantidad de jugadores demo, por defecto `100`.
- `CACHE_TTL_SECONDS` y `CACHE_MAX_ENTRIES`: cache en memoria para vistas pesadas.
- `PLAYER_LIST_PER_PAGE`: paginacion del listado.

## Arranque local minimo

Desde la raíz del repositorio clonado:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt

$env:DEMO_SEED_ON_STARTUP = "1"
$env:APP_DB_URL = "sqlite:///players_updated_v2.db"
.\.venv\Scripts\python.exe .\scouting_app\seed_demo_data.py

$env:ADMIN_USERNAME = "admin_local"
$env:ADMIN_PASSWORD = "Cambiar123"
.\.venv\Scripts\python.exe .\scouting_app\create_admin.py

cd .\scouting_app
$env:APP_SECRET_KEY = "dev-secret-change-me"
$env:TRAINING_DB_URL = "sqlite:///players_training.db"
..\.venv\Scripts\python.exe app.py
```

## Healthcheck

Endpoint:

```text
GET /health
```

Resultado esperado: HTTP `200`, `status=ok`, conectividad de base y datos de calidad operativa.

## Datos

Las bases SQLite locales no forman parte del repositorio. Se generan con:

- `seed_demo_data.py` para una demo rapida.
- `generate_data.py`, `train_model.py`, `evaluate_saved_model.py` y `sync_shortlist.py` para una corrida reproducible completa.

## Artefactos ML

El repo incluye los artefactos necesarios para inferencia:

- `model.pt`
- `preprocessor.joblib`
- `probability_calibrator.joblib`

Si se reentrena el modelo, conviene regenerar los tres artefactos juntos.

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest -q --cov=scouting_app --cov-report=term-missing
```

El smoke visual con Playwright se ejecuta solo con `RUN_PLAYWRIGHT=1`.

## Deploy Render

El blueprint `render.yaml` usa:

- PostgreSQL administrado;
- Gunicorn con `1` worker y `2` threads;
- seed demo idempotente;
- password de administrador cargada como variable secreta.

Despues de publicar, validar al menos:

- `/health`
- `/login`
- login con usuario demo
- `/dashboard`
- `/players`
- `/compare`
- `/compare/multi`
- una accion CRUD minima con CSRF

## Limites del MVP

- Dataset sintetico.
- Cache y rate limiting en memoria.
- Migraciones manuales sin Alembic.
- Locks locales, recomendados con un solo worker.
- La prediccion es soporte de decision, no reemplazo del criterio deportivo.
