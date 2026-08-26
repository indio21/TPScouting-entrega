# TPScouting Entrega

[![CI](https://github.com/indio21/TPScouting-entrega/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/indio21/TPScouting-entrega/actions/workflows/ci.yml)

Repositorio limpio de entrega del MVP TPScouting.

TPScouting es una aplicacion web Flask para registrar jugadores juveniles, cargar historiales, comparar perfiles y estimar potencial con un modelo PyTorch. El alcance del MVP esta acotado a scouting juvenil de 12 a 18 anos para clubes formativos.

## Evaluacion rapida para el profesor

Despues de instalar las dependencias, un solo comando genera `60` jugadores sinteticos, verifica edad/nacimiento/categoria, crea el acceso local e inicia la aplicacion:

```powershell
.\.venv\Scripts\python.exe .\scripts\iniciar_demo.py
```

Abrir `http://127.0.0.1:5000/` e ingresar con:

```text
Usuario: profesor_demo
Contrasena: DemoProfesor123
```

Los datos son ficticios y reproducibles con semilla `42`; no representan futbolistas reales. La base se crea localmente, permite probar altas, ediciones y bajas, y no modifica archivos versionados. Las instrucciones para Windows, Linux/macOS y reinicio de la demo estan en [GUIA_DEMO_PROFESOR.md](GUIA_DEMO_PROFESOR.md).

## Contenido del repositorio

- `scouting_app/`: aplicacion Flask, modelos SQLAlchemy, templates, estilos, pipeline ML y scripts operativos.
- `tests/`: suite automatizada de regresion, seguridad basica, paginas y smoke visual opt-in.
- `scripts/smoke_render.py`: smoke HTTP para validar un despliegue publicado.
- `scripts/iniciar_demo.py`: preparacion y arranque portable de la demo local con 60 jugadores.
- `GUIA_DEMO_PROFESOR.md`: instrucciones de evaluacion rapida y alcance de los datos sinteticos.
- `render.yaml`: blueprint de despliegue en Render con PostgreSQL administrado.
- `requirements.txt`, `requirements-dev.txt` y `requirements-lock.txt`: dependencias.

No se versionan bases de datos locales, documentos Word, notas internas ni backups. La demo genera localmente los datos sinteticos necesarios.

## Stack

- Python + Flask
- SQLAlchemy + SQLite/PostgreSQL
- pandas + scikit-learn
- PyTorch
- Bootstrap + Chart.js
- pytest + GitHub Actions

## Instalacion manual

Desde PowerShell, ubicarse en la raíz del repositorio clonado y ejecutar:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
```

El comando recomendado es `scripts/iniciar_demo.py`. Como alternativa, los pasos manuales son los siguientes.

Crear datos demo locales si la base esta vacia:

```powershell
$env:DEMO_SEED_ON_STARTUP = "1"
$env:DEMO_SEED_PLAYERS = "60"
$env:APP_DB_URL = "sqlite:///players_updated_v2.db"
.\.venv\Scripts\python.exe .\scouting_app\seed_demo_data.py
```

Crear un administrador local:

```powershell
$env:APP_DB_URL = "sqlite:///players_updated_v2.db"
$env:ADMIN_USERNAME = "admin_local"
$env:ADMIN_PASSWORD = "Cambiar123"
.\.venv\Scripts\python.exe .\scouting_app\create_admin.py
```

Levantar la app:

```powershell
cd .\scouting_app
$env:APP_SECRET_KEY = "dev-secret-change-me"
$env:APP_DB_URL = "sqlite:///players_updated_v2.db"
$env:TRAINING_DB_URL = "sqlite:///players_training.db"
..\.venv\Scripts\python.exe app.py
```

Abrir `http://127.0.0.1:5000/`.

## Tests

Estado verificado al 26/08/2026: `87 passed, 1 skipped, 4 warnings`. La CI ejecuta la suite en Python 3.11 y 3.12 y publica los artefactos de cobertura.

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Con cobertura:

```powershell
.\.venv\Scripts\python.exe -m pytest -q --cov=scouting_app --cov-report=term-missing
```

El smoke visual con Playwright es opcional:

```powershell
$env:RUN_PLAYWRIGHT = "1"
.\.venv\Scripts\python.exe -m playwright install chromium
.\.venv\Scripts\python.exe -m pytest -q tests\test_visual_smoke.py
Remove-Item Env:\RUN_PLAYWRIGHT
```

## Imagen de jugador

Los jugadores sin una foto manual utilizan la silueta local `scouting_app/static/img/player-silhouette.svg`. La aplicación también reemplaza los avatares DiceBear heredados por este recurso local, por lo que no depende de un servicio externo para mostrar el listado, la ficha o la proyección.

## Modelo ML

La app incluye tres artefactos pequenos para inferencia de demo:

- `scouting_app/model.pt`
- `scouting_app/preprocessor.joblib`
- `scouting_app/probability_calibrator.joblib`

El modelo se usa como apoyo para priorizar revision de jugadores. La evidencia disponible corresponde a datos sinteticos reproducibles, no a validacion externa con datos reales.

## Deploy

`render.yaml` define un despliegue de referencia en Render:

- servicio web Flask/Gunicorn;
- PostgreSQL administrado;
- seed demo idempotente cuando la base esta vacia;
- `APP_SECRET_KEY` generada por Render;
- `ADMIN_PASSWORD` configurada manualmente como variable secreta.

Para validar un despliegue:

```powershell
$env:RENDER_SMOKE_BASE_URL = "https://TU_SERVICIO.onrender.com"
$env:SMOKE_USERNAME = "usuario"
$env:SMOKE_PASSWORD = "clave"
.\.venv\Scripts\python.exe .\scripts\smoke_render.py
```

La evidencia publica de Render es historica. La disponibilidad actual del servicio no es necesaria para ejecutar o evaluar localmente esta entrega.

## Alcance

Este repositorio representa un MVP academico. Sus limites principales son: dataset sintetico, cache/rate limiting en memoria, migraciones manuales y validacion externa pendiente.
