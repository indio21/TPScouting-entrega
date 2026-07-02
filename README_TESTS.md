# Tests

## Instalacion

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
```

## Suite principal

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Con cobertura:

```powershell
.\.venv\Scripts\python.exe -m pytest -q --cov=scouting_app --cov-report=term-missing --cov-report=xml
```

## Smoke visual opcional

```powershell
$env:RUN_PLAYWRIGHT = "1"
.\.venv\Scripts\python.exe -m playwright install chromium
.\.venv\Scripts\python.exe -m pytest -q tests\test_visual_smoke.py
Remove-Item Env:\RUN_PLAYWRIGHT
```

## Smoke de despliegue

```powershell
$env:RENDER_SMOKE_BASE_URL = "https://TU_SERVICIO.onrender.com"
$env:SMOKE_USERNAME = "usuario"
$env:SMOKE_PASSWORD = "clave"
.\.venv\Scripts\python.exe scripts\smoke_render.py
```

Los tests usan bases temporales y no modifican bases locales reales.
