# Puesta en marcha

La evaluación recomendada usa los tres artefactos ML incluidos y genera una
base SQLite con 60 jugadores ficticios. No requiere entrenar ni usar Render.

## Windows PowerShell

Desde la raíz del repositorio:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe -m pip install -r requirements-torch-cpu.txt
.\.venv\Scripts\python.exe .\scripts\iniciar_demo.py
```

## Linux CPU

```bash
python3.11 -m venv .venv
./.venv/bin/python -m pip install --upgrade pip
./.venv/bin/python -m pip install -r requirements-lock.txt
./.venv/bin/python -m pip install -r requirements-torch-cpu.txt
./.venv/bin/python ./scripts/iniciar_demo.py
```

## macOS Apple Silicon

```bash
python3.11 -m venv .venv
./.venv/bin/python -m pip install --upgrade pip
./.venv/bin/python -m pip install -r requirements.txt
./.venv/bin/python ./scripts/iniciar_demo.py
```

El entorno fijado no soporta macOS Intel. Apple Silicon se verificó con Python
3.11 en GitHub Actions; Linux se verifica con 3.11/3.12 y Windows con 3.11.9.

## Acceso y reinicio

Abrir `http://127.0.0.1:5000/`, usuario `profesor_demo`, contraseña
`DemoProfesor123`. Son credenciales públicas exclusivas de la demo local.
La base `scouting_app/demo_profesor.db` conserva los cambios de la interfaz.

Después de detener el servidor, `scripts/iniciar_demo.py --recrear` regenera
exclusivamente esa base demo. `--solo-preparar` comprueba los datos y el acceso
sin iniciar Flask. Usar el intérprete de la `.venv` de su plataforma.

## Pruebas y operación

Instalar `requirements-dev.txt` después del runtime y ejecutar desde la raíz:

```powershell
.\.venv\Scripts\python.exe -m pytest -q --cov=scouting_app --cov-report=term-missing --cov-fail-under=80
```

En Linux/macOS, usar `./.venv/bin/python`. Véanse
[README_TESTS.md](../README_TESTS.md) y [RUNBOOK.md](../RUNBOOK.md).
No es necesario reemplazar modelos para evaluar la aplicación. El entrenamiento
y la generación web están desactivados en producción. No cargar datos reales.
