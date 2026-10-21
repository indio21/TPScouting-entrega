# RUNBOOK — TPScouting

Guía operativa de la rama vigente `main`. La evaluación académica recomendada
es la demo local descrita en [GUIA_DEMO_PROFESOR.md](GUIA_DEMO_PROFESOR.md).

## 1. Configuración y seguridad

- `APP_SECRET_KEY`: obligatoria en producción/Render; usar un secreto propio.
- `APP_DB_URL` y `TRAINING_DB_URL`: bases operativa y de entrenamiento.
- `LOG_LEVEL`: `INFO` o `DEBUG`.
- `CACHE_TTL_SECONDS`: 60 por defecto; `CACHE_MAX_ENTRIES`: 128.
- `PLAYER_LIST_PER_PAGE`: 50; `MAX_COMPARE_PLAYERS`: 2000.
- `LOGIN_RATE_LIMIT_WINDOW_SECONDS`: 900; `LOGIN_RATE_LIMIT_MAX_ATTEMPTS`: 5.
- `AUTO_TRAIN_ON_STARTUP`: desactivado; mantenerlo así en producción.
- `ALLOW_SQLITE_IN_PRODUCTION=1`: opt-in exclusivo de una demo descartable.
- `ADMIN_USERNAME` / `ADMIN_PASSWORD`: acceso de administración propio.
- `RENDER_SMOKE_BASE_URL`, `SMOKE_USERNAME`, `SMOKE_PASSWORD`: smoke externo.

La plantilla [.env.example](.env.example) describe las variables. La aplicación
no carga `.env` automáticamente; Docker lo recibe con `--env-file`.

El login limita intentos por usuario normalizado, en memoria y por proceso.
No identifica clientes por `X-Forwarded-For`, porque no se ha definido una cadena
de proxies confiables. Las rutas protegidas revalidan existencia y rol del usuario
contra la base. Los POST mutantes y el logout requieren CSRF.

La respuesta pública de `/health` no incluye excepciones ni contadores internos.
Los detalles operativos se consultan como administrador en Configuración.
CSP limita recursos; HSTS y cookies seguras se aplican en modo de producción.

## 2. Instalación y validación

Windows/Linux CPU instalan `requirements-lock.txt` y luego
`requirements-torch-cpu.txt`; las pruebas requieren `requirements-dev.txt`.
macOS Apple Silicon usa `requirements.txt` en lugar de los dos primeros.
No se soporta macOS Intel con este PyTorch. Los comandos completos están en
[docs/puesta_en_marcha.md](docs/puesta_en_marcha.md).

Desde la raíz en Windows:

```powershell
.\.venv\Scripts\python.exe .\scripts\iniciar_demo.py
```

La demo genera 60 jugadores sintéticos, comprueba edad/nacimiento/categoría y
crea `profesor_demo` / `DemoProfesor123`. Guarda los cambios en
`scouting_app/demo_profesor.db`. Detener el servidor antes de usar `--recrear`;
esa opción sustituye exclusivamente dicha base demo.

Última suite local documentada: **131 passed, 1 skipped, 4 warnings, 83,31 %**.
La omitida es Playwright opt-in y las cuatro advertencias All-NaN son conocidas.
Véanse [README_TESTS.md](README_TESTS.md) y el registro público
`VERIFICACION_ENTREGA.md` de entrega para commit, fecha, CI y hashes.
La compatibilidad Apple Silicon sí fue probada en un runner real de GitHub.

## 3. Datos, modelos y nuevas corridas

Los artefactos de runtime incluidos permiten inferencia sin entrenar. No se
deben reemplazar por archivos de terceros: joblib y PyTorch requieren confiar
en su origen. Datos y modelos nuevos deben registrarse como una corrida nueva;
no reconstruir metadatos ausentes de la histórica.

La generación y el entrenamiento web están desactivados en producción.
Una corrida experimental se ejecuta fuera de HTTP, con base y salidas propias.
Ejemplo PowerShell desde `scouting_app/`, usando un nombre de directorio nuevo:

```powershell
New-Item -ItemType Directory -Path ..\artifacts\experimentos\local_01
..\.venv\Scripts\python.exe generate_data.py --num-players 20000 --db-url sqlite:///../artifacts/experimentos/local_01/players_training.db --seed 42 --min-age 12 --max-age 18
..\.venv\Scripts\python.exe train_model.py --db-url sqlite:///../artifacts/experimentos/local_01/players_training.db --model-out ../artifacts/experimentos/local_01/model.pt --preprocessor-out ../artifacts/experimentos/local_01/preprocessor.joblib --calibrator-out ../artifacts/experimentos/local_01/probability_calibrator.joblib --metadata-out ../artifacts/experimentos/local_01/training_metadata.json --splits-out ../artifacts/experimentos/local_01/training_splits.json --epochs 45 --lr 5e-4 --patience 10
..\.venv\Scripts\python.exe evaluate_saved_model.py --db-url sqlite:///../artifacts/experimentos/local_01/players_training.db --model-path ../artifacts/experimentos/local_01/model.pt --preprocessor-path ../artifacts/experimentos/local_01/preprocessor.joblib --calibrator-path ../artifacts/experimentos/local_01/probability_calibrator.joblib --metadata-path ../artifacts/experimentos/local_01/training_metadata.json --splits-path ../artifacts/experimentos/local_01/training_splits.json --cache-path ../artifacts/experimentos/local_01/temporal_training_dataframe.joblib
```

Estos comandos producen una nueva corrida con el pipeline histórico, no la
candidata con política ajustada solo sobre train. La candidata del informe se
mantiene como comparación, sin reemplazar los artefactos entregados.
`scripts/run_ml_blocks_1_3.py` requiere cache temporal y metadata históricos que
no se distribuyen con la demo. El entrenamiento puede agregar un registro local
a `scouting_app/experiments.csv`, ignorado por Git. No sincronizar automáticamente
los datos experimentales con la base de la aplicación.

Para una repetición registrable, guardar también `git rev-parse HEAD`,
`python -m pip freeze`, seed, hashes de datos y artefactos y tiempos medidos.
Las métricas y sus límites están en [docs/modelo_ml.md](docs/modelo_ml.md).

## 4. Salud, permisos y administración

`GET /health` responde 200 con `status=ok` cuando la consulta mínima de base
funciona; ante fallo responde 500 con un estado genérico. La causa queda en logs.

`/settings` y `/register` requieren rol administrador. Ante un 403, revisar el
rol real en base y los logs; cambiar solo `session['role']` no concede permisos.

Creación manual de admin, desde la raíz:

```powershell
$env:APP_DB_URL = "sqlite:///players_updated_v2.db"
$env:ADMIN_USERNAME = "admin_local"
$env:ADMIN_PASSWORD = "REEMPLAZAR_POR_UNA_CLAVE_PROPIA"
.\.venv\Scripts\python.exe .\scouting_app\create_admin.py
```

Configuración incluye controles de calidad y limpieza legacy; realizar backup
antes de una limpieza y revisar su alcance. La escala de atributos e informes
es 1–20; nacimiento determina edad y categoría.

## 5. Backup y restauración

Las rutas predeterminadas son `scouting_app/players_updated_v2.db` y
`scouting_app/players_training.db`; con variables de entorno pueden cambiar.
La demo portable usa su propia `demo_profesor.db`.

No copiar a mano un archivo activo junto con `-wal` y `-shm`: esa copia puede
ser inconsistente. Utilizar la API de backup de SQLite para obtener un snapshot
coherente, o detener todas las conexiones antes de una copia en frío.
Referencia: [SQLite Online Backup API](https://www.sqlite.org/backup.html).

Ejemplo PowerShell desde la raíz; sustituir la ruta por la base elegida y usar
un destino de backup nuevo:

```powershell
@'
import sqlite3
from contextlib import closing
from pathlib import Path

source = Path("scouting_app/players_updated_v2.db").resolve(strict=True)
target = Path("backups/operativa_YYYYMMDD.db")
if target.exists():
    raise FileExistsError("Usar un nombre de backup nuevo")
target.parent.mkdir(parents=True, exist_ok=True)
with closing(sqlite3.connect(source.as_uri() + "?mode=ro", uri=True)) as origin:
    with closing(sqlite3.connect(target)) as backup:
        origin.backup(backup)
'@ | .\.venv\Scripts\python.exe -
```

Para restaurar, detener todas las conexiones, conservar una copia del estado
actual, restaurar el snapshot sin mezclarlo con sidecars de otra base e iniciar
la aplicación. Comprobar `/health`, login y los registros esperados.
PostgreSQL requiere sus herramientas (`pg_dump`, snapshots o backups del
proveedor); no se respalda copiando archivos SQLite.

## 6. Render y contenedor

`render.yaml` fija Python 3.11.9 y los mismos manifests de runtime que CI.
Ejecuta el seed demo antes de Gunicorn y no entrena en producción.
La modalidad Free configurada comparte PostgreSQL para ambas URLs de base.
Según [Render](https://render.com/docs/free), PostgreSQL Free dispone de 1 GB,
expira a los 30 días, no ofrece backups y admite una instancia Free por workspace.
Sirve para demostración temporal; no se contrató ni cambió ningún plan.

El despliegue de mayo se conserva como evidencia histórica. La disponibilidad
actual de Render no se certifica con las pruebas locales ni es requisito para
la evaluación portable. Para revisar un despliegue actual, configurar su URL real:

```powershell
$env:RENDER_SMOKE_BASE_URL = "https://TU_SERVICIO.onrender.com"
$env:SMOKE_USERNAME = "usuario_del_despliegue"
$env:SMOKE_PASSWORD = "clave_del_despliegue"
.\.venv\Scripts\python.exe scripts\smoke_render.py
```

Registrar commit, fecha, URL y resultado, sin publicar claves.
Docker construye un runtime CPU con usuario sin privilegios y healthcheck.
Seguir la sección de contenedor de [README.md](README.md); `.env` se configura
localmente y no se versiona.

## 7. Límites operativos

Cache y rate limiting son en memoria y por proceso. SQLite usa WAL y timeout;
evitar transacciones largas y varias instancias con acceso concurrente.
Las rutas ya están separadas por dominio y registradas como blueprints;
`app.py` aún concentra ensamblaje/configuración y parte de la lógica.
Application factory sigue diferida.

Las bandas visuales predeterminadas son menos de 0,60, desde 0,60 a menos de 0,80
y desde 0,80. El `potential_label` visible/persistido no debe confundirse con
`temporal_target_label`, la etiqueta sintética de entrenamiento.

No cargar datos reales de menores. Los controles legales y operativos futuros
se diseñaron, pero no se implementaron; véase
[docs/REAL_DATA_READINESS.md](docs/REAL_DATA_READINESS.md).
