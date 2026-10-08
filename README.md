# TPScouting - Scouting Inteligente con IA (MVP)

[![CI](https://github.com/indio21/TPScouting/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/indio21/TPScouting/actions/workflows/ci.yml)

Trabajo final orientado al scouting de futbol juvenil, con una app web para:

- registrar jugadores y sus atributos
- cargar historial de rendimiento y evaluaciones
- comparar jugadores (1 vs 1 y multiple)
- estimar potencial con un modelo MLP (PyTorch)
- visualizar datos en dashboard y fichas

## Demo local portable

Despues de instalar `requirements.txt`, este comando genera y verifica `60` jugadores sinteticos, crea un administrador local e inicia la aplicacion:

```powershell
.\.venv\Scripts\python.exe .\scripts\iniciar_demo.py
```

Abrir `http://127.0.0.1:5000/` con usuario `profesor_demo` y contrasena `DemoProfesor123`. Los datos son ficticios y reproducibles con semilla `42`; no representan futbolistas reales. La guia completa para Windows y Linux/macOS esta en [GUIA_DEMO_PROFESOR.md](GUIA_DEMO_PROFESOR.md).

## Stack

- Python + Flask
- SQLAlchemy + SQLite/PostgreSQL
- pandas + scikit-learn para preprocesamiento
- PyTorch + scikit-learn (pipeline y metricas)
- Bootstrap + Chart.js

## Estructura del proyecto

- `scouting_app/`: aplicacion principal (backend, templates, logica y scripts operativos)
- `tests/`: pruebas automatizadas (auth, paginas, permisos)
- `docs/flujo_reproducible_mvp.md`: corrida oficial para regenerar datos, modelo y evaluacion
- `docs/guia_indicadores_app.md`: explicacion de los indicadores visibles de proyeccion
- `docs/contexto_para_nuevo_chat.md`: resumen compacto para continuar el proyecto sin perder contexto
- `docs/comparacion_falencias_codigo_fuente_2026-04-27.md`: estado punto por punto frente al informe de codigo fuente
- `docs/explicacion_cambios_revision_codigo_2026-04-27.md`: explicacion simple de los ultimos cambios de hardening
- `docs/model_training_evidence.md`: evidencia tecnica del entrenamiento y comparacion con baseline
- `docs/model_training_plan.md`: plan tecnico vigente del modelo
- `docs/auditoria_pendientes_2026-05-17.md`: riesgos vivos y cierre por fases de auditoria
- `docs/cierre_pre_entrega_word_render_2026-05-18.md`: cierre previo a entrega con Word final y deploy Render
- `scripts/smoke_render.py`: smoke HTTP contra la URL real de Render
- `scripts/iniciar_demo.py`: genera y ejecuta una demo local portable con 60 jugadores
- `GUIA_DEMO_PROFESOR.md`: instrucciones de evaluacion, reinicio y alcance de los datos demo
- `render.yaml`: configuracion de deploy en Render
- `RUNBOOK.md`: guia operativa (healthcheck, backup/restore, admin, incidentes)

## Ramas de trabajo

- `training`: rama estable cerrada con las correcciones del MVP.
- `ux-crud-polish`: rama actual de pulido UX/UI y cierre de auditoria, sincronizada con GitHub.
- `auditoria-correcciones-mvp`: rama de correcciones de auditoria ya mergeada en `ux-crud-polish`.

## Estado actual

- Fecha de referencia: 2026-08-26.
- Escala de atributos tecnicos, fisicos en escala y reportes scout: `1-20`.
- Potencial bajo: menor a `60%`; medio: `60%` a `79%`; alto: `80%` o mas.
- La edad y categoria juvenil se derivan de `birth_date`; `Player.age` queda como compatibilidad operativa.
- Tests al ultimo cierre: `87 passed, 1 skipped, 4 warnings`.
- La app, los artefactos ML y los diagramas auditados estan alineados con la entrega escrita. La evidencia de Render es historica; la disponibilidad actual del servicio no es un requisito para ejecutar y revisar el MVP localmente.

## Bases de datos del MVP

El proyecto usa dos bases separadas. En local pueden ser SQLite y en despliegue pueden ser PostgreSQL:

- `scouting_app/players_updated_v2.db`: base operativa (maximo 100 jugadores evaluables)
- `scouting_app/players_training.db`: base de entrenamiento (dataset sintetico para el modelo)

En PostgreSQL, la app acepta URLs `postgresql://...` y `postgres://...`; internamente las normaliza para SQLAlchemy con `psycopg`.

Las bases son generadas localmente y no son la fuente principal del repo. Para regenerarlas, seguir `docs/flujo_reproducible_mvp.md`.

Para que Render pueda ejecutar inferencia sin entrenar en produccion, el repo incluye
solo los artefactos chicos de runtime:

- `scouting_app/model.pt`
- `scouting_app/preprocessor.joblib`
- `scouting_app/probability_calibrator.joblib`

Las bases SQLite, metadata de entrenamiento y splits siguen fuera de Git.

## Ejecucion local

Para una demo autocontenida con datos, usar el comando portable indicado arriba. La ejecucion manual basica es:

Windows y Linux CPU usan el mismo snapshot que CI y Render:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe -m pip install -r requirements-torch-cpu.txt
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe scouting_app\app.py
```

En macOS se instala `requirements.txt` en lugar de los dos primeros archivos,
porque PyTorch publica allí el wheel estándar sin sufijo `+cpu`. Esta rama fue
probada localmente en Windows 11 con Python 3.11.9; el workflow de CI configura
Linux con Python 3.11 y 3.12. Este cambio local aún no tiene una corrida remota y
no se afirma una ejecución probada en macOS.

`requirements.txt` contiene dependencias directas, `requirements-lock.txt` fija
el runtime transitivo CPU, `requirements-dev.txt` contiene pruebas/auditoría/lint
y `requirements-docs.txt` separa `python-docx` de la aplicación.

Abrir en navegador:

- `http://127.0.0.1:5000/`

## Usuario local

El usuario administrador se crea por variable de entorno o con `create_admin.py`.

Ejemplo local:

```powershell
$env:APP_DB_URL = "sqlite:///players_updated_v2.db"
$env:ADMIN_USERNAME = "admin"
$env:ADMIN_PASSWORD = "AdminDemo123"
.\.venv\Scripts\python.exe .\scouting_app\create_admin.py
```

Nota: en deploy (Render) la clave de admin se configura por variable de entorno (`ADMIN_PASSWORD`).

## Política de autenticación

- El límite de intentos fallidos se aplica por nombre de usuario normalizado.
  No usa `X-Forwarded-For`, porque el proyecto no tiene documentada una cadena
  fija de proxies confiables y ese header puede ser controlado por el cliente.
- Las solicitudes protegidas vuelven a consultar el usuario y su rol en la base.
  Si el usuario fue eliminado o su rol dejó de ser válido, la sesión se invalida.
- La aplicación no define actualmente un campo de usuario activo/inactivo. La
  disponibilidad comprobable en este MVP es la existencia del usuario y un rol
  reconocido en la base.

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest -q --cov=scouting_app --cov-report=term-missing
```

Con reporte XML de cobertura:

```powershell
.\.venv\Scripts\python.exe -m pytest -q --cov=scouting_app --cov-report=term-missing --cov-report=xml
```

Los controles graduales usados por CI son:

```powershell
.\.venv\Scripts\ruff.exe check scouting_app tests scripts --select E9,F63,F7,F82
.\.venv\Scripts\python.exe -m pip_audit --local --progress-spinner off
```

CI exige cobertura total mínima de `80%`, igual a la cobertura medida el
2026-10-05; el umbral no reemplaza la revisión de cobertura por módulo.

Smoke visual opcional con Playwright:

```powershell
$env:RUN_PLAYWRIGHT = "1"
.\.venv\Scripts\python.exe -m playwright install chromium
.\.venv\Scripts\python.exe -m pytest -q tests\test_visual_smoke.py
Remove-Item Env:\RUN_PLAYWRIGHT
```

## Imagen de jugador

Los jugadores sin una foto manual utilizan la silueta local `scouting_app/static/img/player-silhouette.svg`. La aplicación también reemplaza los avatares DiceBear heredados por este recurso local. Las cargas manuales pueden usar una ruta propia bajo `/static/` o una imagen externa mediante HTTPS; por eso una foto personalizada sí puede depender de su servidor externo.

## Healthcheck

- Endpoint: `GET /health`
- Esperado públicamente: `200` con JSON `status=ok`. Los contadores de calidad
  se consultan con autenticación administrativa desde `Configuracion`.

## Artefactos de inferencia

`model.pt`, `preprocessor.joblib` y `probability_calibrator.joblib` son artefactos
generados y controlados por el proyecto. No existe una función para cargarlos
desde la interfaz. Los formatos PyTorch y joblib usan mecanismos de
deserialización que requieren confiar en el origen: no se deben sustituir por
archivos recibidos de terceros. PyTorch recomienda cargar `state_dict` y usar
`weights_only=True`; TPScouting aplica ambas medidas para `model.pt`. Referencias
oficiales: [persistencia de modelos de scikit-learn](https://scikit-learn.org/stable/model_persistence.html)
y [`torch.load`](https://docs.pytorch.org/docs/stable/generated/torch.load.html).

El 2026-10-05 se comprobó que los tres artefactos existentes cargan con
PyTorch `2.9.1+cpu`, scikit-learn `1.8.0` y joblib `1.5.3`. Esa es una prueba de
compatibilidad actual, no evidencia de las versiones usadas para crearlos: el
metadata histórico no registró las versiones de las bibliotecas.

## Mantenimiento operativo

- `Configuracion` incluye una auditoria de calidad para la base operativa
- La limpieza operativa elimina registros legacy inconsistentes (por ejemplo, jugadores sin identificador) sin inventar datos faltantes

## Deploy (Render)

El repositorio incluye `render.yaml` con:

- `APP_DB_URL` (base operativa)
- `TRAINING_DB_URL` (base de entrenamiento)
- `EVAL_POOL_MAX=100`
- variables de seguridad y logging

El blueprint actual deja preparado el deploy en Render con PostgreSQL administrado.
En modalidad gratuita se usa una sola base PostgreSQL Free (`tpscouting-mvp-db`),
porque Render limita las bases Free activas por workspace. En ese modo,
`APP_DB_URL` y `TRAINING_DB_URL` apuntan a la misma base, `AUTO_TRAIN_ON_STARTUP`
queda desactivado y el deploy ejecuta `seed_demo_data.py` para cargar 100 jugadores
demo solo si la base esta vacia. No ejecutar el pipeline de entrenamiento desde la
web en este modo gratuito.

La política vigente de Render indica que PostgreSQL Free expira 30 días después
de su creación, admite 1 GB, no incluye backups y permite una sola instancia Free
activa por workspace. Por eso sirve para demostración temporal y no debe tratarse
como almacenamiento durable. No se cambió ni contrató ningún plan. Fuente:
[Render, Deploy for Free](https://render.com/docs/free).

Smoke real de Render:

```powershell
$env:RENDER_SMOKE_BASE_URL = "https://TU_SERVICIO.onrender.com"
$env:SMOKE_USERNAME = "admin"
$env:SMOKE_PASSWORD = "AdminDemo123"
.\.venv\Scripts\python.exe scripts\smoke_render.py
```

Deploy MVP verificado el 2026-05-19:

- URL: `https://tpscouting-mvp.onrender.com`
- Rama: `render-free-deploy`
- `/health`, `/login`, login admin, `/dashboard`, `/players`, comparadores,
  `/settings` y `/players/import` respondieron correctamente.
- Ajuste posterior: listado de jugadores y comparador multiple usan cache GET
  in-memory; Render Free usa 20 jugadores por pagina y cache TTL de 300 segundos.

El documento final presenta el alcance real: MVP academico con dataset sintetico, cache y
rate limiting in-memory, migraciones manuales y validacion externa pendiente. La URL de
Render se conserva como evidencia historica y no como garantia de disponibilidad actual.

Seguridad MVP:

- `APP_SECRET_KEY` es obligatoria en producción/Render.
- Los formularios POST mutantes usan CSRF.
- El logout se ejecuta por POST con CSRF.
- El login tiene rate limiting en memoria por usuario normalizado. Es suficiente para el MVP académico, pero no es un limitador distribuido para producción multi-instancia.

## Alcance del MVP

Este proyecto es un MVP academico. La prediccion de potencial se valida con dataset sintetico y se usa como soporte a la toma de decisiones, no como reemplazo del criterio del cuerpo tecnico.
