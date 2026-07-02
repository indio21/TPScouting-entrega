# Puesta en marcha

## 1. Instalar dependencias

```powershell
cd C:\Tesis\TPScouting-entrega
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
```

## 2. Generar datos demo

```powershell
$env:DEMO_SEED_ON_STARTUP = "1"
$env:APP_DB_URL = "sqlite:///players_updated_v2.db"
.\.venv\Scripts\python.exe .\scouting_app\seed_demo_data.py
```

## 3. Crear administrador

```powershell
$env:ADMIN_USERNAME = "admin_local"
$env:ADMIN_PASSWORD = "Cambiar123"
.\.venv\Scripts\python.exe .\scouting_app\create_admin.py
```

## 4. Ejecutar la aplicacion

```powershell
cd .\scouting_app
$env:APP_SECRET_KEY = "dev-secret-change-me"
$env:APP_DB_URL = "sqlite:///players_updated_v2.db"
$env:TRAINING_DB_URL = "sqlite:///players_training.db"
..\.venv\Scripts\python.exe app.py
```

URL local:

```text
http://127.0.0.1:5000/
```

## 5. Ejecutar tests

```powershell
cd C:\Tesis\TPScouting-entrega
.\.venv\Scripts\python.exe -m pytest -q
```

## 6. Regenerar entrenamiento completo

Desde `scouting_app/`:

```powershell
..\.venv\Scripts\python.exe generate_data.py --num-players 20000 --db-url sqlite:///players_training.db --seed 42 --min-age 12 --max-age 18 --reset
..\.venv\Scripts\python.exe train_model.py --db-url sqlite:///players_training.db --model-out model.pt --preprocessor-out preprocessor.joblib --calibrator-out probability_calibrator.joblib --metadata-out training_metadata.json --splits-out training_splits.json --epochs 45 --lr 5e-4 --patience 10
..\.venv\Scripts\python.exe evaluate_saved_model.py --db-url sqlite:///players_training.db --metadata-path training_metadata.json
..\.venv\Scripts\python.exe sync_shortlist.py --src-db sqlite:///players_training.db --dst-db sqlite:///players_updated_v2.db --limit 100 --min-age 12 --max-age 18 --replace
```

Las bases y metadata generadas quedan fuera de Git por diseno.
