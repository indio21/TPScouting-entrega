# Modelo ML

## Objetivo

El modelo estima un score de potencial para priorizar revision de jugadores juveniles. No reemplaza el criterio del cuerpo tecnico.

## Pipeline

- Datos tabulares con atributos, posicion, edad, historial deportivo y senales agregadas.
- Preprocesamiento con `pandas` y `scikit-learn`.
- `ColumnTransformer`, imputacion, escalado y one-hot encoding.
- Modelo PyTorch `PlayerNet`.
- Comparacion con baseline `LogisticRegression(class_weight="balanced")`.

## Artefactos versionados

- `scouting_app/model.pt`
- `scouting_app/preprocessor.joblib`
- `scouting_app/probability_calibrator.joblib`

Estos artefactos permiten inferencia sin reentrenar durante la demo.

## Ultima evidencia documentada

La corrida documentada mas reciente usa dataset sintetico de `20.000` jugadores juveniles, split `14.000 / 3.000 / 3.000`, seed `42` e `input_dim=68`.

Metricas de test registradas para PyTorch:

- Accuracy: `0.9303`
- ROC-AUC: `0.9174`
- PR-AUC: `0.5241`
- F1: `0.5282`

Baseline `LogisticRegression(class_weight="balanced")` en la misma evidencia:

- Accuracy: `0.9310`
- ROC-AUC: `0.9205`
- PR-AUC: `0.5378`
- F1: `0.5327`

Conclusion tecnica: el baseline lineal queda levemente por encima en esa corrida. Por eso la documentacion no afirma superioridad general de PyTorch; el modelo se mantiene como parte del MVP y la comparacion queda explicitada.

## Limites

- Datos sinteticos.
- Sin validacion externa con datos reales.
- La calibracion se conserva como referencia secundaria.
- Cada reentrenamiento debe volver a comparar contra baseline.
