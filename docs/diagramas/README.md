# Diagramas PlantUML TPScouting

Diagramas tecnicos incluidos para explicar arquitectura, modelo de datos, flujos principales y casos de uso del MVP.

## Fuentes

- `plantuml/01_secuencia_prediccion.puml`
- `plantuml/02_secuencia_dashboard.puml`
- `plantuml/03_componentes.puml`
- `plantuml/04_despliegue.puml`
- `plantuml/05_clases.puml`
- `plantuml/06_casos_uso_general.puml`
- `plantuml/07_casos_uso_gestion_jugadores.puml`
- `plantuml/08_casos_uso_analisis_decision.puml`

## Exportaciones

Cada diagrama se incluye en PNG y SVG dentro de `export/`.

## Alcance

Los diagramas reflejan la implementacion real:

- Flask con blueprints por familia funcional.
- SQLAlchemy.
- SQLite para desarrollo local.
- PostgreSQL administrado en despliegue Render.
- PyTorch + preprocesador scikit-learn.
- Artefactos `model.pt`, `preprocessor.joblib` y `probability_calibrator.joblib`.

No se incluye SQL Server porque no forma parte de la implementacion actual.
