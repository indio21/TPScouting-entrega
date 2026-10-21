# Guía de indicadores de la aplicación

## Rendimiento en posición y puesto natural

Es un puntaje de adecuación calculado con atributos ponderados según el puesto.
Se expresa en la escala de atributos 1–20 y no es una probabilidad. En el puesto
natural se aplica a la posición registrada del jugador.

## Puntaje de ficha

Es la salida cruda de `PlayerNet`, obtenida mediante sigmoid y presentada como
porcentaje. Permite priorizar revisión scout; no equivale a una probabilidad
validada de desarrollo deportivo real.

## Probabilidad combinada

Es el nombre de interfaz para el score heurístico combinado. Con los valores
predeterminados mezcla:

- 35 % de salida cruda del modelo;
- 35 % del promedio histórico `final_score`, dividido por 10;
- 30 % de adecuación posicional, dividida por 20.

Si falta una señal opcional, renormaliza los pesos disponibles. El resultado
se limita al intervalo 0–0,99. Los pesos se configuran con `POT_W_MODEL`,
`POT_W_RATING` y `POT_W_FIT`; la fórmula está en `scouting_app/ml/scoring.py`.
La comparación sobre test sintético no mostró una mejora sobre la salida cruda;
el combinado se mantiene como heurística secundaria.

## Referencia calibrada

Cuando hay calibrador, el sistema transforma la salida cruda y aplica a esa
salida la misma combinación con historial y posición. La referencia calibrada
mostrada en Predicción es el combinado desde la salida calibrada, distinto de
la probabilidad calibrada aislada evaluada en las tablas ML.

## Bandas de presentación

Con los valores predeterminados:

- Bajo potencial: menos de 0,60.
- Potencial medio: desde 0,60 hasta menos de 0,80.
- Alto potencial: desde 0,80.

Son bandas visuales, distintas de los umbrales optimizados en validación.
El `potential_label` persistido puede actualizarse según el score, pero no entra
como feature del modelo.

Estos indicadores se demuestran con datos sintéticos y sirven de apoyo a la
revisión humana. No deben utilizarse para decidir sobre una persona real.
