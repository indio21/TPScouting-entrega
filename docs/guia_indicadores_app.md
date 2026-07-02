# Guia de indicadores de la app

## Rendimiento en posicion

Puntaje de ajuste del jugador a una posicion. No es una probabilidad. Se calcula con atributos ponderados segun el puesto.

## Rendimiento en puesto natural

Mismo concepto que el rendimiento en posicion, aplicado a la posicion registrada actualmente para el jugador.

## Puntaje de ficha

Salida principal del modelo PyTorch para ese jugador. Sirve como score de priorizacion para revision scout.

## Probabilidad combinada

Score final mostrado por la app. Combina:

- puntaje de ficha del modelo;
- senales historicas de rendimiento;
- ajuste del jugador a su posicion.

## Referencia calibrada

La calibracion se conserva como referencia secundaria cuando existe calibrador. No reemplaza el score principal del MVP.

## Frase de presentacion

El puntaje de ficha estima el potencial a partir de datos del jugador. La probabilidad combinada ajusta esa estimacion con rendimiento historico y adecuacion posicional. El rendimiento en posicion es un puntaje de 1 a 20, no una probabilidad.
