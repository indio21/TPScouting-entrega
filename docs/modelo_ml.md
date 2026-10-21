# Modelo ML

## Alcance y pipeline

El MVP integra aprendizaje supervisado mediante `PlayerNet`, una red MLP de
PyTorch. Se entrena y evalúa con datos sintéticos juveniles de 12 a 18 años;
sus resultados no demuestran capacidad predictiva sobre futbolistas reales.

`preprocessing.py` transforma edad, posición, atributos y agregados de historial,
partidos, informes scout, evaluaciones físicas y disponibilidad. Usa imputación,
`MinMaxScaler` y `OneHotEncoder`. El runtime incluido transforma 64 columnas
de entrada en 68 columnas; `potential_label` y `temporal_target_label` no son
features. La etiqueta objetivo es `temporal_target_label`, definida por diseño.

El modelo tiene 17.608 parámetros entrenables y 386 elementos de buffers.
El `state_dict` contiene 17.994 elementos, que no son todos parámetros.

## Artefactos de runtime

- `scouting_app/model.pt`
- `scouting_app/preprocessor.joblib`
- `scouting_app/probability_calibrator.joblib`

Permiten ejecutar la demo sin entrenar. Solo deben proceder del proyecto o de
una fuente confiable: joblib y PyTorch deserializan objetos. La aplicación carga
el `state_dict` con `weights_only=True` y no admite cargas arbitrarias de modelos.

## Evidencia histórica y salidas distintas

Los artefactos pertenecen a la corrida histórica del 19/05/2026, con 20.000
jugadores sintéticos, seed 42 y split 14.000/3.000/3.000. Los umbrales de
clasificación se eligieron en validación y se evaluaron finalmente en test.

| Variante histórica | ROC-AUC | PR-AUC |
|---|---:|---:|
| PlayerNet, salida cruda | 0,920341 | 0,546127 |
| PlayerNet, referencia calibrada | 0,917431 | 0,524116 |
| Regresión logística balanceada | 0,920506 | 0,537776 |
| Score combinado desde salida cruda | 0,910888 | 0,506254 |

La accuracy 0,9303 y F1 0,5282 citadas para PlayerNet corresponden a la variante
calibrada; no deben atribuirse a la salida cruda o al score combinado.
El combinado agrega heurísticas de historial y adecuación posicional; esa
evaluación no respalda presentarlo como mejora de la salida cruda.
Tampoco se afirma superioridad general de PlayerNet sobre el baseline.

Las bandas visuales 0,60 y 0,80 son reglas de presentación, distintas de los
umbrales optimizados en validación. La aplicación conserva el combinado como
indicador heurístico visible y la probabilidad cruda como salida del modelo.
Véase [guia_indicadores_app.md](guia_indicadores_app.md).

## Corrida candidata y límites

El trabajo escrito compara la histórica con una corrida candidata cuyos
cuantiles y umbrales de etiqueta se ajustan solo sobre train y se congelan para
validation/test. Esa corrida no reemplazó el runtime incluido. La cuota sintética
del 8 % es una decisión de diseño, no una prevalencia deportiva observada.

La corrida histórica calculaba cuantiles/cuotas antes del split. Sus versiones
originales de bibliotecas, duración y curva de loss de validación están
incompletas y no pueden reconstruirse como hechos históricos.

Los datasets, caches y candidatos no se distribuyen con la demo. Un clon de
entrega puede ejecutar la aplicación y sus pruebas, pero no reconstruir
exactamente la corrida histórica solo con los tres artefactos. Las corridas
nuevas deben guardar dataset, splits, configuración, versiones, commit y hashes
en un directorio separado; véase [RUNBOOK.md](../RUNBOOK.md).
