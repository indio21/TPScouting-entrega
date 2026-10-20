# Verificación de la entrega — 20/10/2026

## Archivos finales

- [Word](documento/TRABAJO_FINAL_TPScouting_ENTREGA_FINAL_ARIAL_2026-10-20.docx)
  - SHA-256: `29ACEE4C6117C143FA7086E7F15A2FDB27F6716ADE3261BD23DC32F960BCB03E`.
- [PDF](documento/TRABAJO_FINAL_TPScouting_ENTREGA_FINAL_ARIAL_2026-10-20.pdf)
  - SHA-256: `E97B4902D8A02AEC2C883CF0A02A96E8782ADBD48C1FCF861BEF755786B2944B`.

El PDF tiene 97 páginas. Se verificaron 88 entradas del índice y 45 entradas de
figuras/tablas, numeración continua e inicios independientes de los preliminares
y la bibliografía. El texto editable está en Arial, con tamaño Word mínimo de
10 pt. Las capturas y diagramas mantienen su tipografía como imágenes.

## Código funcional evaluado

Commit: `2a27d53ae4f8a4849bee7d1bea362fd9146886b5`.

- [CI 39509728373](https://github.com/indio21/TPScouting-entrega/actions/runs/39509728373): aprobada, Python 3.11/3.12, Ruff y auditoría de dependencias.
- [Container 39509728476](https://github.com/indio21/TPScouting-entrega/actions/runs/39509728476): construcción y smoke aprobados.
- [macOS 39509728473](https://github.com/indio21/TPScouting-entrega/actions/runs/39509728473): compatibilidad Apple Silicon aprobada.

La validación local en Windows/Python 3.11.9 obtuvo **131 passed, 1 skipped,
4 warnings y 83,31 % de cobertura**. El smoke visual de Playwright es opt-in y
permanece omitido en esa suite; las cuatro advertencias conocidas corresponden a
columnas completamente NaN en dos pruebas de preprocesamiento. La demo funcional
adicional se comprobó con 60 jugadores sintéticos en una base SQLite temporal.

Las correcciones incluyen rechazo controlado de credenciales incompletas y de
tokens CSRF no ASCII, además del cierre de la sesión de base ante errores del
verificador de contraseña. Los artefactos ML históricos se conservaron.

Las capturas CI de las figuras 6-8 y 10-1 muestran una ejecución anterior, cuyo
alcance histórico se explicita en el Word. La validación de la versión corregida
corresponde a las ejecuciones enlazadas arriba.

## Identificación de la publicación documental

El Word cita el commit funcional evaluado. El commit que incorpora estos archivos
se consulta en el historial de Git sin introducir en el documento su propio hash
futuro:

```bash
git log -1 --format="%H %s" -- documento/TRABAJO_FINAL_TPScouting_ENTREGA_FINAL_ARIAL_2026-10-20.docx documento/TRABAJO_FINAL_TPScouting_ENTREGA_FINAL_ARIAL_2026-10-20.pdf
```

La demostración utiliza datos sintéticos y no constituye validación deportiva
con jugadores reales ni habilita el tratamiento de datos reales de menores.
