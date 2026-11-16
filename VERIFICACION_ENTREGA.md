# Verificación de la entrega — versión del autor, 16/11/2026

## Documento final indicado por el autor

- [Word](<documento/Trabajo Final_Solari Pablo_CORRECCIONES_NOVIEMBRE_16-11-2026.docx>)
  - SHA-256: `487A80BCC882168A59C5E9D952CCA7184FB089AC684B5734AF4508E9B06BB6D5`.
- [PDF](<documento/Trabajo Final_Solari Pablo_CORRECCIONES_NOVIEMBRE_16-11-2026.pdf>)
  - SHA-256: `BC651E98A002E9CA83BF6CACC3CA8C9318CF6F166B37A6BC4801CE3C44D365EF`.

El autor corrigió el Word y lo indicó expresamente como el documento que enviará.
Se publicó una copia idéntica por SHA-256 al archivo recibido del 16/11/2026.
El PDF correspondiente se exportó con Microsoft Word y tiene 97 páginas.
No se editó el Word ni se solicitaron actualizaciones de campos, índices o listas.
Esta copia incluye las últimas correcciones de puntuación y concordancia
realizadas por el autor, y la página corregida de la Tabla 6-3 en la lista (53).

El repaso solicitado se limitó a tipografía, ortografía, gramática y
correspondencia del índice con títulos y páginas. Se comprobaron las 88 entradas
del índice general y las 44 entradas de figuras/tablas, sin diferencias de
títulos o páginas. La tipografía visible es Arial, con tamaños nominales de
10 puntos o más. Fue un repaso rápido, no una corrección lingüística exhaustiva
ni una revisión técnica del contenido o del código. Los archivos publicados
coinciden por SHA-256 con los de esta revisión.

La versión anterior del 12/11 permanece en el historial de Git y como respaldo
local en el proyecto principal. Sus comprobaciones estructurales corresponden
a esa versión y no se presentan como una auditoría del nuevo documento.
No se modificaron código, modelos, dependencias ni configuración de ejecución.

## Código y comprobaciones

El código de aplicación y los modelos siguen siendo los del commit funcional
`2a27d53ae4f8a4849bee7d1bea362fd9146886b5`, citado en el Word.

- [CI funcional](https://github.com/indio21/TPScouting-entrega/actions/runs/39509728373): Python 3.11/3.12, Ruff y auditoría aprobados.
- [Container funcional](https://github.com/indio21/TPScouting-entrega/actions/runs/39509728476): construcción y smoke aprobados.
- [macOS funcional](https://github.com/indio21/TPScouting-entrega/actions/runs/39509728473): Apple Silicon aprobado.

La suite local repetida el 20/10/2026, Windows/Python 3.11.9, obtuvo
**131 passed, 1 skipped, 4 warnings y 83,31 % de cobertura**. La omitida es
Playwright opt-in; las advertencias conocidas provienen de columnas totalmente
NaN en dos pruebas de preprocesamiento. Ruff crítico/gradual, `pip check` y
la demo sintética aislada se comprueban por separado.

Se revisaron los nueve `.md` y cinco `.txt` de entrega: enlaces, ramas/resultados
antiguos, instalación por plataforma, semántica ML y procedimientos operativos.
El lock excluye Gunicorn en Windows y las herramientas de desarrollo fijan
`certifi==2026.10.19`. No se cambió PyTorch ni scikit-learn.

La auditoría `pip-audit` del 20/10/2026 no detectó vulnerabilidades conocidas en los manifiestos revisados
ni en el entorno local tras actualizar certifi y Pillow. Pillow es una
herramienta adicional local y no se añadió al runtime. El wheel `2.14.1+cpu`
se contrasta también como `torch==2.14.1` del manifiesto directo, porque el
escaneo del entorno puede omitir el sufijo local. La auditoría no garantiza
ausencia de vulnerabilidades futuras.

Las capturas CI del documento se mantienen como evidencia histórica; no se
modificaron para representar ejecuciones nuevas.

## Identificación de la publicación

El Word cita el commit funcional probado. Su publicación documental se identifica
externamente para evitar introducir su propio hash futuro dentro del archivo:

```bash
git log -1 --format="%H %s" -- "documento/Trabajo Final_Solari Pablo_CORRECCIONES_NOVIEMBRE_16-11-2026.docx" "documento/Trabajo Final_Solari Pablo_CORRECCIONES_NOVIEMBRE_16-11-2026.pdf"
```

Consultar la [CI de main](https://github.com/indio21/TPScouting-entrega/actions/workflows/ci.yml)
y comprobar que el SHA del run corresponda al commit publicado.
La disponibilidad actual de Render no se certifica con esas pruebas.
La demo usa datos sintéticos: no constituye validación deportiva externa ni
habilita el tratamiento de datos reales de menores.
