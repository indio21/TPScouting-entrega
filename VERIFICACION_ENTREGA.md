# Verificación de la entrega — correcciones de noviembre, 12/11/2026

## Documento final actualizado

- [Word](<documento/Trabajo Final_Solari Pablo_CORRECCIONES_NOVIEMBRE_12-11-2026.docx>)
  - SHA-256: `C6BEB90582E818D371AC9CF96EBAB0C1A1DA025C6DE689DC2383EB606DBFDC79`.
- [PDF](<documento/Trabajo Final_Solari Pablo_CORRECCIONES_NOVIEMBRE_12-11-2026.pdf>)
  - SHA-256: `E43883FC89601B5E14B7144BA885BD6454C84C2FBCF0CC139B9DBC5621EAE5F0`.

Esta copia incorpora exclusivamente las cuatro correcciones docentes de
noviembre: retirada de la leyenda huérfana, descripción de /health alineada
con el código, URL del smoke sin concatenación y explicación del límite de
intentos de acceso. Después se renumeraron las tablas y referencias afectadas.
La tabla de endpoints es ahora Tabla 4-5. La versión original del 21/10 se
conserva como respaldo en el repositorio propio y en el historial de Git.

El PDF tiene 97 páginas. Se verificaron 88 entradas del índice y 44 entradas de
listas (25 figuras y 19 tablas), con sus páginas y alineación correctas.
Las 19 tablas y 25 figuras tienen leyendas asociadas; la Figura 6-5 incluye
sus tres imágenes a), b), c). Se comprobaron 107 referencias, la secuencia de
84 títulos numerados y 23 apariciones de URL, sin destinos internos ausentes,
saltos de numeración ni URL concatenadas o duplicadas accidentalmente.

Se conservan las 28 imágenes, tipografía Arial, tamaños y formato originales,
19 tablas y nueve secciones. No cambiaron métricas, resultados ni bibliografía.
La comparación confirmó que el resto del contenido permanece intacto.
No se modificaron código, modelos, dependencias ni configuración en noviembre.

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
git log -1 --format="%H %s" -- "documento/Trabajo Final_Solari Pablo_CORRECCIONES_NOVIEMBRE_12-11-2026.docx" "documento/Trabajo Final_Solari Pablo_CORRECCIONES_NOVIEMBRE_12-11-2026.pdf"
```

Consultar la [CI de main](https://github.com/indio21/TPScouting-entrega/actions/workflows/ci.yml)
y comprobar que el SHA del run corresponda al commit publicado.
La disponibilidad actual de Render no se certifica con esas pruebas.
La demo usa datos sintéticos: no constituye validación deportiva externa ni
habilita el tratamiento de datos reales de menores.
