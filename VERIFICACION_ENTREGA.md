# Verificación de la entrega — revisión del 20/10/2026

## Documento final designado por el autor

- [Word](<documento/Trabajo Final_Solari Pablo_21-10-2026.docx>)
  - SHA-256: `81A743DF38FB32820EA7DC3BEC247F57CCE6F3F826510DFEA2ED9CF3D63A131C`.
- [PDF](<documento/Trabajo Final_Solari Pablo_21-10-2026.pdf>)
  - SHA-256: `587FB8393C6ED864A2FE6399548E458F8E2654943DD939649ED01856C31DBCC2`.

El nombre con fecha 21/10/2026 fue elegido por el autor. El Word se conserva
idéntico al archivo proporcionado, sin guardar cambios automáticos de campos
ni formato. Su PDF fue exportado con Microsoft Word y tiene 97 páginas.
Se respetan el nuevo título con Machine Learning y los cambios de redacción.

Se verificaron 88 entradas del índice y 45 de figuras/tablas contra las páginas
reales. Los 45 números de página de las listas están alineados a la derecha.
Se conservan 28 imágenes, 19 tablas y 9 secciones, numeración continua e inicios
independientes de preliminares y bibliografía. Sin campos rotos, páginas vacías
ni tamaños Word explícitos menores de 10 pt. El PDF usa únicamente Arial y sus
variantes textuales; las imágenes conservan su tipografía interna.

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

`pip-audit` no detectó vulnerabilidades conocidas en los manifiestos revisados
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
git log -1 --format="%H %s" -- "documento/Trabajo Final_Solari Pablo_21-10-2026.docx" "documento/Trabajo Final_Solari Pablo_21-10-2026.pdf"
```

Consultar la [CI de main](https://github.com/indio21/TPScouting-entrega/actions/workflows/ci.yml)
y comprobar que el SHA del run corresponda al commit publicado.
La disponibilidad actual de Render no se certifica con esas pruebas.
La demo usa datos sintéticos: no constituye validación deportiva externa ni
habilita el tratamiento de datos reales de menores.
