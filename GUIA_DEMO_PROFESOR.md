# Guia De Evaluacion Rapida

## Derechos y licencia

El repositorio se entrega para su revisión académica y no incluye un archivo
`LICENSE`. Todos los derechos se encuentran reservados; la entrega no concede
una autorización general de copia, modificación o redistribución.

Esta guia permite revisar la aplicacion sin descargar una base de datos externa, ejecutar el entrenamiento completo ni depender de Render.

## Alcance de los datos

- La demo genera `60` jugadores juveniles ficticios.
- Todos los nombres, documentos, fechas, atributos, partidos, evaluaciones e informes son sinteticos.
- No cargue información de jugadores reales. El MVP no está habilitado para
  tratar datos personales de menores; consulte `docs/REAL_DATA_READINESS.md`.
- La semilla fija `42` permite reproducir el mismo conjunto inicial.
- Las edades abarcan de 12 a 18 anos.
- La edad se calcula desde la fecha de nacimiento y la categoria corresponde al ano de nacimiento.
- Las predicciones sirven para demostrar el flujo tecnico del MVP; no constituyen validacion con futbolistas reales.

## Preparacion del entorno

Desde la raiz del repositorio:

### Windows PowerShell

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe -m pip install -r requirements-torch-cpu.txt
.\.venv\Scripts\python.exe .\scripts\iniciar_demo.py
```

### Linux CPU

```bash
python3.11 -m venv .venv
./.venv/bin/python -m pip install --upgrade pip
./.venv/bin/python -m pip install -r requirements-lock.txt
./.venv/bin/python -m pip install -r requirements-torch-cpu.txt
./.venv/bin/python ./scripts/iniciar_demo.py
```

### macOS Apple Silicon

```bash
python3.11 -m venv .venv
./.venv/bin/python -m pip install --upgrade pip
./.venv/bin/python -m pip install -r requirements.txt
./.venv/bin/python ./scripts/iniciar_demo.py
```

Luego abrir `http://127.0.0.1:5000/`.

La versión fijada de PyTorch dispone de wheel para macOS arm64, pero no para macOS
Intel. En equipos Intel, la instalación de `requirements.txt` se detendrá de forma
explícita; no se recomienda sustituir PyTorch por otra versión sin repetir las
pruebas de compatibilidad y de los artefactos.

Windows/Linux CPU usan el mismo runtime fijado que CI, Render y Docker. macOS
Apple Silicon usa `requirements.txt`, porque su wheel estándar no proviene del
índice CPU. Python soportado: 3.11 y 3.12; el runner macOS verifica Python 3.11.

Credenciales locales de evaluacion:

```text
Usuario: profesor_demo
Contrasena: DemoProfesor123
```

Estas credenciales son publicas e intencionales para una demostracion local. No deben utilizarse en un despliegue real.

## Que puede verificarse

- Listado, alta, edicion y baja de jugadores.
- Edad, fecha de nacimiento y categoria por ano de nacimiento.
- Historiales tecnicos y de rendimiento.
- Partidos, evaluaciones fisicas, disponibilidad e informes scout.
- Comparacion de dos jugadores y ranking multiple.
- Prediccion de potencial con los artefactos ML incluidos.
- Panel general y control de calidad en Configuracion.

Los cambios realizados desde la interfaz se guardan en `scouting_app/demo_profesor.db`. Esta base es local y esta ignorada por Git.

## Reiniciar la demostracion

Detener primero la aplicacion con `Ctrl+C` y ejecutar:

### Windows PowerShell

```powershell
.\.venv\Scripts\python.exe .\scripts\iniciar_demo.py --recrear
```

### Linux o macOS Apple Silicon

```bash
./.venv/bin/python ./scripts/iniciar_demo.py --recrear
```

La opcion `--recrear` elimina exclusivamente la base local `demo_profesor.db` y vuelve a generar los 60 jugadores sinteticos.

## Verificacion sin iniciar el servidor

```powershell
.\.venv\Scripts\python.exe .\scripts\iniciar_demo.py --recrear --solo-preparar
```

El comando falla con codigo distinto de cero si encuentra fechas faltantes, edades inconsistentes, categorias faltantes o ausencia de administrador.
