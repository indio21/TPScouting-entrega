# Guia De Evaluacion Rapida

Esta guia permite revisar la aplicacion sin descargar una base de datos externa, ejecutar el entrenamiento completo ni depender de Render.

## Alcance de los datos

- La demo genera `60` jugadores juveniles ficticios.
- Todos los nombres, documentos, fechas, atributos, partidos, evaluaciones e informes son sinteticos.
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
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe .\scripts\iniciar_demo.py
```

### Linux o macOS

```bash
python3.11 -m venv .venv
./.venv/bin/python -m pip install --upgrade pip
./.venv/bin/python -m pip install -r requirements.txt
./.venv/bin/python ./scripts/iniciar_demo.py
```

Luego abrir `http://127.0.0.1:5000/`.

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

### Linux o macOS

```bash
./.venv/bin/python ./scripts/iniciar_demo.py --recrear
```

La opcion `--recrear` elimina exclusivamente la base local `demo_profesor.db` y vuelve a generar los 60 jugadores sinteticos.

## Verificacion sin iniciar el servidor

```powershell
.\.venv\Scripts\python.exe .\scripts\iniciar_demo.py --recrear --solo-preparar
```

El comando falla con codigo distinto de cero si encuentra fechas faltantes, edades inconsistentes, categorias faltantes o ausencia de administrador.
