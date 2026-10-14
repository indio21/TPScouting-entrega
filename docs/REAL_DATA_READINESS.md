# Preparación para un uso futuro con datos reales

## Estado actual

TPScouting es un MVP académico para jugadores juveniles de 12 a 18 años. La
demo, el entrenamiento y las verificaciones utilizan datos sintéticos. **No se
deben cargar datos personales reales ni utilizar el score para decidir sobre una
persona real.**

El sistema actual no implementa el conjunto de controles institucionales,
legales y operativos requerido para tratar datos de menores. Los roles, CSRF,
validaciones y controles de sesión existentes son medidas técnicas del MVP; no
constituyen por sí solos una habilitación para ese tratamiento.

## Por qué hace falta una etapa previa

El modelo puede almacenar identificación, fecha de nacimiento, fotografía,
atributos deportivos, reportes cualitativos, lesiones, fatiga, evaluaciones
físicas y una estimación de potencial. En un uso real, esta información permite
identificar a menores y parte de ella puede revelar datos de salud.

La [Ley 25.326](https://www.argentina.gob.ar/normativa/nacional/ley-25326-2000-64790/texto)
establece, entre otros aspectos, finalidad y calidad de los datos, consentimiento
e información previa, seguridad, confidencialidad y derechos de acceso,
rectificación y supresión. La
[Ley 26.061](https://www.argentina.gob.ar/normativa/nacional/ley-26061-110778/texto)
protege el interés superior, la vida privada y el derecho de niñas, niños y
adolescentes a ser oídos según su madurez. La
[Resolución AAIP 47/2018](https://www.argentina.gob.ar/normativa/nacional/resoluci%C3%B3n-47-2018-312662/texto)
ofrece medidas de seguridad recomendadas para el tratamiento y conservación de
datos personales informatizados.

Este documento es un diseño de preparación técnica y operativa. La institución
responsable debe validar con asesoramiento competente la base jurídica concreta,
las responsabilidades y los instrumentos aplicables antes de una prueba real.

## Puertas obligatorias antes de usar datos reales

No se habilitará una prueba real hasta que exista evidencia aprobada para todas
las puertas siguientes.

| Puerta | Evidencia mínima de cierre |
|---|---|
| Responsable y finalidad | Institución responsable identificada; finalidad, población, destinatarios y usos prohibidos documentados |
| Revisión jurídica e institucional | Dictamen sobre base jurídica, consentimiento o autorización, representación, registro de base y transferencias aplicables |
| Información y participación | Aviso claro y versionado; mecanismo para representante y para escuchar al menor conforme a edad y madurez |
| Minimización | Inventario campo por campo; justificación de necesidad; DNI, fotografía y datos de salud excluidos salvo necesidad aprobada |
| Derechos | Procedimiento autenticado para acceso, rectificación, actualización, supresión y bloqueo, con responsables y plazos |
| Retención y baja | Plazos por categoría, disparadores de borrado, excepciones legales, verificación de backups y evidencia de ejecución |
| Seguridad | Evaluación de riesgos, cifrado, secretos, accesos de mínimo privilegio, logs, backups, restauración e incidentes probados |
| Proveedores y transferencias | Contratos, ubicación, subencargados, retención y transferencias revisados antes de alojar o compartir datos |
| Modelo y decisión humana | Validación externa, sesgos por edad/maduración, explicación, impugnación y prohibición de decisiones automáticas adversas |
| Piloto controlado | Entorno separado, conjunto mínimo, autorización formal, capacitación, monitoreo y criterio de suspensión |

## Cambios técnicos que todavía faltan

1. Registrar aviso, versión, finalidad, base habilitante, representante y estado
   de autorización sin guardar documentación excesiva.
2. Separar identificadores directos de las observaciones deportivas y usar
   seudónimos fuera de los flujos que requieren identidad.
3. Aplicar permisos por institución/equipo y por función. Los tres roles actuales
   no representan por sí solos una política completa de mínimo privilegio.
4. Incorporar un registro de auditoría protegido para accesos, consultas,
   exportaciones, cambios, predicciones y eliminaciones, evitando volcar datos
   sensibles en los logs.
5. Implementar solicitudes de derechos con verificación de identidad y
   representación, seguimiento, plazos y evidencia de respuesta.
6. Automatizar retención, bloqueo y borrado, incluyendo copias de seguridad,
   derivados, exportaciones y artefactos de entrenamiento.
7. Proteger datos en tránsito y reposo, rotar secretos, revisar dependencias y
   probar backups, restauración y respuesta a incidentes.
8. Versionar datos, modelo y explicación de cada score. Un resultado debe poder
   revisarse sin presentarse como diagnóstico ni como medida objetiva de valor.
9. Impedir que fotografías o recursos externos comuniquen identificadores a
   terceros sin una política aprobada.
10. Separar completamente datos de producción, datos de prueba y datasets de
    entrenamiento. Ningún dato real debe incorporarse al entrenamiento por
    defecto.

## Pruebas exigidas antes de un piloto

- permisos positivos y negativos para cada rol, institución y tipo de dato;
- aislamiento entre instituciones y ausencia de cambios ante accesos denegados;
- acceso, corrección, exportación, bloqueo y supresión de una identidad de prueba;
- cascada y trazabilidad sobre historiales, archivos, cache, backups y derivados;
- vencimiento de retención con reloj controlado y excepciones documentadas;
- cifrado, rotación de secretos, restauración y simulacro de incidente;
- ausencia de datos personales o sensibles en logs y mensajes de error;
- desactivación inmediata de usuarios y revocación de sesiones;
- evaluación del modelo con datos adecuados, revisión de sesgos y supervisión
  humana;
- prueba de retiro del consentimiento o de la habilitación aplicable, sin
  prometer borrado cuando exista una obligación legal de conservación.

## Responsables mínimos para una decisión real

La aprobación debe involucrar al responsable institucional de los datos, un
referente de protección de datos o asesoramiento jurídico aplicable, seguridad de
la información, responsable deportivo, operación técnica y un canal responsable
por los derechos de titulares y representantes. El desarrollador no debe decidir
en soledad la base jurídica ni la duración de conservación.
