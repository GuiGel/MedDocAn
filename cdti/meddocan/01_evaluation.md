(content:evaluation)=
# Evaluation

La evaluación de las predicciones automáticas para esta tarea tenía dos escenarios o subvías diferentes:

## NER offset y clasificación de tipos de entidades

**Subtrack1**
: La primera tarea se centró en la identificación y clasificación de información sensible (por ejemplo, nombres de pacientes, teléfonos, direcciones, etc.). Se trata de la misma tarea que realizamos al anonimizar documentos legales.

## Detección de span sensibles

**Subtrack2 [Strict]**
: La segunda tarea se centró en la detección de texto sensible más específico para el escenario práctico necesario para la publicación de documentos clínicos desidentificados, donde el objetivo es identificar y enmascarar los datos confidenciales, independientemente del tipo real de entidad o de la identificación correcta del tipo de PHI. En este caso solo nos interesa conocer la ubicación del texto a enmascarar.

**Subtrack2 [Merged]**
: También calculamos adicionalmente otra evaluación en la que fusionamos los tramos de PHI conectados por caracteres no alfanuméricos.

Se puede consultar la {numref}`figura %s <evaluation subtrack comparison>` para entender visualmente qué distingue a cada una de las tareas entre sí.

```{glue:figure} evaluation_subtrack_comparison
:figwidth: 800px
:align: center
:name: evaluation subtrack comparison

Comparación entre lo que se debe detectar en la tarea Subtrack2 [Merged] (index **SPAN MERGED**) y las 2 otras tareas, tanto la Subtrack1 y la Subtrack2 [Strict] en la lineas siguientes (index **NER**).
```