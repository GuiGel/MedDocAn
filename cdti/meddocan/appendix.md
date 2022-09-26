# Anexo

(Appendix-1)=
## Entrenamiento: Método de ajuste fino

Los enfoques de ajuste fino suelen añadir una sola capa lineal a un transformador y ajustan toda la arquitectura en la tarea NER. Para hacer el puente entre el modelado de subtokens y las predicciones a nivel de tokens, aplican la agrupación de subpalabras para crear representaciones a nivel de tokens que luego se pasan a la capa lineal final. Conceptualmente, este enfoque tiene la ventaja de que todo se modela en una única arquitectura que se ajusta en su conjunto.

Una estrategia común de agrupación de subpalabras es "first" {cite:p}`Devlin2019BERTPO`, que utiliza la representación del primer subtoken para todo el token. Véase la {numref}`fig-flert-2` para una ilustración.

```{figure} ../figures/flert-2.png
---
name: fig-flert-2
height: 400px
align: center
---
Ilustración de la primera agrupación de subpalabras. La entrada "The Eiffel Tower" se subdivide, dividiendo la palabra "Eiffel" en tres subpalabras (sombreadas en verde). Sólo la primera ("E") se utiliza como representación de "Eiffel".
```

### Procedimiento de entrenamiento.

Para entrenar esta arquitectura, los trabajos anteriores suelen utilizar el optimizador AdamW {cite}`Loshchilov2019DecoupledWD`, un ritmo de aprendizaje muy pequeña y un número pequeño y fijo de iteraciones como criterio de parada codificado {cite}`Lample2019CrosslingualLM`. En Flert adoptan una estrategia de entrenamiento de un ciclo {cite}`Smith2018ADA`, inspirada en la implementación de los transformadores HuggingFace {cite}`Wolf2019HuggingFacesTS`, en la que el ritmo de aprendizaje disminuye linealmente hasta llegar a 0 al final del entrenamiento. Aquí realizamos un calentamiento lineal antes del decrecimiento de el ritmo de aprendizaje (Linear Warmup With Linear Decay). {numref}`fine-tuning parameters` enumera los parámetros de arquitectura que utilizamos en todos nuestros experimentos.

Traducción realizada con la versión gratuita del traductor www.DeepL.com/Translator

```{table} Parámetros utilizados para el ajuste fino
:name: fine-tuning parameters
|     Parameter      |              Value              |
| :----------------: | :-----------------------------: |
| Transformer layers |              last               |
|   Learning rate    |              5e-6               |
|  Mini Batch size   |                4                |
|     Max epochs     |               150               |
|     Optimizer      |              AdamW              |
|     Scheduler      | Linear Warmup With Linear Decay |
|       Warmup       |               0.1               |
|  Subword pooling   |              first              |

```

(Appendix-2)=
## Entrenamiento: Método basado en características

La {numref}`fig-flert-3` ofrece una visión general del enfoque basado en características: Las representaciones de las palabras se extraen del transformador promediando todas las capas (all-layer-mean) o concatenando las representaciones de las cuatro últimas capas (last four-layers). A continuación, se introducen en una arquitectura LSTM-CRF estándar {cite}`Huang2015BidirectionalLM` como características. Volvemos a utilizar la estrategia de agrupación de subpalabras ilustrada en la {numref}`fig-flert-2`.

```{figure} ../figures/flert-3.png
---
name: fig-flert-3
height: 400px
align: center
---
Visión general del enfoque basado en características. La autoatención se calcula sobre todos los tokens de entrada (incluyendo el contexto izquierdo y derecho).
contexto). La representación final de cada elemento de la frase ("I love Paris", sombreada en verde) puede calcularse como
a) media sobre todas las capas del modelo basado en transformadores o b) concatenando las cuatro últimas capas.
```

En nuestros experimentos hemos elegido la primera variante, crear representaciones de las palabras promediando todas las capas (all-layer-mean).

(procedimiento-caracteristicas)=
### Procedimiento de entrenamiento

Adoptamos el procedimiento de entrenamiento estándar utilizado en trabajos anteriores. Entrenamos la red con SGD con una tasa de aprendizaje mayor que se rectifica con los datos de desarrollo. El entrenamiento finaliza cuando el ritmo de aprendizaje es demasiado pequeño. Los parámetros utilizados para entrenar un modelo basado en características se muestran en {numref}`feature-based parameters`.

```{table} Parámetros utilizados para el método basado en características
:name: feature-based parameters
:align: center
|     Parameter      |       Value       |
| :----------------: | :---------------: |
| Transformer layers |       last        |
|   Learning rate    |        0.1        |
|  Mini Batch size   |         4         |
|     Max epochs     |        500        |
|     Optimizer      |        SGD        |
|     Scheduler      | Anneal On Plateau |
|  Subword pooling   |       first       |
```

(Appendix-3)=
### Entrenamiento: Flair + LSTM-CRF

La {numref}`fig-flair-1` ofrece una visión general del enfoque basado en características: Las representaciones de las palabras se extraen de model de idiomas. A continuación, se introducen en una arquitectura LSTM-CRF estándar {cite}`Huang2015BidirectionalLM` como características.

```{figure} ../figures/flair-1.png
---
name: fig-flair-1
height: 400px
align: center
---
Visión general del enfoque basado en características usando los embeddings contextuales de Flair {cite}`Akbik2018ContextualSE`.
```

### Procedimiento de entrenamiento

Procedemos exactamente como en el [Método basado en características](procedimiento-caracteristicas). Los parámetros utilizados para entrenar un modelo basado en características con Flair se muestran en {numref}`feature-based parameters Flair`.

```{table} Parámetros utilizados para el método basado en características con Flair
:name: feature-based parameters Flair
:align: center
|     Parameter      |       Value       |
| :----------------: | :---------------: |
|   Learning rate    |        0.1        |
|  Mini Batch size   |         4         |
|     Max epochs     |        150        |
|     Optimizer      |        SGD        |
|     Scheduler      | Anneal On Plateau |
```

TODO Average training runtime. 

```{note}
We conduct experiments on a NVIDIA Quadro M6000 (24GB) for fine-tuning
and feature-based approach. We report average training times for our best configurations in Table 9.
```

## Numero de parámetros del modelo

```{glue:figure} model_parameters
:align: center
:name: "model parameters"

Numero de parámetros contenido en cada unos de los modelos utilizados
```
