# Anexo

(transfer-learning)=
## El concepto de transfer learning en PLN

Hoy en día, es una práctica habitual en visión por ordenador utilizar el transfer learning para entrenar una red neuronal convolucional como ResNet en una tarea, y luego adaptarla o ajustarla en una nueva tarea. Esto permite a la red hacer uso de los conocimientos aprendidos en la tarea original. Desde el punto de vista de la arquitectura, esto implica dividir el modelo en un cuerpo y una cabeza, donde la cabeza es una red específica de la tarea. Durante el entrenamiento, los pesos del cuerpo aprenden características generales del dominio de origen, y estos pesos se utilizan para inicializar un nuevo modelo para la nueva tarea [^1]. En comparación con el aprendizaje supervisado tradicional, este enfoque suele producir modelos de alta calidad que pueden entrenarse de forma mucho más eficiente en una variedad de tareas posteriores, y con muchos menos datos etiquetados. En la {numref}`transfert-learning` se muestra una comparación de los dos enfoques.

[^1]: Recordamos que los pesos son los parámetros que se pueden aprender de una red neuronal.

```{figure} ../figures/transformers-1.png
---
name: transfert-learning
align: center
---
Comparación entre el aprendizaje supervisado tradicional (izquierda) y el transfer learning (derecha)
```

En la visión por ordenador, los modelos se entrenan primero en conjuntos de datos a gran escala, como [ImageNet](https://image-net.org/), que contienen millones de imágenes. Este proceso se denomina pre-entrenamiento y su principal objetivo es enseñar a los modelos las características básicas de las imágenes, como los bordes o los colores. A continuación, estos modelos pre-entrenados pueden afinarse en una tarea posterior, como la clasificación de especies florales, con un número relativamente pequeño de ejemplos etiquetados (normalmente unos cientos por clase). Los modelos perfeccionados suelen alcanzar una mayor precisión que los modelos supervisados entrenados desde cero con la misma cantidad de datos etiquetados.

Aunque el transfer learning se ha convertido en el enfoque estándar de la visión por ordenador, durante muchos años no estaba claro cuál era el proceso de pre-entrenamiento análogo para la PNL. En consecuencia, las aplicaciones de PNL solían requerir grandes cantidades de datos etiquetados para lograr un alto rendimiento. E incluso entonces, ese rendimiento no se comparaba con lo que se lograba en el dominio de la visión.

En 2017 y 2018, varios grupos de investigación propusieron nuevos enfoques que finalmente hicieron que el transfert learning funcionara para la PNL. Comenzó con una idea de los investigadores de OpenAI, que obtuvieron un fuerte rendimiento en una tarea de clasificación de sentimientos mediante el uso de características extraídas del pre-entrenamiento no supervisado {cite}`Radford2017LearningTG`.  A esto le siguió ULMFiT, que introdujo un marco general para adaptar los modelos LSTM pre-entrenados para diversas tareas [^2].


[^2]:  Un trabajo relacionado con esta época fue ELMo (Embeddings from Language Models), que mostró cómo el pre-entrenamiento de los LSTM podía producir embeddings de palabras de alta calidad para tareas posteriores.

Como se ilustra en la {numref}`ulmfit`, ULMFiT consta de tres pasos principales:

*pre-entrenamiento*
: El objetivo inicial del entrenamiento es bastante sencillo: predecir la siguiente palabra basándose en las palabras anteriores. Esta tarea se denomina modelado del lenguaje. La elegancia de este enfoque reside en el hecho de que no se necesitan datos etiquetados, y se puede hacer uso de textos disponibles en abundancia en fuentes como Wikipedia [^3].

[^3]: Esto es más cierto en el caso del inglés que en el de la mayoría de las lenguas del mundo, donde puede ser difícil obtener un gran corpus de texto digitalizado. La búsqueda de formas de superar esta carencia es un área activa de la investigación y el activismo en PNL.

*Adaptación de dominio*
: Una vez que el modelo lingüístico se ha pre-entrenado en un corpus a gran escala, el siguiente paso es adaptarlo al corpus del dominio (por ejemplo, de Wikipedia al corpus de críticas de cine de IMDb, como en la {numref}`ulmfit`). En esta etapa se sigue utilizando el modelado del lenguaje, pero ahora el modelo tiene que predecir la siguiente palabra en el corpus de destino.

*Ajuste fino*
: En esta etapa, el modelo lingüístico se ajusta con una capa de clasificación para la tarea objetivo (por ejemplo, clasificar el sentimiento de las críticas de películas en la {numref}`ulmfit`).


```{figure} ../figures/tranformers-2.png
---
name: ulmfit
---
El proceso ULMFiT (Taken from Jeremy Howard course)
```

Al introducir un marco viable para el pre-entrenamiento y el transfer learning en PNL, ULMFiT proporcionó la pieza que faltaba para que los transformadores despegaran. En 2018, se lanzaron dos transformadores que combinaban la auto-atención con el transfer learning:

*GPT*
: Utiliza solo la parte del decodificador de la arquitectura del transformador, y el mismo enfoque de modelado del lenguaje que ULMFiT. GPT fue pre-entrenado en el BookCorpus {cite}`Zhu2015AligningBA`, que consiste en 7.000 libros inéditos de una variedad de géneros, incluyendo Aventura, Fantasía y Romance.

*BERT*
: Utiliza la parte del codificador de la arquitectura Transformer y una forma especial de modelado del lenguaje denominada modelado del lenguaje enmascarado. El objetivo del modelado del lenguaje enmascarado es predecir palabras enmascaradas al azar en un texto. Por ejemplo, dada una frase como "Miré mi [MASK] y vi que [MASK] llegaba tarde", el modelo necesita predecir los candidatos más probables para las palabras enmascaradas que se denotan con [MASK]. BERT se pre-entrenó con el BookCorpus y la Wikipedia en inglés.

GPT y BERT establecieron un nuevo estado de la técnica en una variedad de puntos de referencia de PNL y marcaron el comienzo de la era de los transformadores.

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
## Training time

```{glue:figure} training_time
:name: training time

Evaluación del tiempo de entrenamiento por cada una de la arquitecturas elegidas
```