# Training

## Introduction

Como lo hemos visto en la primera parte, el reconocimiento entidades nombradas (NER) es una tarea de NLP muy estudiada que consiste en predecir etiquetas semánticas superficiales para una secuencia de palabras.

Los enfoques actuales para el NER consisten en aprovechar arquitecturas de transformadores pre-entrenados, como {cite:p}`Devlin2019BERTPO` o {cite:p}`Lample2019CrosslingualLM`. Esos transformadores an sido pre-entrenados en otras tareas sobre un corpus grande y sirven de base para entrenar modelos de NER transfiriendo su aprendizaje previo a esa tarea.

Esos enfoques suelen considerar el texto a nivel de frase, exactamente como lo hemos hecho en el dominio jurídico, y por lo tanto no modelan la información que cruza los límites de la frase. Podemos hacerlo pasando una frase con su contexto circundante. Sin embargo, el uso de un modelo basado en transformadores para NER ofrece una opción natural para capturar características a nivel de documento. Como muestra {numref}`fig-flert-1`, este contexto puede influir en la representación de las palabras de una frase: La frase de ejemplo: "I love Paris", pasa por el transformador junto con la siguiente frase que comienza con "The city is", ayudando potencialmente a resolver la ambigüedad de la palabra "Paris". 

```{figure} ../figures/flert-1.png
---
name: fig-flert-1
---
Para obtener características a nivel de documento para una frase que deseamos etiquetar ("I love Paris", sombreada en verde), añadimos 64 tokens a la izquierda y a la derecha (sombreados en azul). Como el transformador calcula la auto-atención sobre todos los tokens de entrada, la representación de los tokens de la frase está influida por el contexto izquierdo y derecho.
``` 

Esto es exactamente lo que propone el nuevo enfoque Flert {cite:p}`Schweter2020FLERTDF` disponible en la biblioteca **Flair** {cite:p}`Akbik2019FLAIRAE` que roza el estado del arte.

```{note}
**Hacer respetar los límites de los documentos.** Los autores de Flert {cite:p}`Schweter2020FLERTDF` demuestran que el respeto de los límites de los documentos aumenta el escore F1 en casi todos sus experimentos y recomiendan su cumplimiento si es posible. Su expectativa inicial de que los transformadores aprenderían automáticamente a respetar los límites de los documentos no se materializó.
La aplicación de los límites del documento consiste en una ablación en la que se truncan las características del documento en los bordes del mismo, lo que significa que el contexto sólo puede proceder del mismo documento.
```

En la literatura, hay dos enfoques conceptualmente muy diferentes para el NER basado en transformadores que se utilizan actualmente, los dos son basados en el Transfer Learning. Evaluaremos las características de los documentos en ambos:

1. En el primero, *afinamos* el propio transformador en la tarea NER y solo añadimos una capa lineal para las predicciones a nivel de palabra {cite:p}`Devlin2019BERTPO` .
2. En el segundo, utilizamos el transformador solo para proporcionar *características* a una arquitectura de etiquetado de secuencias LSTM-CRF {cite}`Huang2015BidirectionalLM` estándar y, por tanto, no realizamos ningún ajuste fino. Hemos utilizado la arquitectura LSTM-CRF previamente para el dominio judicial pero con los words embeddings contextuales de Flair {cite:p}`Akbik2018ContextualSE` y embeddings estáticos como word2vec {cite:p}`Mikolov2013EfficientEO`.


Discutimos las diferencias entre ambos enfoques y exploramos los mejores hiperparámetros para cada uno, manualmente y con ayuda de la biblioteca [Hyperopt](http://hyperopt.github.io/hyperopt/). En su mejor configuración determinada, realizamos una evaluación comparativa a la cual integramos los mejores modelos entrenados con embeddings contextuales de Flair {cite:p}`Akbik2018ContextualSE` con o sin embeddings estáticos {cite:p}`Mikolov2013EfficientEO` junto con la arquitectura LSTM-CRF (i.e. la mejor architectura sobre la tarea de anonimización de documentos judiciales).

```{admonition} Características a nivel de documento
:class: tip

El enfoque de Flert, que consiste en crear un contexto por cada frase, tiene ventajas computacionales, ya que cada frase y su contexto sólo tienen que pasar por el transformador una vez y el contexto añadido se limita a una ventana relativamente pequeña. Además, sigue siendo posible seguir el procedimiento estándar de mezclar las frases en cada momento del entrenamiento, ya que el contexto se codifica por frases.
```

(experiments)=
## Experimentos con parámetros de referencia

Como se ha mencionado en la introducción, existen dos arquitecturas comunes para el NER basado en transformadores, a saber, los enfoques de ajuste fino y los basados en características.
En esta sección, presentamos brevemente las diferencias entre ambos enfoques y realizamos un estudio para identificar los mejores hiperparámetros para cada uno. Las mejores configuraciones de cada se utilizan luego en la evaluación comparativa final en la sección [](comparative_study).

### Configuración

**Data set.** Utilizamos el conjunto de datos de desarrollo de MEDDOCAN [^1].

[^1]: https://github.com/PlanTL-GOB-ES/SPACCC_MEDDOCAN/tree/master/corpus

**Modelo de transformador**. En todos los experimentos de esta sección, empleamos 2 modelos de transformadores:
1.  El modelo de transformador XLM-Roberta (XLMR) propuesto por {cite:p}`Lample2019CrosslingualLM`. En nuestros experimentos utilizamos *xlm-roberta large*, entrenado en 2,5TB de datos del corpus limpio Commom Crawl {cite:p}`Wenzek2020CCNetEH` para 100 idiomas diferentes.
2.  Modelo transformador BERT propuesto por {cite:p}`Devlin2019BERTPO`. En nuestros experimentos utilizamos *BETO*, un modelo bert entrenado en el gran corpus español {cite}`CaneteCFP2020`.

**Embeddings (+ WE)**. Para cada configuración experimentamos concatenando embeddings de palabras clásicas a las representaciones a nivel de palabra obtenidas del modelo transformador. Utilizamos los embeddings de FastText en español {cite}`Bojanowski2017EnrichingWV` estabilizados {cite:p}`Antoniak2018EvaluatingTS`.

### Primera estrategia: Ajuste fino

Las estrategias de ajuste fino suelen añadir una sola capa lineal a un transformador y ajustan toda la arquitectura en la tarea NER. Para hacer el puente entre el modelado de subtokens y las predicciones a nivel de tokens, aplican la agrupación de subpalabras para crear representaciones a nivel de tokens que luego se pasan a la capa lineal final. Conceptualmente, este enfoque tiene la ventaja de que todo se modela en una única arquitectura que se ajusta en su conjunto. En el anexo {ref}`Appendix-1` se dan más detalles sobre los parámetros y la arquitectura.

Evaluamos esta estrategia con los transformadores BETO {cite}`CaneteCFP2020` y XLMR {cite:p}`Lample2019CrosslingualLM`. Los resultados figuran en la {numref}`tabla %s <finetuning approach>`.

```{glue:figure} table_finetune_dev
:figwidth: 800px
:align: center
:name: finetuning approach

Evaluación de diferentes transformadores mediante el proceso de ajuste fino. La evaluación se realiza contra el conjunto de desarrollo.
```

### Segunda estrategia: Basado en características

En cambio, los enfoques basados en características utilizan el transformador solo para generar embeddings para cada palabra de una frase y las utilizan como entrada en una arquitectura de etiquetado de secuencias estándar, normalmente una LSTM-CRF {cite:p}`Huang2015BidirectionalLM`. Los pesos del transformador se congelan para que el entrenamiento se limite al LSTM-CRF. Conceptualmente, este enfoque se beneficia de un procedimiento de entrenamiento del modelo bien entendido que incluye un criterio de parada real. En el anexo {ref}`Appendix-2` se dan más detalles sobre los parámetros y la arquitectura. En nuestros experimentos solo evaluamos una variante de las dos propuestas en {cite:p}`Schweter2020FLERTDF` porque suele dar los mejores resultados.

**Media de todas las capas** Obtenemos embeddings para cada token utilizando la media de todas las capas producidas por el transformador, incluida la capa de embeddings de palabras. Esta representación tiene la misma longitud que el tamaño oculto de cada capa transformadora. Este enfoque se inspira del "scalar mix" del estilo ELMO {cite:p}`Peters2018DeepCW`.

Los resultados se encuentran en la {numref}`tabla %s <feature-based approach>`.

```{glue:figure} table_feature_based_dev
:figwidth: 800px
:align: center
:name: feature-based approach

Evaluación de la estrategia basada en características. La evaluación se realiza contra el conjunto de desarrollo.
```

### Flair reference

De la misma mañera usamos los embeddings contextuales de Flair {cite}`Akbik2018ContextualSE` como entrada de la arquitectura LSTM-CRF {cite:p}`Huang2015BidirectionalLM`. En el anexo {ref}`Appendix-3` se dan más detalles sobre los parámetros y la arquitectura.

Los resultados se encuentran en la {numref}`tabla %s <flair approach>`.

```{glue:figure} table_flair_dev
:figwidth: 800px
:align: center
:name: flair approach

Evaluación de la estrategia basada en características con los embeddings de Flair. La evaluación se realiza contra el conjunto de desarrollo.
```

### Resultados: Mejor configuración

Evaluamos ambos enfoques en cada variante en todas las combinaciones posibles añadiendo embeddings de palabras estándar "(+ WE)" y características a nivel de documento "(+ Contexto)". Cada configuración se ejecuta tres veces para reportar el promedio de F1 y la desviación estándar para cada una de las 3 opciones: NER, Span y Span Merged.  
**Results**. Para el modelo de referencia con Flair + LSTM CRF la adición de embeddings estáticos (+ WE) tiene un impacto negativo (ver {numref}`tabla %s <flair approach>`).
Para el ajuste fino, vemos que la adición de embeddings estática, así como el uso del contexto, parece bastante convincente, incluso si la diferencia con la versión simple no es tan clara (véase {numref}`tabla %s <finetuning approach>`.)
Para el enfoque basado en características, encontramos que la adición de embeddings de palabras produce muy claramente los mejores resultados (véase {numref}`tabla %s <feature-based approach>`).

(comparative_study)=
## Evaluación comparativa

Con las configuraciones identificadas en [](experiments) en los datos de desarrollo, realizamos una evaluación comparativa final en los datos de test, con y sin características de los documentos.

### Principales resultados

Los resultados de la evaluación se recogen en la {numref}`Tabla %s <table test>`. Hacemos las siguientes observaciones:

```{glue:figure} table_test
:name: table test

Evaluación comparativa de las mejores configuraciones de los enfoques de ajuste fino y basados en características en los datos de test.
```

El uso de Transformers permite una ligera ganancia en comparación con el uso de Flair (véase la {numref}`Tabla %s <flair comparison>`).

```{note}
Con los resultados obtenidos, Flert habría ganado la competición {cite}`Marimon2019AutomaticDO` por delante del actual ganador Lukas Lange {cite}`Lange2019NLNDETN` (véase {numref}`Tabla %s <lukas lange>`).

```{table} Mejor score F1 sobres cada una de las 3 Subtracks obtenidos por Luckas Lange.
:name: lukas lange
| Subtrack1 | Subtrack2 [Strict] | Subtrack2 [Merged] |
| :-------- | :----------------- | :----------------- |
| 96.96     | 97.49              | 98.53              |
```


En cuanto a los enfoques de "FINETUNE + LINEAR" y basado en "FEATURE BASED + LSTM CRF", los resultados son bastante similares, aunque el segundo enfoque se ve afectado negativamente por el contexto y el primero positivamente.
Por otro lado, la estrategia "FEATURE BASED + LSTM CRF + CONTEXTO + WE" es la más beneficiada con la mejor puntuación F1 en las 3 tareas.
Al contrario de los resultados obtenidos por los autores de Flert {cite}`Schweter2020FLERTDF` los resultados no son tan claros entre cada opción como en el caso del dataset CONLL03.

```{glue:figure} compare_with_flair
:name: flair comparison

Evaluación de las mejoras en score F1 en los datos de test en comparación con la opción CARACTERÍSTICAS + FLAIR + LSTM CRF.
```

## Conclusión

El sistema basado en Transfer Learning y el uso de Transformadores (BERT / XLMR- Large) nos ha permitido obtener muy buenos resultados sobre el dataset MEDDOCAN, mostrando un score F1 superior en todas las variantes de evaluación en comparación con el uso de las tecnologías precedentes.
No obstante esa mejorara conlleva un tiempo de entrenamiento mas largo asi que modelos con mas parámetros y entonces mas voluminosos (véase la {numref}`Tabla %s <model parameters>`).