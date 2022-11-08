---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.1
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

```{code-cell} ipython3
:tags: [hide-output, remove-input]

import warnings
warnings.filterwarnings("ignore", category=FutureWarning) 
```

# Código

```{note}
El proyecto completo está disponible en el repositorio [MedDocAn](https://github.com/GuiGel/MedDocAn) de nuestro github donde todo el código así como los experimentos del proyecto son disponibles.
```

+++

(meddocan-pipeline)=
## Meddocan pipeline

+++

Una de las tareas es obtener un preprocesso correcto de los informes clínicos a través  de un objeto `spacy.tokens.Doc` a partir de cualquier cadena de caracteres. Por ello utilizaremos el `meddocan.language.pipeline.meddocan_pipeline` creado con la ayuda de la biblioteca [spaCy](https://spacy.io/) para adaptarlo a nuestras necesidades. El código relativo a la creación del pipeline se puede encontrar en el paquete [meddocan/language](https://github.com/GuiGel/MedDocAn/tree/master/meddocan/language).

Por otra parte, tenemos que poder leer y extraer la información relevante de los documentos provisto en el formato Brat. Es decir que cada informe clínico esta compuesto de 2 ficheros. Uno contiene el texto bruto y el otro las anotaciones. El código relativo de esa parte se encuentra en el paquete [meddocan/data](https://github.com/GuiGel/MedDocAn/tree/master/meddocan/data).

Para ver cómo funciona, seleccionamos un informe médico gracias al objeto `meddocan.data.docs_iterators.GsDocs` que permite acceder a los documentos del conjunto de datos meddocan directamente como objetos `spacy.tokens.Doc` con varios atributos específicos.

```{code-cell} ipython3
from pathlib import Path
from typing import List, Tuple

import pandas as pd
from spacy import displacy

from meddocan.data import meddocan_zip, ArchiveFolder
from meddocan.data.containers import BratAnnotations, BratFilesPair, BratSpan
from meddocan.data.docs_iterators import GsDocs

gs_docs = GsDocs(ArchiveFolder.train)
docs_with_brat_pair = iter(gs_docs)
doc_with_brat_pair = next(docs_with_brat_pair)
```

El objeto `doc_with_brat_pair` creado por `GsDocs` tiene 2 atributos.

```{code-cell} ipython3
[attr for attr in vars(doc_with_brat_pair).keys()]
```

El atributo `brat_files_pair` es un objeto `meddocan.data.docs_iterators.BratFilesPair` que indica la ubicación de los ficheros originales correspondiendo al atributo `doc`.

```{code-cell} ipython3
pd.DataFrame(
    [type(doc_with_brat_pair.brat_files_pair).__qualname__,
     doc_with_brat_pair.brat_files_pair.ann.name,
     doc_with_brat_pair.brat_files_pair.txt.name],
    index=["type", "txt", "ann"]
).T
```

Lo que hace `GsDocs` es crear un objeto `Doc` a partir de un objeto `meddocan.data.docs_iterators.DocWithBratPair` utilizando el `MedocanPipeline`.

En una primera fase el ``MeddocanPipeline`` recibe el texto contenido en el fichero original *S0004-06142005000500011-1.txt* como argumento.  

Miramos el contenido del fichero utilizando el atributo `doc`.

```{code-cell} ipython3
gold = doc_with_brat_pair.doc
gold
```

  En una segunda fase se le asigna las entidades extraídas del fichero *S0004-06142005000500011-1.ann* utilizando el método ``spacy.tokens.Doc.set_ents``.   
  Miramos el fichero al formato *.brat* que contiene las anotaciones para hacerse una idea.

```{code-cell} ipython3
print(doc_with_brat_pair.brat_files_pair.ann.read_text())
```

Ahora que tenemos una idea más clara de nuestros datos originales, observamos la serie de pre-procesos aplicados por el `meddocan_pipeline` y ``GsDocs`` gracias a la instancia `gold` del objeto ``Doc`` para preparar el conjunto de datos a fin de entrenar una red neuronal con Flair.

```{note}
En el ejemplo solo miramos las 3 primeras lineas del objeto ``Doc``.
```

```{code-cell} ipython3
max_lines = 3

for i, sent in enumerate(gold.sents):
    print(f"--------------- Sentence {i + 1} ---------------")
    a = zip(*((tok.text, tok.ent_iob_, tok.ent_type_) for tok in sent))
    df = pd.DataFrame(a, index=["text", "bio", "etiqueta"])
    display(df.T)
    if i >= max_lines - 1:
        break
```

Para entender un poco mejor lo que hacemos miramos los diferentes componentes del `MeddocanPipeline`.

```{code-cell} ipython3
pd.DataFrame(gs_docs.nlp.pipe_names, columns=["componentes"]).T
```

1. El primer elemento de nuestro pipeline es el tokenizer seguido del componente `missaligned_splitter` que nos permite afinar la tokenización de tal forma que cada token se corresponda exactamente con una etiqueta al formato BIO.
2. El segundo componente, `line_sentencizer` permite partir el texto en frases. En este caso se corresponde a un párrafo.
3. El componente `predictor` nos permite utilizar un modelo de Flair de tal forma que se integra al pipeline. De esa mañera se puede hacer directamente predicciones utilizando un objeto `Doc` y un modelo entrenado previamente.
4. El componente `write_methods` añade los métodos ``to_connl03`` y ``to_ann`` al objeto ``Doc`` que sirven a crear los ficheros necesarios para:
    - Crear un `flair.data.Corpus` que va a permitir entrenar un modelo con Flair.
    - Evaluar un modelo utilizando el script de evaluación propio de la competición.

``````{note}
Hemos integrado el [script de evaluación](https://github.com/PlanTL-GOB-ES/MEDDOCAN-Evaluation-Script) dentro de nuestra librería con algunas modificaciones y un poco más de documentación, con el objetivo de unificar el workflow del entrenamiento hasta la evaluación.  
La evaluación se hace entonces directamente desde nuestra librería gracias al commando: 

```console
$ meddocan eval --help
Usage: meddocan eval [OPTIONS] MODEL NAME
Evaluate the model with the `meddocan` metrics.
    
    Compute f1-score for Ner (start, end, tag), Span (start, end) and merged
    span if not there is no number or letter between consecutive span.

    The function produce the following temporary folder hierarchy:

    evaluation_root
    ├── golds
    │   ├── dev
    |   |    └── brat
    |   |       ├── file-1.ann
    |   |       ├── file-1.txt
    |   |       ├── ...
    |   |       └── file-n.ann
    |   └── test
    |        └── brat
    |           ├── file-1.ann
    |           ├── file-1.txt
    |           ├── ...
    |           └── file-n.ann
    │       
    └── name
        ├── dev
        |    └── brat
        |       ├── file-1.ann
        |       ├── file-1.txt
        |       ├── ...
        |       └── file-n.ann
        └── test
             └── brat
                ├── file-1.ann
                ├── file-1.txt
                ├── ...
                └── file-n.ann

    Then the model is evaluate producing the following files:

    evaluation_root/name
    ├── dev
    │   ├── ner
    │   └── spans
    └── test
        ├── ner
        └── spans

    And the temporary folder are removed.

    Args:
        model (str): Path to the Flair model to evaluate.
        name (str): Name of the folder that will holds the results produced by\
            the Flair model.
        evaluation_root (str): Path to the root folder where the
            results will be stored.
        sentence_splitting (Path): Path to the sub-directory
            `sentence_splitting`. This directory is mandatory to compute the
            `leak score` evaluation metric.
        force (bool, optional): Force to create again the golds standard files.
            Defaults to False.

Arguments:
  MODEL  Path to the Flair model to evaluate.  [required]
  NAME   Name of the folder that will holds the results produced by the
         Flair model.  [required]

Options:
  --evaluation-root PATH     Path to the root folder where the results will be
                             stored.
  --sentence-splitting PATH  The sub-directory `sentence_splitting` is
                             mandatory to compute the `leak score` evaluation
                             metric.
  --device TEXT              Device to use.  [default: cuda:0]
  --help                     Show this message and exit.        
```
``````

+++

La entidades anotadas se obtienen utilizando el atributo ``ents`` de nuestro objeto ``spacy.tokens.Doc``.

```{code-cell} ipython3
pd.DataFrame(
    zip(*[(ent.text, ent.start_char, ent.end_char) for ent in gold.ents]),
    index=["Tag", "start", "end"]
).T
```

Nuestras entidades son en este ejemplo compuestas del tag y del a position de la entidad en la cadena de caracteres original.

Ahora si queremos algo más visual podemos utilizar la function ``spacy.displacy.render()`` que nos permite trabajar con el objeto ``Doc``.

```{code-cell} ipython3
displacy.render(gold, style="ent")
```

## Entrenar un modelo con Flair

+++

La clase ``GsDocs`` sirve a preparar los datos de forma que puedan ser leídos por Flair.  Esto lo hacemos a través del objeto ``meddocan.data.corpus.MEDDOCAN`` que hereda de ``flair.datasets.ColumnCorpus``.  

Este corpus permite a la biblioteca Flair acceder directamente a los conjuntos de datos de entrenamiento, validación y prueba.

```{code-cell} ipython3
from meddocan.data.corpus import MEDDOCAN

corpus = MEDDOCAN(sentences=True, in_memory=True, document_separator_token="-DOCSTART-")
```

Una vez creado nuestro corpus podemos utilizar los métodos de los que hereda nuestro objeto como el método especial ``__str__`` que escribe en stdout el número de objetos ``flair.tokens.Sentence`` que contiene cada subconjunto del conjunto de datos. Es decir, cuántos párrafos tenemos en total en cada uno de nuestros conjuntos de datos.

```{code-cell} ipython3
print(corpus)
```

Para entrenar los modelos con la librería Flair basta seguir el ejemplo siguiente. 

```{note}
Todos nuestros experimentos se pueden encontrar en la carpeta [experiments](https://github.com/GuiGel/MedDocAn/tree/master/experiments) y siguen este formato.  
La única diferencia es la adición de las librerías hyperopt [^1] y Tensorboard [^2] para obtener una trazabilidad de los entrenamientos mediante el registro de diversos parámetros y resultados.
```

[^1]: https://github.com/hyperopt/hyperopt
[^2]: https://pytorch.org/docs/stable/tensorboard.html

+++

```python
from flair.data import Corpus
from flair.embeddings import TransformerWordEmbeddings
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer

from meddocan.data.corpus import MEDDOCAN

# 1. Obtener el corpus
corpus: Corpus = MEDDOCAN(
    sentences=True, document_separator_token="-DOCSTART-"
)
print(corpus)

# 2. Que label se quiere predecir?
label_type = 'ner'

# 3. Crear el diccionario de labels a partir del corpus
label_dict = corpus.make_label_dictionary(label_type=label_type, add_unk=False)
print(label_dict)

# 4. Inicializar los embeddings generados por el transformador utilizando el contexto
embeddings = TransformerWordEmbeddings(model='dccuchile/bert-base-spanish-wwm-cased',
                                       layers="-1",
                                       subtoken_pooling="first",
                                       fine_tune=True,
                                       use_context=True,
                                       )

# 5. Inicializar etiquedator simple (no CRF, no RNN, no reprojección)
tagger = SequenceTagger(hidden_size=256,
                        embeddings=embeddings,
                        tag_dictionary=label_dict,
                        tag_type='ner',
                        use_crf=False,
                        use_rnn=False,
                        reproject_embeddings=False,
                        )

# 6. Initializar el trainer
trainer = ModelTrainer(tagger, corpus)

# 7. Ejecutar el fine-tuning
trainer.fine_tune('experiments/meddocan',
                  learning_rate=5.0e-6,
                  mini_batch_size=4,
                  )
```

+++

## Inferencia

+++

Una vez el modelo entrenado, la inferencia se hace gracias al ``meddocan_pipeline`` justamente porque nos permite integrar un modelo de ``Flair `` gracias a su componente ``predictor`` como lo vamos a ver a continuación. 

```{note}
Aquí utilizamos por el ejemplo un modelo de FLair pre-entrenado con los embeddings de Flair y una red LSTM-CRF y entrenado sobre el conjunto de datos *CONLL-2002* en inglés. Este conjunto de datos consiste en artículos de noticias anotados con las categorías LOC, PER y ORG que son diferentes de las categorías de MEDDOCAN.
```

```{code-cell} ipython3
from meddocan.language.pipeline import meddocan_pipeline

nlp = meddocan_pipeline("flair/ner-english-fast")
sys = nlp(gold.text)
```

El objeto ``sys`` es un objeto ``spacy.tokens.Doc`` al igual de el objeto ``gold``.  

La única diferencia entre ``sys`` y ``gold`` es que ``sys`` contiene entidades que le son asignadas por un modelo entrenado con los algoritmos de Flair, mientras que en el caso de ``gold`` provienen de la lectura de un archivo *.ann*.

Entonces para visualizar las predicciones de nuestro model, lo tenemos igual de fácil que antes. Basta utilizar la function ``spacy.displacy.render()``.

```{code-cell} ipython3
displacy.render(sys, style="ent")
```

```{note}
Vemos a ojo que el modelo de Flair ``flair/ner-english-fast`` parece detectar correctamente el span de ciertas entidades. Les asigna una etiqueta como ``LOC`` o ``PERS`` y efectivamente son direcciones o personas aunque por supuesto no tiene el mismo etiquetado.
```

+++

## Evaluation

+++

La evaluación originalmente provistas a través  del [script de evaluación](https://github.com/PlanTL-GOB-ES/MEDDOCAN-Evaluation-Script) que re-utilizamos, utiliza el texto original así como su anotación al formato *brat* para calcular las métricas según las tareas ``Subtrack1``, ``Subtrack2 [Strict]`` y ``SubTrack2 [Merged]``.

Para evaluar nuestros modelos, utilizamos el texto original a partir del cual se ha creado el documento ``sys`` así como su atributo ``_.to_ann``. Ese método permite codificar el atributo ``ents`` del objeto ``sys`` en un fichero siguiendo el formado brat como se puede ver a continuación.

```{code-cell} ipython3
from tempfile import TemporaryDirectory

with TemporaryDirectory() as td:
    pth = Path(td, "file.txt")
    sys._.to_ann(pth)
    for i, line in enumerate(pth.read_text().split("\n")):
        print(line)
```

Para hacer esto más automatizado, al igual que la clase ``meddocan.data.docs_iterators.GsDocs``, tenemos la clase ``from meddocan.data.docs_iterators.SysDocs`` que utiliza un modelo de Flair para detectar entidades sobre los documentos de cada sub-conjunto de datos. Veamos un ejemplo:

```{code-cell} ipython3
from meddocan.data.docs_iterators import SysDocs
from meddocan.data import ArchiveFolder

import torch
import flair

flair.device = torch.device("cuda:1")

sys_docs = iter(SysDocs(archive_name=ArchiveFolder.test, model="flair/ner-spanish-large"))
```

Gracias a las clase ``GsDocs`` y ``SysDocs`` podemos producir fácilmente los ficheros requeridos para usar el script de evaluación provisto a través  de la la linea de comando *meddocan eval*.

+++

Como curiosidad podemos explicar como se pueden calcular las métricas.

```{code-cell} ipython3
gs_docs = iter(GsDocs(archive_name=ArchiveFolder.test))
```

Recuperamos los iteradores ``sys_doc`` y ``gold_doc`` y comprobamos que corresponden a los mismos documentos originales, tanto en su origen como en su contenido.

```{code-cell} ipython3
sys_doc, gold_doc = next(sys_docs), next(gs_docs)

assert sys_doc.brat_files_pair.ann.name == gold_doc.brat_files_pair.ann.name
assert sys_doc.brat_files_pair.txt.name == gold_doc.brat_files_pair.txt.name
assert sys_doc.doc.text == gold_doc.doc.text
```

Ahora recuperamos las entidades originales en ``gold_labels`` y las predicciones ``sys_labels``.

```{code-cell} ipython3
from meddocan.evaluation.classes import Ner, Span

gold_labels = set(Ner(ent.start, ent.end, ent.label_) for ent in gold.ents)
sys_labels = set(Ner(ent.start, ent.end, ent.label_) for ent in sys.ents)
```

Por fin calculamos el score $F_{1} micro$ para el documento así como el recall y la precision.

```{code-cell} ipython3
from typing import TypeVar

T = TypeVar("T", Ner, Span)

def f1(gold_label: List[T], sys_label: List[T]) -> float:
    tp = gold_label.intersection(sys_label)
    fp = sys_label - gold_label
    fn = gold_label - sys_label
    try:
        recall = len(tp) / float(len(tp) + len(fp))
    except ZeroDivisionError:
        recall = 0.0
    try:
        precision = len(tp) / float(len(tp) + len(fn))
    except ZeroDivisionError:
        precision = 0.0
    try:
        f1 = 2 * (recall * precision) / (recall + precision)
    except ZeroDivisionError:
        f1 = 0.0
    return f1, recall, precision

pd.DataFrame(
    f1(gold_labels, sys_labels),
    index=["f1", "recall", "precisión"],
    columns=["Subtrack1"]
).T
```

¡El score $F_{1} micro$ esta nulo simplemente porque el modelo utilizado no predice las mismas entidades y ademas esta entrenado sobre un conjunto de datos en inglés! Pero si se trata únicamente de anonimizar y que usamos solo los ``Span`` puede que no sea lo mismo dado que ahora no importan las etiquetas sino solo la posición de las entidades.

```{code-cell} ipython3
gold_spans = set(Span(ent.start, ent.end) for ent in gold.ents)
sys_spans = set(Span(ent.start, ent.end) for ent in sys.ents)
pd.DataFrame(
    f1(gold_spans, sys_spans),
    index=["f1", "recall", "precision"],
    columns=["Subtrack2[strict]"]
).T
```

De hecho, vemos que incluso con un modelo entrenado en un conjunto de datos muy diferente con tan sólo 4 entidades, conseguimos anonimizar un poco más de un cuartos de los span deseados.

```{code-cell} ipython3

```
