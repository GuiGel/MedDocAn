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
import warnings
warnings.filterwarnings("ignore", category=FutureWarning) 
```

```{code-cell} ipython3
import pandas as pd
from pathlib import Path
from meddocan.data import meddocan_zip, ArchiveFolder
from meddocan.data.containers import BratAnnotations, BratSpan, BratFilesPair
from typing import List, Tuple
from spacy import displacy
from meddocan.data.docs_iterators import GsDocs
```

# Resumen del código [^1]

[^1]: El proyecto completo está disponible en [github](https://github.com/GuiGel/MedDocAn).

+++

## Meddocan pipeline

+++

Una de las tareas es obtener un preprocesso correcto de los informes clínicos a traves de un objeto `spacy.tokens.Doc` a partir de cualquier cadena de caracteres. Por ello utilizaremos el `meddocan.language.pipeline.meddocan_pipeline` creado con la ayuda de la biblioteca [spaCy](https://spacy.io/) para adaptarlo a nuestras necesidades. El código relativo a la creación del pipeline se puede encontrar en el paquete [meddocan/language](https://github.com/GuiGel/MedDocAn/tree/master/meddocan/language).

Por otra parte tenemos que poder leer y extraer la información relevante de los documentos provisto en el formato Brat. Es decir que cada informe clínico esta compuesto de 2 ficheros. Uno contiene el texto bruto encodificado en el formato utf-8. Otro las anotaciónes al formato Brat tambien al formato utf-8. El código relativo de esa parte se encuentra en el paquete [meddocan/data](https://github.com/GuiGel/MedDocAn/tree/master/meddocan/data)

Para ver cómo funciona, seleccionamos un informe médico gracias al objeto `meddocan.data.docs_iterators.GsDocs` que permite acceder a los documentos del dataset meddocan directamente como objetos `spacy.tokens.Doc` con varios atributos específicos.

```{code-cell} ipython3
gs_docs = GsDocs(ArchiveFolder.train)
docs_with_brat_pair = iter(gs_docs)
doc_with_brat_pair = next(docs_with_brat_pair)
```

El objecto `doc_with_brat_pair` creado por `GsDocs` tiene 2 atributos.

```{code-cell} ipython3
[attr for attr in vars(doc_with_brat_pair).keys()]
```

El atributo `brat_files_pair` es un objeto `meddocan.data.docs_iterators.BratFilesPair` que indica la ubicación de los ficheros originales correspondiendo al attributo `doc`.

```{code-cell} ipython3
pd.DataFrame([type(doc_with_brat_pair.brat_files_pair).__qualname__, doc_with_brat_pair.brat_files_pair.ann.name, doc_with_brat_pair.brat_files_pair.txt.name], index=["type", "txt", "ann"]).T
```

Lo que hace `GsDocs` es crear un objecto `Doc` a partir de un objeto `meddocan.data.docs_iterators.DocWithBratPair` utilizando el `MedocanPipeline`.  

En una primera fase el pipeline recibe el texo contenido en el fichero original *S0004-06142005000500011-1.txt* como argumento y en un segunda fase se le asigña las entidades extraidas del fichero *S0004-06142005000500011-1.ann* utilizando el metodo ``spacy.tokens.Doc.set_ents``.

```{code-cell} ipython3
gold = doc_with_brat_pair.doc
gold
```

Observamos la serie de pre-processos applicados por el `meddocan_pipeline` y ``GsDocs`` observando la instancia `gold` del objeto ``Doc``.

```{note}
Por el ejemplo solo miramos las 3 primeras lineas del objeto ``Doc``.
```

```{code-cell} ipython3
max_lines = 3

for i, sent in enumerate(gold.sents):
    print(f"---------------------- Sentence {i + 1} ------------------------------")
    a = zip(*((tok.text, tok.ent_iob_, tok.ent_type_) for tok in sent))
    df = pd.DataFrame(a, index=["text", "bio", "etiqueta"])
    display(df.T)
    if i >= max_lines - 1:
        break
```

Para entender un poco mejor lo que hacemos miramos los differentes componentes del `MeddocanPipeline`.

```{code-cell} ipython3
gs_docs.nlp.pipe_names
```

1. El primer elemento de nuestro pipeline es el tokenizer seguido del componente `missaligned_splitter` que nos permite afinar la tokenización de tal forma que cada token se corresponda exactamente con una etiqueta al formato BIO.
2. El segundo componente, `line_sentencizer` permite partir el texto en frases. En este caso se corresponden a un párafo.
3. El componente `predictor` nos permite utilizar un modelo de `Flair` de tal forma que se integré al pipeline. De esa mañera se puede hacer directamente prediciones utilizando un objeto `Doc` y un modelo entrenado previamente.
4. El componente `write_methods` añade los metodos ``to_connl03`` y ``to_ann`` al objecto ``Doc`` que sirven a crear los ficheros necesarios para:
    - Crear un `flair.data.Corpus` que va a permitir entrenar un modelo con `Flair`.
    - Evaluar un modelo utilizando el script de evaluación propio de la competición.

``````{note}
Hemos integrado el [script de evaluation](https://github.com/PlanTL-GOB-ES/MEDDOCAN-Evaluation-Script) dentro de nuestra librería con algunas modificaciones y un poco mas de documentación, con el objetivo de unificar el workflow del entrenamiento a la evaluación.  
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
        model (str): Path to the ``Flair`` model to evaluate.
        name (str): Name of the folder that will holds the results produced by\
            the ``Flair`` model.
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
         ``Flair`` model.  [required]

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

Si queremos obtener las entidades basta con hacer:

```{code-cell} ipython3
pd.DataFrame([ent.text for ent in gold.ents], columns=["Entidad"]).T
```

```{code-cell} ipython3
pd.DataFrame(zip(*[(ent.text, ent.start, ent.end) for ent in gold.ents]), index=["Tag", "start", "end"])
```

O si queremos algo mas visual:

```{code-cell} ipython3
displacy.render(gold, style="ent")
```

## Flair corpus y entrenamiento de modelos

+++

Ahora que tenemos los datos preparados, utilizamos el objecto ``meddocan.data.corpus.MEDDOCAN`` que hereda de ``flair.datasets.ColumnCorpus`` para entrenar nuestro modelos con la libreria ``Flair``.  

+++

Por ello vamos a ver en un ejemplo como proceder.

```{note}
Todos nuestros experimentos se pueden encontrar en la carpeta [experiments](https://github.com/GuiGel/MedDocAn/tree/master/experiments).
```

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

## Utilizar el modelo

+++

El ``meddocan_pipeline`` nos permitte integrar un modelo de ``Flair `` gracias al componente ``predictor`` de la siguiente mañera. 

```{note}
Aquí utilizamos por el ejemplo un modelo de FLair pre-entreando con los embeddings de Flair y una red LSTM-CRF.
```

```{code-cell} ipython3
from meddocan.language.pipeline import meddocan_pipeline

nlp = meddocan_pipeline("flair/ner-spanish-large")
sys = nlp(gold.text)
```

Gracias a esta integración entre Flair y spaCy, producimos tanto los datos que nos permitten hacer

+++

1. la evaluación (los datos se esciben en ficheros al formato IOB.

```{code-cell} ipython3
from tempfile import TemporaryDirectory

with TemporaryDirectory() as td:
    pth = Path(td, "file.txt")
    sys._.to_connl03(pth)
    for i, line in enumerate(pth.read_text().split("\n")):
        print(line)
        if i > 18:
            break
```

Para tener una idea de como se puede calcular las metricas lo hacemos sobre el documento de ejemplo:

```{code-cell} ipython3
from meddocan.evaluation.classes import Ner

gold_label = set(Ner(ent.start, ent.end, ent.label_) for ent in gold.ents)
sys_label = set(Ner(ent.start, ent.end, ent.label_) for ent in sys.ents)
tp = gold_label.intersection(sys_label)
fp = sys_label - gold_label
fn = gold_label - sys_label

recall = len(tp) / float(len(tp) + len(fp))
precision = len(tp) / float(len(fn) + len(tp))
try:
    f1 = (recall + precision) / (recall * precision)
except ZeroDivisionError:
    f1 = 0.0
```

2. La visualización

```{code-cell} ipython3
displacy.render(sys, style="ent")
```
