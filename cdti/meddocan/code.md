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

Para obtener un objeto `spacy.tokens.Doc` a partir de cualquier cadena de caracteres utilizaremos el `meddocan.language.pipeline.meddocan_pipeline` creado con la ayuda de la biblioteca [spaCy](https://spacy.io/) para adaptarlo a nuestras necesidades.  
  
Para ver cómo funciona, seleccionamos un informe médico gracias al objeto `meddocan.data.docs_iterators.GsDocs` que permiten acceder a los documentos del dataset meddocan directamente como objetos `spacy.tokens.Doc` con varios atributos específicos.

```{note}
Utilizamos solo los 101 primeros caracteres para que el informemcínico sea mas leible
```

```{code-cell} ipython3
gs_docs = GsDocs(ArchiveFolder.train)
docs_with_brat_pair = iter(gs_docs)
doc_with_brat_pair = next(docs_with_brat_pair)
doc = doc_with_brat_pair.doc[:101]
doc
```

Observamos la serie de pre-processos que nos permite hacer el `meddocan_pipeline` con la ayuda del objeto `doc`.

```{note}
Miramos las 3 primeras lineas del objeto ``Doc``.
```

```{code-cell} ipython3
max_lines = 3

for i, sent in enumerate(doc.sents):
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
3. EL componente `predictor` nos permite utilizar un modelo de `Flair` de tal forma que se integré al pipeline. De esa mañera se puede hacer directamente prediciones utilizando un objeto `Doc` y un modelo entrenado previamente.
4. El componente `write_methods` sirve a partir del objecto `Doc` a crear los ficheros necesarios para
    - Entrenar un modelo de `Flair`
    - Evaluar un modelo utilizando el script de evaluación propio de la competición.

```{note}
Hemos integrado el [script de evaluation](https://github.com/PlanTL-GOB-ES/MEDDOCAN-Evaluation-Script) dentro de nuestra librería con algunas modificaciones y un poco mas de documentación, con el objetivo de facilitarnos la vida a la hora de correr las evaluaciones.  
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
```



+++

El objecto `doc_with_brat_pair` creado por `GsDocs` tiene 2 atributos.

```{code-cell} ipython3
[attr for attr in vars(doc_with_brat_pair).keys()]
```

El atributo `brat_files_pair` es un objeto `meddocan.data.docs_iterators.BratFilesPair` que indica la ubicación de los ficheros originales 

```{code-cell} ipython3
pd.DataFrame([type(doc_with_brat_pair.brat_files_pair).__qualname__, doc_with_brat_pair.brat_files_pair.ann.name, doc_with_brat_pair.brat_files_pair.txt.name], index=["type", "txt", "ann"]).T
```

Lo que hace `GsDocs` es crear un objecto `Doc` a partir de un objeto `meddocan.data.docs_iterators.DocWithBratPair` utilizando el `MedocanPipeline`.  
En una primera fase el pipeline recibe el texo contenido en el fichero original *S0004-06142005000500011-1.txt* como argumento y en un segunda fase se le asigña las entidades extraidas del fichero *S0004-06142005000500011-1.ann*.  

Si queremos obtener las entidades:

```{code-cell} ipython3
pd.DataFrame([ent.text for ent in doc.ents], columns=["Entidad"]).T
```

O si queremos algo mas didactico:

```{code-cell} ipython3
displacy.render(doc, style="ent")
```

```{code-cell} ipython3

```
