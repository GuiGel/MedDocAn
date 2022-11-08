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
from myst_nb import glue
```

# Evaluation

```{code-cell} ipython3
from contextlib import contextmanager
from typing import List, Iterable, Tuple
import base64
import tempfile
import requests
from pathlib import Path

from meddocan.evaluation.classes import BratAnnotation, EvaluateSubtrack1, EvaluateSubtrack2, EvaluateSubtrack2merged, Evaluate, Span


base = "https://api.github.com/repos/PlanTL-GOB-ES/MEDDOCAN-Evaluation-Script/contents"
# The api where the text can be reach.

def get_sample(base: str, name: str) -> str:
    # Get sample content from the folder located at https://github.com/PlanTL-GOB-ES/MEDDOCAN-Evaluation-Script/tree/master/gold/brat/sample via the Github api.
    # Use the Stackoverflow reponse: https://stackoverflow.com/questions/38491722/reading-a-github-file-using-python-returns-html-tags
    url = "/".join([base, name])
    req = requests.get(url)
    if req.status_code == requests.codes.ok:
        req = req.json() # the response is JSON
        # req is now a dict with keys: name, encoding, url, size ...
        # and content. But it is encoded with base64.
        content = base64.b64decode(req['content'])
        return content.decode("utf-8")
    else:
        print('Content was not found')

@contextmanager
def write_text_to_tempdir(seq_file_text: Iterable[Tuple[str, str]]) -> Path:
    # Context manager that write a sequence of (filename, text content) tuple
    # to a temporary directory and return the directory name.
    with tempfile.TemporaryDirectory() as tmpdirname:
        root = Path(tmpdirname)
        for loc, content in seq_file_text:
            (root / loc).write_text(content)
        yield  Path(tmpdirname)

def get_brat_annotation_from_github(file: str) -> BratAnnotation:
    ann = Path(file).with_suffix(".ann")
    txt = Path(file).with_suffix(".txt")

    seq_file_text = ((loc.name, get_sample(base, str(loc))) for loc in [ann, txt])

    with write_text_to_tempdir(seq_file_text) as dir_loc:
        gold_annotation = BratAnnotation(dir_loc / Path(ann).name)
        return gold_annotation
```

```{code-cell} ipython3
from meddocan.evaluation.classes import BratAnnotation, EvaluateSubtrack1, EvaluateSubtrack2, EvaluateSubtrack2merged, Evaluate, Span

gold_annotation = get_brat_annotation_from_github("gold/brat/sample/S0004-06142005000700014-1.ann")
sys_annotation = get_brat_annotation_from_github("system/brat/subtrack2/sample/baseline/S0004-06142005000700014-1.ann")
```

```{code-cell} ipython3
e = EvaluateSubtrack1({sys_annotation.id: sys_annotation}, {gold_annotation.id: gold_annotation})
e.print_report()
```

```{code-cell} ipython3
e = EvaluateSubtrack1({sys_annotation.id: sys_annotation}, {gold_annotation.id: gold_annotation})
e.print_docs()
```

Extraemos los distintos objetos que contienen la información a detectar.

```{code-cell} ipython3
ner = sorted(Evaluate.get_tagset_ner(gold_annotation), key=lambda a: a.start)
span = sorted(Evaluate.get_tagset_span(gold_annotation), key=lambda a: a.start)
span_merged = sorted(Evaluate.get_tagset_span_merged(gold_annotation), key=lambda a: a.start)
```

Podemos visualizar un ejemplo al azar para hacerse una idea de las características que se debe detectar en cada variante.

```{code-cell} ipython3
import pandas as pd
from itertools import zip_longest

df = pd.DataFrame(zip_longest((ner[1][i] for i in [1,2,0]), span[1], span_merged[1]), columns=["NER", "SPAN", "SPAN MERGED"], index=["START", "END", "TAG"])
df
```

Para entender las diferencias, pongamos un ejemplo ilustrativo donde span y spam merged son distintos.

```{code-cell} ipython3
from typing import List, Tuple

diff = set(span_merged) - set(span)  # difference between merge span and span

for i, _diff in enumerate(diff):

    indexes: List[Tuple[str, int]] = []
    span_merged_array = (_diff.start, _diff.end, f"{gold_annotation.text[_diff.start: _diff.end]!r}", None)
    indexes.append(("SPAN MERGED", i+1))

    ner_array: List[List[int]] = []  # List of ner object "include" in span_merged

    cpt = 0
    for _ner in ner:
        if _ner.end <= _diff.end and _ner.start >= _diff.start:
            cpt += 1
            ner_array.append((_ner.start, _ner.end, f"{gold_annotation.text[_ner.start: _ner.end]!r}", _ner.tag))
            indexes.append(["NER", cpt])
        else:
            cpt = 0

    df_index = pd.DataFrame(indexes, columns=["Track", "Num"])
    df = pd.DataFrame(
        [span_merged_array] + ner_array,
        columns=["START", "END", "TEXT", "TAG"],
        index=pd.MultiIndex.from_frame(df_index)
    )

    with pd.option_context(
        "display.min_rows", 50, "display.max_rows", 100, \
        "display.max_columns", 15, 'display.max_colwidth', 150):
        glue("evaluation_subtrack_comparison", df)
        pass
    print("\n")
    break
    
```

Obtain the ner annotation in common between gold and sys doc

```{code-cell} ipython3
gold_ner = set(Evaluate.get_tagset_ner(sys_annotation))
sys_ner = set(Evaluate.get_tagset_ner(gold_annotation))

fp = sys_ner - gold_ner  # Annotation detected that are not in the gold standard Annotation set.
tp = gold_ner.intersection(sys_ner)
fn = gold_ner - sys_ner
```

```{code-cell} ipython3
precision = len(tp) / (len(tp) + len(fn))
precision
```

```{code-cell} ipython3

```

```{code-cell} ipython3

```
