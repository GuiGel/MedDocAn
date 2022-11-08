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
from meddocan.data.docs_iterators import GsDocs
from meddocan.data import ArchiveFolder, meddocan_zip
from typing import Type
from spacy import displacy
import pandas as pd
from spacy.tokens import Token
from pathlib import Path

def get_number_of_doc(archive_folder: Type[ArchiveFolder], attr: str) -> int:
    total = sum([1 for _ in meddocan_zip.brat_files(getattr(archive_folder, attr))])
    return total

def get_labels(token: Token) -> str:
    if token.ent_iob_ == "O":
        return token.ent_iob_
    else:
        return f"{token.ent_iob_}_{token.ent_type_}"
```

# Data exploration

+++

Extract a clinical case as well as the associated annotation at the **Brat** format.

```{code-cell} ipython3
gs_docs = list(iter(GsDocs(ArchiveFolder.train)))  # Gold Standard Document
doc = gs_docs[0].doc
brat = gs_docs[0].brat_files_pair.ann.read_text()

glue("txt_ex", doc.text, display=False)
glue("brat_ex", brat, display=False)
```

Visualize the document as rendered by **spaCy**

```{code-cell} ipython3
displacy_ex = displacy.render(doc[17: 44], style="ent", page=False, minify=True, jupyter=False)
# glue("display_ex", displacy_ex, display=False)

# output_path = Path("../figures/displacy_ner.svg") # you can keep there only "dependency_plot.svg" if you want to save it in the same folder where you run the script
# output_path.touch(exist_ok=True)
# output_path.open("w", encoding="utf-8").write(displacy_ex)
```

Visualize data sentence, tokenization and labels

```{code-cell} ipython3
sentences = list(doc.sents)[3:7]

df = pd.DataFrame(
    {
        "Sentence": [sentence.text for sentence in sentences],
        "Tokens": [f"{[token.text for token in sentence]}" for sentence in sentences],
        "Labels": [f"{[get_labels(token) for token in sentence]}" for sentence in sentences],
        "Idx": range(len(sentences)),
    },
).set_index("Idx")
pd.set_option('display.max_colwidth', None)
glue("data_preparation", df, display=True)
```

The special example of "DominguezCorreo"

```{code-cell} ipython3
from meddocan.language.pipeline import meddocan_pipeline
nlp = meddocan_pipeline()
sentence = "DominguezCorreo"
spacing_error_ex = f"{sentence!r} -> {[token.text for token in nlp(sentence)]!r}"
spacing_error_ex = glue("spacing_error_ex", spacing_error_ex)
```

Extract statistics about corpus

```{code-cell} ipython3
from meddocan.data.corpus import MEDDOCAN

corpus = MEDDOCAN(sentences=True, document_separator_token="-DOCSTART-", in_memory=True)
stats = corpus.obtain_statistics(pretty_print=False)
```

```{code-cell} ipython3
v_name = "number_of_documents_per_class"
df_phi_stats = (
    pd.DataFrame({k: stats[k][v_name] for k in stats.keys()})
    .fillna(0)
    .astype(int)
    .sort_values(by=["TRAIN"], ascending=False)
)
df_phi_stats.loc["TOTAL"] = df_phi_stats.sum()
glue("phi_statistics", df_phi_stats)
unique_phi_classes = len(df_phi_stats) - 1
glue("phi_class_num", unique_phi_classes)
```

```{code-cell} ipython3
from typing import List
from collections import Counter
from flair.data import Dataset, _iter_dataset, Dictionary

def _get_all_tokens(dataset: Dataset) -> List[str]:
    assert dataset
    tokens = list(map((lambda s: s.tokens), _iter_dataset(dataset)))
    tokens = [token for sublist in tokens for token in sublist]
    return list(map((lambda t: t.text), tokens))

def _get_most_common_tokens(dataset: Dataset, max_tokens, min_freq) -> List[str]:
    tokens_and_frequencies = Counter(_get_all_tokens(dataset))

    tokens: List[str] = []
    for token, freq in tokens_and_frequencies.most_common():
        if (min_freq != -1 and freq < min_freq) or (max_tokens != -1 and len(tokens) == max_tokens):
            break
        tokens.append(token)
    return tokens

def make_vocab_dictionary(dataset: Dataset, max_tokens=-1, min_freq=1) -> Dictionary:
    """
    Creates a dictionary of all tokens contained in the corpus.
    By defining `max_tokens` you can set the maximum number of tokens that should be contained in the dictionary.
    If there are more than `max_tokens` tokens in the corpus, the most frequent tokens are added first.
    If `min_freq` is set the a value greater than 1 only tokens occurring more than `min_freq` times are considered
    to be added to the dictionary.
    :param max_tokens: the maximum number of tokens that should be added to the dictionary (-1 = take all tokens)
    :param min_freq: a token needs to occur at least `min_freq` times to be added to the dictionary (-1 = there is no limitation)
    :return: dictionary of tokens
    """
    tokens = _get_most_common_tokens(dataset, max_tokens, min_freq)

    vocab_dictionary: Dictionary = Dictionary()
    for token in tokens:
        vocab_dictionary.add_item(token)

    return vocab_dictionary


df_doc_stats = pd.DataFrame(
    {k: {
            "num docs": get_number_of_doc(ArchiveFolder, k.lower()),
            "num sentences": stats[k]["total_number_of_documents"],
            "num tokens": stats[k]["number_of_tokens"]["total"],
            "vocabulary": len(make_vocab_dictionary(getattr(corpus, k.lower()))),
            "Min token per sentence": stats[k]["number_of_tokens"]["min"],
            "Max token per sentence": stats[k]["number_of_tokens"]["max"],
            "Avg token per sentence": stats[k]["number_of_tokens"]["avg"],
            "num PHI": df_phi_stats.loc["TOTAL"][k]
        } for k in ["TRAIN", "DEV", "TEST"]
    }
).astype(int).T

df_sentence_stats = pd.DataFrame(
    {k: {
            "num sentences": stats[k]["total_number_of_documents"],
            "Min token per sentence": stats[k]["number_of_tokens"]["min"],
            "Max token per sentence": stats[k]["number_of_tokens"]["max"],
            "Avg token per sentence": stats[k]["number_of_tokens"]["avg"],
        } for k in ["TRAIN", "DEV", "TEST"]
    }
).astype(int).T

glue("doc_statistics", df_doc_stats)
glue("sent_statistics", df_sentence_stats)
```

```{code-cell} ipython3

```
