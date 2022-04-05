"""This module implements a custom spacy component that splits a document into
sentences.   
Sentences are split when the sentencizer encounters a new line marker "\n".
In this way, when we load a meddocan file ``.txt.``, we can load the file as
``spacy.tokens.Document`` and split the document into lines easily this way:

```python
for sent in doc.sents:
    ... # Do some stuff on the sentence
```

Ref: https://spacy.io/usage/processing-pipelines#custom-components
"""
from spacy.language import Language


@Language.component("line_sentencizer")
def line_sentencizer(doc):
    for i, token in enumerate(doc[:-2]):
        # Define sentence start if pipe + titlecase token
        if "\n" in token.text:
            doc[i + 1].is_sent_start = True
        else:
            # Explicitly set sentence start to False otherwise, to tell
            # the parser to leave those tokens alone
            doc[i + 1].is_sent_start = False
    return doc
