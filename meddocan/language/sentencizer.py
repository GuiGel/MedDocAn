r"""This module implements a custom a spaCy `custom component`_ that splits a
document into sentences.   
Sentences are split when the sentencizer encounters a new line marker "\\n".
In this way, when we load a ``meddocan`` file ``.txt``, we can load the file as
|Doc| and split the document into lines easily this way:

.. code-block:: python

    for sent in doc.sents:
        ...

.. _`custom component`:
  https://spacy.io/usage/processing-pipelines#custom-components
"""
from spacy.language import Language
from spacy.tokens import Doc


@Language.component("line_sentencizer")
def line_sentencizer(doc: Doc) -> Doc:
    for i, token in enumerate(doc[:-2]):
        # Define sentence start if pipe + titlecase token
        if "\n" in token.text:
            doc[i + 1].is_sent_start = True
        else:
            # Explicitly set sentence start to False otherwise, to tell
            # the parser to leave those tokens alone
            doc[i + 1].is_sent_start = False
    return doc
