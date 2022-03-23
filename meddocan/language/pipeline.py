from typing import NewType

import spacy
from spacy.language import Language

from .splitter import missaligned_splitter  # To register the component.
from .tokenizer import meddocan_tokenizer

MeddocanLanguage = NewType("MeddocanLanguage", Language)


def meddocan_pipeline() -> MeddocanLanguage:
    """Create meddocan language.

    Returns:
        MeddocanLanguage: A spacy.language.Language that preprocess the
            meddocan text in order to have alignment with the entities offsets.
    """
    nlp = spacy.blank("es")
    nlp.tokenizer = meddocan_tokenizer(nlp)
    # nlp.add_pipe("sentencizer")
    nlp.add_pipe(
        "missaligned_splitter",
        config={"words": ["NºCol", "Correo", "años", "DR", "\.$"]},
    )
    return MeddocanLanguage(nlp)
