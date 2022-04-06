from typing import NewType, Optional

from spacy import blank
from spacy.language import Language
from spacy.tokens import Doc

# TODO Define the tokenizer as a language for using entry points and give it a
# real type at the same time.
from meddocan.language.tokenizer import meddocan_tokenizer

MeddocanLanguage = NewType("MeddocanLanguage", Language)


def meddocan_pipeline(
    model_loc: Optional[str] = None, mini_batch_size: int = 8
) -> MeddocanLanguage:
    """Create meddocan language.

    Example:

    >>> nlp = meddocan_pipeline()
    >>> doc = nlp("Vivo en Bilbao.")
    >>> doc
    Vivo en Bilbao.

    The document has an extension ``is_meddocan_doc`` that permit to
    know that the document has been produced by the ``meddocan_pipeline``.

    >>> doc._.is_meddocan_doc
    True

    Verify the default value of the extension
    >>> doc._.is_meddocan_doc
    False

    .. warning:
        The ``is_meddocan_doc`` extension is mutable!

    Returns:
        MeddocanLanguage: A ``spacy.language.Language`` that preprocess the
            meddocan text in order to have alignment with the entities offsets.
    """
    nlp = blank("es")

    Doc.set_extension("is_meddocan_doc", default=False, force=True)
    nlp.tokenizer = meddocan_tokenizer(nlp)
    nlp.add_pipe(
        "missaligned_splitter",
        config={"words": ["NºCol", "Correo", "años", "DR", "\.$", "\n"]},
    )
    nlp.add_pipe("line_sentencizer")
    nlp.add_pipe(
        "predictor",
        config={"model_loc": model_loc, "mini_batch_size": mini_batch_size},
    )
    nlp.add_pipe("write_methods")
    return MeddocanLanguage(nlp)
