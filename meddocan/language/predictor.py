"""Create a component to predict tags on the |Doc| with a
``flair.models.SequenceTagger`` object.

Look the spaCy `stateful components`_ example for more information.

.. _stateful components: https://spacy.io/usage/processing-pipelines#example-stateful-components
"""
import logging
from typing import Callable, List, Optional

import flair.data
from flair.models import SequenceTagger
from spacy.language import Language
from spacy.tokens import Doc, Span

from meddocan.language.logutils import all_logging_disabled


class PredictorComponent:
    """A custom component that adds |Span| entities  to a
    |Doc| based on predictions made by an entity detection model
    trained by the ``Flair`` library.

    If no model is passed as an argument, the component does nothing.

    Args:
        nlp (Language): spaCy Language.
        model_loc (Optional[str], optional): Location of the
        ``flair.models.SequenceTagger`` model on disk. Defaults to None.
        mini_batch_size (int, optional): Batch size for the prediction.
        Defaults to 8.
    """

    def __init__(
        self,
        nlp: Language,
        model_loc: Optional[str] = None,
        mini_batch_size: int = 8,
    ) -> None:
        self.nlp = nlp
        self.model_loc = model_loc  # Required by spacy
        if self.model_loc is not None:
            self.model = SequenceTagger.load(model_loc)
        else:
            self.model = None
        self.mini_batch_size = mini_batch_size

        # Set that a prediction as been made on the doc.
        Doc.set_extension(name="predicted", default=False, force=True)

    def __repr__(self) -> str:
        return (
            f"PredictorComponent(nlp={self.nlp.__class__.__qualname__}, "
            f"model_loc={self.model_loc}, "
            f"mini_batch_size={self.mini_batch_size})"
        )

    @staticmethod
    def flair_sentence(spacy_sentence: Span) -> flair.data.Sentence:
        """Get a Flair Sentence from a spaCy sentence.

        Args:
            spacy_sentence (Span): The spacy sentence.

        Returns:
            flair.data.Sentence: The obtained Flair sentence.
        """
        with all_logging_disabled(logging.WARNING):
            flair_sentence = flair.data.Sentence("")
        flair_sentence.language_code = "es"
        flair_sentence.start_pos = spacy_sentence.start_char
        flair_tokens: List[flair.data.Token] = []
        previous_flair_token: Optional[flair.data.Token] = None
        for spacy_token in spacy_sentence:
            if spacy_token.text.isspace():
                continue
            else:
                flair_token = flair.data.Token(
                    text=spacy_token.text,
                    start_position=spacy_token.idx,
                    whitespace_after=True,
                )
                flair_tokens.append(flair_token)
                if (previous_flair_token is not None) and (
                    flair_token.start_pos
                    == previous_flair_token.start_pos
                    + len(previous_flair_token.text)
                ):
                    previous_flair_token.whitespace_after = False
            previous_flair_token = flair_token
            flair_sentence.add_token(flair_token)
        return flair_sentence

    def set_ents(self, doc: Doc) -> Doc:
        """Transform each flair entity of type ''flair.data.Span`` to a
        ``spacy.tokens.Span`` list with. Fill the ``spacy.tokens.Doc.ents``
        list with the new ``Span``.

        Args:
            doc (Doc): The spacy ``Doc`` to which the
                method attaches entities.

        Returns:
            Doc: The document with the added entities if they
                exist.
        """
        doc_spans: List[Span] = []

        flair_sentences = list(map(self.flair_sentence, doc.sents))
        self.model.predict(flair_sentences)

        for flair_sentence in flair_sentences:
            flair_spans = flair_sentence.get_spans("ner")
            for flair_span in flair_spans:
                doc_span = doc.char_span(
                    flair_span.start_position,
                    flair_span.end_position,
                    flair_span.tag,
                    alignment_mode="strict",
                )
                doc_spans.append(doc_span)
        doc.set_ents(doc_spans)
        return doc

    def __call__(self, doc: Doc) -> Doc:
        if self.model is not None:
            self.set_ents(doc)
            doc._.predicted = True
        return doc


@Language.factory(
    "predictor",
    default_config={"model_loc": None, "mini_batch_size": 8},
)
def create_predictor(
    nlp: Language,
    name: str,
    model_loc: Optional[str],
    mini_batch_size: int = 8,
) -> Callable[[Doc], Doc]:
    return PredictorComponent(nlp, model_loc, mini_batch_size=mini_batch_size)
