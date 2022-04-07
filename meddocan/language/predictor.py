# Create a component to predict tags from ``flair.models.SequenceTagger``
# https://spacy.io/usage/processing-pipelines#example-stateful-components
from typing import Callable, List, Optional

import flair.data
from flair.models import SequenceTagger
from flair.tokenization import SpacyTokenizer
from spacy.language import Language
from spacy.tokens import Doc, Span


class PredictorComponent:
    """A custom component that adds entities (``spacy.tokens.Span``) to a
    ``spacy.tokens.Doc`` based on predictions made by an entity detection model
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

        # Tokenize words in order to create ``flair.data.Sentence``.
        # TODO How to avoid retokenize the doc using the SpacyTokenizer.
        self.tokenizer = SpacyTokenizer(nlp)

        # Set that a prediction as been made on the doc.
        Doc.set_extension(name="predicted", default=False, force=True)

    def _repr__(self) -> str:
        return (
            f"PredictorComponent(nlp={self.nlp}, model_loc={self.model_loc}, "
            f"mini_batch_size={self.mini_batch_size})"
        )

    def predict(
        self,
        doc: Doc,
    ) -> List[flair.data.Sentence]:
        """Apply ``flair.models.SequenceTagger`` to spacy ``Doc`` sentence by
        sentence. We create from each sentence ``spacy.tokens.Span`` of the
        ``spacy.tokens.Doc`` a ``flair.data.Sentence`` with prediction.
        """
        flair_sentences = []
        for spacy_sentence in doc.sents:
            flair_sentence = flair.data.Sentence(
                spacy_sentence.text,
                use_tokenizer=self.tokenizer,
                language_code="es",
                start_position=spacy_sentence[0].idx,
            )
            flair_sentences.append(flair_sentence)

        self.model.predict(
            flair_sentences, mini_batch_size=self.mini_batch_size
        )
        return flair_sentences

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

        flair_sentences = self.predict(doc)

        for flair_sentence in flair_sentences:
            flair_spans = flair_sentence.get_spans("ner")
            for flair_span in flair_spans:
                doc_span = doc.char_span(
                    flair_sentence.start_pos + flair_span[0].start_pos,
                    flair_sentence.start_pos + flair_span[-1].end_pos,
                    flair_span.tag,
                    alignment_mode="strict",
                )
                assert (
                    doc_span.text == flair_span.to_original_text()
                ), f"{doc_span.text!r} == {flair_span.to_original_text()!r}"
                doc_spans.append(doc_span)
        doc.set_ents(doc_spans)
        return doc

    def __call__(self, doc: Doc) -> Doc:
        if self.model is not None:
            self.set_ents(doc)
            doc._.predicted = True
        return doc


@Language.factory(
    "predictor", default_config={"model_loc": None, "mini_batch_size": 8}
)
def create_predictor(
    nlp: Language,
    name: str,
    model_loc: Optional[str],
    mini_batch_size: int = 8,
) -> Callable[[Doc], Doc]:
    return PredictorComponent(nlp, model_loc, mini_batch_size=mini_batch_size)
