"""This module contains the tests done to validate the functionality of the
:class:`PredictorComponent`.
"""
from __future__ import annotations

from typing import Iterator, List, Literal, Tuple
from unittest.mock import patch

import pytest
from flair.data import Sentence, Span
from flair.models.sequence_tagger_utils.bioes import get_spans_from_bio
from spacy.tokens import Doc
from spacy.vocab import Vocab

from meddocan.language.predictor import PredictorComponent


def get_doc(
    words: List[str],
    spaces: List[bool],
    sent_starts: List[Literal[1, -1]],
) -> None:
    """Create a ``spacy.tokens.Doc`` object.

    Example:

    >>> words = ["Vivo", "en", "Bilbao", "."]
    >>> spaces = [True, True, False, False]
    >>> sent_starts = [True, False, False, False]
    >>> doc = get_doc(words=words, spaces=spaces, sent_starts=sent_starts)
    >>> for sent in doc.sents:
    ...     print(f"{sent=}")
    sent=Vivo en Bilbao.

    Args:
        words (List[str]): The words that will be represent as a
            ``spacy.tokens.Token`` object
        spaces (List[bool]): The spaces associated with the ``Token``.
        sent_starts (List[Literal[-1, 1]]): The token is a sentence start or not

    Returns:
        _type_: _description_
    """
    vocab = Vocab(strings=words)
    doc = Doc(vocab, words=words, spaces=spaces, sent_starts=sent_starts)
    return doc


def get_docs(
    attrs: List[List[Tuple[str, bool, Literal[-1, 1]]]]
) -> Iterator[Doc]:
    """

    Example:

    >>> docs = get_docs(
    ...     [
    ...         [
    ...             ["Vivo", False, 1],
    ...             ["en", True, -1],
    ...             ["Bilbao", False, -1],
    ...             [".", False, -1],
    ...         ],
    ...     ]
    ... )
    >>> for doc1 in docs:
    ...     for sent in doc1.sents:
    ...         print(f"{sent=}")
    sent=Vivoen Bilbao.

    Args:
        attrs (List[List[Tuple[str, bool, Literal[-1, 1]]]): Each tuple must
            contains a string (the future token), a bool indicating if there
            is a space after the token or not and a literal that indicate if
            1 that the token is a sentence start and if -1 that no sentence
            begins.

    Yields:
        Iterator[Doc]: The created ``spacy.tokens.Doc`` objects.
    """
    for attr in attrs:
        (
            words,
            spaces,
            sent_starts,
        ) = map(list, zip(*attr))
        doc = get_doc(words, spaces, sent_starts)
        yield doc


def add_tags(sentences: List[Sentence], tags: List[List[Tuple[str, str]]]):
    """Add tags to flair sentences.

    Example:

    >>> sentences = [Sentence("Vivo en Bilbo.")]
    >>> add_tags(sentences, [[("Vivo", "O"), ("en", "O"), ("Bilbo", "B-LOC"), (".", "O")]])
    >>> sentences[0].get_labels("ner")
    ['Span[2:3]: "Bilbo"'/'LOC' (1.0)]

    Args:
        sentences (List[Sentence]): A list of flair sentence to tag.
        tags (List[List[Tuple[str, str]]]): A list that contains lists of tag\
        to add to the sentence.
            Each tuple has the form (text, bio_tag-label)
    """
    for flair_sentence, flair_tags in zip(sentences, tags):
        sentence_strings, sentence_tags = zip(*flair_tags)

        # Verify text is set correctly
        original_sentence_strings = tuple(
            [t.text for t in flair_sentence.tokens]
        )
        assert (
            original_sentence_strings == sentence_strings
        ), f"{original_sentence_strings} != {sentence_strings}"

        # Set scores to 1.0 for this test.
        sentence_scores = [1.0 for _ in flair_tags]
        predicted_spans = get_spans_from_bio(
            list(sentence_tags), sentence_scores
        )
        for predicted_span in predicted_spans:
            span: Span = flair_sentence[
                predicted_span[0][0] : predicted_span[0][-1] + 1
            ]
            span.add_label(
                "ner", value=predicted_span[2], score=predicted_span[1]
            )


@pytest.mark.parametrize(
    "sentences, tags, expected",
    [
        (
            [Sentence("Vivo en Bilbao.")],
            [[("Vivo", "O"), ("en", "O"), ("Bilbao", "B-LOC"), (".", "O")]],
            "['Span[2:3]: \"Bilbao\"'/'LOC' (1.0)]",
        )
    ],
)
def test_add_tags(
    sentences: List[Sentence],
    tags: List[List[Tuple[str, str]]],
    expected: str,
) -> None:
    add_tags(sentences, tags)
    assert str(sentences[0].get_labels("ner")) == expected


class TestPredictorComponent:
    @pytest.mark.parametrize(
        argnames="text_space_sent,text_start_end",
        argvalues=[
            (
                (
                    ("Nombre", False, True),
                    (":", True, False),
                    ("Mariano", False, False),
                    (".", False, False),
                    ("\n", True, False),
                    ("Remitidos", True, True),
                    ("  \t   ", False, False),
                    ("por", True, False),
                    ("Oscar", False, False),
                    (".", False, False),
                ),
                (
                    [
                        ("Nombre", 0, 6),
                        (":", 6, 7),
                        ("Mariano", 8, 15),
                        (".", 15, 16),
                    ],
                    [
                        ("Remitidos", 18, 27),
                        ("por", 34, 37),
                        ("Oscar", 38, 43),
                        (".", 43, 44),
                    ],
                ),
            ),
            (
                (
                    ("Apellidos", False, True),
                    (":", True, False),
                    ("Ramos", True, False),
                    ("Baez", False, False),
                    (".", False, False),
                    ("\n", True, False),
                    ("NHC", False, True),
                    (":", True, False),
                    ("2594890", False, False),
                    (".", False, False),
                ),
                (
                    [
                        ("Apellidos", 0, 9),
                        (":", 9, 10),
                        ("Ramos", 11, 16),
                        ("Baez", 17, 21),
                        (".", 21, 22),
                    ],
                    [
                        ("NHC", 24, 27),
                        (":", 27, 28),
                        ("2594890", 29, 36),
                        (".", 36, 37),
                    ],
                ),
            ),
            (
                (
                    ("Apellidos", False, True),
                    (":", True, False),
                    ("Ramos", True, False),
                    ("  ", True, False),
                    ("Baez", False, False),
                    (".", False, False),
                ),
                (
                    [
                        ("Apellidos", 0, 9),
                        (":", 9, 10),
                        ("Ramos", 11, 16),
                        ("Baez", 20, 24),
                        (".", 24, 25),
                    ],
                ),
            ),
        ],
    )
    def test_flair_sentence_bis(self, text_space_sent, text_start_end):
        """Test that the conversion from spaCy Span to Flair sentence is correct.
        The transformation is destructive but preserves the position of tokens
        that are not considered as spaces.

        Args:
            text_space_sent (_type_): _description_
            text_start_end (_type_): _description_
        """
        for doc in get_docs([text_space_sent]):
            for spacy_sent, _text_start_end in zip(doc.sents, text_start_end):
                flair_sentence = PredictorComponent.flair_sentence(spacy_sent)
                for token, (text, start, end) in zip(
                    flair_sentence, _text_start_end
                ):
                    assert token.text == text
                    assert token.start_pos == start
                    assert token.end_pos == end

    @pytest.mark.parametrize(
        argnames="spacy_text_space_sentstart_label,flair_text_label,expected_ents",
        argvalues=[
            (
                (
                    ("Nombre", False, True, "O"),
                    (":", True, False, "O"),
                    ("Mariano", False, False, "B-PERS"),
                    (".", False, False, "O"),
                    (
                        "\n",
                        True,
                        False,
                        "O",
                    ),  #  for "/n" count as 2 chars for spacy.
                    ("Remitidos", True, True, "O"),
                    ("  \t   ", False, False, "O"),
                    ("por", True, False, "O"),
                    ("Oscar", False, False, "B-PERS"),
                    (".", False, False, "O"),
                ),
                (
                    [
                        [
                            ("Nombre", "O"),
                            (":", "O"),
                            ("Mariano", "B-PERS"),
                            (".", "O"),
                        ],
                        [
                            ("Remitidos", "O"),
                            ("por", "O"),
                            ("Oscar", "B-PERS"),
                            (".", "O"),
                        ],
                    ],
                ),
                (("Mariano", 8, 15, "PERS"), ("Oscar", 38, 43, "PERS")),
            ),
            (
                (
                    ("Apellidos", False, True, "O"),
                    (":", True, False, "O"),
                    ("Ramos", True, False, "B-PERS"),
                    ("Baez", False, False, "I-PERS"),
                    (".", False, False, "O"),
                    ("\n", True, False, "O"),
                    ("NHC", False, True, "O"),
                    (":", True, False, "O"),
                    ("2594890", False, False, "B-NUM"),
                    (".", False, False, "O"),
                ),
                (
                    [
                        [
                            ("Apellidos", "O"),
                            (":", "O"),
                            ("Ramos", "B-PERS"),
                            ("Baez", "I-PERS"),
                            (".", "O"),
                        ],
                        [
                            ("NHC", "O"),
                            (":", "O"),
                            ("2594890", "B-NUM"),
                            (".", "O"),
                        ],
                    ],
                ),
                (("Ramos Baez", 11, 21, "PERS"), ("2594890", 29, 36, "NUM")),
            ),
            (
                (
                    ("Apellidos", False, True, "O"),
                    (":", True, False, "O"),
                    ("Ramos", True, False, "B-PERS"),
                    ("  ", True, False, "I-PERS"),
                    ("Baez", False, False, "I-PERS"),
                    (".", False, False, "O"),
                ),
                (
                    [
                        [
                            ("Apellidos", "O"),
                            (":", "O"),
                            ("Ramos", "B-PERS"),
                            ("Baez", "I-PERS"),
                            (".", "O"),
                        ],
                    ],
                ),
                (("Ramos    Baez", 11, 24, "PERS"),),
            ),
        ],
    )
    def test_set_ents(
        self,
        spacy_text_space_sentstart_label: Tuple[Tuple[str, bool, bool, str]],
        flair_text_label: Tuple[List[List[Tuple[str, str]]]],
        expected_ents: Tuple[Tuple[str, int, int, str]],
    ):
        """Test the ``set_ents`` method of the
        ``meddocan.language.predictor.Predictor`` class.

        1. Create ``spacy.tokens.Doc`` object given the argument pass to\
            ``spacy_text_space_sentstart_label``. The Doc object will be pass\
            to the ``Predictor.set_ents`` method in order to predict the\
            entities.
        2. To predict the entities we need to use the flair model. As we don't\
            want to load a model to slow down the tests, we use a mock that\
            create a fake model. To create predictions we use the function\
            ``add_tags`` to which we pass the appropriate labels via the\
            ``flair_text_label``\
            parameter.
        3. Finally we check that the entities added to the ``Doc`` object are\
            the expected ones with the help of the ``expected_ents`` parameter.
        """
        # Use flair_text_label to mock prediction with the ``add_tags`` function.
        texts, spaces, sents_starts, labels = zip(
            *spacy_text_space_sentstart_label
        )
        docs = get_docs([list(zip(texts, spaces, sents_starts))])

        for doc, tags in zip(docs, flair_text_label):
            print(f"------------ {doc=}")
            print(f"------------ {tags=}")
            # tag_flair_sentence = partial(add_tags, tags)
            with patch(
                "meddocan.language.predictor.SequenceTagger"
            ) as MockSequenceTagger:

                # Instantiate the MockSequenceTagger object and set methods
                mst = MockSequenceTagger.return_value
                MockSequenceTagger.load.return_value = mst
                mst.predict.side_effect = lambda s: add_tags(s, tags)

                # Pass an argument to the model_loc parameter in order to load
                # the mock model internally.
                predictor = PredictorComponent(None, "model_loc", None)
                doc = predictor.set_ents(doc)

                # Test that the predicted entities are the expected ones.
                for predict_ent, expected_ent in zip(doc.ents, expected_ents):
                    text, start_char, end_char, label = expected_ent
                    assert predict_ent.text == text
                    assert predict_ent.start_char == start_char
                    assert predict_ent.end_char == end_char
                    assert predict_ent.label_ == label
