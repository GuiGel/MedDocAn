from itertools import zip_longest

import pytest


@pytest.mark.parametrize(
    "text,expected_sentences",
    [
        (
            "Datos del paciente.\nNombre: Pedro.\n  Apellidos: Rob Rivera.\n",
            [
                "Datos del paciente.\n",
                "Nombre: Pedro.\n  ",
                "Apellidos: Rob Rivera.\n",
            ],
        )
    ],
)
def test_line_sentencizer(blank_language, text, expected_sentences):
    """Test the ``line_sentencizer`` custom component.

    1.  Create a blank language
    2.  Add a the ``line_sentencizer`` component.
    3.  Create the ``Doc`` from the given text
    4.  Check that the sentences obtained are those expected.
    """
    nlp = blank_language
    nlp.add_pipe("line_sentencizer")
    doc = nlp(text)
    for sentence, expected_sentence in zip_longest(
        doc.sents, expected_sentences
    ):
        assert sentence.text == expected_sentence
