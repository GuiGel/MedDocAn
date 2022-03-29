# First test that the lines are split as expected
from itertools import zip_longest

import pytest
from spacy import blank


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
def test_line_sentencizer(text, expected_sentences):
    nlp = blank("es")
    nlp.add_pipe("line_sentencizer")
    doc = nlp(text)
    for sentence, expected_sentence in zip_longest(
        doc.sents, expected_sentences
    ):
        assert sentence.text == expected_sentence
