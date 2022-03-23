import pytest
import spacy

from meddocan.language.tokenizer import meddocan_tokenizer


@pytest.mark.parametrize(
    argnames="text,tokens",
    argvalues=[
        ("H.", ["H", "."]),
        ("M.", ["M", "."]),
        ("F.", ["F", "."]),
        ("nhc-150679.", ["nhc", "-", "150679", "."]),
        ("CP:28029", ["CP", ":", "28029"]),
        ("2014.El", ["2014", ".", "El"]),
        ("11A.", ["11A", "."]),
        ("s/n.", ["s/n", "."]),
        ("Der..", ["Der", ".", "."]),
        ("Izq..", ["Izq", ".", "."]),
        ("der..", ["der", ".", "."]),
        ("izq..", ["izq", ".", "."]),
        ("1964)10", ["1964", ")", "10"]),
        ("+34679802102", ["+", "34679802102"]),
        (
            "E-mail:guacorch@yahoo.es",
            ["E", "-", "mail", ":", "guacorch@yahoo", ".", "es"],
        ),
        (
            "E-mail.hleonbrito@hotmail.com",
            ["E", "-", "mail", ".", "hleonbrito@hotmail", ".", "com"],
        ),
        ("nhc/976421.", ["nhc", "/", "976421", "."]),
    ],
)
def test_meddocan_tokenizer(text, tokens):
    """Test that the tokenizer works as expected, i.e that the given "text"
    is tokenize into the given "tokens".
    """
    nlp = spacy.blank("es")
    nlp.tokenizer = meddocan_tokenizer(nlp)
    assert [token.orth_ for token in nlp(text)] == tokens
