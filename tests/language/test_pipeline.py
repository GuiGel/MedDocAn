import pytest

from meddocan.language.pipeline import meddocan_pipeline


@pytest.mark.parametrize(
    "text,expected",
    [
        ("NºCol FerrándezCorreo", ["NºCol", "Ferrández", "Correo"]),
        ("NºColMartínez", ["NºCol", "Martínez"]),
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
        ("c.", ["c", "."]),
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
def test_meddocan_pipeline(text, expected):
    """Test that the ``meddocan_pipeline`` tokenize the sentence as expected."""
    nlp = meddocan_pipeline()
    doc = nlp(text)
    for token, expected_text in zip(doc, expected):
        assert token.text == expected_text