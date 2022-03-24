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
    nlp = meddocan_pipeline()
    doc = nlp(text)
    for token, expected_text in zip(doc, expected):
        assert token.text == expected_text


if __name__ == "__main__":
    nlp = meddocan_pipeline()
    texts = [
        # "FerrándezCorreo",
        # "Remitido por: Dra. Lucrecia Sánchez-Rubio FerrándezCorreo electrónico: lsanchez@riojasalud.es",
        # "NºColMartínez",
        # "Médico: Gastón Demaría MartínezNºCol: 28 28 98702.",
        # "Médico: Gastón Demaría MartínezNºColumn: 28 28 98702.",
        # If 2 words are in the same document the test fails but this does not
        # happen in this dataset.
        # If this happens there will be non-detection or detection of the
        # entire text...
        # This is a limitation of tokenization on a character-based model only.
        "NºColFerrándezCorreo",
        "NºCol FerrándezCorreos",
    ]
    for doc in nlp.pipe(texts):
        print("=" * 100)
        for t in doc:
            print(t)
    print("=" * 100)
