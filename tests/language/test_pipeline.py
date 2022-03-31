from itertools import zip_longest

import pytest

from meddocan.language.pipeline import meddocan_pipeline


class TestMeddocanPipeline:
    """Test the different components of the ``meddocan_pipeline`` with their
    interdependencies.
    """

    nlp = meddocan_pipeline()

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
            ("    \n   ", ["    ", "\n", "   "]),
        ],
    )
    def test_tokenizer(self, text, expected):
        """Test that the ``meddocan_pipeline`` tokenize the sentence as
        expected.
        """
        nlp = meddocan_pipeline()
        doc = nlp(text)
        for token, expected_text in zip(doc, expected):
            assert token.text == expected_text

    @pytest.mark.parametrize(
        argnames="text,expected",
        argvalues=[
            ("  \n   ", "  |\n|   "),
            ("Der..", "Der|.|."),
            ("der..", "der|.|."),
            ("Izq..", "Izq|.|."),
            ("izq..", "izq|.|."),
            ("MartínezNºCol", "Martínez|NºCol"),
            ("DominguezCorreo", "Dominguez|Correo"),
            ("3 c.", "3|c|."),
            ("DRAlberto", "DR|Alberto"),
            ("añosingresó", "años|ingresó"),
        ],
    )
    def test_missaligned_splitter(self, text, expected):
        """Test that the missaligned splitter works as expected, i.e that the
        given ``text`` argument is tokenize as indicated by ``expected``.
        """
        doc = self.nlp(text)
        output = "|".join(token.text for token in doc)
        assert expected == output

    @pytest.mark.parametrize(
        "text,expected_sentences",
        [
            (
                (
                    "Datos del paciente.\nNombre: Pedro.   \n  Apellidos: "
                    "Rob Rivera.\n"
                ),
                [
                    "Datos del paciente.\n",
                    "Nombre: Pedro.   \n",
                    "  Apellidos: Rob Rivera.\n",
                ],
            )
        ],
    )
    def test_line_sentencizer(self, text, expected_sentences):
        doc = self.nlp(text)
        for sentence, expected_sentence in zip_longest(
            doc.sents, expected_sentences
        ):
            assert sentence.text == expected_sentence
