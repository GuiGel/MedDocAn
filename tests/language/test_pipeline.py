from itertools import zip_longest
from unittest.mock import call, mock_open, patch

import pytest

from meddocan.data.containers import BratSpan
from meddocan.data.utils import set_ents_from_brat_spans


class TestMeddocanPipeline:
    """Test the different components of the ``meddocan_pipeline`` with their
    interdependencies.
    """

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
    def test_tokenizer(self, meddocan_language, text, expected):
        """Test that the ``meddocan_pipeline`` tokenize the sentence as
        expected.
        """
        doc = meddocan_language(text)
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
    def test_missaligned_splitter(self, meddocan_language, text, expected):
        """Test that the missaligned splitter works as expected, i.e that the
        given ``text`` argument is tokenize as indicated by ``expected``.
        """
        doc = meddocan_language(text)
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
    def test_line_sentencizer(
        self, meddocan_language, text, expected_sentences
    ):
        doc = meddocan_language(text)
        for sentence, expected_sentence in zip_longest(
            doc.sents, expected_sentences
        ):
            assert sentence.text == expected_sentence

    @pytest.mark.parametrize(
        argnames="text,brat_spans_args,write_sentences,expected",
        argvalues=[
            (
                "Me llamo Veranika. Vivo en Minsk.",
                [
                    (None, "PERS", 9, 17, "Veranika"),
                    (None, "LOC", 27, 32, "Minsk"),
                ],
                True,
                [
                    "Me O\n",
                    "llamo O\n",
                    "Veranika B-PERS\n",
                    ". O\n",
                    "Vivo O\n",
                    "en O\n",
                    "Minsk B-LOC\n",
                    ". O\n",
                    "\n",
                ],
            ),
            (
                "Me llamo Veranika.\nVivo en Minsk.\n",
                [
                    (None, "PERS", 9, 17, "Veranika"),
                    (None, "LOC", 27, 32, "Minsk"),
                ],
                True,
                [
                    "Me O\n",
                    "llamo O\n",
                    "Veranika B-PERS\n",
                    ". O\n",
                    "\n",
                    "Vivo O\n",
                    "en O\n",
                    "Minsk B-LOC\n",
                    ". O\n",
                    "\n",
                ],
            ),
            (
                "Me llamo Veranika.\n  Vivo en Minsk.\n",
                [
                    (None, "PERS", 9, 17, "Veranika"),
                    (None, "LOC", 29, 34, "Minsk"),
                ],
                True,
                [
                    "Me O\n",
                    "llamo O\n",
                    "Veranika B-PERS\n",
                    ". O\n",
                    "\n",  # A new line is created here.
                    # The space is removed.
                    "Vivo O\n",
                    "en O\n",
                    "Minsk B-LOC\n",
                    ". O\n",
                    "\n",
                ],
            ),
            (
                "Es un tumor maligno.",
                [],
                True,
                [
                    "Es O\n",
                    "un O\n",
                    "tumor O\n",
                    "maligno O\n",
                    ". O\n",
                    "\n",
                ],
            ),
        ],
    )
    def test_doc_to_connl03(
        self,
        meddocan_language,
        text,
        brat_spans_args,
        write_sentences,
        expected,
    ):
        """Test that the ``spacy.tokens.Doc`` object produce by the
        ``meddocan.language.pipeline.MedocanPipeline`` is written correctly to
        the desire ``CoNNL03`` format by the method ``to_connl03`` that is
        attached to the ``spacy.tokens.Doc`` object by the pipeline.

        (Perhaps a fixture to generate data can be the good thing here?)

        1.  Create a doc from blank_language and text.
        2.  Create BratSpan from brat_spans_args.
        3.  Set doc entities from BratSpans list.
        4.  Write doc to file.
        5.  Verify that ``write`` or ``writelines`` methods received the correct
            parameters.
        """
        doc = meddocan_language(text)
        if len(brat_spans_args):
            brat_spans = [
                BratSpan(*brat_span_args) for brat_span_args in brat_spans_args
            ]
            doc = set_ents_from_brat_spans(doc, brat_spans)

        # Verify that the file annotations are the expected one.
        m = mock_open()
        with patch("pathlib.Path.open", m):
            doc._.to_connl03(file="file.tsv", write_sentences=write_sentences)
        m.assert_called_once_with(mode="w", encoding="utf-8", newline="\n")

        # Recuperate the calls made to the "writelines" method of the mock
        # object.
        handle = m()

        if write_sentences:
            execute_calls = handle.writelines.call_args
        else:
            execute_calls = handle.write.call_args

        # Verify that the written lines are the expected ones.
        assert execute_calls == call(expected)

    @pytest.mark.parametrize(
        argnames="text,brat_spans_args,expected_lines",
        argvalues=[
            (
                "Me llamo Veranika.\nVivo en Minsk.",
                [
                    (None, "PERS", 9, 17, "Veranika"),
                    (None, "LOC", 27, 32, "Minsk"),
                ],
                [
                    "T_0\tPERS 9 17\tVeranika\n",
                    "T_1\tLOC 27 32\tMinsk\n",
                ],
            ),
        ],
    )
    def test_to_ann(
        self, meddocan_language, text, brat_spans_args, expected_lines
    ):
        """Test that the ``spacy.tokens.Doc`` object produce by the
        ``meddocan.language.pipeline.MedocanPipeline`` is written correctly to
        the desire ``ann`` format by the method ``to_ann`` that is attached to
        the ``spacy.tokens.Doc`` object by the pipeline.

        1.  Create a doc from blank_language and text.
        2.  Create BratSpan from brat_spans_args.
        3.  Set doc entities from BratSpans list.
        4.  Write doc to file.
        5.  Verify that the ``writelines`` method received the correct
            parameters.
        """
        doc = meddocan_language(text)
        brat_spans = [
            BratSpan(*brat_span_args) for brat_span_args in brat_spans_args
        ]
        doc = set_ents_from_brat_spans(doc, brat_spans)

        # Verify that the file annotations are the expected one.
        m = mock_open()
        with patch("pathlib.Path.open", m):
            doc._.to_ann(file="test_file.ann")

        m.assert_called_once_with(mode="w", encoding="utf-8")

        # Recuperate the calls made to the "write" method of the mock object.
        handle = m()
        writelines_call = handle.writelines.call_args

        # Verify that the written lines are the expected ones.
        assert writelines_call == call(expected_lines)
