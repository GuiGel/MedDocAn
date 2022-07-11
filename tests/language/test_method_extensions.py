"""This module contains the tests done to validate the functionality of the
:class:`WriteMethods` and particularly the ability to write a
``spacy.tokens.Doc`` in a file at the *IOB* format.
"""
from __future__ import annotations

import itertools as it
from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple
from unittest.mock import call, mock_open, patch

import pytest
from spacy.tokens import Doc
from spacy.vocab import Vocab

from meddocan.language.method_extensions import WriteMethods


@dataclass
class TokenElements:
    word: str
    space: Optional[bool] = False
    sent_start: bool = False
    iob: str = "O"


@dataclass
class MockDoc:
    tokens: List[TokenElements]
    """``spacy.tokens.Doc`` raw tokens attribute"""

    def get_spacy_doc(self) -> Doc:
        """Create a ``spacy.tokens.Doc``.

        Returns:
            Doc: The doc created from ``MockDoc.tokens``attribute.
        """
        words = [t.word for t in self.tokens]

        spaces = None
        if None not in (_spaces := [t.space for t in self.tokens]):
            spaces = _spaces

        return Doc(
            Vocab(strings=words),
            words=words,
            spaces=spaces,  # type: ignore[arg-type]
            sent_starts=[t.sent_start for t in self.tokens],
            ents=[t.iob for t in self.tokens],
        )


class TestMockDoc:
    """Test :class:`MockDoc`."""

    @pytest.mark.parametrize(
        argnames="tokens, expected_text, expected_sents, expected_ents",
        argvalues=(
            (
                [
                    TokenElements("Me", True, True),
                    TokenElements("llamo", True, False),
                    TokenElements("Pedro", True, False, "B-PERS"),
                    TokenElements("Fuerte", False, False, "I-PERS"),
                    TokenElements("!", True, False),
                    TokenElements("Vivo", True, True),
                    TokenElements("en", True, False),
                    TokenElements("Salamanca", False, False, "B-LOC"),
                    TokenElements(".", False, False),
                ],
                "Me llamo Pedro Fuerte! Vivo en Salamanca.",
                ["Me llamo Pedro Fuerte!", "Vivo en Salamanca."],
                ["Pedro Fuerte", "Salamanca"],
            ),
            ([TokenElements("Bilbao")], "Bilbao", ("Bilbao",), None),
            (
                [TokenElements("Bilbao", iob="B-LOC")],
                "Bilbao",
                [
                    "Bilbao",
                ],
                [
                    "Bilbao",
                ],
            ),
        ),
    )
    def test_get_doc(
        self,
        tokens: List[TokenElements],
        expected_text: str,
        expected_sents: Optional[List[str]],
        expected_ents: Optional[List[str]],
    ) -> None:

        _expected_sents: List[str] = []
        if expected_sents is not None:
            _expected_sents = expected_sents

        _expected_ents: List[str] = []
        if expected_ents is not None:
            _expected_ents = expected_ents

        doc = MockDoc(tokens).get_spacy_doc()

        # text spaces and str are rendered correctly
        assert doc.text == expected_text

        # sentences are correct
        for found_sent, expected_sent in it.zip_longest(
            doc.sents, _expected_sents
        ):
            assert found_sent.text == expected_sent

        # entities are correct
        for found_ent, expected_ent in it.zip_longest(
            doc.ents, _expected_ents
        ):
            assert found_ent.text == expected_ent


class TestWriteMethods:
    @pytest.mark.parametrize(
        "tokens",
        (
            [
                TokenElements("Me", True, True),
                TokenElements("llamo", True, False),
                TokenElements("Pedro", True, False, "B-PERS"),
                TokenElements("Fuerte", False, False, "I-PERS"),
                TokenElements("!", True, False),
                TokenElements("Vivo", True, True),
                TokenElements("en", True, False),
                TokenElements("Salamanca", False, False, "B-LOC"),
                TokenElements(".", False, False),
            ],
        ),
    )
    @pytest.mark.parametrize("mode", ("a", "w"))
    @pytest.mark.parametrize(
        argnames="write_sentences, document_separator_token, expected",
        argvalues=(
            (
                True,
                None,
                (
                    "Me O\n",
                    "llamo O\n",
                    "Pedro B-PERS\n",
                    "Fuerte I-PERS\n",
                    "! O\n",
                    "\n",
                    "Vivo O\n",
                    "en O\n",
                    "Salamanca B-LOC\n",
                    ". O\n",
                    "\n",
                ),
            ),
            (
                True,
                "-DOCSTART-",
                (
                    "Me O\n",
                    "llamo O\n",
                    "Pedro B-PERS\n",
                    "Fuerte I-PERS\n",
                    "! O\n",
                    "\n",
                    "Vivo O\n",
                    "en O\n",
                    "Salamanca B-LOC\n",
                    ". O\n",
                    "\n",
                ),
            ),
            (
                False,
                None,
                (
                    "Me O\nllamo O\nPedro B-PERS\nFuerte I-PERS\n!"
                    " O\n\\n O\nVivo O\nen O\nSalamanca B-LOC\n. O\n\\n O\n\n"
                ),
            ),
            (
                False,
                "-DOCSTART-",
                (
                    "Me O\nllamo O\nPedro B-PERS\nFuerte I-PERS\n!"
                    " O\n\\n O\nVivo O\nen O\nSalamanca B-LOC\n. O\n\\n O\n\n"
                ),
            ),
        ),
    )
    def test__doc_to_connl03_one_doc(
        self,
        tokens: List[TokenElements],
        write_sentences: bool,
        document_separator_token: Optional[str],
        mode: Literal["w", "a"],
        expected: Tuple[str, ...],
    ) -> None:

        doc = MockDoc(tokens).get_spacy_doc()

        m = mock_open()
        with patch("pathlib.Path.open", m):
            WriteMethods._WriteMethods__doc_to_connl03(  # type: ignore[attr-defined]  # private method
                doc,
                file="file.tsv",
                write_sentences=write_sentences,
                document_separator_token=document_separator_token,
            )
        m.assert_called_once_with(mode="w", encoding="utf-8", newline="\n")

        # Recuperate the calls made to the "writelines" method of the mock
        # object.
        handle = m()

        # Verify that the written lines are the expected ones.
        if write_sentences:
            # Verify that writelines methods is call with the expected lines
            assert handle.writelines.call_args == call(list(expected))
            if document_separator_token is not None:
                assert handle.write.call_args == call(
                    f"{document_separator_token} O\n\n"
                )
        else:
            if document_separator_token is not None:
                assert handle.write.call_args_list == [
                    call(expected),
                    call(f"{document_separator_token} O\n\n"),
                ]
            else:
                assert handle.write.call_args == call(expected)
