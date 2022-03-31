from itertools import zip_longest
from pathlib import Path
from unittest.mock import call, mock_open, patch

import pytest
from spacy import blank

from meddocan.data.containers import BratAnnotations, BratDoc, BratSpan


class TestBratDoc:
    @pytest.mark.parametrize(
        "text,lines,sentences,expected_lines",
        [
            (
                "Vivo en Bilbao.\nMe llamo Guillaume.\n",
                ["T1\tLOC 8 14\tBilbao", "T2\tPERS 25 34\tGuillaume"],
                True,
                [
                    "Vivo O\n",
                    "en O\n",
                    "Bilbao B-LOC\n",
                    ". O\n",
                    "\n",
                    "Me O\n",
                    "llamo O\n",
                    "Guillaume B-PERS\n",
                    ". O\n",
                    "\n",
                ],
            ),
            (
                "Vivo en Bilbao.\nMe llamo Guillaume.\n",
                ["T1\tLOC 8 14\tBilbao", "T2\tPERS 25 34\tGuillaume"],
                False,
                [
                    "Vivo O\n",
                    "en O\n",
                    "Bilbao B-LOC\n",
                    ". O\n",
                    "\\n O\n",
                    "Me O\n",
                    "llamo O\n",
                    "Guillaume B-PERS\n",
                    ". O\n",
                    "\\n O\n",
                    "\n",
                ],
            ),
            (
                "Vivo en Bilbao.  \n   \tMe llamo Guillaume.\n",
                ["T1\tLOC 8 14\tBilbao", "T2\tPERS 31 40\tGuillaume"],
                True,
                [
                    "Vivo O\n",
                    "en O\n",
                    "Bilbao B-LOC\n",
                    ". O\n",
                    "\n",
                    "Me O\n",
                    "llamo O\n",
                    "Guillaume B-PERS\n",
                    ". O\n",
                    "\n",
                ],
            ),
            (
                "Vivo en Bilbao.  \n   \tMe llamo Guillaume.\n",
                ["T1\tLOC 8 14\tBilbao", "T2\tPERS 31 40\tGuillaume"],
                False,
                [
                    "Vivo O\n",
                    "en O\n",
                    "Bilbao B-LOC\n",
                    ". O\n",
                    # Remplace "\n" by "\\n O\n" and append " O\n" at the end of
                    # the line.
                    " \\n O\n   \t O\n",
                    "Me O\n",
                    "llamo O\n",
                    "Guillaume B-PERS\n",
                    ". O\n",
                    "\\n O\n",
                    "\n",
                ],
            ),
        ],
    )
    def test_write(self, text, lines, sentences, expected_lines):
        """Test that an annotated document in the BRAT format is written
        correctly.
        """
        nlp = blank("es")
        brat_spans = [
            BratSpan.from_bytes(line.encode("utf-8")) for line in lines
        ]
        brat_annotations = BratAnnotations(text, brat_spans)
        brat_doc = BratDoc.from_brat_annotations(nlp, brat_annotations)

        # Verify that the entities are assigned correctly to the Doc attribute
        assert [ent.text for ent in brat_doc.doc.ents] == [
            brat_span.text for brat_span in brat_spans
        ]

        # https://docs.python.org/3/library/unittest.mock.html#mock-open
        m = mock_open()

        with patch("meddocan.data.containers.Path.open", m):
            brat_doc.write_connl03(
                Path("test_file"), mode="w", sentences=sentences
            )

        m.assert_called_once_with(mode="w", encoding="utf-8", newline="\n")

        # Recuperate the calls made to the "write" method of the mock object.
        handle = m()
        write_calls = handle.write.call_args_list

        # Verify that the written lines are the expected ones.
        for write_call, expected_line in zip_longest(
            write_calls, expected_lines
        ):
            assert write_call == call(expected_line)
