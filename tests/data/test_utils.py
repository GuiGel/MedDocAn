from itertools import zip_longest
from unittest.mock import call, mock_open, patch

import pytest

from meddocan.data.containers import BratSpan
from meddocan.data.utils import doc_to_ann, set_ents_from_brat_spans


@pytest.mark.parametrize(
    argnames="text,brat_spans_args",
    argvalues=[
        (
            "Me llamo Veranika.\nVivo en Minsk.\n",
            [
                (None, "PERS", 9, 17, "Veranika"),
                (None, "LOC", 27, 32, "Minsk"),
            ],
        ),
    ],
)
def test_set_ents_from_brat_spans(blank_language, text, brat_spans_args):
    """Test the `set_ents_from_brat_spans` function.

    1.  Create a doc from blank_language and text
    2.  Create BratSpan from brat_spans_args
    3.  Set doc entities from BratSpans list.
    4.  Verify that the ents are the expected one by comparing to BratSpans
    list.
    """
    doc = blank_language(text)
    brat_spans = [
        BratSpan(*brat_span_args) for brat_span_args in brat_spans_args
    ]
    doc = set_ents_from_brat_spans(doc, brat_spans)
    for ent, brat_span in zip_longest(doc.ents, brat_spans):
        assert ent.start_char == brat_span.start
        ent.end_char == brat_span.end
        ent.text == brat_span.text
        ent.label_ == brat_span.entity_type


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
def test_doc_to_ann(blank_language, text, brat_spans_args, expected_lines):
    """Test the `doc_to_ann` function.

    1.  Create a doc from blank_language and text
    2.  Create BratSpan from brat_spans_args
    3.  Set doc entities from BratSpans list.
    4.  Verify that the entities of the object ``Doc`` are well serialized in
    the format ``ann`` in a file. To do this, we observe that the data written in the file are those we expect.
    """
    doc = blank_language(text)
    brat_spans = [
        BratSpan(*brat_span_args) for brat_span_args in brat_spans_args
    ]
    doc = set_ents_from_brat_spans(doc, brat_spans)

    # Verify that the file annotations are the expected one.
    m = mock_open()
    with patch("pathlib.Path.open", m):
        doc_to_ann(doc, file="test_file.ann")

    m.assert_called_once_with(mode="w", encoding="utf-8")

    # Recuperate the calls made to the "write" method of the mock object.
    handle = m()
    writelines_call = handle.writelines.call_args

    # Verify that the written lines are the expected ones.
    assert writelines_call == call(expected_lines)
