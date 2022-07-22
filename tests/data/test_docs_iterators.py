import logging

logging.basicConfig(level=logging.DEBUG)

import itertools as it
from typing import List

import pytest
from spacy.tokens import Doc

from meddocan.data import ArchiveFolder, meddocan_zip
from meddocan.data.containers import BratSpan, ExpandedEntity
from meddocan.data.docs_iterators import (
    DocWithBratPair,
    GsDoc,
    get_expanded_entities,
)
from tests.language.test_method_extensions import MockDoc, TokenElements

logger = logging.getLogger("meddocan")


class TestDocWithBratPair:
    def test(self):
        doc = (
            MockDoc(
                [
                    TokenElements("Vivo", True, True),
                    TokenElements("en", True, False),
                    TokenElements("Aix", True, False, "B-LOC"),
                    TokenElements("en", True, False, "I-LOC"),
                    TokenElements("Provence", False, False, "I-LOC"),
                    TokenElements("!", True, False),
                    TokenElements("Soy", True, True),
                    TokenElements("Eric", True, False, "B-PERS"),
                    TokenElements("Laffont", False, False, "I-PERS"),
                    TokenElements(".", True, False),
                ],
            ).get_spacy_doc(),
        )
        brat_file_pair = next(meddocan_zip.brat_files(ArchiveFolder.train))
        doc_with_brat_pair = DocWithBratPair(brat_file_pair, doc)
        assert str(doc_with_brat_pair).strip() == (
            "DocWithBratPair("
            "brat_files_pair=BratFilesPair("
            "ann=Path('/home/wave/.meddocan/datasets/meddocan/train-set.zip', "
            "'train/brat/S0004-06142005000500011-1.ann'), "
            "txt=Path('/home/wave/.meddocan/datasets/meddocan/train-set.zip', "
            "'train/brat/S0004-06142005000500011-1.txt')), "
            "doc=(Vivo en Aix en Provence! Soy Eric Laffont. ,))"
        ).strip()


@pytest.mark.parametrize(
    argnames="doc",
    argvalues=(
        [
            MockDoc(
                [
                    TokenElements("Vivo", True, True),
                    TokenElements("en", True, False),
                    TokenElements("Aix", True, False, "B-LOC"),
                    TokenElements("en", True, False, "I-LOC"),
                    TokenElements("Provence", False, False, "I-LOC"),
                    TokenElements("!", True, False),
                    TokenElements("Soy", True, True),
                    TokenElements("Eric", True, False, "B-PERS"),
                    TokenElements("Laffont", False, False, "I-PERS"),
                    TokenElements(".", True, False),
                ],
            ).get_spacy_doc(),
        ]
    ),
)
@pytest.mark.parametrize(
    argnames="brat_spans, expected",
    argvalues=(
        (
            [
                BratSpan(
                    id=None,
                    entity_type="LOC",
                    start=2,
                    end=5,
                    text="Aix en Provence",
                ),
                BratSpan(
                    id=None,
                    entity_type="PERS",
                    start=7,
                    end=9,
                    text="Eric Laffont",
                ),
            ],
            [],
        ),
        (
            [
                BratSpan(
                    id=None,
                    entity_type="LOC",
                    start=2,
                    end=5,
                    text="Aix en Provence",
                ),
                BratSpan(
                    id=None,
                    entity_type="PERS",
                    start=7,
                    end=8,
                    text="EricLaffont",
                ),
            ],
            [ExpandedEntity("EricLaffont", "Eric Laffont")],
        ),
    ),
)
def test_get_expanded_entities(
    doc: Doc,
    brat_spans: List[BratSpan],
    expected: List[ExpandedEntity],
) -> None:
    """Test ``meddocan.data.docs_iterators.get_expanded_entities``.

    Args:
        doc (Doc): A spacy.tokens.Doc.
        brat_spans (List[BratSpan]): The entities that the doc must contains.
        expected (List[ExpandedEntity]): The expected results.
    """
    obtained = get_expanded_entities(doc, brat_spans)
    for obtained_ee, expected_ee in it.zip_longest(obtained, expected):
        assert obtained_ee == expected_ee


@pytest.mark.parametrize(
    argnames="doc",
    argvalues=(
        [
            MockDoc(
                [
                    TokenElements("Vivo", True, True),
                    TokenElements("en", True, False),
                    TokenElements("Aix", True, False, "B-LOC"),
                    TokenElements("en", True, False, "I-LOC"),
                    TokenElements("Provence", False, False, "I-LOC"),
                    TokenElements("!", True, False),
                    TokenElements("Soy", True, True),
                    TokenElements("Eric", True, False),
                    TokenElements("Laffont", False, False),
                    TokenElements(".", True, False),
                ],
            ).get_spacy_doc(),
        ]
    ),
)
@pytest.mark.parametrize(
    argnames="brat_spans",
    argvalues=(
        [
            BratSpan(
                id=None,
                entity_type="LOC",
                start=2,
                end=5,
                text="Aix en Provence",
            ),
            BratSpan(
                id=None,
                entity_type="PERS",
                start=7,
                end=9,
                text="Eric Laffont",
            ),
        ],
    ),
)
def test_get_expanded_entities_bad_ents_num(
    doc: Doc,
    brat_spans: List[BratSpan],
) -> None:
    """Test that ``meddocan.data.docs_iterators.get_expanded_entities`` is
    raising an :class:``AssertionError`` when the numbers of entities detected
    are not the same a the expected ones.

    Args:
        doc (Doc): A spacy.tokens.Doc.
        brat_spans (List[BratSpan]): The entities that the doc must contains.
        expected (List[ExpandedEntity]): The expected results.
    """
    with pytest.raises(AssertionError):
        get_expanded_entities(doc, brat_spans)


class GsDoc:
    @pytest.mark.parametrize("text,expected_lines")
    def test_to_connl03(self, text, expected_lines):
        assert True


if __name__ == "__main__":
    a = (
            "DocWithBratPair("
            "brat_files_pair=BratFilesPair("
            "ann=Path('/home/wave/.meddocan/datasets/meddocan/train-set.zip', "
            "'train/brat/S0004-06142005000500011-1.ann'), "
            "txt=Path('/home/wave/.meddocan/datasets/meddocan/train-set.zip', "
            "'train/brat/S0004-06142005000500011-1.txt')), "
            "doc=(Vivo en Aix en Provence! Soy Eric Laffont. ,))"
        )
    print(a == a.strip())