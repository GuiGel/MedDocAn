from turtle import right
from typing import List, Iterator, Tuple, Union
from unittest.mock import patch, Mock
import pytest

from flair.data import Sentence

from meddocan.data.corpus import MEDDOCAN
from tests.language.test_method_extensions import MockDoc, TokenElements

from meddocan.data.docs_iterators import (
    GsDocs,
    BratFilesPair,
    DocWithBratPair,
    GsDoc,
)


class MockBratDocs:
    def __init__(self, docs: Union[List[MockDoc], MockDoc]) -> None:
        if isinstance(docs, MockDoc):
            docs = [docs]
        self.docs = docs

    def __iter__(self):
        for a in self.docs:
            print("inside __iter__")
            yield a.get_spacy_doc()

    def __call__(self):
        print("inside call")
        return self.__iter__()


class TestMeddocan:
    """

    An example of how we can test this class:
    Cf: https://docs.python.org/3/library/unittest.mock.html#patch

        >>> from unittest.mock import patch
        >>> class Class:
        ...     def __iter__(self):
        ...         for a in [1, 2, 3]:
        ...             yield a
        ...
        >>> def function():
        ...     return sum(a for a in Class())
        ...
        >>> with patch("__main__.Class") as MockClass:
        ...     instance = MockClass.return_value
        ...     instance.__iter__.return_value = [1, 2, 3]
        ...     assert Class() is instance
        ...     assert [a for a in Class()] == [1, 2, 3]
        ...     assert function() == 6
    """

    @pytest.mark.parametrize(
        "docs, expected_contexts",
        (
            (
                [
                    MockDoc(
                        [
                            TokenElements("Vivo", True, True),
                            TokenElements("en", True, False),
                            TokenElements("Aix", False, False),
                            TokenElements("!", True, False),
                            TokenElements("Soy", True, True),
                            TokenElements("Eric", True, False, "B-PERS"),
                            TokenElements("Laffont", False, False, "I-PERS"),
                            TokenElements(".", True, False),
                        ],
                    ),
                    MockDoc(
                        [
                            TokenElements("Vivo", True, True),
                            TokenElements("en", True, False),
                            TokenElements("Bilbao", False, False),
                            TokenElements("!", True, False),
                            TokenElements("Soy", True, True),
                            TokenElements("Zaira", True, False, "B-PERS"),
                            TokenElements(
                                "Aurrekoetxea", False, False, "I-PERS"
                            ),
                            TokenElements(".", True, False),
                        ],
                    ),
                ],
                [
                    ([], ["Soy", "Eric", "Laffont", "."]),
                    (["Vivo", "en", "Aix", "!"], []),
                    (
                        [
                            "Vivo",
                            "en",
                            "Aix",
                            "!",
                            "Soy",
                            "Eric",
                            "Laffont",
                            ".",
                        ],
                        [
                            "Vivo",
                            "en",
                            "Bilbao",
                            "!",
                            "Soy",
                            "Zaira",
                            "Aurrekoetxea",
                            ".",
                        ],
                    ),
                    ([], ["Soy", "Zaira", "Aurrekoetxea", "."]),
                    (["Vivo", "en", "Bilbao", "!"], []),
                    (
                        [
                            "Vivo",
                            "en",
                            "Bilbao",
                            "!",
                            "Soy",
                            "Zaira",
                            "Aurrekoetxea",
                            ".",
                        ],
                        [],
                    ),
                ],
            ),
        ),
    )
    def test_init(
        self,
        docs: Union[List[MockDoc], MockDoc],
        expected_contexts: List[Tuple[List[str], List[str]]],
    ) -> None:
        with patch.object(
            GsDocs, "__iter__", new_callable=lambda: MockBratDocs(docs)
        ):
            # Cf: https://docs.python.org/3/library/unittest.mock.html#where-to-patch
            # nonlocal expected
            meddocan = MEDDOCAN(
                sentences=True, document_separator_token="-DOCSTART-"
            )
            sentence: Sentence
            for sentence, (left_context, right_context) in zip(
                meddocan.train, expected_contexts
            ):
                assert sentence.left_context(64) == left_context
                assert sentence.right_context(64) == right_context


if __name__ == "__main__":
    TestMeddocan().test_init()

    # from flair.datasets import CONLL_03, CONLL_03_SPANISH
    # from flair.data import Sentence
    # from meddocan import cache_root
    #
    # corpus = CONLL_03(cache_root / "datasets")
    # corpus = CONLL_03_SPANISH()
    # sentence: Sentence
    # for i, sentence in enumerate(corpus.train):
    #     if i < 10:
    #         print("\t", sentence, " ".join(sentence.left_context(64)), " ".join(sentence.right_context(64)))
    #     else:
    #         break
