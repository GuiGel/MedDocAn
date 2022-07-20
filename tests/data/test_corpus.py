"""Module where we test that the MEDDOCAN corpus implementation is correct.
"""
import logging
from typing import List, Tuple, Union
from unittest.mock import patch

import pytest
from flair.data import Label, Sentence, Span

from meddocan.data.corpus import MEDDOCAN
from meddocan.data.docs_iterators import GsDocs
from tests.language.test_method_extensions import MockDoc, TokenElements

logger = logging.getLogger("flair")


class MockBratDocs:
    def __init__(self, docs: Union[List[MockDoc], MockDoc]) -> None:
        if isinstance(docs, MockDoc):
            docs = [docs]
        self.docs = docs

    def __iter__(self):
        for a in self.docs:
            yield a.get_spacy_doc()

    def __call__(self):
        return self.__iter__()


class TestMeddocan:
    """In order to test if the corpus is implemented correctly,
    we look at the ``flair.data.Sentence.left_context`` and
    ``flair.data.Sentence.right_context`` output.

    An example of how we can test this class:
    Cf: https://docs.python.org/3/library/unittest.mock.html#patch
    """

    @pytest.mark.parametrize(
        "docs",
        (
            [
                MockDoc(
                    [
                        TokenElements("Vivo", True, True),
                        TokenElements("en", True, False),
                        TokenElements("Aix", False, False, "B-LOC"),
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
                        TokenElements("Bilbao", False, False, "B-LOC"),
                        TokenElements("!", True, False),
                        TokenElements("Soy", True, True),
                        TokenElements("Zaira", True, False, "B-PERS"),
                        TokenElements("Aurrekoetxea", False, False, "I-PERS"),
                        TokenElements(".", True, False),
                    ],
                ),
            ],
        ),
    )
    @pytest.mark.parametrize(
        "document_separator_token, expected_contexts",
        (
            (
                "-DOCSTART-",
                [
                    (
                        [],
                        "-DOCSTART-",
                        ["Vivo", "en", "Aix", "!", "Soy", "Eric"],
                        None,
                    ),
                    (
                        [],
                        "Vivo en Aix !",
                        ["Soy", "Eric", "Laffont", "."],
                        (2, 3, "LOC"),
                    ),
                    (
                        ["Vivo", "en", "Aix", "!"],
                        "Soy Eric Laffont .",
                        [],
                        (1, 3, "PERS"),
                    ),
                    (
                        ["Aix", "!", "Soy", "Eric", "Laffont", "."],
                        "-DOCSTART-",
                        ["Vivo", "en", "Bilbao", "!", "Soy", "Zaira"],
                        None,
                    ),
                    (
                        [],
                        "Vivo en Bilbao !",
                        ["Soy", "Zaira", "Aurrekoetxea", "."],
                        (2, 3, "LOC"),
                    ),
                    (
                        ["Vivo", "en", "Bilbao", "!"],
                        "Soy Zaira Aurrekoetxea .",
                        [],
                        (1, 3, "PERS"),
                    ),
                ],
            ),
            (
                None,
                [
                    (
                        [],
                        "Vivo en Aix !",
                        ["Soy", "Eric", "Laffont", ".", "Vivo", "en"],
                        (2, 3, "LOC"),
                    ),
                    (
                        ["Vivo", "en", "Aix", "!"],
                        "Soy Eric Laffont .",
                        ["Vivo", "en", "Bilbao", "!", "Soy", "Zaira"],
                        (1, 3, "PERS"),
                    ),
                    (
                        ["Aix", "!", "Soy", "Eric", "Laffont", "."],
                        "Vivo en Bilbao !",
                        ["Soy", "Zaira", "Aurrekoetxea", "."],
                        (2, 3, "LOC"),
                    ),
                    (
                        ["Laffont", ".", "Vivo", "en", "Bilbao", "!"],
                        "Soy Zaira Aurrekoetxea .",
                        [],
                        (1, 3, "PERS"),
                    ),
                ],
            ),
        ),
    )
    def test_init(
        self,
        docs: Union[List[MockDoc], MockDoc],
        document_separator_token: str,
        expected_contexts: List[
            Tuple[List[str], str, List[str], Tuple[int, int, str]]
        ],
    ) -> None:
        """Test that the MEDDOCAN corpus is implemented correctly.
        To do that we mock the
        :meth:`~meddocan.data.docs_iterators.GsDocs.__iter__` method of the
        class :class:`~meddocan.data.docs_iterators.GsDocs`. This mocked |Doc|
        is then internally copied to a temporary file by the custom
        :meth:`~meddocan.language.method_extensions.WriteMethods.__doc_to_connl03`
        of the Doc object. This is possible because the ``meddocan`` language
        set the |Doc| extensions `to_connl03` and thus allow to use it on the
        mocked documents.

        Args:
            docs (Union[List[MockDoc], MockDoc]): The documents retrieve by
                the mocked `__iter__` method.
            document_separator_token (str): Document separator argument for the
                :class:`~meddocan.data.corpus.MEDDOCAN` corpus.
            expected_contexts (
                List[ Tuple[List[str], str, List[str], Tuple[int, int, str]] ]
                ): The expected contexts. A tuple made of the following parts:
                (Left context, tokenized sentence as str, right context, label)

                THe chosen context size is of 6 tokens.

                For Example:

                >>> expected_contexts = (
                ...     ["Me", "llamo", "Bertrand", "."],
                ...     "Vivo en Aix .",
                ...     ["Es", "una", "ciudad", "bonita", "."],
                ...     (2, 3, "LOC"),
                ... )
        """
        with patch.object(
            GsDocs,
            "__iter__",
            new_callable=lambda: MockBratDocs(docs),
        ):
            # Cf: https://docs.python.org/3/library/unittest.mock.html#where-to-patch
            # nonlocal expected
            meddocan = MEDDOCAN(
                sentences=True,
                document_separator_token=document_separator_token,
            )
            sentence: Sentence
            for sentence, (
                left_context,
                expected_sentence,
                right_context,
                entity,
            ) in zip(meddocan.train, expected_contexts):

                # Construct Label from expected_contexts

                labels: List[Label] = []
                if entity is not None:
                    span = Span(sentence.tokens[entity[0] : entity[1]])
                    labels.append(Label(span, value=entity[2]))

                assert sentence.to_original_text() == expected_sentence
                assert sentence.left_context(6) == left_context
                assert sentence.right_context(6) == right_context
                assert sentence.get_labels() == labels


if __name__ == "__main__":
    # TestMeddocan().test_init()
    from flair.data import Sentence

    sentence = Sentence("Vivo en Aix!")
    print(
        sentence.to_original_text(),
        [token.start_position for token in sentence],
    )

    from flair.data import Sentence
    from flair.datasets import CONLL_03

    from meddocan import cache_root

    corpus = CONLL_03(cache_root / "datasets")
    # corpus = CONLL_03_SPANISH()
    sentence: Sentence
    for i, sentence in enumerate(corpus.train):
        if i < 100:
            print(sentence)
            print("\t", " ".join(sentence.left_context(4)))
            print("\t", " ".join(sentence.right_context(4)))
        else:
            break
