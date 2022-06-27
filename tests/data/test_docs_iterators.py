import itertools as it
from typing import List, Tuple
from unittest.mock import MagicMock

import pytest
from spacy.tokens import Doc
from spacy.vocab import Vocab

from meddocan.data.docs_iterators import GsDoc


def create_docs(
    texts: List[str],
    sents_start: List[Tuple[int, ...]],
    spaces: List[Tuple[int, ...]],
) -> List[Doc]:
    splitted_texts = [text.split("|") for text in texts]
    vocab = Vocab(strings=list(it.chain(*splitted_texts)))
    docs = [Doc(vocab, words) for words in splitted_texts]
    for doc, sent_start, space in zip(docs, sents_start, spaces):
        for i, token in enumerate(doc):
            if i in sent_start:
                token.sent_start = True
            else:
                token.sent_start = False
            if i in space:
                token.is_space = True
    return docs


class GsDoc:
    @pytest.mark.parametrize("text,expected_lines")
    def test_to_connl03(self, text, expected_lines):
        assert True


if __name__ == "__main__":

    sentences_1 = "Datos|del|paciente|.|\n|Remitido|por|:|Dra|.|Estefanía|Romero|Selas|.|Email|:|eromeroselas@yahoo|.|es"
    sent_start_1 = [0, 5]
    is_space_1 = [4]
    sentences_2 = "Datos|del|paciente|.|\n|Remitido|por|:|Dr|:|Rodrigo|Martínez|Mansur|.|Email|:|drrmmmansur@yahoo|.|com"
    sent_start_2 = [0, 5]
    is_space_2 = [4]

    docs = create_docs(
        [sentences_1, sentences_2],
        [sent_start_1, sent_start_2],
        [is_space_1, is_space_2],
    )
    for doc in docs:
        for sent in doc.sents:
            print(f"{sent.text!r}")

        # TODO Test the function create_docs

        doc._.to_connl03 = MagicMock(return_value=1)
        print(doc._.to_connl03("test.tsv", mode="w", write_sentences=True))

    # Otherwise a more functional solution than a unitary one is to make a
    # patch on __iter__ so that we return documents created by
    # meddocan_pipeline
