from typing import Union, Iterable, Dict, Any
from pathlib import Path

from thinc.api import Config

# set library-specific custom warning handling before doing anything else
from spacy.errors import setup_default_warnings

setup_default_warnings()  # noqa: E402

from spacy.language import Language
from spacy.vocab import Vocab
from spacy import util


def load(
    name: Union[str, Path],
    *,
    vocab: Union[Vocab, bool] = True,
    disable: Iterable[str] = util.SimpleFrozenList(),
    exclude: Iterable[str] = util.SimpleFrozenList(),
    config: Union[Dict[str, Any], Config] = util.SimpleFrozenDict(),
) -> Language:
    ...

def blank(
    name: str,
    *,
    vocab: Union[Vocab, bool] = True,
    config: Union[Dict[str, Any], Config] = util.SimpleFrozenDict(),
    meta: Dict[str, Any] = util.SimpleFrozenDict(),
) -> Language:
    ...