from typing import Any, Optional, Union, Match, Callable, Dict, Iterator, Iterable, List, Tuple

from spacy.vocab import Vocab

from spacy import symbols
from spacy.tokens import Doc

RE_SEARCH = Callable[[str, Optional[int], Optional[int]], Union[Match[str], None]]
RE_FINDITER = Callable[[str, Optional[int], Optional[int]], Iterator[Match[str]]]
RE_MATCH = Callable[[str, Optional[int], Optional[int]], Union[Match[str], None]]

class Tokenizer:
    def __init__(
        self,
        vocab: Vocab,
        rules: Optional[Dict[str, Dict[str, Any]]] = None,
        prefix_search: Optional[RE_SEARCH]  = None,
        suffix_search: Optional[RE_SEARCH]  = None,
        infix_finditer: Optional[RE_FINDITER] = None,
        token_match: Optional[RE_MATCH] = None,
        url_match: Optional[RE_MATCH] = None,
    ) -> None: ...
    def add_special_case(self, string: str, *substrings: Iterable[Dict[int, str]]) -> Any: ...
    def __call__(str) -> Doc: ...
    @property
    def rules(self) -> Optional[Dict[str, List[Dict[int, str]]]]: ...
    @rules.setter
    def rules(self, value: Optional[Dict[str, List[Dict[int, str]]]]) -> None: ...
    @property
    def prefix_search(self) -> Optional[Callable[[str], Optional[Match]]]: ...
    @prefix_search.setter
    def prefix_search(self, value: Optional[Callable[[str], Optional[Match]]]) -> None: ...
    @property
    def infix_finditer(self) -> Optional[Callable[[str], Iterator[Match]]]: ...
    @infix_finditer.setter
    def infix_finditer(self, value: Optional[Callable[[str], Iterator[Match]]]) -> None: ...
    @property
    def url_match(self) -> Optional[Dict[str, List[Dict[int, str]]]]: ...
    @url_match.setter
    def url_match(self, value: Optional[Dict[str, List[Dict[int, str]]]]) -> None: ...
    def explain(self, text: str) -> List[Tuple[str, str]]: ...