r"""The tokenizer module provide a tokenizer based on the `spaCy`_ library.
The tokenizer must allow for an alignment between the offsets of each entity
and the tokens such that an entity begins with the first character of a token
and ends with the last character of a token.

.. _spaCy: https://spacy.io/
"""
from typing import NewType

import spacy
from spacy.symbols import ORTH
from spacy.tokenizer import Tokenizer

MeddocanTokenizer = NewType("MeddocanTokenizer", Tokenizer)


def meddocan_tokenizer(nlp: spacy.language.Language) -> MeddocanTokenizer:
    """Update `spaCy <https://spacy.io/>`_ tokenizer in order to align the
    entities offset with the token offsets.

    Modifying spaCy tokenizer existing rules set. For more information look
    `here <https://spacy.io/usage/linguistic-features#native-tokenizer-additions>`_

    Args:
        nlp (spacy.language.Language): A spaCy language.
        inplace (bool, optional): Update tokenizer inplace. Defaults to True.

    Returns:
        Optional[spacy.language.Language]: If inplace is ``False``, return the
        ``spaCy.language.Language`` with the new tokenizer.
    """

    # Add special case rule
    # https://spacy.io/usage/linguistic-features#special-cases

    # The following special case, "Der..", "der..", "Izq..", "izq.." can also
    # be solved by adding a rule to split the "." at the end of each token in
    # the ``meddocan.language.splitter.MissalignedSplitter`` object. But it
    # will separate all the ending point in each token.

    special_cases = [
        # blank tokenization is 's', '/', 'n.'
        ("s/n.", [{ORTH: "s/n"}, {ORTH: "."}]),
    ]
    for special_case in special_cases:
        nlp.tokenizer.add_special_case(*special_case)

    # Remove special case that cause ``Flair`` to be unable to read correctly
    # the data from a line.
    # "UU. EE." is a special case that create a unique token "UU. EE." with a
    # space in it. When this token is written as a line "UU. EE. S-LOC" for
    # example, spacy is not able to read the tag as it make a confusion
    # with the "EE." part of the token.

    nlp.tokenizer.rules.pop("EE. UU.")
    nlp.tokenizer.rules.pop("Ee. Uu.")

    # Add prefix to spaCy default prefix
    prefixes = list(nlp.Defaults.prefixes)
    if prefixes is not None:
        prefixes += [r"\+"]
        prefix_regex = spacy.util.compile_prefix_regex(prefixes)
        nlp.tokenizer.prefix_search = prefix_regex.search

    # Add infixes to spaCy default infixes
    infixes = list(nlp.Defaults.infixes)
    if infixes is not None:
        infixes += [r"""[-:\.\)\+/]"""]
        infix_regex = spacy.util.compile_infix_regex(infixes)
        nlp.tokenizer.infix_finditer = infix_regex.finditer

    # Add suffixes "\." to spaCy default suffixes. This is finally done by the
    # MissalignedSplitter component.

    # To be able to tokenize correctly the span of the form:
    #               ORIGINAL | EXPANDED
    #      guacorch@yahoo.es | E-mail:guacorch@yahoo.es
    # hleonbrito@hotmail.com | E-mail.hleonbrito@hotmail.com
    # that are not splitted correctly because they are recognize as url.

    nlp.tokenizer.url_match = None

    # Idem for the intend of token match.
    # url_regex = re.compile(
    #     r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
    # )
    # nlp.tokenizer.token_match = url_regex.match

    return MeddocanTokenizer(nlp.tokenizer)
