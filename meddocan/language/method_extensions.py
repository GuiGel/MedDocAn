"""This module implements the methods that are attached to the Doc object
produce by the :func:`meddocan_pipeline`.
By assigning a function that becomes accessible as its own method of a |Doc|
object, we have more reliable information about the type that can be passed as
a parameter to the function.
Indeed, if the method only accepts |Doc| s produced by the :func:`meddocan_pipeline`, this
can only be made mandatory by attaching an extension of type ``_.meddocan_doc``
to the document.

The information needed to understand the adjunction of methods to the
|Doc| object that are accessed through the personalize attribute
can be found in the `custom components attributes`_ documentation.

The information needed to understand how to create custom components can be
found in the |spaCy| documentation `Creating custom pipeline components`_.

.. _custom components attributes: https://spacy.io/usage/processing-pipelines#custom-components-attributes
.. _Creating custom pipeline components: https://spacy.io/usage/processing-pipelines#custom-components
"""
import codecs
from pathlib import Path
from typing import Callable, List, Literal, Optional, Union

import spacy.tokens
from spacy.language import Language
from spacy.tokens import Doc, Token


class WriteMethods:
    """Add a set of methods to a ``spacy.tokens.Doc`` that write relevant
    information to a file."""

    name = "WriteMethods"

    def __init__(self, nlp: Language) -> None:
        self.nlp = nlp

        # The ``method`` argument accept a parameter of the type ``DocMethod``
        # defined by spaCy as a Protocol with a __call__ method that take a doc
        # as first parameter.
        # As we want to attach the method and hide them in some
        # way there is not evident solution to this complains so we silent it.

        Doc.set_extension(
            "to_connl03", method=self.__doc_to_connl03, force=True  # type: ignore[arg-type]
        )
        # error: error: Argument "method" to "set_extension" of "Doc" has
        # incompatible type "Callable[[Doc, Union[str, Path],
        # Union[Literal['w'], Literal['a']], bool], None]"; expected
        # "Optional[DocMethod]"

        Doc.set_extension(
            "to_ann", method=self.__doc_to_ann, force=True  # type: ignore[arg-type]
        )
        # Argument "method" to "set_extension" of "Doc" has incompatible type
        # "Callable[[Doc, Union[str, Path]], None]"; expected
        # "Optional[DocMethod]"

    @staticmethod
    def __doc_to_connl03(
        doc: Doc,
        file: Union[str, Path],
        mode: Literal["w", "a"] = "w",
        write_sentences: bool = True,
        document_separator_token: Optional[str] = None,
    ) -> None:
        # ----------- Write a Doc to the given file at the CoNLL03 format.
        if isinstance(file, str):
            file = Path(file)

        def get_line(token: Token) -> str:
            if (tag := token.ent_iob_) in ["", "O"]:
                tag = "O"
            else:
                tag = f"{tag}-{token.ent_type_}"

            text = token.text

            if text[0].encode("utf-8") == codecs.BOM_UTF8:

                # Remove the detected BOOM.
                # If not removed, the ﻿ sign sometimes appears in
                # documents in BIO format opened by vscode.

                text = text[1:]

            line = f"{text} {tag}\n"
            return line

        with file.open(mode=mode, encoding="utf-8", newline="\n") as f:
            lines: List[str] = []

            if document_separator_token is not None:
                document_separator_line = f"{document_separator_token} O\n\n"
                f.write(document_separator_line)

            for sent in doc.sents:
                for token in sent:
                    if token.is_space:
                        # We can do that because the previous pipeline split
                        # spaCy spaces of the forms "  \n  " in 3 tokens
                        # ["  ", "\n", " "].
                        continue
                    lines.append(get_line(token))
                if write_sentences:
                    line = "\n"
                else:
                    line = "\\n O\n"
                lines.append(line)

            if write_sentences:
                f.writelines(lines)
            else:
                joined_lines = "".join((*lines, "\n"))
                f.write(joined_lines)

    @staticmethod
    def __doc_to_ann(doc: Doc, file: Union[str, Path]) -> None:
        from meddocan.data.utils import doc_to_ann  # To avoid circular import

        doc_to_ann(doc, file)

    def __call__(self, doc: Doc) -> Doc:
        doc._.is_meddocan_doc = True
        return doc


@Language.factory(
    "write_methods",
)
def create_write_methods(
    nlp: Language,
    name: str,
) -> Callable[[spacy.tokens.Doc], spacy.tokens.Doc]:
    """A Component factory that return an instance of the
    :class:`WriteMethods` custom components.

    Example:

    >>> from spacy import blank
    >>> nlp = blank("es")
    >>> _ = nlp.add_pipe("write_methods")
    >>> doc = nlp("El paciente sufre una enfermedad grave.")
    >>> doc.has_extension("to_connl03")
    True
    >>> doc.has_extension("to_ann")
    True

    Args:
        nlp (Language): The current nlp object. Can be used to access the \
            shared vocab.
        name (str): The instance name of the component in the pipeline. This \
            lets us identify different instances of the same component.

    Returns:
        Callable[[spacy.tokens.Doc], spacy.tokens.Doc]: The pipeline \
            component function.
    """
    return WriteMethods(nlp)
