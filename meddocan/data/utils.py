"""Module that implements utility functions.
"""
from enum import Enum
from pathlib import Path
from typing import List, Union

from spacy.tokens import Doc

from meddocan.data.containers import BratAnnotations, BratSpan


class AlignmentMode(Enum):
    """Enumeration that enumerate the possible parameters for
    :func:`meddocan.data.utils.set_ents_from_brat_spans`.

    Define how character indices snap to token boundaries.

    Example:

    >>> AlignmentMode.STRICT == AlignmentMode("strict")
    True
    >>> AlignmentMode.STRICT.value
    'strict'
    """

    STRICT = "strict"
    """no snapping
    """
    CONTRACT = "contract"
    """span of all tokens completely within the character span
    """
    EXPAND = "expand"
    """span of all tokens at least partially covered by the character span
    """


def set_ents_from_brat_spans(
    doc: Doc,
    brat_spans: List[BratSpan],
    alignment_mode: AlignmentMode = AlignmentMode.EXPAND,
) -> Doc:
    """Add entities to a |Doc| object from a list of
    :class:`meddocan.data.containers.BratSpan`.

    Example:

    >>> from spacy import blank
    >>> nlp = blank("es")
    >>> doc = nlp("Vivo en Bilbao.")
    >>> brat_spans = [BratSpan(None, "LOC", 8, 14, "Bilbao")]
    >>> doc = set_ents_from_brat_spans(doc, brat_spans)
    >>> doc.ents
    (Bilbao,)

    Args:
        doc (Doc): A spaCy |Doc|.
        brat_spans (List[BratSpan]): The :class:`BratSPan`\s objects.
        alignment_mode (AlignmentMode, optional): Option for the alignment \
            between the Tokens of the Doc object and the BratSpan. \
            How character indices snap to token boundaries. Options: "strict" (no snapping), "contract" (span of all tokens completely within the character span), "expand" (span of all tokens at least partially covered by the character span). Defaults to "strict".Defaults to AlignmentMode.EXPAND.

    Returns:
        Doc: An object |Doc| to which entities have been added.
    """
    doc.set_ents(
        [
            doc.char_span(
                span.start,
                span.end,
                label=span.entity_type,
                alignment_mode=alignment_mode.value,
            )
            for span in brat_spans
        ]
    )
    return doc


def doc_to_ann(doc: Doc, file: Union[str, Path]) -> None:
    """Writes the entities of the |Doc| object to a file in ``ann`` format.

    Example:

    >>> import os
    >>> from tempfile import NamedTemporaryFile
    >>> from spacy import blank
    >>> nlp = blank("es")
    >>> doc = nlp("Vivo en Bilbao.")
    >>> f = NamedTemporaryFile(delete=False)
    >>> doc_to_ann(doc, f.name)
    >>> f.close()
    >>> os.unlink(f.name)
    >>> os.path.exists(f.name)
    False

    Args:
        doc (Doc): The |Doc| from which the entities are serialized.
        file (Union[str, Path]): The file to write in.
    """
    if isinstance(file, str):
        file = Path(file)

    if not file.parent.exists():
        file.parent.mkdir(parents=True, exist_ok=True)

    brat_spans = [
        BratSpan.from_spacy_span(entity, i)
        for i, entity in enumerate(doc.ents)
    ]
    brat_annotations = BratAnnotations(doc.text, brat_spans)
    with Path(file).open(mode="w", encoding="utf-8") as f:
        f.writelines(brat_annotations.to_ann_lines)
