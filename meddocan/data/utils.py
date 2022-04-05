"""Module that implements utils functions.
"""
from enum import Enum
from pathlib import Path
from typing import List, Union

from spacy.tokens import Doc

from meddocan.data.containers import BratAnnotations, BratSpan


class AlignmentMode(Enum):
    STRICT = "strict"
    CONTRACT = "contract"
    EXPAND = "expand"


def set_ents_from_brat_spans(
    doc: Doc,
    brat_spans: List[BratSpan],
    alignment_mode: AlignmentMode = AlignmentMode.EXPAND,
) -> Doc:
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

    if isinstance(file, str):
        file = Path(file)

    if not file.parent.exists():
        file.parent.mkdir(parents=True, exist_ok=True)

    brat_spans = [
        BratSpan.from_spacy_span(entity, i)
        for i, entity in enumerate(doc.ents)
    ]
    brat_annotations = BratAnnotations(doc.text, brat_spans, sep="")
    with Path(file).open(mode="w", encoding="utf-8") as f:
        f.writelines(brat_annotations.to_ann_lines)
