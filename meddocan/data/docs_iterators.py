"""Module that contains the containers for a ``spacy.tokens.Doc`` produced
by the :`meddocan_pipeline` with entities tagged by a trained model or tagger
from an annotation file at the brat format.
"""
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, List, NewType, Optional, Tuple, Union

from spacy.tokens import Doc

from meddocan.data import ArchiveFolder, BratFilesPair, meddocan_zip
from meddocan.data.containers import BratAnnotations, BratSpan, ExpandedEntity
from meddocan.data.utils import set_ents_from_brat_spans
from meddocan.language.pipeline import MeddocanLanguage, meddocan_pipeline


@dataclass
class DocWithBratPair:
    brat_files_pair: BratFilesPair
    doc: Doc


# GoldStandard document container.
GsDoc = NewType("GsDoc", DocWithBratPair)

# Sys document container
SysDoc = NewType("SysDoc", DocWithBratPair)


def get_expanded_entities(
    doc: Doc, brat_spans: List[BratSpan]
) -> List[ExpandedEntity]:
    """Get the missaligned entities that have been expanded to force the
    alignment.
    """
    ents = doc.ents

    assert len(ents) == len(brat_spans), (
        f"entities numbers {len(ents)} != " f"spans numbers {len(brat_spans)}."
    )

    # The ``spacy.tokens.Doc.ents`` attributes are sorted by they order
    # of apparition in the text. To compare with the brat_spans we must
    # to the same i.e sort the ``brat_spans`` list in place.

    brat_spans.sort(key=lambda brat_span: brat_span.start)

    expanded_entities: List[ExpandedEntity] = []

    for ent, brat_span in zip(ents, brat_spans):

        # Verify that the doc entity label is the same as the brat label.

        if ent.label_ != brat_span.entity_type:
            print(f"{ent.label_=} != {brat_span.entity_type=}")

        # Verify that the doc entity text is the same as the brat text.

        if ent.text != brat_span.text:
            expanded_entity = ExpandedEntity(brat_span.text, ent.text)
            expanded_entities.append(expanded_entity)

    return expanded_entities


@dataclass
class GsDocs:
    """Iterable of gold standard documents :class:``GsDoc``.

    Example:

    >>> gs_docs = GsDocs(ArchiveFolder.train)
    >>> gs_doc = next(iter(gs_docs))
    >>> gs_doc
    DocWithBratPair(brat_files_pair=BratFilesPair(ann=..., txt=...), \
doc=Datos del ...)

    Verify that the produce doc is a doc made by the meddocan pipeline

    >>> gs_doc.doc._.is_meddocan_doc
    True

    Args:
        archive_name (ArchiveFolder):
        model (str, Optional): Path to a serialize ``flair.models.Sequence\
            Tagger`` model.
        nlp (MeddocanLanguage, Optional): A meddocan pipeline.
            The meddocan pipeline work with the given ``model`` only if \
            ``nlp`` is None.
    """

    archive_name: ArchiveFolder
    model: Optional[str] = None
    batch_size: int = 128
    nlp: MeddocanLanguage = field(
        default=meddocan_pipeline(), init=True, repr=False
    )

    def __post_init__(self) -> None:
        if self.model is not None:
            self.nlp = meddocan_pipeline(model_loc=self.model)
        elif self.model is not None:
            warnings.warn(
                "Don't used the provided model because "
                f"{self.__class__.__qualname__} as received an argument for "
                f"the nlp parameter."
            )

    @staticmethod
    def print_expanded_entities(
        expanded_entities: List[ExpandedEntity],
    ) -> None:
        print(f"{'ORIGINAL ENTITY':>40} | {'EXPANDED ENTITY'}")
        print(f"{'---------------':>40}{'---'}{'---------------'}")
        for expanded_entity in expanded_entities:
            print(
                f"{expanded_entity.original:>40}"
                f" | {expanded_entity.expanded}"
            )
        print(f"{'---------------':>40}{'---'}{'---------------'}")
        msg = f"There is {len(expanded_entities)}"
        print(f"{msg:>40} expanded entities")

    def __gen_tuple_for_pipe(
        self,
    ) -> Iterator[Tuple[str, Tuple[List[BratSpan], BratFilesPair]]]:
        brat_files_pairs = meddocan_zip.brat_files(self.archive_name)
        for brat_files_pair in brat_files_pairs:
            brat_annotations = BratAnnotations.from_brat_files(brat_files_pair)
            context = brat_annotations.brat_spans, brat_files_pair
            yield (brat_annotations.text, context)

    def __iter__(self) -> Iterator[GsDoc]:
        expanded_entities: List[ExpandedEntity] = []

        for doc, (brat_spans, brat_files_pair) in self.nlp.pipe(
            self.__gen_tuple_for_pipe(),
            as_tuples=True,
            batch_size=self.batch_size,
            # The tuple (brat_spans, brat_files_pair) is not pickable...
            # The speed is not important here so let n_process to 1.
            n_process=1,
        ):
            doc = set_ents_from_brat_spans(doc, brat_spans=brat_spans)
            doc._.is_meddocan_doc = True
            expanded_entities_in_doc = get_expanded_entities(doc, brat_spans)
            expanded_entities.extend(expanded_entities_in_doc)
            doc_with_brat_pair = DocWithBratPair(brat_files_pair, doc)
            yield GsDoc(doc_with_brat_pair)

        if expanded_entities:
            self.print_expanded_entities(expanded_entities)
            print("\n")

    def to_connl03(
        self, file: Union[str, Path], write_sentences: bool = True
    ) -> None:
        for i, gs_doc in enumerate(self):
            (mode := "a") if i else (mode := "w")
            gs_doc.doc._.to_connl03(
                file, mode=mode, write_sentences=write_sentences
            )

    def to_gold_standard(
        self, parent: Union[str, Path], force: bool = False
    ) -> None:
        # Write all the files need for the evaluation process. We unzip the
        # zip archive and copy them to the parent folder.
        parent = Path(parent)

        for brat_files_pair in meddocan_zip.brat_files(self.archive_name):
            for brat_file_path in brat_files_pair:
                content = brat_file_path.read_text()
                file: Path = parent / brat_file_path.at  # type: ignore[attr-defined]
                if not file.parent.exists():
                    file.parent.mkdir(parents=True, exist_ok=True)
                file.write_text(content)


@dataclass
class SysDocs:
    """Iterable of :class:`SysDoc` produced by a
    ``flair.models.SequenceTagger`` model and a :class:`BratFilesPair` coming
    from the given :class:`ArchiveFolder`.

    Example:

    >>> sys_docs = SysDocs(ArchiveFolder.train, model=None)
    >>> sys_docs
    SysDocs(archive_name=<ArchiveFolder.train: 'train'>, model=None, \
model_batch_size=8, batch_size=32)

    SysDocs is also an ``Iterable`` of :class:`SysDoc` object.
    >>> sys_doc = next(iter(sys_docs))
    >>> sys_doc
    DocWithBratPair(brat_files_pair=BratFilesPair(ann=..., txt=...), \
doc=Datos del ...)

    Verify that the produce doc is a doc made by the meddocan pipeline
    >>> sys_doc.doc._.is_meddocan_doc
    True

    Args:
        archive_name (ArchiveFolder): An attribute of the ArchiveFolder
            enumeration.
        model (str, optional): Path to a serialize ``flair.models.Sequence\
            Tagger`` model.
        nlp (MeddocanLanguage, Optional): A meddocan pipeline.
            The meddocan pipeline work with the given ``model`` only if \
            ``nlp`` is None.
    """

    archive_name: ArchiveFolder

    # --- Flair ----
    model: str
    model_batch_size: int = 8

    # --- spaCy pipeline ----
    batch_size: int = 32

    def __post_init__(self) -> None:
        self.nlp = meddocan_pipeline(model_loc=self.model)

    def __gen_brat_annotations_as_tuple(
        self,
    ) -> Iterator[Tuple[str, BratFilesPair]]:
        brat_files_pairs = meddocan_zip.brat_files(self.archive_name)
        for brat_files_pair in brat_files_pairs:
            brat_annotations = BratAnnotations.from_brat_files(brat_files_pair)
            yield (brat_annotations.text, brat_files_pair)

    def __iter__(self) -> Iterator[SysDoc]:
        for doc, brat_files_pair in self.nlp.pipe(
            self.__gen_brat_annotations_as_tuple(),
            as_tuples=True,
            batch_size=self.batch_size,
            n_process=1,
        ):
            doc._.is_meddocan_doc = True
            doc_with_brat_pair = DocWithBratPair(brat_files_pair, doc)
            yield SysDoc(doc_with_brat_pair)

    def to_ann(self, parent: Path) -> None:

        if isinstance(parent, str):
            parent = Path(parent)

        # Example
        # sys_doc.brat_files_pair.ann.at
        # == 'train/brat/S0004-06142005000500011-1.ann'
        # We have thus all the path required to make the evaluation
        # process.

        for sys_doc in self:
            file: Path = parent / sys_doc.brat_files_pair.ann.at  # type: ignore[attr-defined]
            sys_doc.doc._.to_ann(file)

    def to_txt(self, parent: Path) -> None:

        if isinstance(parent, str):
            parent = Path(parent)

        for sys_doc in self:
            file: Path = parent / sys_doc.brat_files_pair.txt.at  # type: ignore[attr-defined]
            text = sys_doc.brat_files_pair.txt.read_text()
            assert sys_doc.doc.text == text
            file.write_text(text)

    def to_evaluation_folder(self, parent: Union[str, Path]) -> None:
        # Write the ``.ann`` and ``.txt`` file for the evaluation process.
        if isinstance(parent, str):
            parent = Path(parent)

        if not parent.exists():
            parent.mkdir(parents=True, exist_ok=True)

        # Example
        # sys_doc.brat_files_pair.ann.at
        # == 'train/brat/S0004-06142005000500011-1.ann'
        # We have thus all the path required to make the evaluation
        # process.

        self.to_ann(parent)
        self.to_txt(parent)
