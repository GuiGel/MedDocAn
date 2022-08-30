from __future__ import annotations

from dataclasses import dataclass
from io import TextIOWrapper
from typing import List, NamedTuple, Sized, Tuple
from zipfile import ZipExtFile

from spacy.tokens import Span

from meddocan.data import BratFilesPair


@dataclass
class BratSpan:
    """Container for a line of a ``.ann`` file in the BRAT format.

    .. note::
        The ``ID``, ``TEXT``, and ``COMMENT`` fields are not used for any of
        the evaluation metrics. The number in the ``ID`` field and the
        ``COMMENT`` field are arbitrary, and the evaluation of the ``TEXT``
        field is implicit in the offset evaluation, as the text is the same for
        the GS and the systems.

    Example:

    >>> BratSpan(id=None, entity_type="LOC", start=8, end=14, text="Bilbao")
    BratSpan(id=None, entity_type='LOC', start=8, end=14, text='Bilbao')

    Args:
        id (str): ID of the entity.
        entity_type (str): Entity type.
        start (int): Start offset.
        end (int): End offset.
        text (str): Text snippet between the previous offsets.
    """

    id: str
    entity_type: str
    start: int
    end: int
    text: str

    @property
    def entity(self) -> Tuple[int, int, str]:
        return (self.start, self.end, self.entity_type)

    @classmethod
    def from_bytes(cls, byte_line: bytes) -> BratSpan:
        """Read a line of a brat file and return a :class:`BratSpan` object.

        Example:

        >>> line = "T1\\tCALLE 2365 2391\\tc/ del Abedul 5-7, 2º dcha\\n"
        >>> byte_line = line.encode("utf-8")
        >>> BratSpan.from_bytes(byte_line)
        BratSpan(id='T1', entity_type='CALLE', start=2365, end=2391, \
text='c/ del Abedul 5-7, 2º dcha')
        """
        line = byte_line.decode("utf-8").strip()
        id, phi_start_end, text = line.split("\t")
        entity_type, start, end = phi_start_end.split(" ")
        return BratSpan(
            id=id,
            entity_type=entity_type,
            start=int(start),
            end=int(end),
            text=text,
        )

    @classmethod
    def from_txt(cls, line: str) -> BratSpan:
        """Read a line of a brat file and return a :class:`BratSpan` object.

        Example:

        >>> line = "T1\\tCALLE 2365 2391\\tc/ del Abedul 5-7, 2º dcha\\n"
        >>> BratSpan.from_txt(line)
        BratSpan(id='T1', entity_type='CALLE', start=2365, end=2391, \
text='c/ del Abedul 5-7, 2º dcha')
        """
        line = line.strip()
        id, phi_start_end, text = line.split("\t")
        entity_type, start, end = phi_start_end.split(" ")
        return BratSpan(
            id=id,
            entity_type=entity_type,
            start=int(start),
            end=int(end),
            text=text,
        )

    @classmethod
    def from_spacy_span(cls, entity: Span, i: int) -> BratSpan:
        """Create a BratSpan from a spacy Span.

        Example:

        >>> from spacy import blank
        >>> es = blank("es")
        >>> doc = es("Nombre: Ernesto.")
        >>> doc.set_ents([doc.char_span(8, 15, "PERS")])
        >>> entity = doc.ents[0]
        >>> brat_span = BratSpan.from_spacy_span(entity, 0)
        >>> print(brat_span)
        BratSpan(id='T_0', entity_type='PERS', start=8, end=15, text='Ernesto')

        Args:
            entity (Span): spaCy Span.
            i (int): The position of the ``spacy.tokens.Span`` among the other
                Spans in the document from which they are taken.

        Returns:
            BratSpan: A new :class:``BratSpan`` object.
        """
        return BratSpan(
            f"T_{i}",
            entity_type=entity.label_,
            start=entity.start_char,
            end=entity.end_char,
            text=entity.text,
        )

    @property
    def to_ann_line(self) -> str:
        r"""Represent the :class:``BratSpan`` object as a line of a ``.ann``
        file. The line does contains the ``\\n`` at the end.

        Example:

        >>> brat_span = BratSpan(
        ...     id='T1',
        ...     entity_type='CALLE',
        ...     start=2365,
        ...     end=2391,
        ...     text='c/ del Abedul 5-7, 2º dcha',
        ... )
        >>> brat_span.to_ann_line
        'T1\tCALLE 2365 2391\tc/ del Abedul 5-7, 2º dcha\n'

        Returns:
            str: A line of a ``.ann`` file.
        """
        return (
            f"{self.id}\t{self.entity_type} {self.start} {self.end}\t"
            f"{self.text}\n"
        )


@dataclass
class BratAnnotations(Sized):
    """Container for a document annotated in the brat format.

    :class:`BratAnnotations` object are construct by the classmethod
    ``from_brat_files``.

    Args:
        lines (List[str]): Document lines.
        brat_spans (List[BratSpan]): Document annotations.
    """

    # lines: List[str]
    text: str
    brat_spans: List[BratSpan]

    def __post_init__(self) -> None:
        """Verify that the str selected by the ``BratSpan.start`` and
        ``BratSpan.end`` attributes of each ``brat_spans`` element in the
        ``text`` attribute identical to the ``BratSpan.text`` attribute.
        """
        for brat_span in self.brat_spans:
            assert (
                self.text[brat_span.start : brat_span.end] == brat_span.text
            ), f"{self.text[brat_span.start : brat_span.end]} != {brat_span.text}"

    def __len__(self) -> int:
        return len(self.text.split("\n")) - 1

    @classmethod
    def from_brat_files(
        cls,
        brat_files_pair: BratFilesPair,
    ) -> BratAnnotations:
        with brat_files_pair.ann.open() as archive:
            if isinstance(archive, ZipExtFile):
                brat_spans = list(map(BratSpan.from_bytes, archive))
            elif isinstance(archive, TextIOWrapper):
                brat_spans = list(map(BratSpan.from_txt, archive))
            else:
                raise TypeError(
                    f"archive {archive} must be an instance of"
                    "BytesIO or TextIOWrapper"
                )

        # TODO Talks about the BOM (Byte Ordering Mark). Cf: Fluent Python.
        # Some file contains the BOM and other not. If we remove the BOM
        # Manually or reading file with ``UTF-8-SIG`` encoding, with have
        # misalignment between the spans and the text entities. This suggest
        # that the annotations process has been saved on file with BOM.
        # A solution is to remove the BOM manually when writing file or remove
        # 1 to all the offsets in the corresponding brat_span.

        text = brat_files_pair.txt.read_text(encoding="utf-8")

        # import codecs
        # if text[0].encode("utf-8") == codecs.BOM_UTF8  # b'\xef\xbb\xbf':
        #     print(f"UTF-8-SIG {text[0:2].encode('utf-8')}")

        return BratAnnotations(text=text, brat_spans=brat_spans)

    @property
    def to_ann_lines(self) -> List[str]:
        """List of the :class:``BratAnnotations`` brat_spans ready to be
        written to a file as lines.

        Returns:
            List[str]: list of a ``file.ann`` lines.
        """
        return [brat_span.to_ann_line for brat_span in self.brat_spans]


class ExpandedEntity(NamedTuple):
    original: str
    expanded: str
