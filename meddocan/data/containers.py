from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, List, Literal, Tuple

from spacy.tokens import Doc

from meddocan.data import ArchiveFolder, BratFilesPair, meddocan_zip
from meddocan.language.pipeline import MeddocanLanguage


@dataclass
class BratSpan:
    """Container for a line of a ``.ann`` file in the BRAT format.

    The line doesn't contains the ``\\n`` at the end.

    Args:
        id (str):
        entity_type (str):
        start (int):
        end (int):
        text (str):
    """

    id: str
    entity_type: str
    start: int
    end: int
    text: str
    entity: Tuple[int, int, str] = field(init=False)

    def __post_init__(self):
        self.entity = (self.start, self.end, self.entity_type)

    @classmethod
    def from_bytes(cls, byte_line: bytes) -> BratSpan:
        """Read a line of a brat file and return a :py:class:`BratSpan` object.

        Example:

        >>> line = "T1\tCALLE 2365 2391\tc/ del Abedul 5-7, 2º dcha"
        >>> byte_line = line.encode("utf-8")
        >>> BratLine.from_bytes(byte_line)
        BratSpan(
            id='T1',
            entity_type='CALLE',
            start=2365,
            end=2391,
            text='c/ del Abedul 5-7, 2º dcha,
        )
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


@dataclass
class BratAnnotations:
    """Container for a document annotated in the brat format.

    TODO Document!

    Args:
        text (str): Document text.
        brat_spans (List[BratSpan]): Document annotations.
    """

    lines: List[str]
    brat_spans: List[BratSpan]
    sep: str
    text: str = field(init=False)
    entities: List[Tuple[int, int, str]] = field(init=False)

    def __post_init__(self):
        """Verify that the str selected by the ``BratSpan.start`` and
        ``BratSpan.end`` attributes of each ``brat_spans`` element in the
        ``text`` attribute identical to the ``BratSpan.text`` attribute."""

        self.text = self.sep.join(self.lines)

        for brat_span in self.brat_spans:
            assert self.text[brat_span.start : brat_span.end] == brat_span.text

        self.entities = [bs.entity for bs in self.brat_spans]

    def __len__(self):
        return len(self.lines)

    @classmethod
    def from_brat_files(
        cls, brat_files_pair: BratFilesPair, sep: str
    ) -> BratAnnotations:
        with brat_files_pair.ann.open() as archive:
            brat_spans = list(map(BratSpan.from_bytes, archive))

        with brat_files_pair.txt.open() as archive:
            brat_txt_lines = [line.decode("utf-8") for line in archive]

        return BratAnnotations(
            lines=brat_txt_lines, brat_spans=brat_spans, sep=sep
        )


@dataclass
class BratDoc:
    doc: Doc

    @classmethod
    def from_brat_annotations(
        cls, nlp: MeddocanLanguage, brat_annotations: BratAnnotations
    ) -> BratDoc:
        doc = nlp(brat_annotations.text)
        spans = brat_annotations.brat_spans
        doc.set_ents(
            [
                doc.char_span(
                    span.start,
                    span.end,
                    label=span.entity_type,
                    alignment_mode="expand",
                )
                for span in spans
            ]
        )

        return BratDoc(doc)

    def check_doc(self, brat_spans: List[BratSpan]) -> int:
        """No exception occurs if alignment_mode="expand".
        Let's check that the number of entities in the
        ``spacy.tokens.doc.Doc`` is the same as in the ``brat_spans``
        argument.
        """

        missalignment_count = 0

        assert len(self.doc.ents) == len(brat_spans), (
            f"entities numbers {len(self.doc.ents)} != "
            f"spans numbers {len(brat_spans)}."
        )

        # The ``spacy.tokens.Doc.ents`` attributes are sorted by they order
        # of apparition in the text. To compare with the brat_spans we must
        # to the same i.e sort the ``brat_spans`` list in place.

        brat_spans.sort(key=lambda brat_span: brat_span.start)

        for ent, brat_span in zip(self.doc.ents, brat_spans):

            # Verify that the doc entity label is the same as the brat label.

            if ent.label_ != brat_span.entity_type:
                print(f"{ent.label_=} != {brat_span.entity_type=}")

            # Verify that the doc entity text is the same as the brat text.

            if ent.text != brat_span.text:
                print(
                    f"The selected entity '{ent.text}' expand the original "
                    f"entity '{brat_span.text}'."
                )
                missalignment_count += 1

        return missalignment_count

    def write(
        self,
        file: Path,
        mode: Literal["a", "w"] = "w",
        sentences: bool = False,
    ) -> None:
        """Write a SpacyBratDoc in a file. Write a new line at the end of each
        document.

        Args:
            file (Path): The file where the data must be written.
            mode (Literal["a", "w"], optional): Append or write text to the
                given file. Defaults to "w".
            sentences (bool): If false, write the document as a whole with each
                new line represented as ``"\\n O"``. It True, separate each
                line of the document with each new line represented as
                ``"\\n"``. Defaults to False.
        """

        # This function is important because it writes the information that
        # will be used to train the models.

        # A solution to have one document per line or whole is to look at "\n"
        # lines in the SpacyBratDoc.

        with file.open(mode=mode, encoding="utf-8", newline="\n") as f:
            for token in self.doc:

                if (tag := token.ent_iob_) != "O":
                    assert tag in "BIO"
                    tag = f"{tag}-{token.ent_type_}"

                # To remove special case as token.text == "\n             "
                # In this case, the if statement add a new line made of space
                # that lead to more document creation.
                # text = f"{token.orth_!r}"
                # What spacy does when it receives a string of the form
                # "str1 str2 str2" which corresponds here to 2 sentences of
                # our datasets, is that it will tokenize it by grouping the
                # spaces in a single token "/n". In this way we end up with a
                # document which will be of the form "str1", "/n", "str2",
                # "/n".
                # The easiest way to do this is to remove the spaces by
                # replacing them with "/n" if we desire create sentences or
                # just remplace it by "\n O" if we want to make docs with
                # sentences in it.

                if "\n" in token.text:
                    if sentences:
                        f.write("\n")
                    """else:
                        f.write("\\n O\n")"""
                else:

                    # https://stackoverflow.com/questions/17912307/u-ufeff-in-python-string
                    # Sometimes at the beginning of a new document, the U+FEFF
                    # character is write in the file.

                    line = f"{token.text} {tag}\n"
                    line = line.replace("\ufeff", "")
                    f.write(line)
                    assert line != "\n"
            f.write("\n")


@dataclass
class BratDocs:
    archive_name: ArchiveFolder
    sep: str = ""

    def __iter__(self) -> Iterator[BratDoc]:
        from meddocan.language.pipeline import meddocan_pipeline

        nlp = meddocan_pipeline()

        missalignment_count = 0

        for brat_files_pair in meddocan_zip.brat_files(self.archive_name):
            brat_annotations = BratAnnotations.from_brat_files(
                brat_files_pair, sep=self.sep
            )
            brat_doc = BratDoc.from_brat_annotations(nlp, brat_annotations)
            missalignment_count += brat_doc.check_doc(
                brat_annotations.brat_spans
            )
            yield brat_doc

        print(f"{missalignment_count=}")

    def write(self, output: Path, sentences: bool = False) -> None:
        for i, brat_doc in enumerate(self):
            mode = "a"
            if i:
                mode = "w"
            brat_doc.write(output, mode, sentences=sentences)
