from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, List, Literal, NamedTuple, Tuple

from spacy.tokens import Doc, Span

from meddocan.data import ArchiveFolder, BratFilesPair, meddocan_zip
from meddocan.language.pipeline import MeddocanLanguage


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
    BratSpan(id=None, entity_type='LOC', start=8, end=14, text='Bilbao', \
entity=(8, 14, 'LOC'))

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
    entity: Tuple[int, int, str] = field(init=False)

    def __post_init__(self):
        self.entity = (self.start, self.end, self.entity_type)

    @classmethod
    def from_bytes(cls, byte_line: bytes) -> BratSpan:
        """Read a line of a brat file and return a :class:`BratSpan` object.

        The line doesn't contains the ``\\n`` at the end.

        Example:

        >>> line = "T1\\tCALLE 2365 2391\\tc/ del Abedul 5-7, 2º dcha"
        >>> byte_line = line.encode("utf-8")
        >>> BratSpan.from_bytes(byte_line)
        BratSpan(id='T1', entity_type='CALLE', start=2365, end=2391, \
text='c/ del Abedul 5-7, 2º dcha', entity=(2365, 2391, 'CALLE'))
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
        BratSpan(id='T_0', entity_type='PERS', start=8, end=15, text='Ernesto', entity=(8, 15, 'PERS'))

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

    def to_ann_line(self) -> str:
        r"""Represent the :class:``BratSpan`` object as a line of a ``.ann``
        file.

        Example:

        >>> BratSpan(
        ...     id='T1',
        ...     entity_type='CALLE',
        ...     start=2365,
        ...     end=2391,
        ...     text='c/ del Abedul 5-7, 2º dcha',
        ... ).to_ann_line()
        ...
        'T1\tCALLE 2365 2391\tc/ del Abedul 5-7, 2º dcha'

        Returns:
            str: A line of a ``.ann`` file.
        """
        return f"{self.id}\t{self.entity_type} {self.start} {self.end}\t{self.text}"


@dataclass
class BratAnnotations:
    """Container for a document annotated in the brat format.

    A :class:`BratAnnotations` object are construct by the classmethod
    :method:`from_brat_files`

    Args:
        text (str): Document text.
        brat_spans (List[BratSpan]): Document annotations.
    """

    lines: List[str]
    brat_spans: List[BratSpan]
    sep: str = ""
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


class ExpandedEntity(NamedTuple):
    original: str
    expanded: str


@dataclass
class BratDoc:

    # TODO Can have a NewType to specified that this doc as the entities
    # attached?

    doc: Doc

    @classmethod
    def from_brat_annotations(
        cls, nlp: MeddocanLanguage, brat_annotations: BratAnnotations
    ) -> BratDoc:
        """Instantiate a ``BratDoc`` directly from a ``BratAnnotations`` and
        the ``MeddocanLanguage``. The created ``spacy.tokens.Doc`` has the
        meddocan annotations set as entities.

        Args:
            nlp (MeddocanLanguage): A specific ``spacy.language.Language``
                for the meddocan corpus.
            brat_annotations (BratAnnotations): The ``dataclasses.dataclass``
                that contains the ``BratAnnotations`` attributes.

        Returns:
            BratDoc: Container for a ``spacy.tokens.Doc`` with entities that
                correspond to the offsets of the meddocan entities.
        """
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

    def check_doc(self, brat_spans: List[BratSpan]) -> List[ExpandedEntity]:
        """No exception occurs if alignment_mode="expand".
        Let's check that the number of entities in the
        ``spacy.tokens.doc.Doc`` is the same as in the ``brat_spans``
        argument.
        """

        assert len(self.doc.ents) == len(brat_spans), (
            f"entities numbers {len(self.doc.ents)} != "
            f"spans numbers {len(brat_spans)}."
        )

        # The ``spacy.tokens.Doc.ents`` attributes are sorted by they order
        # of apparition in the text. To compare with the brat_spans we must
        # to the same i.e sort the ``brat_spans`` list in place.

        brat_spans.sort(key=lambda brat_span: brat_span.start)

        expanded_entities: List[ExpandedEntity] = []

        for ent, brat_span in zip(self.doc.ents, brat_spans):

            # Verify that the doc entity label is the same as the brat label.

            if ent.label_ != brat_span.entity_type:
                print(f"{ent.label_=} != {brat_span.entity_type=}")

            # Verify that the doc entity text is the same as the brat text.

            if ent.text != brat_span.text:
                expanded_entity = ExpandedEntity(brat_span.text, ent.text)
                expanded_entities.append(expanded_entity)

        return expanded_entities

    def write(
        self,
        file: Path,
        mode: Literal["a", "w"] = "w",
        sentences: bool = False,
    ) -> None:
        """Writes the BratDoc.doc attribute to the file provided in CONLL03 \
        format, encoded with the ``BIO`` scheme.

        Args:
            file (Path): The file where the data must be written.
            mode (Literal["a", "w"], optional): Append or write text to the
                given file. Defaults to "w".
            sentences (bool): If false, write the document as a whole with each
                new line represented as ``"\\\\n O"``. If True, separate each
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

                # On sait que toutes les phrases dans les documents originaux se termine par un point et qu'il n'y a pas de phrase vide.
                # Par contre il y a parfois des lignes qui commencent par des espaces.
                # Or lorsque l'on crée l'oject ``BratAnnotation`` en utilisant sa method ``from_brat_file``, on utilise le parametre ``sep`` pour agrouper les lignes du fichier.txt afin de créer une seule chaine de charactère que l'on passera a spaCy.
                # Une ligne à généralement la forme suivante: ``'Apellidos: Rivera Bueno.\n'``. Toutes les lignes se terminent par un passage à la ligne ``'\n'``.
                # Le **``flair.datasets.ColumnCorpus``** de ``Flair``
                # Instantiates a Corpus from CoNLL column-formatted task data such as CoNLL03 or CoNLL2000.
                # Il lit des documents tokenizés depuis un fichier ``.txt``. Chaque ligne contient un mot ainsi que le tag ``BIO`` qui lui est associé séparé par un espace.
                # Pour detecter si l'on est en présence d'une nouvelle phrase, **``Flair``** regarde juste si la ligne est un espace ou non:
                # >>> "\n     ".isispace()
                # True
                # Dans notre case parfois la ligne suivante commence par un espace comme dans le cas suivant:
                # "Nombre: Carlos Lopèz.\n"
                # "  Tel: 645 394 261\n"
                # Dans ce cas prècis lorsque l'on lira le text avec ``BratAnnotations.from_brat_files`` on obtiendra l'attribut ``txt`` suivant:
                # "Nombre: Carlos Lopèz.\n  Tel: 645 394 261\n"
                # Spacy au travers du pipeline ``meddocan_pipeline`` transformera alors le text en un document qui va regouper les espaces pour former un unique token:
                # from meddocan.language.pipeline import meddocan_pipeline
                # >>> lines = ["Nombre: Carlos Lopèz.\n", "     Tel: 645 394 261\n"]
                # >>> text = "".join(lines)
                # >>> nlp = meddocan_pipeline()
                # >>> doc = nlp(text)
                # >>> for token in doc:
                # >>>     print(f"{token.text!r}")
                # 'Nombre'
                # ':'
                # 'Carlos'
                # 'Lopèz'
                # '.'
                # '\n     '
                # 'Tel'
                # ':'
                # '645'
                # '394'
                # '261'
                # '\n'

                if "\n" in token.text:
                    if sentences:
                        # Here spacy tokenize the following doc like this:
                        # text = "w1\n   \tw2\n"
                        # "|".join(f"{t.text!}" for t in nlp(text)) == "'w1'|'\n   \t'|'w2'|'\n'"
                        # So the token "\n   \t" is replace by "\n".
                        # The sentence becomes: "w1\nw2\n".
                        # This choice strip the sentences for the training.
                        f.write("\n")
                    else:
                        orth_ = token.text
                        if "\n" == orth_:
                            line = orth_.replace("\n", "\\n O\n")
                        else:
                            # We are in the case orth_ == "\n     ".
                            line = orth_.replace("\n", "\\n O\n")
                            line = f"{line} O\n"
                        f.write(line)
                else:

                    # https://stackoverflow.com/questions/17912307/u-ufeff-in-python-string
                    # Sometimes at the beginning of a new document, the U+FEFF
                    # character is write in the file.

                    line = f"{token.text} {tag}\n"
                    line = line.replace("\ufeff", "")
                    f.write(line)
                    assert line != "\n"
            # The last line seems to be a \n.. How to detect it?
            if not sentences:
                f.write("\n")


@dataclass
class BratDocs:
    """BratDoc is an Iterable of BratDoc for a given
    :class:`meddocan.data.ArchiveFolder`.
    It is used with its method ``write`` to write all the data present in one
    of the train, dev or test archive to a file in a column
    format compatible with the ``CoNLL03`` or ``CoNLL2000`` task type as
    required by the ``Flair.datasets.ColumnCorpus`` class.

    >>> from typing import Iterable
    >>> issubclass(BratDocs, Iterable)
    True

    Args:
        archive_name: (ArchiveFolder): The name of the folder contained in the
            zip file containing the compressed data.
        sep (str): The separator used to paste the lines of each text file in
            the archive together.

    Yields:
        BratDoc: The spacy documents with entity span with BIO notation.
    """

    archive_name: ArchiveFolder
    sep: str = ""

    def __iter__(self) -> Iterator[BratDoc]:
        from meddocan.language.pipeline import meddocan_pipeline

        nlp = meddocan_pipeline()

        expanded_entities: List[ExpandedEntity] = []

        for brat_files_pair in meddocan_zip.brat_files(self.archive_name):
            brat_annotations = BratAnnotations.from_brat_files(
                brat_files_pair, sep=self.sep
            )
            brat_doc = BratDoc.from_brat_annotations(nlp, brat_annotations)
            expanded_entities += brat_doc.check_doc(
                brat_annotations.brat_spans
            )
            yield brat_doc

        if expanded_entities:
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
        print("\n")

    def write(self, output: Path, sentences: bool = False) -> None:
        for i, brat_doc in enumerate(self):
            mode = "w"
            if i:
                mode = "a"
            brat_doc.write(output, mode, sentences=sentences)


if __name__ == "__main__":
    from flair.datasets import ColumnCorpus

    from meddocan.data.corpus import flair

    corpus: ColumnCorpus = flair.datasets.MEDDOCAN(sentences=True)

    total_lines = 0
    for dataset in ("test", "dev", "train"):
        archive_folder = getattr(ArchiveFolder, dataset)
        subtotal_lines = 0
        for brat_files_pair in meddocan_zip.brat_files(archive_folder):
            brat_annotations = BratAnnotations.from_brat_files(
                brat_files_pair, sep=""
            )
            subtotal_lines += len(brat_annotations)
        print(f"{archive_folder.value} had {subtotal_lines} lines.")
        total_lines += subtotal_lines
        assert len(getattr(corpus, dataset)) == subtotal_lines
    print(f"The whole datasets contains {total_lines} lines.")
