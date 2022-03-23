from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List

from spacy.tokens.doc import Doc

from meddocan.data import ArchiveFolders, meddocan_zip
from meddocan.data.containers import BratAnnotations, BratDoc

# TODO With the function ``brat_annotations_gen`` one can extract the number of
# rows and documents per archive and compare it to those obtained in the
# flair.datasets.Corpus. But it is quicker to do it directly by reading the
# files.

# TODO Create a ``BratDocs`` class that is an iterator and have a ``write``
# method? Could be more convenient?

# TODO Change ArchiveFolders to ArchiveFolder


@dataclass
class BratDocs:
    archive_name: ArchiveFolders
    sep: str = ""

    def __iter__(self) -> Iterator[BratDoc]:
        yield from brat_doc_gen(self.archive_name, self.sep)

    def write(self, output: Path, sentences: bool = False) -> None:
        for i, brat_doc in enumerate(self):
            mode = "w"
            if i != 0:
                mode = "a"
            brat_doc.write(output, mode, sentences=sentences)


def brat_doc_gen(
    zip_folder_name: ArchiveFolders,
    sep: str = "",
) -> Iterator[BratDoc]:
    """Iterate over the dataset and embed each file in a :class:SpacyDoc
    object.

    Args:
        zip_folder_name (MeddocanZip): The folder inside the zip archive
            where are located the data from which the BratAnnotations will be
            extracted.
        sep (str): The separator that will join the lines of the
            :class:BratAnnotations to create it's ``text`` attribute.
            Defaults "".

    Returns:
        Iterator[BratAnnotations]: The document contained in a
            :class:SpacyDoc object.

    Yields:
        BratAnnotations: The next :class:SpacyDoc object.
    """
    # BratSpan.from_bytes(
    # "T17	CALLE 107 130	Calle Miguel Benitez 90\n".encode("utf-8")
    # )
    # first sort the ``brat_spans`` attribute of the
    # ``meddocan.brat_containers.BratAnnotations`` class.
    from meddocan.language.pipeline import meddocan_pipeline

    nlp = meddocan_pipeline()

    missalignment_count = 0

    for brat_files_pair in meddocan_zip.brat_files(zip_folder_name):
        brat_annotations = BratAnnotations.from_brat_files(
            brat_files_pair, sep=sep
        )
        brat_doc = BratDoc.from_brat_annotations(nlp, brat_annotations)
        missalignment_count += brat_doc.check_doc(brat_annotations.brat_spans)
        yield brat_doc

    print(f"{missalignment_count=}")

    """brat_text = brat_annotations.text
    brat_spans = brat_annotations.brat_spans

    # We have to link both info, the annotation and the document.
    # particularly the tokenization and the annotation offsets.
    # Take care of the alignment.

    # For the alignment part, using spaCy help reduce the programming
    # time offering great functionalities!

    # Create a **``spacy.tokens.Doc``** object.
    brat_doc = nlp(brat_text)

    # Don't sentencize the doc for now.

    # Add entities to the doc using the
    # ``spacy.training.offsets_to_biluo_tags`` function.
    # https://spacy.io/api/top-level#offsets_to_biluo_tags

    # First transform the list of BratSPan to a sequence of
    # (start, end, label) triples. Start and end should be
    # character-offset integers denoting the slice into the original
    # string.
    # If there is some entity that are not aligned in the text a
    # warning message is send by spaCy.
    # In this case a specific tokenizer must be use.

    entities = [brat_span.entity for brat_span in brat_spans]

    from spacy.training import offsets_to_biluo_tags

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tags = offsets_to_biluo_tags(
            brat_doc, entities=entities, missing="O"
        )

    # Check if a tag and a token are note aligned.
    # If it's the case the tokenizer must be modified.

    # As we have decided to extend the spans to all tokens that include
    # offsets, in case of an alignment error, it is sufficient to compare
    # the spans of the Doc with the corresponding BratSpan.text attribute.

    tag_alignment_warning(brat_doc, tags)

    # The token "H." is not split into ["H", "."] by spaCy.
    # We create a special case for this one as well a the "M." case with
    # stand for "Mujer".

    # If the tokens are aligned, we can add the spans to the document and
    # split it into sentence using a sentencizer.

    # Add span to doc.
    # For this task use the method ``set_ents`` of the
    # **``spacy.tokens.Doc``** object that accept a list of
    # **``spacy.tokens.Span``** object as argument. This method set the
    # name entities in the documents.

    # Concerning the entities that are not aligned, we can do the
    # following: If the entities are not aligned with the tokens, we
    # modify the ``spacy.tokens.span`` so that all tokens are included in
    # the span. In this case we would need to use post-detection
    # processing to clean up the detected entities from unwanted
    # characters.
    # For this we can use the ``spacy.tokens.doc.Doc.char_span`` methods
    # https://spacy.io/api/doc#char_span
    # and pass to the parameter ``alignment_mode`` the argument
    # ``"expand"``. In this case, we create a span of all tokens at
    # least partially covered by the character span.

    brat_doc.set_ents(
        [
            brat_doc.char_span(
                brat_span.start,
                brat_span.end,
                label=brat_span.entity_type,
                alignment_mode="expand",
            )
            for brat_span in brat_spans
        ]
    )
    # No exception occurs if alignment_mode="expand".
    # Let's check that the number of entities in the
    # ``spacy.tokens.doc.Doc`` is the same as in ``brat_spans``.

    assert len(brat_doc.ents) == len(brat_spans), (
        f"entities numbers {len(brat_doc.ents)} != "
        f"spans numbers {len(brat_spans)}."
    )

    # The ``spacy.tokens.Doc.ents`` attribute are sorted by they order
    # of apparition in the text. To compare with the brat_spans we must
    # to the same.
    # Sort the ``brat_spans`` list in place.

    brat_spans.sort(key=lambda brat_span: brat_span.start)

    print_brat_text = False
    for ent, brat_span in zip(brat_doc.ents, brat_spans):

        # Verify that the doc entity label is the same as the brat label.

        if ent.label_ != brat_span.entity_type:
            print(f"{ent.label_=} != {brat_span.entity_type=}")

        # Verify that the doc entity text is the same as the brat text.

        if ent.text != brat_span.text:
            print(
                f"selected entity '{ent.text}' expand original entity  "
                f"'{brat_span.text}'."
            )
            print_brat_text = True
            missalignment_count += 1

    if print_brat_text and verbose:
        print(brat_text)
        print("=" * 100)
    """


def tag_alignment_warning(brat_doc: Doc, tags: List[str]) -> None:
    import warnings

    msg = ["\n"]
    for token, tag in zip(brat_doc, tags):

        # The tag of the token indicate that the token is not aligned with the
        # entity.

        if tag == "-":
            msg.append(f" {token.orth_:>20} | {tag:<20}")
    if msg != ["\n"]:
        warnings.warn("\n".join(msg).center(20, "="))


def write_docs(
    data_path: ArchiveFolders, output: Path, sentences: bool = False
) -> None:
    for i, brat_doc in enumerate(BratDocs(data_path)):
        mode = "w"
        if i != 0:
            mode = "a"
        brat_doc.write(output, mode, sentences=sentences)


if __name__ == "__main__":

    for brat_doc in BratDocs(ArchiveFolders.dev):
        print(brat_doc)
        break

    raise StopIteration

    folder = Path("/home/ggelabert/Projects/MedDocAn/data")

    if not folder.exists():
        folder.mkdir(parents=True, exist_ok=True)

    for data_path in (
        ArchiveFolders.train,
        ArchiveFolders.dev,
        ArchiveFolders.test,
    ):

        file_path: Path = folder / data_path.value  # File Path to write in.
        print(f"========================== {data_path.value=}")

        for i, brat_doc in enumerate(brat_doc_gen(data_path)):

            # brat_doc contains the entities span. We can now transform them to
            # the flair format easily. The sentencizer step should be perform
            # before...

            mode = "w"
            if i != 0:
                mode = "a"
            write_docs(brat_doc, file_path, mode)

            # TODO We can think of a small function that does this job
            # but first we need to feel it!
            # We can look at the flair code to create the correct ColumnCorpus
            # in order to load the training data for flair.
            # from flair.datasets import CONLL_03_DUTCH

            # Works on the sentencizer first. Remove the . from the tokenizer
            # as it is yet added as a special case in the merger. We can thus
            # do it before the sentencizer step...

            # TODO You should check that you have the right number of documents
            # in each file 500, 250, 250.

        print(i)
        sentence_number = 0
        with file_path.open(mode="r", encoding="utf-8") as f:
            for line in f:
                if line == "\n":
                    sentence_number += 1
        print(f"{sentence_number=} {i+1=}")
