import tempfile
from pathlib import Path

import flair
import flair.datasets
from flair.datasets import ColumnCorpus

from meddocan.data import ArchiveFolder
from meddocan.data.containers import BratDocs


class MEDDOCAN(ColumnCorpus):
    def __init__(
        self, sentences: bool = False, in_memory: bool = True, **corpusargs
    ):
        with tempfile.TemporaryDirectory() as tmpdirname:
            for data_pth in (
                ArchiveFolder.train,
                ArchiveFolder.dev,
                ArchiveFolder.test,
            ):
                output_pth: Path = Path(tmpdirname) / data_pth.value
                msg = f"Write {data_pth} data to {output_pth=}".center(50, "=")
                print(msg)
                brat_docs = BratDocs(archive_name=data_pth)
                brat_docs.write(output=output_pth, sentences=sentences)

            # Column format.
            columns = {0: "text", 1: "ner"}

            # This dataset name.
            self.__class__.__name__.lower()

            super().__init__(
                data_folder=tmpdirname,
                column_format=columns,
                tag_to_bioes="ner",
                in_memory=in_memory,
                **corpusargs,
            )


flair.datasets.MEDDOCAN = MEDDOCAN
