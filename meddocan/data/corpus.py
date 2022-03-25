import tempfile
from pathlib import Path

import flair
import flair.datasets
from flair.datasets import ColumnCorpus

from meddocan.data import ArchiveFolder
from meddocan.data.containers import BratDocs


class MEDDOCAN(ColumnCorpus):
    """Flair column corpus that hold the train, dev and test sets.
    Two options are available:

    - The documents are not cut into lines but are presented as one long \
        sentence.
    - The documents are split into lines as is customary for sequence \
        labeling tasks.

    Args:
        sentences (bool, optional): Divide the original documents into
            sentences or group the document as a whole.  
            Defaults to False.
        in_memory (bool, optional): Keep the data into memory or not.  
            Defaults to True.
    """

    def __init__(
        self, sentences: bool = False, in_memory: bool = True, **corpusargs
    ):
        with tempfile.TemporaryDirectory() as tmpdirname:
            for data_pth in (
                ArchiveFolder.train,
                ArchiveFolder.dev,
                ArchiveFolder.test,
            ):
                msg = f"{data_pth.value}".center(33, "-")
                print(f"{msg:>58}")
                output_pth: Path = Path(tmpdirname) / data_pth.value
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
