import logging
import tempfile
from pathlib import Path
from typing import Optional

import flair
import flair.datasets
from flair.datasets import ColumnCorpus

from meddocan.data import ArchiveFolder
from meddocan.data.docs_iterators import GsDocs

logger = logging.getLogger(__name__)


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
        document_separator_token (str = optional): Separate each document by
            new line for `Flert` architecture. Usually *-DOCSTART-* in `FLair`.
            Defaults to None.
    """

    def __init__(
        self,
        sentences: bool = False,
        window: Optional[int] = None,
        in_memory: bool = True,
        document_separator_token: Optional[str] = None,
        **corpusargs,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdirname:
            for data_pth in (
                ArchiveFolder.train,
                ArchiveFolder.dev,
                ArchiveFolder.test,
            ):
                msg = f"{data_pth.value}".center(33, "-")
                logger.info(f"{msg:>58}")

                output_pth: Path = Path(tmpdirname) / data_pth.value
                brat_docs = GsDocs(archive_name=data_pth)
                brat_docs.to_connl03(
                    file=output_pth,
                    write_sentences=sentences,
                    window=window,
                    document_separator_token=document_separator_token,
                )

            # Column format.
            columns = {0: "text", 1: "ner"}

            # This dataset name.
            self.__class__.__name__.lower()

            super().__init__(
                data_folder=tmpdirname,
                column_format=columns,
                tag_to_bioes="ner",
                in_memory=in_memory,
                document_separator_token=document_separator_token,
                **corpusargs,
            )


flair.datasets.MEDDOCAN = MEDDOCAN
