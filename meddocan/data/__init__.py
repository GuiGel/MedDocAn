"""
``meddocan.data`` provides class and function to deals with the original
meddocan datasets located on the web.
"""
from dataclasses import dataclass
from enum import Enum
from itertools import tee
from pathlib import Path
from typing import Iterator, NamedTuple
from zipfile import Path as ZipPath

import meddocan


class BratFilesPair(NamedTuple):
    """A pair of files composing the Brat format.

    Args:
        ann (ZipPath): The ``file_name.ann`` file. This file contains the
            span information related to the ``file_name.txt`` file.
        txt (ZipPath): The ``file_name.txt`` file. This file contains the
            text of a medical document.
    """

    ann: ZipPath
    txt: ZipPath


class ArchiveFolder(Enum):
    """The names of the decompressed zip files.

    This Enumeration is for self-documentation purposes only.

    Args:
        train (str, optional): Defaults to "train"
        dev (str, optional): Defaults to "dev"
        test (str, optional): Defaults to "test"
        sample (str, optional): Defaults to "sample"
        background (str, optional): Defaults to "background"
    """

    train = "train"
    dev = "dev"
    test = "test"
    sample = "sample"
    background = "background"


@dataclass
class MeddocanUrl:
    """The url to reach the distinct meddocan dataset.
    These url are available from `here_`

    Example:

    >>> meddocan_url = MeddocanUrl()
    >>> meddocan_url.train
    'http://temu.bsc.es/meddocan/wp-content/uploads/2019/03/train-set.zip'

    .. _here: https://temu.bsc.es/meddocan/index.php/datasets/
    """

    sample: str
    train: str
    dev: str
    test: str
    background: str
    base: str = "http://temu.bsc.es/meddocan/wp-content/uploads/2019"

    def __init__(self) -> None:
        self.sample = self._get_loc("03/sample-set.zip")
        self.train = self._get_loc("03/train-set.zip")
        self.dev = self._get_loc("04/dev-set-1.zip")
        self.test = self._get_loc("05/test-set.zip")
        self.background = self._get_loc("10/background-set.zip")

    def _get_loc(self, loc: str) -> str:
        return "/".join([self.base, loc])


meddocan_url = MeddocanUrl()


@dataclass
class MeddocanZip:
    """Names of the meddocan zip files.

    Example:

    >>> meddocan_zip = MeddocanZip()
    >>> base = meddocan.cache_root / "datasets" / "meddocan"
    >>> assert meddocan_zip.base == base
    >>> assert meddocan_zip.train = base / "train-set.zip"
    """

    train: Path
    dev: Path
    test: Path
    sample: Path
    background: Path
    base = meddocan.cache_root / "datasets" / "meddocan"

    def __init__(self) -> None:
        for attr in ["train", "dev", "test", "sample", "background"]:
            value = self._get_loc(attr)
            self.__setattr__(attr, value)

    def _get_loc(self, loc: str) -> Path:
        dir_name = Path(getattr(meddocan_url, loc)).name
        return self.base / dir_name

    def __iter__(self) -> Iterator[Path]:
        for attr in self.__dataclass_fields__:
            yield getattr(self, attr)

    def brat_files(self, dir_name: ArchiveFolder) -> Iterator[BratFilesPair]:

        # Save the zip files to disk if the files are not already present.

        from meddocan.data.cached_files import cached_meddocan_zipfile

        cached_meddocan_zipfile()

        # The desired zip file from which the files must be yield.

        zip_file: str = getattr(self, dir_name.value)

        zip_path = ZipPath(zip_file)

        # The path to the directory named ``dir_name`` inside the ``.zip``
        # archive.

        train_brat_path = zip_path / dir_name.value / "brat"

        files_txt, files_ann = tee(train_brat_path.iterdir(), 2)

        ann_files = [
            file for file in files_ann if Path(file.name).suffix == ".ann"
        ]
        txt_files = [
            file for file in files_txt if Path(file.name).suffix == ".txt"
        ]

        for ann_file, txt_file in zip(ann_files, txt_files):

            # The file name must be the same to be sure that the brat
            # annotation correspond to the wright txt file.

            assert Path(ann_file.name).stem == Path(txt_file.name).stem

            yield BratFilesPair(ann_file, txt_file)


meddocan_zip = MeddocanZip()
