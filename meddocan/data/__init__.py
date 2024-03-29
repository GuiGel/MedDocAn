"""
``meddocan.data`` provides class and function to deals with the original
meddocan datasets located on the web.

This module also provides two instances of objects that are useful for the rest
of the project.

The :obj:`meddocan.data.meddocan_url` that contains the url link to reach the zipped data folders.

    Examples:

    >>> meddocan_url.train
    'http://temu.bsc.es/meddocan/wp-content/uploads/2019/03/train-set.zip'

The :obj:`meddocan.data.meddocan_zip` object which gets the brat file pair for
a given folder in the cached dataset through its ``brat_files`` method.

    Example:

    >>> brat_files_pairs = meddocan_zip.brat_files(ArchiveFolder.train)
    >>> next(brat_files_pairs)
    BratFilesPair(ann=Path('/.../.meddocan/datasets/meddocan/train-set.zip', \
'train/brat/S0004-06142005000500011-1.ann'), \
txt=Path('/.../.meddocan/datasets/meddocan/train-set.zip', \
'train/brat/S0004-06142005000500011-1.txt'))

"""
from dataclasses import dataclass
from enum import Enum
from itertools import tee
from pathlib import Path
from typing import Iterator, NamedTuple, Union
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

    ann: Union[ZipPath, Path]
    txt: Union[ZipPath, Path]

    @property
    def is_zipfile_path(self) -> bool:
        """Verify that BratFilesPair attributes are of type ZipPath

        Returns:
            bool: If True :class:`BratFilesPair` attributes are of type ZipPath

        Example:

        Consider a zip file with this structure:

        .
        └── a.txt
        └── a.ann
        >>> from io import BytesIO
        >>> from zipfile import ZipFile, Path
        >>> data = BytesIO()
        >>> zf = ZipFile(data, 'w')
        >>> zf.writestr('a.txt', 'content of txt')
        >>> zf.writestr('a.ann', 'content of ann')
        >>> zf.filename = "folder.zip"
        >>> zfp = Path(zf)
        >>> txt, ann = zfp.iterdir()

        Then

        >>> BratFilesPair(ann, txt).is_zipfile_path
        True

        """
        if isinstance(self.ann, ZipPath) and isinstance(self.txt, ZipPath):
            return True
        else:
            return False

    @property
    def is_pathlib_path(self) -> bool:
        """Verify that BratFilesPair attributes are of type pathlib.Path

        Returns:
            bool: If True :class:`BratFilesPair` attributes are of type
                pathlib.Path

        Example:

        Consider a the following directory

        .
        └── folder
            ├── a.txt
            └── a.ann
        >>> from pathlib import Path
        >>> txt = Path("file.txt")
        >>> txt.write_text('content of txt')
        14
        >>> ann = Path("file.ann")
        >>> ann.write_text('content of ann')
        14

        Then

        >>> BratFilesPair(ann, txt).is_pathlib_path
        True

        Remove the created files

        >>> _, _ = txt.unlink(), ann.unlink()

        """
        if isinstance(self.ann, Path) and isinstance(self.txt, Path):
            return True
        else:
            return False


class ArchiveFolder(Enum):
    """The names of the decompressed zip files.

    This Enumeration is for self-documentation purposes only.

    Example:

    >>> ArchiveFolder("train")
    <ArchiveFolder.train: 'train'>

    If we pass a folder that doesn't exist:

    >>> ArchiveFolder("not_exist")
    Traceback (most recent call last):
    ...
    ValueError: 'not_exist' is not a valid ArchiveFolder

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
    >>> base
    PosixPath('/.../.meddocan/datasets/meddocan')
    >>> assert meddocan_zip.base == base
    >>> assert meddocan_zip.train == base / "train-set.zip"

    You can also iterate over the MeddocanZip attributes:

    >>> for folder_path in MeddocanZip():
    ...     print(folder_path)
    /.../.meddocan/datasets/meddocan/train-set.zip
    ...

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
        """Yield :class:`BratFilesPair` given a
        :class:`meddocan.data.ArchiveFolder`.

        Example:

        >>> brat_files_pairs = MeddocanZip().brat_files(ArchiveFolder.train)
        >>> next(brat_files_pairs)
        BratFilesPair(ann=Path('/.../.meddocan/datasets/meddocan/train-set.\
zip', 'train/brat/S0004-06142005000500011-1.ann'), txt=Path('/.../\
.meddocan/datasets/meddocan/train-set.zip', \
'train/brat/S0004-06142005000500011-1.txt'))


        Args:
            dir_name (ArchiveFolder): Name of the folder inside the downloaded\
                zip file.

        Yields:
            Iterator[BratFilesPair]: Pair of brat files.
        """

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

            # The file name must be the same to ensure that the brat annotation
            # matches the wright txt file.

            assert Path(ann_file.name).stem == Path(txt_file.name).stem

            yield BratFilesPair(ann_file, txt_file)


meddocan_zip = MeddocanZip()
