"""Test the MeddocanZip __iter__ method."""

from meddocan import cache_root
from meddocan.data import MeddocanZip


class TestMeddocanZip:
    def test___iter__(self):
        for folder_path in MeddocanZip():
            assert folder_path == (
                cache_root / "datasets/meddocan/train-set.zip"
            )
            break
