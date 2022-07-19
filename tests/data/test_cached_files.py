"""module where the ``meddocan.data.cached_files`` is tested"""
from pathlib import Path
from unittest.mock import patch

from meddocan import cache_root
from meddocan.data import meddocan_zip
from meddocan.data.cached_files import cached_meddocan_zipfile, cached_path


def test_cached_path():
    with patch.object(Path, "exists", return_value=True) as mock_method:
        output = cached_path("filename", "cached_dir")
        assert output == Path("filename")
        import flair

        assert flair.cache_root == cache_root
    mock_method.assert_called_once_with()


def test_cached_meddocan_zipfile():
    assert meddocan_zip.base == cached_meddocan_zipfile()
