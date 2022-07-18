"""module where the ``meddocan.data.cached_files`` is tested"""
from black import out
import pytest
from meddocan.data.cached_files import cached_meddocan_zipfile, cached_path
from meddocan.data import meddocan_zip
from unittest.mock import call, patch, MagicMock
from pathlib import Path
from meddocan import cache_root


def test_cached_path():
    with patch.object(Path, "exists", return_value=True) as mock_method:
        output = cached_path("filename", "cached_dir")
        assert output == Path("filename")
        import flair
        assert flair.cache_root == cache_root
    mock_method.assert_called_once_with()


def test_cached_meddocan_zipfile():
    assert meddocan_zip.base == cached_meddocan_zipfile()

