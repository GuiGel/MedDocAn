from __future__ import annotations

from pathlib import Path

import meddocan
from meddocan.data import meddocan_url, meddocan_zip


def cached_path(url_or_filename: str, cache_dir: str | Path) -> Path:
    """Given something that might be a URL (or might be a local path),
    determine which. If it's a URL, download the file and cache it, and
    return the path to the cached file. If it's already a local path,
    make sure the file exists and then return the path.
    """
    import flair

    # Patch ``cache_root`` to download files inside meddocan cache_root
    flair.cache_root = meddocan.cache_root

    from flair.data_fetcher import cached_path

    return cached_path(url_or_filename, cache_dir)


def cached_meddocan_zipfile() -> Path:
    """Cached the meddocan files directly from their url and return the path to
    their root folder on disk.

    The cached files are located in the ``$/.meddocan/data/meddocan``
    directory.
    """
    dir_name = meddocan_zip.base
    for attr in meddocan_zip.__dataclass_fields__:
        cached_path(getattr(meddocan_url, attr), dir_name)

    return dir_name
