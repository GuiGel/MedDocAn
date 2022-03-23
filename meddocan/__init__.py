"""The meddocan package contains various results obtained by training the
neural network on the medical document anonymisation track. This track
corresponds to track 9 of the `Iberian Language Evaluation Forum 2019`_..

.. _Iberian Languages Evaluation Forum 2019:
    http://ceur-ws.org/Vol-2421/
"""
import os
from pathlib import Path

__version__ = "0.1.0"

# global variable: cache_root
cache_root = Path(
    os.getenv("MEDDOCAN_CACHE_ROOT", Path(Path.home(), ".meddocan"))
)
