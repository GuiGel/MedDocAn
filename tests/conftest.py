import pytest
from spacy import blank

from meddocan.language.pipeline import meddocan_pipeline


@pytest.fixture(scope="function")
def blank_language():
    """Fixture that return a blank language for Spanish.
    .. note::
        We add custom components to the ``blank_language`` to test how
        they work. If the scope is session, all these pipelines will be add one
        afteranother...
    """
    yield blank("es")


@pytest.fixture(scope="session")  # scope session to seed up tests
def meddocan_language():
    """Fixture that return the ``meddocan_pipeline``."""
    nlp = meddocan_pipeline()
    yield nlp
