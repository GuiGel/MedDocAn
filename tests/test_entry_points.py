import importlib.metadata
import itertools as it
import logging

logger = logging.getLogger(__name__)


def test_entry_points():
    """Test that the custom spacy components have been correctly installed."""
    existing_eps = importlib.metadata.entry_points()["spacy_factories"]

    expected_eps = (
        importlib.metadata.EntryPoint(
            "line_sentencizer",
            value="meddocan.language.sentencizer:line_sentencizer",
            group="spacy_factories",
        ),
        importlib.metadata.EntryPoint(
            "missaligned_splitter",
            value="meddocan.language.splitter:missaligned_splitter",
            group="spacy_factories",
        ),
        importlib.metadata.EntryPoint(
            "predictor",
            value="meddocan.language.predictor:create_predictor",
            group="spacy_factories",
        ),
        importlib.metadata.EntryPoint(
            "write_methods",
            value="meddocan.language.method_extensions:create_write_methods",
            group="spacy_factories",
        ),
    )

    for existing_ep, expected_ep in it.zip_longest(existing_eps, expected_eps):
        assert existing_ep.__class__ == expected_ep.__class__
        for existing_attr, expected_attr in it.zip_longest(
            existing_ep, expected_ep
        ):
            assert existing_attr == expected_attr
        assert existing_ep == expected_ep
        logger.info(f"{existing_ep=}")
        logger.info(f"{expected_ep=}")
    # assert eps == entry_points, f"{eps} != {entry_points}"


if __name__ == "__main__":
    test_entry_points()
