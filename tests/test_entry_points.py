# import importlib.metadata
import itertools as it
import logging

import importlib_metadata as metadata

logger = logging.getLogger(__name__)


def test_entry_points():
    """Test that the custom spacy components have been correctly installed."""
    existing_eps = metadata.entry_points().select(group="spacy_factories")

    expected_eps = (
        metadata.EntryPoint(
            "line_sentencizer",
            value="meddocan.language.sentencizer:line_sentencizer",
            group="spacy_factories",
        ),
        metadata.EntryPoint(
            "missaligned_splitter",
            value="meddocan.language.splitter:missaligned_splitter",
            group="spacy_factories",
        ),
        metadata.EntryPoint(
            "predictor",
            value="meddocan.language.predictor:create_predictor",
            group="spacy_factories",
        ),
        metadata.EntryPoint(
            "write_methods",
            value="meddocan.language.method_extensions:create_write_methods",
            group="spacy_factories",
        ),
    )

    for existing_ep, expected_ep in it.zip_longest(existing_eps, expected_eps):
        assert existing_ep.__class__.__name__ == expected_ep.__class__.__name__
        for existing_attr, expected_attr in it.zip_longest(
            existing_ep, expected_ep
        ):
            assert existing_attr == expected_attr
        assert existing_ep == expected_ep

    # NOTE:
    # assert eps == entry_points, f"{eps} != {entry_points}" works on local
    # but fails on Github Actions...


if __name__ == "__main__":
    test_entry_points()
