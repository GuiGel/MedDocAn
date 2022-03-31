from importlib import metadata


def test_entry_points():
    """Test that the custom spacy components have been correctly installed."""
    eps = metadata.entry_points()["spacy_factories"]

    entry_points = (
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
    )

    assert eps == entry_points, f"{eps} != {entry_points}"
