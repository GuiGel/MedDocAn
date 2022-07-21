from tests.cli.utils import meddocan_cli


def test_app():
    result = meddocan_cli("--help")
    expected = (
        "Usage: Meddocan [OPTIONS] COMMAND [ARGS]...\n\n  "
        "Meddocan Command-Line Interface\n\n"
        "Options:\n"
        "  --install-completion  Install completion for the current shell.\n"
        "  --show-completion     Show completion for the current shell, to "
        "copy it or\n                        customize the installation.\n"
        "  --help                Show this message and exit.\n\n"
        "Commands:\n"
        "  eval         Evaluate the model with the `meddocan` metrics.\n"
        "  train-flert"
    )
    assert result == expected
