import pytest
from spacy.tokens import Doc

from tests.language.test_method_extensions import MockDoc, TokenElements


@pytest.fixture(scope="package")
def mock_docs() -> Doc:
    """Fixture that create a list of `MockDoc` objects.

    To obtain a mocked |Doc| just use the `MockDoc.get_spacy_doc` method.
    """
    docs = [
        MockDoc(
            [
                TokenElements("Vivo", True, True),
                TokenElements("en", True, False),
                TokenElements("Aix", True, False, "B-LOC"),
                TokenElements("en", True, False, "I-LOC"),
                TokenElements("Provence", False, False, "I-LOC"),
                TokenElements("!", True, False),
                TokenElements("Soy", True, True),
                TokenElements("Eric", True, False, "B-PERS"),
                TokenElements("Laffont", False, False, "I-PERS"),
                TokenElements(".", True, False),
            ],
        ),
        MockDoc(
            [
                TokenElements("Vivo", True, True),
                TokenElements("en", True, False),
                TokenElements("Bilbao", False, False, "B-LOC"),
                TokenElements("!", True, False),
                TokenElements("Soy", True, True),
                TokenElements("Zaira", True, False, "B-PERS"),
                TokenElements("Aurrekoetxea", False, False, "I-PERS"),
                TokenElements(".", True, False),
            ],
        ),
        MockDoc(
            [
                TokenElements("Vivo", True, True),
                TokenElements("en", True, False),
                TokenElements("Aix", True, False, "B-LOC"),
                TokenElements("en", True, False, "I-LOC"),
                TokenElements("Provence", False, False, "I-LOC"),
                TokenElements("!", True, False),
                TokenElements("Soy", True, True),
                TokenElements("Eric", True, False),
                TokenElements("Laffont", False, False),
                TokenElements(".", True, False),
            ],
        ),
    ]
    return docs
