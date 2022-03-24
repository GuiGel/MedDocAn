import pytest
import spacy


@pytest.mark.parametrize(
    argnames="text,expected,regex",
    argvalues=[
        # Test that Lola is split.
        ("LolaLopezGarcia", "Lola|LopezGarcia", "Lola"),
        # Test that Lopez is split.
        ("LolaLopezGarcia", "Lola|Lopez|Garcia", "Lopez"),
        # Test that Lopez doesn't split because must be at the beginning.
        ("LolaLopezGarcia", "LolaLopezGarcia", "^Lopez"),
        # Test that Lopez doesn't split because must be at the end.
        ("LolaLopezGarcia", "LolaLopezGarcia", "Lopez$"),
    ],
)
def test_missaligned_splitter(text, expected, regex):
    """Test that the missaligned splitter works as expected, i.e that the given
    ``text`` argument is tokenize as indicated by ``expected`` and
    utilizing the ``regex`` value as split parameter.
    """
    nlp = spacy.blank("es")
    nlp.add_pipe("missaligned_splitter", config={"words": [regex]})
    doc = nlp(text)
    output = "|".join(token.text for token in doc)
    assert expected == output
