from flair.data import Sentence

from meddocan.models import SequenceTagger


def test_predict():
    model = SequenceTagger.load("ner-fast")
    sentences = [Sentence("My name is Guillaume")]
    model.predict(sentences)
    assert (
        sentences.__str__()
        == '[Sentence: "My name is Guillaume" → ["Guillaume"/PER]]'
    )
