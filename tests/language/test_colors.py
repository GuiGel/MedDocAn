from spacy.util import registry

from meddocan.language.colors import displacy_colors


def test_displacy_colors():
    user_colors = registry.displacy_colors.get_all()
    assert user_colors.get("spacy_displacy_colors", None) == displacy_colors
