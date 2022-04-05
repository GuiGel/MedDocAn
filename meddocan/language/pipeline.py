from typing import NewType, Optional

from spacy import blank
from spacy.language import Language

# TODO Set tokenizer as a language to use entry points
from .tokenizer import meddocan_tokenizer

MeddocanLanguage = NewType("MeddocanLanguage", Language)


def meddocan_pipeline(
    model_loc: Optional[str] = None, mini_batch_size: int = 8
) -> MeddocanLanguage:
    """Create meddocan language.

    Returns:
        MeddocanLanguage: A ``spacy.language.Language`` that preprocess the
            meddocan text in order to have alignment with the entities offsets.
    """
    nlp = blank("es")
    nlp.tokenizer = meddocan_tokenizer(nlp)
    nlp.add_pipe(
        "missaligned_splitter",
        config={"words": ["NºCol", "Correo", "años", "DR", "\.$", "\n"]},
    )
    nlp.add_pipe("line_sentencizer")
    nlp.add_pipe(
        "predictor",
        config={"model_loc": model_loc, "mini_batch_size": mini_batch_size},
    )
    return MeddocanLanguage(nlp)


if __name__ == "__main__":

    nlp = meddocan_pipeline()

    to_tokenize = [
        ("H.", ["H", "."]),
        ("M.", ["M", "."]),
        ("F.", ["F", "."]),
        ("nhc-150679.", ["nhc", "-", "150679", "."]),
        ("CP:28029", ["CP", ":", "28029"]),
        ("2014.El", ["2014", ".", "El"]),
        ("11A.", ["11A", "."]),
        # Occurs a lots in the text. Perhaps making a special rules make sense?
        ("s/n.", ["s/n", "."]),
        ("Der..", ["Der", ".", "."]),
        ("Izq..", ["Izq", ".", "."]),
        ("der..", ["der", ".", "."]),
        ("izq..", ["izq", ".", "."]),
        ("1964)10", ["1964", ")", "10"]),
        ("+34679802102", ["+", "34679802102"]),
        (
            "E-mail:guacorch@yahoo.es",
            ["E", "-", "mail", ":", "guacorch@yahoo", ".", "es"],
        ),
        (
            "E-mail.hleonbrito@hotmail.com",
            ["E", "-", "mail", ".", "hleonbrito@hotmail", ".", "com"],
        ),
        ("nhc/976421.", ["nhc", "/", "976421", "."]),
    ]

    for text, tokens in to_tokenize:
        if [token.orth_ for token in nlp(text)] != tokens:
            print("-" * 100)
            for t in nlp.tokenizer.explain(text):
                print(t[1], "\t", t[0])
