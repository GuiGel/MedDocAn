"""This module contains a custom spaCy component that retokenize the document. 
"""
import logging
import re
from typing import List, Tuple, Union

from spacy.language import Language
from spacy.matcher import Matcher
from spacy.tokens import Doc, Token

logger = logging.getLogger(__name__)


class MissalignedSplitter:
    """When we process the text for feeding some neural network, we encounter
    some offsets that correspond to words that are inside a token that hasn't
    been split by the tokenizer.
    These words, like  "NºCol" can be in a token like "LopezNºCol". These
    cases observe most of the time, a repeated pattern that can be catch and
    split in two words ["Lopez", "NºCol"] where "Lopez" is one of the
    words that must be anonymised.
    This class use a spaCy token matcher to catch the specific pattern and
    then use the spaCy retokenize context manager to split the matched words in
    2 words that made sense in term of offsets.
    """

    def __init__(
        self,
        nlp: Language,
        name: str = "missaligned_splitter",
        words: List[str] = ["NºCol"],
    ) -> None:
        """Initialize the class

        Args:
            nlp (Language): spaCy Language.
            name (str, optional): Component's name. Defaults to
                "missaligned_splitter".
        """
        self.name = name
        self.nlp = nlp

        # The regex to match the missaligned token: r"^words_1words_2$".

        self.wds1 = words
        # wd2 = "[A-ZÓÍÉ]+([a-zóéí]|[A-ZÓ])*"
        patterns = [[{"ORTH": {"REGEX": f"^.*{wd1}.*$"}}] for wd1 in self.wds1]

        # Register a new token extension to flag to_split.

        Token.set_extension("split", default=False, force=True)
        self.matcher = Matcher(nlp.vocab)
        self.matcher.add("TO_SPLIT", patterns)

    def __call__(self, doc: Doc) -> Doc:
        """Apply component to doc.

        Args:
            doc (Doc): The spaCy Doc to process.

        Returns:
            Doc: The process Doc.
        """
        # https://spacy.io/usage/linguistic-features#retokenization
        # This method is invoked when the component is called on a Doc

        matches = self.matcher(doc)

        # The doc size change has we split some span.
        # cpt is used to update the next span position inside doc.

        cpt = 0

        for match_id, start, end in matches:

            # Catch the span that correspond to a given pattern.
            # For example if the pattern is r"^NºCol*$" the selected span
            # can be "NºColMartinez"

            span = doc[start + cpt : end + cpt]

            # Now that we have a span containing the desire pattern, it's time
            # to retokenize it in order to obtained the desired tokens.

            with doc.retokenize() as retokenizer:

                # Create the token into which the span must be split
                # Check first the patterns of the form word1WORD2.
                # For example we have in the judgment a lot of times
                # the following sort of words: apelanteCARLOS
                # and just CARLOS have to be anonymised.

                regex = r"|".join([f"({word})" for word in self.wds1])
                re_words = re.compile(regex)
                match = re_words.search(span.text)

                # Warning: If a word is repeated more than once in a span, only
                # the first word will be detected.

                if match:

                    # You have to detect where the match is located, for
                    # example "NºCol" in the span for example "CarlosNºCol".
                    # Here token_start and token_end are the start en end value
                    # inside the span.

                    token_start, token_end = match.start(), match.end()

                    tokens = [
                        span.text[:token_start],
                        span.text[token_start:token_end],
                        span.text[token_end:],
                    ]
                    tokens = [token for token in tokens if token]

                    # If we have only one token in tokens, it means that the
                    # word to be detected is alone. There is therefore no
                    # retokenization to perform.

                    if len(tokens) - 1:

                        # We don't care about heads so we attribute each
                        # subtoken to itself.

                        heads: List[Union[Token, Tuple[Token, int]]]  # mypy
                        heads = [(doc[start], i) for i, _ in enumerate(tokens)]

                        retokenizer.split(doc[span.start], tokens, heads=heads)

                        for token in span:
                            token._.split = True  # Mark token as split

                        # We update the cpt by 1 because we know
                        # that the span is split in 2 tokens that remplace one token.
                        # The difference is 1.

                        cpt += 1
        return doc


@Language.factory(
    name="missaligned_splitter",
    default_config={"words": ["NºCol"]},
    assigns=["token._.split"],
    retokenizes=True,
)
def missaligned_splitter(
    nlp: Language, name: str, words: List[str]
) -> MissalignedSplitter:
    # https://spacy.io/api/language#factory
    return MissalignedSplitter(nlp, name=name, words=words)
