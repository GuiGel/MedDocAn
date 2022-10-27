import spacy
import streamlit as st

from meddocan.language.pipeline import meddocan_pipeline


@st.cache(
    allow_output_mutation=True,
    suppress_st_warning=True,
    hash_funcs={"MyUnhashableClass": lambda _: None},
)
def load_model(name: str = None) -> spacy.language.Language:
    """Load a model into the meddocan pipeline."""
    return meddocan_pipeline(name)


@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def get_sys_doc(model_name: str, text: str) -> spacy.tokens.Doc:
    """Process a text and create a Doc object."""
    nlp = load_model(model_name)
    return nlp(text)


from meddocan.data.containers import BratSpan
from meddocan.data.utils import set_ents_from_brat_spans


@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def get_gold_doc(text: str, annotations: str) -> spacy.tokens.Doc:
    """Process a text and create a Doc object."""
    nlp = load_model()
    doc = nlp(text)
    brat_spans = list(map(BratSpan.from_txt, annotations.strip().split("\n")))
    return set_ents_from_brat_spans(doc, brat_spans)
