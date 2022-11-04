from pathlib import Path
from typing import List, Tuple

import flair
import pandas as pd
import streamlit as st
import torch
from spacy.tokens import Doc
from spacy_streamlit import visualize_ner

from application.model import get_gold_doc, get_sys_doc
from application.samples import (
    FOLDER_PATH,
    GIT_REPO,
    get_samples,
    get_text_from_name,
)
from application.visualizer import visualize_ner
from meddocan.evaluation.classes import Evaluate, Ner, Span
from meddocan.language.colors import displacy_colors

flair.device = torch.device("cpu")

# ------ APP
st.set_page_config(
    page_title="Anonimización aplicada al ámbito medico",
    page_icon="👨🏼‍⚕️",
    layout="wide",
)

MODEL = "GuiGel/meddocan"


class App:
    # --------- Select all file available
    files = get_samples()
    names = {
        i: name
        for i, name in enumerate(
            list(set(Path(file).stem for file in files.keys()))
        )
    }

    @staticmethod
    def _max_width_():
        max_width_str = f"max-width: 1400px;"
        st.markdown(
            f"""
        <style>
        .reportview-container .main .block-container{{
            {max_width_str}
        }}
        </style>    
        """,
            unsafe_allow_html=True,
        )

    @staticmethod
    def _visualize_chosen_report(
        text: str, annotations: str
    ) -> Tuple[str, str]:
        with st.expander("Gold Files", expanded=False):
            c11, c12 = st.columns([1, 1])
            with c11:
                text = st.text_area(
                    label="Report",
                    value=text,
                    height=1000,
                    placeholder="Insert your text here!",
                )
            with c12:
                annotations = st.text_area(
                    label="Annotations",
                    value=annotations,
                    height=1000,
                    placeholder="Insert your text here!",
                )
        return text, annotations

    @classmethod
    def get_text(cls, name: str):
        return get_text_from_name(cls.files, name, "txt")

    @classmethod
    def get_annotations(cls, name: str):
        return get_text_from_name(cls.files, name, "ann")

    @classmethod
    def chose_reports(cls) -> List[str]:
        return st.multiselect(
            label="Select a Clinical Report from the list",
            default=["S0365-66912011000900005-1"],
            options=list(cls.names.values()),
            help=f"The files come from the test set located in the following github repository {GIT_REPO}/{FOLDER_PATH}",
        )

    @staticmethod
    def get_sys(model: str, text: str) -> Doc:
        return get_sys_doc(model, text)

    @staticmethod
    def get_gold(text: str, annotations: str) -> Doc:
        return get_gold_doc(text, annotations)

    @staticmethod
    def subtrack1(gold_doc: Doc, sys_doc: Doc) -> pd.DataFrame:
        sys = set(
            Ner(ent.label_, ent.start_char, ent.end_char)
            for ent in sys_doc.ents
        )
        gold = set(
            Ner(ent.label_, ent.start_char, ent.end_char)
            for ent in gold_doc.ents
        )

        tp = gold.intersection(sys)
        fp = sys - gold
        fn = gold - sys

        p = Evaluate.precision(tp, fp)
        r = Evaluate.recall(tp, fn)
        f1 = Evaluate.F_beta(p, r)

        return pd.DataFrame(
            [p, r, f1],
            index=["precision", "recall", "f1"],
            columns=["SubTrack1"],
        )

    @staticmethod
    def subtrack2_strict(gold_doc: Doc, sys_doc: Doc) -> pd.DataFrame:
        sys = set(Span(ent.start_char, ent.end_char) for ent in sys_doc.ents)
        gold = set(Span(ent.start_char, ent.end_char) for ent in gold_doc.ents)

        tp = gold.intersection(sys)
        fp = sys - gold
        fn = gold - sys

        p = Evaluate.precision(tp, fp)
        r = Evaluate.recall(tp, fn)
        f1 = Evaluate.F_beta(p, r)

        return pd.DataFrame(
            [p, r, f1],
            index=["precision", "recall", "f1"],
            columns=["SubTrack2[Strict]"],
        )

    @staticmethod
    def compute_scores(gold: Doc, sys: Doc):
        subtrack1 = App.subtrack1(gold, sys)
        subtrack2_strict = App.subtrack2_strict(gold, sys)
        return pd.concat(
            [subtrack1, subtrack2_strict],
            axis=1,
        )

    @staticmethod
    def visualize_ner(gold: Doc, sys: Doc) -> None:
        # Visualize the golds standard and system documents.
        c21, c22 = st.columns([1, 1])

        # ----- The NER labels
        labels = list(displacy_colors.keys())

        # ----- Set annotation to doc...
        with c21:
            visualize_ner(gold, labels=labels, key=hash(gold), title="Gold")

        # ----- Pass trough meddocan pipeline and visualize detected entities
        with c22:
            visualize_ner(
                sys,
                labels=labels,
                key=hash(sys),
                title="System",
                show_table=True,
            )


st.title("👨🏼‍⚕️ Anonimización aplicada al ámbito medico")
st.markdown(
    """     
-   The *MEDDOCAN* app is an easy-to-use interface built in Streamlit for the MedDocAn library!
-   It use Spacy and Flair to perform NER on the the SPACCC MEDDOCAN dataset.

----
"""
)


app = App()

app._max_width_()

with st.sidebar:
    names = app.chose_reports()


if names:
    c11, c12, c13 = st.columns([0.5, 0.5, 0.3])
    with c11:
        st.markdown(f"File")
    with c12:
        st.markdown(f"f1 micro")
    st.markdown("---")

    for i, name in enumerate(names):

        text = app.get_text(name)
        annotations = app.get_annotations(name)

        # ------ Predict
        gold = app.get_gold(text, annotations)
        sys = app.get_sys(MODEL, text)

        # Compute metrics
        df = app.compute_scores(gold, sys)

        c11, c12, c13 = st.columns([0.5, 0.5, 0.3])
        with c11:
            st.markdown(f"**{name}**")
        with c12:
            st.markdown(f"**{df['SubTrack1']['f1']}**")
        with c13:
            visualize = st.button(
                f"Visualize",
                key=name,
                help="""Click on this button to view the reference document \
                    and the prediction made by the model. To remove this view\
                    , simply deselect the corresponding file in the sidebar.\
                """,
            )

        if visualize:
            app._visualize_chosen_report(text, annotations)

            with st.expander(f"Detailed metrics"):
                st.table(df)

            # ------ Visualize NER
            with st.expander("Visualize NER"):
                app.visualize_ner(gold, sys)

        st.markdown("---")
