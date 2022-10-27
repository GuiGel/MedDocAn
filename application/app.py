from pathlib import Path

import flair
import streamlit as st
import torch
from spacy_streamlit import visualize_ner

from application.model import get_gold_doc, get_sys_doc
from application.samples import get_samples, get_text_from_name
from meddocan.language.colors import displacy_colors

flair.device = torch.device("cpu")


st.set_page_config(
    page_title="Anonimización aplicada al ámbito medico",
    page_icon="👨🏼‍⚕️",
    layout="wide",
)


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


_max_width_()


# with c30:
# st.image("logo.png", width=400)
st.title("👨🏼‍⚕️ Anonimización aplicada al ámbito medico")
st.header("")


with st.expander("ℹ️ - About this app", expanded=False):

    st.write(
        """     
-   The *MEDDOCAN* app is an easy-to-use interface built in Streamlit for the MedDocAn library!
-   It use Spacy and Flair to perform NER on the the SPACC MEDDOCAN dataset.
	    """
    )

    st.markdown("")


model = "/home/wave/Project/MedDocAn/experiments/corpus_sentence_bert_context_we_finetune_it_150/an_wh_rs_False_dpt_0_emb_Stack(0_es-wiki-fasttext-300d-1M, 1_1-beto-cased_FT_True_Ly_-1_seed_33)_lr_5e-06_it_150_bs_4_opti_AdamW_pjct_emb_False_sdl_LinearSchedulerWithWarmup_use_crf_False_use_rnn_False_wup_0.1/0/final-model.pt"


# --------- Select all file available
files = get_samples()
names = list(set((Path(file).stem for file in files.keys())))

selected_report = st.selectbox(
    label="Select a Clinical Report from the list!", options=names
)

# ----- Extract text from clinical report
txt = get_text_from_name(files, selected_report, "txt")

# ----- Extract annotation from clinical report
ann = get_text_from_name(files, selected_report, "ann")

# ------ Display the text and the annotations in 2 distinct columns
with st.expander("ℹ️ - About the Clinical Report", expanded=False):
    c11, c12 = st.columns([1, 1])
    with c11:
        text = st.text_area(
            label="Report",
            value=txt,
            height=1000,
            placeholder="Insert your text here!",
        )
    with c12:
        annotations = st.text_area(
            label="Annotations",
            value=ann,
            height=1000,
            placeholder="Insert your text here!",
        )

# Visualize the golds standard and system documents.
c21, c22 = st.columns([1, 1])

# ----- The NER labels
labels = list(displacy_colors.keys())

# ----- Set annotation to doc...
with c21:
    gold_doc = get_gold_doc(text, annotations)
    visualize_ner(gold_doc, labels=labels, key="gold", title="Gold")

# ----- Pass trough meddocan pipeline and visualize detected entities
with c22:
    sys_doc = get_sys_doc(model, text)
    visualize_ner(sys_doc, labels=labels, key="sys", title="System")

st.text(f"Analyzed using Flair model {Path(model).parents[1].name}")
