from pathlib import Path

from flair.models.sequence_tagger_model import SequenceTagger
import pandas as pd
import torch
from meddocan.language.pipeline import MeddocanLanguage

def get_parameter_num(model_path: Path) -> int:
    """Obtain the number of parameter in a model

    Args:
        model_path (Path): Path to a SequenceTagger model

    Returns:
        int: Number of parameters of the model
    """
    model = SequenceTagger.load(model_path)
    parameters = model.parameters()
    return sum(map(torch.numel, parameters))

def make_pretty(styler):
    styler.background_gradient(axis=None, cmap="YlGnBu")
    styler.set_table_styles([
        {'selector': 'th.col_heading.level0', 'props': 'text-align: center; font-size: 1.5em;'},
        {"selector": "", "props": [("border", "1px solid grey")]},
        {"selector": "tbody td", "props": [("border", "1px solid grey")]},
        {"selector": "th", "props": [("border", "1px solid grey")]},
    ], overwrite=False)
    styler.format( "{:.0f}",)
    return styler

def get_models_parameters() -> pd.DataFrame:
    # The model are located on my public HuggingFace account
    mdpths = {
        ("FINETUNE", "XLMR + WE"): "GuiGel/xlm-roberta-large-flert-we-finetune-meddocan",
        ("FINETUNE", "XLMR"): "GuiGel/xlm-roberta-large-flert-finetune-meddocan",
        ("FINETUNE", "BETO + WE"): "GuiGel/beto-uncased-flert-context-we-finetune-meddocan", 
        ("FINETUNE", "BETO"): "GuiGel/beto-uncased-flert-finetune-meddocan",
        ("LSTM CRF", "BETO"): "GuiGel/beto-uncased-flert-lstm-crf-meddocan",
        ("LSTM CRF", "BETO + WE"): "GuiGel/beto-uncased-flert-context-we-lstm-crf-meddocan",
        ("LSTM CRF", "FLAIR"): "GuiGel/meddocan-flair-lstm-crf",
        ("LSTM CRF", "FLAIR + WE"): "GuiGel/meddocan-flair-we-lstm-crf",
    }

    model_parameters_num = {k: get_parameter_num(v) for k, v in mdpths.items()}

    df = (pd.DataFrame(model_parameters_num, index=["Numero de parámetros en Million"], dtype=int) / 1e6)

    return df.style.pipe(make_pretty)


def num_param(nlp: MeddocanLanguage) -> int:
    # The model are located on my public HuggingFace account
    parameters = nlp._components[2][1].model.parameters()  # Bad code but it works...
    return sum(map(torch.numel, parameters))
