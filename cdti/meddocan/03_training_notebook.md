---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.1
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Generate training results

```{code-cell} ipython3
from myst_nb import glue
```

## Results on dev set
The evaluation of automatic predictions had two different scenarios or sub-tracks:

1.  **NER offset and entity type classification**: the first sub-track was focused
on the identification and classification of sensitive information (e.g., patient
names, telephones, addresses, etc.).  

2.  **Sensitive span detection**: the second sub-track was focused on the detection
of sensitive text more specific to the practical scenario necessary for the
release of de-identified clinical documents, where the objective is to identify
and to mask confidential data, regardless of the real type of entity or the
correct identification of PHI type.

We evaluate our models using the various evaluation scripts and report averaged F1-Score over tree runs.

+++

First create some code to automatically extract the evaluation results.

```{code-cell} ipython3
import pandas as pd
from collections import defaultdict
from pathlib import Path
from typing import Callable, DefaultDict, List, NamedTuple


class SubtrackScores(NamedTuple):
    precision: float
    recall: float

def _get_scores(folder_path: Path, filename: str, precision_line: int, recall_line: int) -> SubtrackScores:
    fpth = Path(folder_path / filename)
    if not fpth.exists():
        raise FileNotFoundError(f"{fpth} not found!")

    lines = fpth.read_text().split("\n")

    precision = float(lines[precision_line].split("=")[-1])
    recall = float(lines[recall_line].split("=")[-1])

    return SubtrackScores(precision, recall)

def get_subtrack1_scores(folder_path: Path) -> SubtrackScores:
    return _get_scores(folder_path, "ner", -3, -2)

def get_subtrack2_strict_scores(folder_path: Path) -> SubtrackScores:
    return _get_scores(folder_path, "spans", -6, -5)

def get_subtrack2_merged_scores(folder_path: Path) -> SubtrackScores:
    return _get_scores(folder_path, "spans", -3, -2)

def get_scores_as_df(seeds: List[int], get_folder: Callable[[int], Path]) -> pd.DataFrame:
    subtracks_scores: DefaultDict[List, float] = defaultdict(list)

    for seed in seeds:
        fpth = get_folder(seed)

        p, r = get_subtrack1_scores(fpth)
        subtracks_scores["1_p"].append(p)
        subtracks_scores["1_r"].append(r)

        p, r = get_subtrack2_strict_scores(fpth)
        subtracks_scores["2_1_p"].append(p)
        subtracks_scores["2_1_r"].append(r)

        p, r = get_subtrack2_merged_scores(fpth)
        subtracks_scores["2_2_p"].append(p)
        subtracks_scores["2_2_r"].append(r)

    df = pd.DataFrame.from_dict(subtracks_scores)
    for col in ["1", "2_1", "2_2"]:
        df[f"{col}_f1"] = 2*df[f"{col}_p"]*df[f"{col}_r"] / (df[f"{col}_p"] + df[f"{col}_r"])

    # Reorder columns
    new_columns = ["1_p", "1_r", "1_f1", "2_1_p", "2_1_r", "2_1_f1", "2_2_p", "2_2_r", "2_2_f1"]
    df = df[new_columns]

    # Prepare multi index names
    multi_index = pd.MultiIndex.from_product(
        [
            ["Subtrack 1", "Subtrack 2 [Strict]", "Subtrack 2 [Merged]"],
            ["precision", "recall", "f1"]
        ],
        names=["Track", "Scores"]
    )
    # Give multi index to df
    return pd.DataFrame(df.to_numpy().T, index=multi_index)
```

2. Define code to visualize the results in a convenient way

```{code-cell} ipython3
import pandas as pd
import matplotlib.pyplot as plt  
from matplotlib import colors

def make_pretty(styler):
    styler.set_table_styles([
        {'selector': '.index_name', 'props': 'font-style: italic; color: darkgrey; font-weight:normal;'},
        {'selector': 'th.level1', 'props': 'text-align: left;'},
        {'selector': 'th.level0', 'props': 'text-align: center;'},
        {'selector': 'th.col_heading', 'props': 'text-align: center;'},
        {'selector': 'th.col_heading.level0', 'props': 'font-size: 1.5em;'},
        {'selector': 'td', 'props': 'text-align: center; font-weight: bold;'},
    ], overwrite=False)
    # .set_caption("Ajuste fino evaluado con distintas métricas")
    styler.hide(axis="index", level=2)
    styler.hide(axis="columns", level=1)
    styler.format(precision=2)
    return styler

def visualize_df(df: pd.DataFrame):
    # Get the text that will be display in the form mean plus minus std
    std = (df*100).iloc[1::2, ::].round(2).astype(str).droplevel(2)
    mean = (df*100).iloc[::2, ::].round(2).astype(str).droplevel(2)
    df_txt = (mean + " \u00b1 " + std)

    # Extract the mean value that will serve to create the gradient map
    background_df = df.iloc[::2, ::]

    def b_g(s, cmap='PuBu', low=0, high=0):
        # Taken from https://stackoverflow.com/questions/47391948/pandas-style-background-gradient-using-other-dataframe
        nonlocal background_df
        # Pass the columns from Dataframe background_df
        a = background_df.loc[:,s.name].copy()
        rng = a.max() - a.min()
        norm = colors.Normalize(a.min() - (rng * low), a.max() + (rng * high))
        normed = norm(a.values)
        c = [colors.rgb2hex(x) for x in plt.cm.get_cmap(cmap)(normed*0.9)]
        return ['background-color: %s' % color for color in c]

    return df_txt.style.apply(b_g, cmap='plasma').pipe(make_pretty)
```

Define the root folder where all the results and the trained models are stored

```{code-cell} ipython3
base_folder = Path.cwd().parents[1]
```

Store the evaluation for each model in a ``pandas.DataFrame``.

```{code-cell} ipython3
from functools import partial

metric = "f1"
dataset = "dev"

def get_results(metric: str = "f1", dataset: str = "dev") -> pd.DataFrame:
    get_scores_as_df_with_seed = partial(get_scores_as_df,  [1, 12, 33])
    get_folders_all = {
        # ("LSTM CRF", "FLAIR + WE"): lambda seed: base_folder / f"experiments/corpus_sentence_flair_we_lstm_crf/results_seed_{seed}/evals/{dataset}",
        ("LSTM CRF", "FLAIR"): lambda seed: base_folder / f"experiments/corpus_sentence_flair_lstm_crf/an_wh_rs_True_dpt_0.08716810045694838_emb_seed_{seed}_Stack(0_lm-es-forward.pt, 1_lm-es-backward.pt)_hdn_sz_256_lr_0.1_it_150_bs_4_opti_SGD_pjct_emb_True_rnn_ly_2_sdl_AnnealOnPlateau_use_crf_True_use_rnn_True/0/evals/{dataset}",
        ("LSTM CRF", "BETO + CONTEXT"): lambda seed: base_folder / f"experiments/corpus_sentence_bert_context_lstm_crf/an_wh_rs_False_dpt_0_emb_beto_Ly_all_mean_context_seed_{seed}_hdn_sz_256_lr_0.1_it_500_bs_4_opti_SGD_pjct_emb_False_rnn_ly_2_sdl_AnnealOnPlateau_use_crf_True_use_rnn_True/0/evals/{dataset}",
        ("LSTM CRF", "BETO"): lambda seed: base_folder / f"experiments/corpus_sentence_bert_lstm_crf/an_wh_rs_False_dpt_0_emb_beto_Ly_all_mean_seed_{seed}_hdn_sz_256_lr_0.1_it_500_bs_4_opti_SGD_pjct_emb_False_rnn_ly_2_sdl_AnnealOnPlateau_use_crf_True_use_rnn_True/0/evals/{dataset}",
        ("LSTM CRF", "BETO + WE + CONTEXT"): lambda seed: base_folder / f"experiments/corpus_sentence_bert_context_we_lstm_crf/an_wh_rs_False_dpt_0_emb_Stack(0_es-wiki-fasttext-300d-1M, 1_1-beto_Ly_all_mean_context_seed_{seed})_hdn_sz_256_lr_0.1_it_500_bs_4_opti_SGD_pjct_emb_False_rnn_ly_2_sdl_AnnealOnPlateau_use_crf_True_use_rnn_True/0/evals/{dataset}",
        ("LSTM CRF", "BETO + WE"): lambda seed: base_folder / f"experiments/corpus_sentence_bert_we_lstm_crf/an_wh_rs_False_dpt_0_emb_Stack(0_es-wiki-fasttext-300d-1M, 1_1-beto_Ly_all_mean_seed_{seed})_hdn_sz_256_lr_0.1_it_500_bs_4_opti_SGD_pjct_emb_False_rnn_ly_2_sdl_AnnealOnPlateau_use_crf_True_use_rnn_True/0/evals/{dataset}",
        ("FINETUNE", "BETO + CONTEXT"): lambda seed: base_folder / f"experiments/corpus_sentence_bert_context_finetune/an_wh_rs_False_dpt_0_emb_beto-cased-context_FT_True_Ly_-1_seed_{seed}_lr_5e-06_it_150_bs_4_opti_AdamW_pjct_emb_False_sdl_LinearSchedulerWithWarmup_use_crf_False_use_rnn_False_wup_0.1/0/evals/{dataset}",
        ("FINETUNE", "BETO + WE + CONTEXT"): lambda seed: base_folder / f"experiments/corpus_sentence_bert_context_we_finetune_it_150/an_wh_rs_False_dpt_0_emb_Stack(0_es-wiki-fasttext-300d-1M, 1_1-beto-cased_FT_True_Ly_-1_seed_{seed})_lr_5e-06_it_150_bs_4_opti_AdamW_pjct_emb_False_sdl_LinearSchedulerWithWarmup_use_crf_False_use_rnn_False_wup_0.1/0/evals/{dataset}",
        ("FINETUNE", "BETO"): lambda seed: base_folder / f"experiments/corpus_sentence_bert_finetune_it_150/an_wh_rs_False_dpt_0_emb_beto-cased_FT_True_Ly_-1_seed_{seed}_lr_5e-06_it_150_bs_4_opti_AdamW_pjct_emb_False_sdl_LinearSchedulerWithWarmup_use_crf_False_use_rnn_False_wup_0.1/0/evals/{dataset}",
        ("FINETUNE", "BETO + WE"): lambda seed: base_folder / f"experiments/corpus_sentence_bert_we_finetune_it_150/an_wh_rs_False_dpt_0_emb_Stack(0_es-wiki-fasttext-300d-1M, 1_1-beto-cased_FT_True_Ly_-1_seed_{seed})_lr_5e-06_it_150_bs_4_opti_AdamW_pjct_emb_False_sdl_LinearSchedulerWithWarmup_use_crf_False_use_rnn_False_wup_0.1/0/evals/{dataset}",
        ("FINETUNE", "XLMRL + CONTEXT"): lambda seed: base_folder / f"experiments/corpus_sentence_xlmrl_context_finetune/an_wh_rs_False_dpt_0_emb_xlm-roberta-large-cased-context_FT_True_Ly_-1_seed_{seed}_lr_5e-06_it_40_bs_4_opti_AdamW_pjct_emb_False_sdl_LinearSchedulerWithWarmup_use_crf_False_use_rnn_False_wup_0.1/0/evals/{dataset}",
        ("FINETUNE", "XLMRL"): lambda seed: base_folder / f"experiments/corpus_sentence_xlmrl_finetune/an_wh_rs_False_dpt_0_emb_xlm-roberta-large-cased_FT_True_Ly_-1_seed_{seed}_lr_5e-06_it_40_bs_4_opti_AdamW_pjct_emb_False_sdl_LinearSchedulerWithWarmup_use_crf_False_use_rnn_False_wup_0.05/0/evals/{dataset}",
        ("FINETUNE", "XLMRL + WE"): lambda seed: base_folder / f"experiments/corpus_sentence_xlmrl_we_finetune/an_wh_rs_False_dpt_0_emb_Stack(0_es-wiki-fasttext-300d-1M, 1_1-xlm-roberta-large-cased_FT_True_Ly_-1_seed_{seed})_lr_5e-06_it_40_bs_4_opti_AdamW_pjct_emb_False_sdl_LinearSchedulerWithWarmup_use_crf_False_use_rnn_False_wup_0.05/0/evals/{dataset}",
    }
    dfs_all = {k: get_scores_as_df_with_seed(v).T.describe().T[["mean", "std"]].loc[pd.IndexSlice[:, [f"{metric}"]], :] for k,v in get_folders_all.items()}


    get_scores_as_df_with_seed = partial(get_scores_as_df,  [1, 10, 25, 33, 42])
    get_folders_flair = {
        ("LSTM CRF", "FLAIR + WE"): lambda seed: base_folder / f"experiments/corpus_sentence_flair_we_lstm_crf/results_seed_{seed}/evals/{dataset}",
    }
    dfs_flair = {k: get_scores_as_df_with_seed(v).T.describe().T[["mean", "std"]].loc[pd.IndexSlice[:, [f"{metric}"]], :] for k,v in get_folders_flair.items()}

    dfs = {**dfs_all, **dfs_flair}
    return dfs

dfs = get_results()
result_metrics = pd.concat(dfs.values(), axis=1, keys=sorted(dfs.keys()), names=["Embedding", "Estrategia"]).T
visualize_df(result_metrics)
```

Visualize the results for the finetuning strategy with different flavours and the 2 chosen transformers.

```{code-cell} ipython3
data = {
        ("XLMR LARGE", "Transformador lineal"): dfs[("FINETUNE", "XLMRL")],
        ("XLMR LARGE", "+ Context"): dfs[("FINETUNE", "XLMRL + CONTEXT")],
        ("XLMR LARGE", "+ WE"): dfs[("FINETUNE", "XLMRL + WE")],
        ("BETO", "Transformador lineal"): dfs[("FINETUNE", "BETO")],
        ("BETO", "+ Context"): dfs[("FINETUNE", "BETO + CONTEXT")],
        ("BETO", "+ WE"): dfs[("FINETUNE", "BETO + WE")],
        ("BETO", "+ WE + Context"): dfs[("FINETUNE", "BETO + WE + CONTEXT")],
    }
df = pd.concat(data.values(), axis=1, keys=data.keys(), names=["Transformador", "Estrategia"]).T
glue("table_finetune_dev", visualize_df(df))
```

Visualize the results for the lstm+crf strategy with different flavours and the 2 chosen transformers.

```{code-cell} ipython3
data = {
        ("LSTM CRF", "BETO (Ultimas 4 capas)"): dfs[("LSTM CRF", "BETO")],
        ("LSTM CRF", "+ Context"): dfs[("LSTM CRF", "BETO + CONTEXT")],
        ("LSTM CRF", "+ WE"): dfs[("LSTM CRF", "BETO + WE")],
        ("LSTM CRF", "+ WE + Context"): dfs[("LSTM CRF", "BETO + WE + CONTEXT")],
    }
df = pd.concat(data.values(), axis=1, keys=data.keys(), names=["Estrategia", "computation"]).T
glue("table_feature_based_dev", visualize_df(df))
```

Visualize the results for the lstm+crf strategy with different flavours with Flair embeddings.

```{code-cell} ipython3
data = {
        ("LSTM CRF", "FLAIR"): dfs[("LSTM CRF", "FLAIR")],
        ("LSTM CRF", "+ WE"): dfs[("LSTM CRF", "FLAIR + WE")],
    }
df = pd.concat(data.values(), axis=1, keys=data.keys(), names=["Estrategia", "Embeddings"]).T
glue("table_flair_dev", visualize_df(df))
```

## Results on the test set

```{code-cell} ipython3
data = get_results(dataset="test")
result_metrics = pd.concat(data.values(), axis=1, keys=sorted(data.keys()), names=["Estrategia", "Embeddings"]).T

glue("table_test", visualize_df(result_metrics))
```

Compare the previous results with the best score obtained with ``Flair`` contextual embeddings.

```{code-cell} ipython3
diff = result_metrics[::2] - result_metrics.loc[("LSTM CRF", "FLAIR")].loc["mean"].values.squeeze()
diff = diff.drop([("LSTM CRF", "FLAIR"), ("LSTM CRF", "FLAIR + WE")])*100
glue("compare_with_flair", diff.style.pipe(make_pretty).hide(axis="index", level=[2, 3]).background_gradient("plasma"))
```

```{code-cell} ipython3

```
