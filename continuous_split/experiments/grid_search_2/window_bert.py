import json
from pathlib import Path
import itertools as it

import flair
import torch
from flair.data import Corpus
from flair.embeddings import TransformerWordEmbeddings
from flair.optim import LinearSchedulerWithWarmup
from hyperopt import hp
from torch.optim import AdamW
from transformers import AdamW

from meddocan.data.corpus import MEDDOCAN
from meddocan.hyperparameter.param_selection import (
    OptimizationValue,
    Parameter,
    SearchSpace,
    SequenceTaggerParamSelector,
)
from meddocan.hyperparameter.parameter import Parameter

base_path = Path(__file__).parent

SEED = 1

flair.set_seed(SEED)
flair.device = torch.device("cuda:0")

def training(sentences, window, epochs):  
    # 1. get the corpus
    corpus: Corpus = MEDDOCAN(sentences=sentences, window=window, document_separator_token="-DOCSTART-")
    print(corpus)

    if sentences:
        emb_name = f"beto-cased-context_FT_True_Ly_-1_seed_{SEED}"
    else:
        emb_name = f"beto-cased-context_window_{window}_FT_True_Ly_-1_seed_{SEED}"

    # 3. make the label dictionary from the corpus
    label_dict = corpus.make_label_dictionary(label_type="ner")

    statistics = corpus.obtain_statistics("ner", pretty_print=False)
    for set, value in statistics.items():
        for label in value["number_of_documents_per_class"].keys():
            label_dict.add_item(label)

    print(json.dumps(statistics, indent=4))

    # 4. Define your search space
    search_space = SearchSpace()
    search_space.add(
        Parameter.EMBEDDINGS,
        hp.choice,
        options=[
            TransformerWordEmbeddings(
                model="dccuchile/bert-base-spanish-wwm-cased",
                fine_tune=True,
                layers="-1",
                use_context=64,
                layer_mean=True,
                name=emb_name,
                subtoken_pooling="first",
                allow_long_sentences=True,
            ),
        ],
    )
    search_space.add(Parameter.USE_CRF, hp.choice, options=[False])
    search_space.add(Parameter.USE_RNN, hp.choice, options=[False])
    search_space.add(Parameter.REPROJECT_EMBEDDINGS, hp.choice, options=[False])
    search_space.add(Parameter.NUM_WORKERS, hp.choice, options=[4])
    search_space.add(Parameter.DROPOUT, hp.choice, options=[0])
    search_space.add(Parameter.LEARNING_RATE, hp.choice, options=[5e-6])
    search_space.add(Parameter.MINI_BATCH_SIZE, hp.choice, options=[4])
    search_space.add(Parameter.ANNEAL_WITH_RESTARTS, hp.choice, options=[False])
    search_space.add(Parameter.OPTIMIZER, hp.choice, options=[AdamW])
    search_space.add(
        Parameter.SCHEDULER, hp.choice, options=[LinearSchedulerWithWarmup]
    )
    search_space.add(Parameter.WARMUP_FRACTION, hp.choice, options=[0.1])
    search_space.add(Parameter.EMBEDDINGS_STORAGE_MODE, hp.choice, options=["gpu"])
    search_space.add(Parameter.MAX_EPOCHS, hp.choice, options=[epochs])

    # 5. Create the parameter selector
    param_selector = SequenceTaggerParamSelector(
        corpus,
        "ner",
        base_path,
        training_runs=1,
        optimization_value=OptimizationValue.DEV_SCORE,
        tensorboard_logdir=base_path / "logs",
        save_model=True,
    )
    param_selector.tag_dictionary = label_dict

    # 6. Start the optimization
    param_selector.optimize(search_space, max_evals=1)

sentences_yet = False

sentences = False  # We know this case yet...

for window, epochs in [(60, 150)]:
    print(f"sentences is {sentences}")
    training(sentences, window, epochs)