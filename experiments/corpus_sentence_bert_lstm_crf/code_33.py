import json
from pathlib import Path

import flair
import torch
from flair.data import Corpus
from flair.embeddings import TransformerWordEmbeddings
from flair.training_utils import AnnealOnPlateau
from hyperopt import hp
from torch.optim import SGD

from meddocan.data.corpus import MEDDOCAN
from meddocan.hyperparameter.param_selection import (
    OptimizationValue,
    Parameter,
    SearchSpace,
    SequenceTaggerParamSelector,
)
from meddocan.hyperparameter.parameter import Parameter

base_path = Path(__file__).parent

SEED = 33

flair.set_seed(SEED)
flair.device = torch.device("cuda:1")

# 1. get the corpus
corpus: Corpus = MEDDOCAN(sentences=True)
print(corpus)

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
            fine_tune=False,
            layers="all",
            use_context=False,
            layer_mean=True,
            name=f"beto_Ly_all_mean_seed_{SEED}",
            subtoken_pooling="first",
            allow_long_sentences=True,
        ),
    ],
)
search_space.add(Parameter.USE_CRF, hp.choice, options=[True])
search_space.add(Parameter.USE_RNN, hp.choice, options=[True])
search_space.add(Parameter.RNN_LAYERS, hp.choice, options=[2])
search_space.add(Parameter.HIDDEN_SIZE, hp.choice, options=[256])
search_space.add(Parameter.REPROJECT_EMBEDDINGS, hp.choice, options=[False])
search_space.add(Parameter.NUM_WORKERS, hp.choice, options=[4])
search_space.add(Parameter.DROPOUT, hp.choice, options=[0])
search_space.add(Parameter.LEARNING_RATE, hp.choice, options=[0.1])
search_space.add(Parameter.MINI_BATCH_SIZE, hp.choice, options=[4])
search_space.add(Parameter.ANNEAL_WITH_RESTARTS, hp.choice, options=[False])
search_space.add(Parameter.OPTIMIZER, hp.choice, options=[SGD])
search_space.add(Parameter.SCHEDULER, hp.choice, options=[AnnealOnPlateau])
# search_space.add(Parameter.WARMUP_FRACTION, hp.choice, options=[0.1])
search_space.add(Parameter.EMBEDDINGS_STORAGE_MODE, hp.choice, options=["gpu"])
search_space.add(Parameter.MAX_EPOCHS, hp.choice, options=[500])

# 5. Create the parameter selector
param_selector = SequenceTaggerParamSelector(
    corpus,
    "ner",
    base_path,
    training_runs=1,
    optimization_value=OptimizationValue.DEV_SCORE,
    tensorboard_logdir=Path(__file__).parent / "logs",
    save_model=True,
)
param_selector.tag_dictionary = label_dict

# 6. Start the optimization
param_selector.optimize(search_space, max_evals=1)
