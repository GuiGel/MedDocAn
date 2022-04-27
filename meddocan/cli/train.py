import logging
from enum import Enum
from pathlib import Path

import torch
from flair.embeddings import TransformerWordEmbeddings
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer

from meddocan.data.corpus import flair

from .utils import Arg, app

logger = logging.getLogger(__name__)

COMMAND_NAME = "train-flert"


class Device(str, Enum):
    cpu = "cpu"
    gpu_0 = "cuda:0"
    gpu_1 = "cuda:1"


@app.command(
    name=COMMAND_NAME,
)
def train_flert(
    seed: int = Arg(..., help="Random seed"),
    device: Device = Arg(..., help="The device to use"),
) -> None:

    flair.set_seed(seed)
    flair.device = torch.device(device.value)

    # 1. get the corpus
    corpus = flair.datasets.MEDDOCAN(sentences=True)
    logger.info(corpus)

    # 2. what label do we want to predict?
    label_type = "ner"

    # 3. make the label dictionary from the corpus
    label_dict = corpus.make_label_dictionary(label_type=label_type)
    logger.info(label_dict)

    # 4. initialize fine-tuneable transformer embeddings WITH document context
    embeddings = TransformerWordEmbeddings(
        model="dccuchile/bert-base-spanish-wwm-cased",
        layers="-1",
        subtoken_pooling="first",
        fine_tune=True,
        use_context=True,
    )

    # 5. initialize bare-bones sequence tagger (no CRF, no RNN, no reprojection)
    tagger = SequenceTagger(
        hidden_size=256,
        embeddings=embeddings,
        tag_dictionary=label_dict,
        tag_type="ner",
        use_crf=False,
        use_rnn=False,
        reproject_embeddings=False,
    )

    # 6. initialize trainer
    trainer = ModelTrainer(tagger, corpus)

    # 7. run fine-tuning
    trainer.fine_tune(
        Path(__file__).parent / f"results_seed_{seed}",
        max_epochs=150,
        learning_rate=5.0e-6,
        mini_batch_size=4,
        # Remove this parameter to speed up computation if you have a big GPU
        # mini_batch_chunk_size=1,
        monitor_train=True,
        monitor_test=True,
    )
