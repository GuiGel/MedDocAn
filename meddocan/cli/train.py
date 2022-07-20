import logging
from enum import Enum
from pathlib import Path
from typing import List, Optional

import torch
from flair.embeddings import TransformerWordEmbeddings
from flair.models import SequenceTagger
from flair.optim import (
    ExpAnnealLR,
    LinearSchedulerWithWarmup,
    ReduceLROnPlateau,
)
from flair.trainers import ModelTrainer
from flair.training_utils import AnnealOnPlateau
from torch.optim.adamw import AdamW
from torch.optim.sgd import SGD

from meddocan.cli.utils import Arg, Opt, app
from meddocan.data.corpus import flair

logger = logging.getLogger(__name__)

COMMAND_NAME = "train-flert"


class Device(str, Enum):
    cpu = "cpu"
    gpu_0 = "cuda:0"
    gpu_1 = "cuda:1"


class Scheduler(str, Enum):
    ANNEAL_ON_PLATEAU = "AnnealOnPlateau"
    LINEAR_SCHEDULER_WITH_WARMUP = "LinearSchedulerWithWarmup"
    EXP_ANNEAL_LR = "ExpAnnealLR"
    REDUCE_LR_ON_PLATEAU = "ReduceLROnPlateau"


class Schedulers:
    AnnealOnPlateau = AnnealOnPlateau
    LinearSchedulerWithWarmup = LinearSchedulerWithWarmup
    ExpAnnealLR = ExpAnnealLR
    ExpAnnealLR = ReduceLROnPlateau


class Optimizer(str, Enum):
    SGD = "SGD"
    ADAMW = "AdamW"


class Optimizers:
    SGD = SGD
    AdamW = AdamW


@app.command(
    name=COMMAND_NAME,
)
def train_flert(
    # ---- GENERAL --------
    base_path: Path = Arg(
        ...,
        help="Main path to which all output during training is logged and models are saved",
    ),
    seed: int = Arg(..., help="Random seed."),
    device: Device = Arg(..., help="The device to use."),
    # ---- FLERT ----------
    model: str = Arg(
        ...,
        help="name of transformer model (see https://huggingface.co/transformers/pretrained_models.html for options).",
    ),
    # ---- TAGGER --------
    hidden_size: int = Opt(256, help="number of hidden states in RNN."),
    # ---- TRAINING -------
    learning_rate: float = Opt(
        5e-5,
        help="Initial learning rate (or max, if scheduler is OneCycleLR).",
    ),
    mini_batch_size: int = Opt(
        4, help="Size of mini-batches during training."
    ),
    mini_batch_chunk_size: Optional[int] = Opt(
        None,
        help=(
            "If mini-batches are larger than this number, they get broken "
            "down into chunks of this size for processing purposes."
        ),
    ),
    max_epochs: int = Opt(
        10,
        help=(
            "Maximum number of epochs to train. Terminates training if this "
            "number is surpassed."
        ),
    ),
    train_with_dev: bool = Opt(
        False,
        help="If True, the data from dev split is added to the training data.",
    ),
    train_with_test: bool = Opt(
        False,
        help=(
            "If True, the data from test split is added to the training data."
        ),
    ),
    monitor_train: bool = Opt(
        False,
        help=("If True, training data is evaluated at end of each epoch."),
    ),
    monitor_test: bool = Opt(
        False,
        help="If True, test data is evaluated at end of each epoch.",
    ),
    main_evaluation_metric: List[str] = Opt(
        ["micro avg", "f1-score"],
        help=(
            "Type of metric to use for best model tracking and learning rate "
            "scheduling (if dev data is available, otherwise loss will be "
            "used), currently only applicable for text_classification_model."
        ),
    ),
    scheduler: Scheduler = Opt(
        Scheduler.LINEAR_SCHEDULER_WITH_WARMUP,
        help="The learning rate scheduler to use.",
    ),
    anneal_factor: float = Opt(
        0.5, help="The factor by which the learning rate is annealed."
    ),
    patience: int = Opt(
        3,
        help=(
            "Patience is the number of epochs with no improvement the Trainer "
            "waits."
        ),
    ),
    min_learning_rate: float = Opt(
        0.0001,
        help=(
            "If the learning rate falls below this threshold, training "
            "terminates."
        ),
    ),
    initial_extra_patience: int = Opt(0, help=""),
    optimizer: Optimizer = Opt(
        Optimizer.ADAMW,
        help="The optimizer to use (typically SGD or Adam)",
    ),
    cycle_momentum: bool = Opt(
        False,
        help=(
            "If scheduler is OneCycleLR, whether the scheduler should cycle "
            "also the momentum"
        ),
    ),
    warmup_fraction: float = Opt(
        0.1,
        help=(
            "Fraction of warmup steps if the scheduler is "
            "LinearSchedulerWithWarmup"
        ),
    ),
    embeddings_storage_mode: str = Opt(
        "cpu",
        help=(
            "One of 'none' (all embeddings are deleted and freshly "
            "recomputed), 'cpu' (embeddings are stored on CPU) or 'gpu' "
            "(embeddings are stored on GPU)"
        ),
    ),
    checkpoint: bool = Opt(
        False,
        help="If True, a full checkpoint is saved at end of each epoch",
    ),
    save_final_model: bool = Opt(True, help="If True, final model is saved"),
    anneal_with_restarts: bool = Opt(
        False,
        help=(
            "If True, the last best model is restored when annealing the "
            "learning rate"
        ),
    ),
    anneal_with_prestarts: bool = Opt(
        False,
        help=(
            "If True, the last model before the best model is restored when "
            "annealing the learning rate"
        ),
    ),
    anneal_against_dev_loss: bool = Opt(False, help=""),
    batch_growth_annealing: bool = Opt(
        False,
        help=(
            "Multiply batch size by two when learning rate change. Batch "
            "growth with OneCycle policy is not implemented"
        ),
    ),
    shuffle: bool = Opt(True, help=""),
    param_selection_mode: bool = Opt(False, help=""),
    write_weights: bool = Opt(False, help="Write weights to file"),
    num_workers: int = Opt(6, help="Number of workers in your data loader."),
    sampler=Opt(
        None,
        help="You can pass a data sampler here for special sampling of data.",
    ),
    use_amp: bool = Opt(False, help="Use automatic mixed precision"),
    amp_opt_level: str = Opt(
        "O1", help="Level of apex pure of mixed precision training"
    ),
    eval_on_train_fraction: float = Opt(0.0, help=""),
    eval_on_train_shuffle: bool = Opt(False, help=""),
    save_model_each_k_epochs: int = Opt(
        0,
        help=(
            "Each k epochs, a model state will be written out. If set to '5', "
            "a model will be saved each 5 epochs. Default is 0 which means no "
            "model saving"
        ),
    ),
    tensorboard_comment: str = Opt(
        "", help="Comment to use for tensorboard logging"
    ),
    use_swa: bool = Opt(False, help="Use Sochatic Weight Averaging"),
    use_final_model_for_eval: bool = Opt(True, help=""),
    # gold_label_dictionary_for_eval: Optional[Dictionary] = Opt(None),
    create_file_logs: bool = Opt(
        True,
        help=(
            "If True, the logs will also be stored in a file 'training.log' "
            "in the model folder"
        ),
    ),
    create_loss_file: bool = Opt(
        True,
        help=(
            "If True, the loss will be written to a file 'loss.tsv' in the "
            "model folder"
        ),
    ),
    epoch: int = Opt(
        0,
        help=(
            "The starting epoch (normally 0 but could be higher if you "
            "continue training model)"
        ),
    ),
    use_tensorboard: bool = Opt(
        False, help="Number of workers in your data loader."
    ),
    tensorboard_log_dir=Opt(
        None,
        help="Directory into which tensorboard log files will be written",
    ),
    metrics_for_tensorboard: List[str] = Opt(
        [],
        help=(
            "List of str that specify which metrics (in addition to the "
            "main_score) shall be plotted in tensorboard, could be "
            "[('macro avg | f1-score'), ('macro avg | precision')] for example"
        ),
    ),
    # optimizer_state_dict: Optional = Opt(None, help=""),
    # scheduler_state_dict: Optional = Opt(None, help=""),
    save_optimizer_state: bool = Opt(False, help=""),
    # ----- Optimizer
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
        model=model,  # "dccuchile/bert-base-spanish-wwm-cased"
        layers="-1",
        subtoken_pooling="first",
        fine_tune=True,
        use_context=True,
        # ----- Transformer Tokenizer
        model_max_length=4096,  # Cf https://huggingface.co/docs/transformers/main_classes/tokenizer#transformers.PreTrainedTokenizer.model_max_length
        fast_tokenizer=True,
    )

    # 5. initialize bare-bones sequence tagger (no CRF, no RNN, no reprojection)
    tagger = SequenceTagger(
        hidden_size=hidden_size,
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
    trainer.train(
        # ----- FINE TUNE
        base_path=base_path,
        learning_rate=learning_rate,
        max_epochs=max_epochs,
        optimizer=getattr(Optimizers, optimizer.value),
        scheduler=getattr(Schedulers, scheduler.value),
        warmup_fraction=warmup_fraction,
        mini_batch_size=mini_batch_size,
        mini_batch_chunk_size=mini_batch_chunk_size,
        embeddings_storage_mode=embeddings_storage_mode,
        use_final_model_for_eval=use_final_model_for_eval,
        # ----- TRAIN
        train_with_dev=train_with_dev,
        train_with_test=train_with_test,
        monitor_train=monitor_train,
        monitor_test=monitor_test,
        patience=patience,
        main_evaluation_metric=main_evaluation_metric,
        anneal_factor=anneal_factor,
        min_learning_rate=min_learning_rate,
        checkpoint=checkpoint,
        initial_extra_patience=initial_extra_patience,
        cycle_momentum=cycle_momentum,
        save_final_model=save_final_model,
        anneal_with_restarts=anneal_with_restarts,
        batch_growth_annealing=batch_growth_annealing,
        anneal_with_prestarts=anneal_with_prestarts,
        anneal_against_dev_loss=anneal_against_dev_loss,
        shuffle=shuffle,
        param_selection_mode=param_selection_mode,
        write_weights=write_weights,
        use_amp=use_amp,
        num_workers=num_workers,
        sampler=sampler,
        amp_opt_level=amp_opt_level,
        eval_on_train_fraction=eval_on_train_fraction,
        eval_on_train_shuffle=eval_on_train_shuffle,
        save_model_each_k_epochs=save_model_each_k_epochs,
        tensorboard_comment=tensorboard_comment,
        use_swa=use_swa,
        # gold_label_dictionary_for_eval=gold_label_dictionary_for_eval,
        create_file_logs=create_file_logs,
        create_loss_file=create_loss_file,
        epoch=epoch,
        use_tensorboard=use_tensorboard,
        tensorboard_log_dir=tensorboard_log_dir,
        metrics_for_tensorboard=metrics_for_tensorboard,
        save_optimizer_state=save_optimizer_state,
    )
