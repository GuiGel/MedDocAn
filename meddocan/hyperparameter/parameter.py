from enum import Enum


class Parameter(Enum):
    EMBEDDINGS = "embeddings"
    HIDDEN_SIZE = "hidden_size"
    USE_CRF = "use_crf"
    USE_RNN = "use_rnn"
    RNN_LAYERS = "rnn_layers"
    DROPOUT = "dropout"
    WORD_DROPOUT = "word_dropout"
    LOCKED_DROPOUT = "locked_dropout"
    LEARNING_RATE = "learning_rate"
    WARMUP_FRACTION = "warmup_fraction"
    MINI_BATCH_SIZE = "mini_batch_size"
    ANNEAL_FACTOR = "anneal_factor"
    ANNEAL_WITH_RESTARTS = "anneal_with_restarts"
    ANNEAL_WITH_PRESTARTS = "anneal_with_prestarts"
    BATCH_GROWTH_ANNEALING = "batch_growth_annealing"
    PATIENCE = "patience"
    OPTIMIZER = "optimizer"
    MOMENTUM = "momentum"
    DAMPENING = "dampening"
    WEIGHT_DECAY = "weight_decay"
    NESTEROV = "nesterov"
    AMSGRAD = "amsgrad"
    BETAS = "betas"
    EPS = "eps"
    TRANSFORMER_MODEL = "model"
    LAYERS = "LAYERS"
    SCHEDULER = "scheduler"
    USE_AMP = "use_amp"


TRAINING_PARAMETERS = [
    Parameter.LEARNING_RATE.value,
    Parameter.MINI_BATCH_SIZE.value,
    Parameter.OPTIMIZER.value,
    Parameter.ANNEAL_FACTOR.value,
    Parameter.PATIENCE.value,
    Parameter.ANNEAL_WITH_RESTARTS.value,
    Parameter.MOMENTUM.value,
    Parameter.DAMPENING.value,
    Parameter.WEIGHT_DECAY.value,
    Parameter.NESTEROV.value,
    Parameter.AMSGRAD.value,
    Parameter.BETAS.value,
    Parameter.EPS.value,
    Parameter.SCHEDULER.value,
    Parameter.WARMUP_FRACTION.value,
    Parameter.ANNEAL_WITH_PRESTARTS.value,
    Parameter.BATCH_GROWTH_ANNEALING.value,
    Parameter.USE_AMP.value,
]


SEQUENCE_TAGGER_PARAMETERS = [
    Parameter.EMBEDDINGS.value,
    Parameter.HIDDEN_SIZE.value,
    Parameter.RNN_LAYERS.value,
    Parameter.USE_CRF.value,
    Parameter.USE_RNN.value,
    Parameter.DROPOUT.value,
    Parameter.LOCKED_DROPOUT.value,
    Parameter.WORD_DROPOUT.value,
]


TEXT_CLASSIFICATION_PARAMETERS = [
    Parameter.LAYERS.value,
    Parameter.TRANSFORMER_MODEL.value,
]