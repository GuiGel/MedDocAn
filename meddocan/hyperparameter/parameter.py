from enum import Enum


class Parameter(Enum):
    EMBEDDINGS = "embeddings"
    HIDDEN_SIZE = "hidden_size"
    USE_CRF = "use_crf"
    USE_RNN = "use_rnn"
    REPROJECT_EMBEDDINGS = "reproject_embeddings"
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
    MINI_BATCH_CHUNK_SIZE = "mini_batch_chunk_size"
    NUM_WORKERS = "num_workers"
    EMBEDDINGS_STORAGE_MODE = "embeddings_storage_mode"


class ParameterName(Enum):
    embeddings = "emb"
    hidden_size = "hdn_sz"
    use_crf = "use_crf"
    use_rnn = "use_rnn"
    reproject_embeddings = "pjct_emb"
    rnn_layers = "rnn_ly"
    dropout = "dpt"
    word_dropout = "wd_dpt"
    locked_dropout = "lck_dpt"
    learning_rate = "lr"
    warmup_fraction = "wup"
    mini_batch_size = "bs"
    anneal_factor = "an_fr"
    anneal_with_restarts = "an_wh_rs"
    anneal_with_prestarts = "an_wh_ps"
    batch_growth_annealing = "bs_gh_an"
    patience = "pa"
    optimizer = "opti"
    momentum = "momentum"
    dampening = "damp"
    weight_decay = "wt_dcy"
    nesterov = "nstv"
    amsgrad = "amsgd"
    betas = "betas"
    eps = "eps"
    model = "model"
    LAYERS = "LAYERS"
    scheduler = "sdl"
    use_amp = "amp"
    mini_batch_chunk_size = None
    num_workers = None
    embeddings_storage_mode = None


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
    Parameter.MINI_BATCH_CHUNK_SIZE.value,
    Parameter.NUM_WORKERS.value,
    Parameter.EMBEDDINGS_STORAGE_MODE.value,
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
    Parameter.REPROJECT_EMBEDDINGS.value,
]


TEXT_CLASSIFICATION_PARAMETERS = [
    Parameter.LAYERS.value,
    Parameter.TRANSFORMER_MODEL.value,
]

if __name__ == "__main__":
    print(getattr(ParameterName, "mini_batch_size").value)
