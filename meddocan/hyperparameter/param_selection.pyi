import abc
from enum import Enum
from flair.data import Corpus as Corpus
from flair.training_utils import EvaluationMetric
from meddocan.hyperparameter.parameter import Parameter
from pathlib import Path
from typing import Any, Callable, NamedTuple, Union

log: Any

class OptimizationValue(Enum):
    DEV_LOSS: str
    DEV_SCORE: str

class HpMetrics(NamedTuple):
    hp_dev_score: float
    hp_dev_score_var: float
    hp_test_score: float

class SearchSpace:
    search_space: Any
    def __init__(self) -> None: ...
    def add(self, parameter: Parameter, func: Callable[..., Any], **kwargs) -> None: ...
    def get_search_space(self): ...

class ParamSelector(metaclass=abc.ABCMeta):
    corpus: Any
    base_path: Any
    evaluation_metric: Any
    run: int
    training_runs: Any
    optimization_value: Any
    param_selection_file: Any
    tensorboard_logdir: Any
    def __init__(self, corpus: Corpus, base_path: Union[str, Path], evaluation_metric: EvaluationMetric, training_runs: int, optimization_value: OptimizationValue, tensorboard_logdir: Union[str, Path] = ...) -> None: ...
    def optimize(self, space: SearchSpace, max_evals: int = ...) -> None: ...

class SequenceTaggerParamSelector(ParamSelector):
    tag_type: Any
    tag_dictionary: Any
    def __init__(self, corpus: Corpus, tag_type: str, base_path: Union[str, Path], evaluation_metric: EvaluationMetric = ..., training_runs: int = ..., optimization_value: OptimizationValue = ..., tensorboard_logdir: Union[str, Path] = ...) -> None: ...
