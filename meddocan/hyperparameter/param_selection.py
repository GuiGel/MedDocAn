import logging
from abc import abstractmethod
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, NamedTuple, Union

import flair.nn
import numpy as np
from flair.data import Corpus
from flair.embeddings import StackedEmbeddings
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer
from flair.training_utils import EvaluationMetric, init_output_file, log_line
from hyperopt import fmin, hp, tpe

from meddocan.hyperparameter.parameter import (
    SEQUENCE_TAGGER_PARAMETERS,
    TRAINING_PARAMETERS,
    Parameter,
)
from meddocan.hyperparameter.utils import (
    get_model_card,
    get_tensorboard_dirname,
)

log = logging.getLogger("flair")


class OptimizationValue(Enum):
    DEV_LOSS = "loss"
    DEV_SCORE = "score"


class HpMetrics(NamedTuple):
    hp_dev_score: float
    hp_dev_score_var: float
    hp_test_score: float


class SearchSpace(object):
    def __init__(self) -> None:
        self.search_space: Dict[str, Any] = {}

    def add(
        self, parameter: Parameter, func: Callable[..., Any], **kwargs
    ) -> None:
        self.search_space[parameter.value] = func(parameter.value, **kwargs)

    def get_search_space(self):
        return hp.choice("parameters", [self.search_space])


class ParamSelector(object):
    def __init__(
        self,
        corpus: Corpus,
        base_path: Union[str, Path],
        evaluation_metric: EvaluationMetric,
        training_runs: int,
        optimization_value: OptimizationValue,
        tensorboard_logdir: Union[str, Path] = None,
    ) -> None:
        if isinstance(base_path, str):
            base_path = Path(base_path)

        self.corpus = corpus
        # self.max_epochs = max_epochs
        self.base_path = base_path
        self.evaluation_metric = evaluation_metric
        self.run = 1
        self.training_runs = training_runs
        self.optimization_value = optimization_value

        self.param_selection_file = init_output_file(
            base_path, "param_selection.txt"
        )
        self.tensorboard_logdir = tensorboard_logdir

    @abstractmethod
    def _set_up_model(self, params: dict) -> flair.nn.Model:
        pass

    def _objective(self, params: dict) -> Dict[str, Any]:
        log_line(log)
        log.info(f"Evaluation run: {self.run}")
        log.info("Evaluating parameter combination:")
        for k, v in params.items():
            if isinstance(v, tuple):
                v = ",".join([str(x) for x in v])
            log.info(f"\t{k}: {str(v)}")
        log_line(log)

        scores = []
        vars = []
        test_score = []

        for i in range(0, self.training_runs):
            log_line(log)
            log.info(f"Training run: {i + 1}")

            for sent in self.corpus.get_all_sentences():  # type: ignore
                sent.clear_embeddings()

            model = self._set_up_model(params)

            training_params = {
                key: params[key]
                for key in params
                if key in TRAINING_PARAMETERS
            }

            trainer: ModelTrainer = ModelTrainer(model, self.corpus)

            if self.tensorboard_logdir is None:
                tbd_base_dir = self.base_path / "tensorboard_logdir"
            else:
                tbd_base_dir = Path(self.tensorboard_logdir)

            tbd_training_name = get_tensorboard_dirname(params)
            tbd_log_dir = tbd_base_dir / tbd_training_name

            tensorboard_comment = "Grid Search"
            if not tbd_log_dir.exists():
                tbd_log_dir.mkdir(parents=True)

            result = trainer.train(
                self.base_path,
                param_selection_mode=True,
                tensorboard_comment=tensorboard_comment,
                use_tensorboard=True,
                tensorboard_log_dir=tbd_log_dir / f"run_{i}",
                metrics_for_tensorboard=[
                    ("micro avg", "f1-score"),
                    ("micro avg", "precision"),
                    ("micro avg", "recall"),
                    ("macro avg", "f1-score"),
                    ("macro avg", "precision"),
                    ("macro avg", "recall"),
                ],
                **training_params,
            )

            from torch.utils.tensorboard import SummaryWriter

            with SummaryWriter(
                log_dir=tbd_log_dir,
                comment=tensorboard_comment,
            ) as writer:

                # ------- Write model card
                model_card = get_model_card(trainer.model)
                writer.add_text("model_card", model_card)

                # ------ Average the last three training scores
                if self.optimization_value == OptimizationValue.DEV_LOSS:
                    curr_scores = result["dev_loss_history"][-3:]
                else:
                    curr_scores = list(
                        map(
                            lambda s: 1 - s,
                            result["dev_score_history"][-3:],
                        )
                    )

                # ----- Compute scores for the current training run
                score = np.mean(curr_scores)
                var = np.var(curr_scores)
                scores.append(score)
                vars.append(var)
                test_score.append(result["test_score"])

                # ----- Tensorboard logs
                def get_hparam_dict(params) -> dict:
                    hparam_dict = {}
                    for key, val in params.items():
                        if type(val) in [str, int, float]:
                            _val = val
                        else:
                            if hasattr(val, "__name__"):
                                _val = val.__name__
                            elif hasattr(val, "name"):

                                def get_emb_name(emb):
                                    return emb.name.split("/")[-1]

                                if isinstance(val, StackedEmbeddings):
                                    e = ", ".join(
                                        [
                                            f"{i}_{get_emb_name(emb)}"
                                            for i, emb in enumerate(
                                                val.embeddings
                                            )
                                        ]
                                    )
                                    _val = f"{val.name}({e})"
                                else:
                                    _val = f"{get_emb_name(val)}"
                            else:
                                _val = val
                        hparam_dict[key] = str(_val)
                    return hparam_dict

                # ----- Hyperopt minimize
                if self.optimization_value != OptimizationValue.DEV_LOSS:
                    hp_metrics = HpMetrics(
                        1 - score, var, result["test_score"]
                    )
                else:
                    hp_metrics = HpMetrics(score, var, result["test_score"])

                # for epoch, epoch_loss in enumerate(result["dev_loss_history"]):
                #     writer.add_scalar("dev_loss", epoch_loss, epoch + 1)
                # for epoch, epoch_score in enumerate(result["dev_score_history"]):
                #     writer.add_scalar("dev_score", epoch_loss, epoch + 1)

                # ----- Add hyperparameters to tensorboard
                writer.add_hparams(
                    get_hparam_dict(params),
                    hp_metrics._asdict(),
                    run_name=f"run_{i}",
                )

        # ------ Take average over the scores from the different training runs
        final_score = np.mean(scores)
        final_var = np.var(scores)
        final_test_score = sum(test_score) / float(len(test_score))

        hp_metrics = HpMetrics(1 - final_score, final_var, final_test_score)

        # ----- Write the final runs result to tensorboard

        with SummaryWriter(
            log_dir=tbd_log_dir,
            comment=tensorboard_comment,
        ) as writer:
            writer.add_hparams(
                get_hparam_dict(params),
                hp_metrics._asdict(),
                run_name="final",
            )

        # ----- Log results
        test_score = result["test_score"]
        log_line(log)
        log.info("Done evaluating parameter combination:")
        for k, v in params.items():
            if isinstance(v, tuple):
                v = ",".join([str(x) for x in v])
            log.info(f"\t{k}: {v}")
        log.info(f"{self.optimization_value.value}: {final_score}")
        log.info(f"variance: {final_var}")
        log.info(f"test_score: {test_score}\n")
        log_line(log)

        with open(self.param_selection_file, "a") as f:
            f.write(f"evaluation run {self.run}\n")
            for k, v in params.items():
                if isinstance(v, tuple):
                    v = ",".join([str(x) for x in v])
                f.write(f"\t{k}: {str(v)}\n")
            f.write(f"{self.optimization_value.value}: {final_score}\n")
            f.write(f"variance: {final_var}\n")
            f.write(f"test_score: {test_score}\n")
            f.write("-" * 100 + "\n")

        self.run += 1

        return {
            "status": "ok",
            "loss": final_score,
            "loss_variance": final_var,
        }

    def optimize(self, space: SearchSpace, max_evals=100) -> None:
        search_space = space.search_space
        best = fmin(
            self._objective,
            search_space,
            algo=tpe.suggest,
            max_evals=max_evals,
        )

        log_line(log)
        log.info("Optimizing parameter configuration done.")
        log.info("Best parameter configuration found:")
        for k, v in best.items():
            log.info(f"\t{k}: {v}")
        log_line(log)

        with open(self.param_selection_file, "a") as f:
            f.write("best parameter combination\n")
            for k, v in best.items():
                if isinstance(v, tuple):
                    v = ",".join([str(x) for x in v])
                f.write(f"\t{k}: {str(v)}\n")


class SequenceTaggerParamSelector(ParamSelector):
    def __init__(
        self,
        corpus: Corpus,
        tag_type: str,
        base_path: Union[str, Path],
        evaluation_metric: EvaluationMetric = EvaluationMetric.MICRO_F1_SCORE,
        training_runs: int = 1,
        optimization_value: OptimizationValue = OptimizationValue.DEV_LOSS,
        tensorboard_logdir: Union[str, Path] = None,
    ) -> None:
        """
        :param corpus: the corpus
        :param tag_type: tag type to use
        :param base_path: the path to the result folder (results will be written to that folder)
        :param max_epochs: number of epochs to perform on every evaluation run
        :param evaluation_metric: evaluation metric used during training
        :param training_runs: number of training runs per evaluation run
        :param optimization_value: value to optimize
        :param tensorboard_logdir: Tensorboard log folder name. The logs are
            located in base_path / tensorboard_logdir if a not `logdir` is
            given else use tensorboard_logdir.
        """
        super().__init__(
            corpus,
            base_path,
            evaluation_metric,
            training_runs,
            optimization_value,
            tensorboard_logdir=tensorboard_logdir,
        )

        self.tag_type = tag_type
        self.tag_dictionary = self.corpus.make_label_dictionary(self.tag_type)

    def _set_up_model(self, params: dict):
        sequence_tagger_params = {
            key: params[key]
            for key in params
            if key in SEQUENCE_TAGGER_PARAMETERS
        }

        tagger: SequenceTagger = SequenceTagger(
            tag_dictionary=self.tag_dictionary,
            tag_type=self.tag_type,
            **sequence_tagger_params,
        )
        return tagger
