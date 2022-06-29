import logging
from abc import abstractmethod
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Union

import flair.nn
import numpy as np
from flair.data import Corpus
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer
from flair.training_utils import EvaluationMetric, init_output_file, log_line
from hyperopt import fmin, hp, tpe

from meddocan.hyperparameter.parameter import (SEQUENCE_TAGGER_PARAMETERS,
                                               TRAINING_PARAMETERS, Parameter)
from meddocan.hyperparameter.utils import get_tensorboard_dirname

log = logging.getLogger("flair")


class OptimizationValue(Enum):
    DEV_LOSS = "loss"
    DEV_SCORE = "score"


class SearchSpace(object):
    def __init__(self) -> None:
        self.search_space = {}

    def add(self, parameter: Parameter, func, **kwargs) -> None:
        self.search_space[parameter.value] = func(parameter.value, **kwargs)

    def get_search_space(self):
        return hp.choice("parameters", [self.search_space])


class ParamSelector(object):
    def __init__(
        self,
        corpus: Corpus,
        base_path: Union[str, Path],
        max_epochs: int,
        evaluation_metric: EvaluationMetric,
        training_runs: int,
        optimization_value: OptimizationValue,
    ) -> None:
        if isinstance(base_path, str):
            base_path = Path(base_path)

        self.corpus = corpus
        self.max_epochs = max_epochs
        self.base_path = base_path
        self.evaluation_metric = evaluation_metric
        self.run = 1
        self.training_runs = training_runs
        self.optimization_value = optimization_value

        self.param_selection_file = init_output_file(
            base_path, "param_selection.txt"
        )

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

            tbd_base_dir = self.base_path / "tensorboard_logdir"
            tbd_training_name = get_tensorboard_dirname(params)
            tbd_log_dir = tbd_base_dir / tbd_training_name

            tensorboard_comment = "Grid Search"
            if not tbd_log_dir.exists():
                tbd_log_dir.mkdir(parents=True)

            result = trainer.train(
                self.base_path,
                max_epochs=self.max_epochs,
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
                embeddings_storage_mode="cpu",
                **training_params,
            )

            from torch.utils.tensorboard import SummaryWriter

            with SummaryWriter(
                log_dir=tbd_log_dir,
                comment=tensorboard_comment,
            ) as writer:

                # ------ Take the average over the last three scores of training
                if self.optimization_value == OptimizationValue.DEV_LOSS:
                    curr_scores = result["dev_loss_history"][-3:]
                else:
                    curr_scores = list(
                        map(lambda s: 1 - s, result["dev_score_history"][-3:])
                    )

                # ----- Compute scores for the current training run
                score = sum(curr_scores) / float(len(curr_scores))
                var = np.var(curr_scores)
                scores.append(score)
                vars.append(var)
                test_score.append(result["test_score"])

                # ----- Tensorboard logs
                def get_hparam_dict(params) -> dict:
                    hparam_dict = {}
                    for key, val in params.items():
                        if not type(val) in [str, int, float]:
                            _val = str(params[key])
                        else:
                            _val = val
                        hparam_dict[key] = str(_val)
                    return hparam_dict

                # ----- Hyperopt minimize
                if self.optimization_value != OptimizationValue.DEV_LOSS:
                    metrics_dict = {
                        "score_dev": 1 - score,
                        "score_dev_var": var,
                        "score_test": result["test_score"],
                    }
                else:
                    metrics_dict = {
                        "score_dev": score,
                        "score_dev_var": var,
                        "test_score": result["test_score"],
                    }

                # for epoch, epoch_loss in enumerate(result["dev_loss_history"]):
                #     writer.add_scalar("dev_loss", epoch_loss, epoch + 1)
                # for epoch, epoch_score in enumerate(result["dev_score_history"]):
                #     writer.add_scalar("dev_score", epoch_loss, epoch + 1)

                # ----- Add hyperparameters to tensorboard
                writer.add_hparams(
                    get_hparam_dict(params),
                    metrics_dict,
                    run_name=f"run_{i}",
                )

        # ------ Take average over the scores from the different training runs
        final_score = sum(scores) / float(len(scores))
        final_var = sum(vars) / float(len(vars))
        final_test_score = sum(test_score) / float(len(test_score))

        metrics_dict = {
            "score_dev": 1 - final_score,
            "score_dev_var": final_var,
            "test_score": final_test_score,
        }

        # ----- Write the final runs result to tensorboard

        with SummaryWriter(
            log_dir=tbd_log_dir,
            comment=tensorboard_comment,
        ) as writer:
            writer.add_hparams(
                get_hparam_dict(params),
                metrics_dict,
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
        max_epochs: int = 50,
        evaluation_metric: EvaluationMetric = EvaluationMetric.MICRO_F1_SCORE,
        training_runs: int = 1,
        optimization_value: OptimizationValue = OptimizationValue.DEV_LOSS,
    ) -> None:
        """
        :param corpus: the corpus
        :param tag_type: tag type to use
        :param base_path: the path to the result folder (results will be written to that folder)
        :param max_epochs: number of epochs to perform on every evaluation run
        :param evaluation_metric: evaluation metric used during training
        :param training_runs: number of training runs per evaluation run
        :param optimization_value: value to optimize
        """
        super().__init__(
            corpus,
            base_path,
            max_epochs,
            evaluation_metric,
            training_runs,
            optimization_value,
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
