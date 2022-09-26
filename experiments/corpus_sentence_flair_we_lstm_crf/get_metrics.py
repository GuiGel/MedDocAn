"""Compute the metrics for the meddocan tasks."""
from typer.testing import CliRunner

from meddocan.cli import app

runner = CliRunner()


if __name__ == "__main__":

    for SEED in [12]: # [1, 10, 25, 33, 42]:
        model = f"experiments/corpus_sentence_flair_we_lstm_crf/results_seed_{SEED}/final-model.pt"
        result = runner.invoke(
            app, ["eval", model, "evals", "--device", "cuda:1"]
        )
        print(result)
