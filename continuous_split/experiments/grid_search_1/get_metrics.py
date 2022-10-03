"""Compute the metrics for the meddocan tasks."""

from typer.testing import CliRunner
from pathlib import Path
from meddocan.cli import app

runner = CliRunner()

if __name__ == "__main__":
    for folder in Path("continuous_split/experiments/grid_search_1").iterdir():
        if folder.is_dir():
            model = folder / "0" / "final-model.pt"
            result = runner.invoke(
                app, ["eval", model, "evals", "--device", "cuda:0"]
            )