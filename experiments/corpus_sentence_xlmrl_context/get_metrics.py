"""Compute the metrics for the meddocan tasks."""
from typer.testing import CliRunner

from meddocan.cli import app

runner = CliRunner()


if __name__ == "__main__":

    for SEED in [1, 12, 33]:
        model = f"/home/wave/Project/MedDocAn/experiments/corpus_sentence_grid_search_xlm-roberta_docstart/an_wh_rs_False_dpt_0_emb_xlm-roberta-large-cased-context_FT_True_Ly_-1_seed_{SEED}_lr_5e-06_it_150_bs_4_opti_AdamW_pjct_emb_False_sdl_LinearSchedulerWithWarmup_use_crf_False_use_rnn_False_wup_0.1/0/final-model.pt"
        result = runner.invoke(
            app, ["eval", model, "evals" "--device", "cuda:1"]
        )
        print(result)