"""Compute the metrics for the meddocan tasks."""
from typer.testing import CliRunner

from meddocan.cli import app

runner = CliRunner()


if __name__ == "__main__":

    for SEED in [1, 12, 33]:
        model = f"/home/wave/Project/MedDocAn/experiments/corpus_sentence_flair_models/an_wh_rs_True_dpt_0.08716810045694838_emb_seed_{SEED}_Stack(0_lm-es-forward.pt, 1_lm-es-backward.pt)_hdn_sz_256_lr_0.1_it_150_bs_4_opti_SGD_pjct_emb_True_rnn_ly_2_sdl_AnnealOnPlateau_use_crf_True_use_rnn_True/0/final-model.pt"
        result = runner.invoke(
            app, ["eval", model, "evals", "--device", "cuda:0"]
        )
        print(result)
