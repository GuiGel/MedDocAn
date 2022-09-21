"""Compute the metrics for the meddocan tasks."""
from typer.testing import CliRunner

from meddocan.cli import app

runner = CliRunner()


if __name__ == "__main__":

    for SEED in [1, 12, 33]:
        model = f"experiments/corpus_sentence_bert_lstm_crf/an_wh_rs_False_dpt_0_emb_beto_Ly_all_mean_seed_{SEED}_hdn_sz_256_lr_0.1_it_500_bs_4_opti_SGD_pjct_emb_False_rnn_ly_2_sdl_AnnealOnPlateau_use_crf_True_use_rnn_True/0/final-model.pt"
        result = runner.invoke(
            app, ["eval", model, "evals", "--device", "cuda:1"]
        )
        print(result)
