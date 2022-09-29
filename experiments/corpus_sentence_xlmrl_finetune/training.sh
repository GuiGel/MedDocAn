#!/bin/sh -e
set -x

python experiments/corpus_sentence_xlmrl_finetune/training_seed_1.py
python experiments/corpus_sentence_xlmrl_finetune/training_seed_12.py
python experiments/corpus_sentence_xlmrl_finetune/training_seed_33.py
python experiments/corpus_sentence_xlmrl_finetune/get_metrics.py
