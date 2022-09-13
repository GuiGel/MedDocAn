#!/bin/sh -e
set -x

python experiments/corpus_sentence_flair_models/training_seed_1.py
python experiments/corpus_sentence_flair_models/training_seed_12.py
python experiments/corpus_sentence_flair_models/training_seed_33.py
python experiments/corpus_sentence_flair_models/get_metrics.py