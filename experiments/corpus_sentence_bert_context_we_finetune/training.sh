#!/bin/sh -e
set -x

python experiments/corpus_sentence_bert_context_we_finetune/code_1.py
python experiments/corpus_sentence_bert_context_we_finetune/code_12.py
python experiments/corpus_sentence_bert_context_we_finetune/code_33.py
python experiments/corpus_sentence_bert_context_we_finetune/get_metrics.py
