#!/bin/sh -e
set -x

# python experiments/corpus_sentence_bert_context_we_finetune_it_150/code_1.py
# python experiments/corpus_sentence_bert_context_we_finetune_it_150/code_12.py
python experiments/corpus_sentence_bert_context_we_finetune_it_150/code_33.py
python experiments/corpus_sentence_bert_context_we_finetune_it_150/get_metrics.py
