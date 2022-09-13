# %% [markdown]
# Train FLair NER model on the Meddocan corpus sentence by sentences.
# %%
from pathlib import Path

import torch
from flair.embeddings import FlairEmbeddings, StackedEmbeddings, WordEmbeddings
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer

from meddocan.data.corpus import flair

seed = 33

flair.set_seed(seed)
flair.device = torch.device("cuda:1")

# 1. get the corpus
corpus = flair.datasets.MEDDOCAN(sentences=True)
print(corpus)

# 2. what label do we want to predict?
label_type = "ner"

# 3. make the label dictionary from the corpus
label_dict = corpus.make_label_dictionary(label_type=label_type)
print(label_dict)

# 4. initialize embedding stack with Flair and GloVe
embedding_types = [
    WordEmbeddings("es", stable=True),
    FlairEmbeddings("spanish-forward"),
    FlairEmbeddings("spanish-backward"),
]

embeddings = StackedEmbeddings(embeddings=embedding_types)

# 5. initialize sequence tagger
tagger = SequenceTagger(
    hidden_size=256,
    embeddings=embeddings,
    tag_dictionary=label_dict,
    tag_type=label_type,
    use_crf=True,
)

# 6. initialize trainer
trainer = ModelTrainer(tagger, corpus)

# 7. start training
trainer.train(
    Path(__file__).parent / f"results_seed_{seed}",
    max_epochs=150,
    embeddings_storage_mode="gpu",
)
