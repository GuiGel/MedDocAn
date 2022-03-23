# %%
import spacy
from meddocan.data.corpus import flair
from meddocan.language.tokenizer import meddocan_tokenizer
from meddocan.data import meddocan_zip
from flair.datasets import ColumnCorpus

corpus: ColumnCorpus = flair.datasets.MEDDOCAN(sentences=False)
print(corpus.obtain_statistics("ner"))
# %%
print(corpus)
# %%
corpus: ColumnCorpus = flair.datasets.MEDDOCAN(sentences=True)
print(corpus.obtain_statistics("ner"))
# %%
print(corpus)
# %%
for sentences in corpus.test:
    print(sentences)
# %%
from meddocan.data.brat_containers import brat_annotations_gen
from meddocan.data import ZipFolder

lines_numbers = 0
for brat_annotations in brat_annotations_gen(ZipFolder.train, sep=""):
    lines_numbers += len(brat_annotations)
print(lines_numbers)
# %%
from zipfile import ZipFile
from zipfile import Path as ZipPath
from pathlib import Path
from meddocan.data import meddocan_zip as zipfile_path
from itertools import tee

zip_path = ZipPath(zipfile_path.train)
train_brat_path = zip_path / "train" / "brat"
files_1, files_2 = tee(train_brat_path.iterdir(), 2)
ann_files = [file for file in files_1 if Path(file.name).suffix == ".ann"]
txt_files = [file for file in files_2 if Path(file.name).suffix == ".txt"]
for ann_file, txt_file in zip(ann_files, txt_files):
    # The file name must be the same to be sure that the brat annotation
    # correspond to the wright file.
    assert Path(ann_file.name).stem == Path(txt_file.name).stem

# %%
from pathlib import Path

with Path(".test.txt").open(mode="w", encoding="utf-8", newline="\n") as f:
    f.write("1. coucou NEW \\n Data\n")
    f.write("2. hola NEW \\n datos")
# %%
