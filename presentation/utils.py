import base64
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Iterator, Tuple

import requests
from IPython.display import Markdown, display

from meddocan.evaluation.classes import BratAnnotation


def display_script(loc: Path, language: str) -> None:
    script = Path(loc).read_text()
    display(Markdown(f"```{language}\n{script}\n```"))


BASE = "https://api.github.com/repos/PlanTL-GOB-ES/MEDDOCAN-Evaluation-Script/contents"
# The api where the text can be reach.


def get_sample(base: str, name: str) -> str:
    # Get sample content from the folder located at https://github.com/PlanTL-GOB-ES/MEDDOCAN-Evaluation-Script/tree/master/gold/brat/sample via the Github api.
    # Use the Stackoverflow response: https://stackoverflow.com/questions/38491722/reading-a-github-file-using-python-returns-html-tags
    url = "/".join([base, name])
    req = requests.get(url)
    if req.status_code == requests.codes.ok:
        req = req.json()  # the response is JSON
        # req is now a dict with keys: name, encoding, url, size ...
        # and content. But it is encoded with base64.
        content = base64.b64decode(req["content"])
        return content.decode("utf-8")
    else:
        raise ValueError("Content was not found!")


@contextmanager
def write_text_to_tempdir(
    seq_file_text: Iterable[Tuple[str, str]]
) -> Iterator[Path]:
    # Context manager that write a sequence of (filename, text content) tuple
    # to a temporary directory and return the directory name.
    with tempfile.TemporaryDirectory() as tmpdirname:
        root = Path(tmpdirname)
        for loc, content in seq_file_text:
            (root / loc).write_text(content)
        yield Path(tmpdirname)


def get_brat_annotation_from_github(
    file: str, base: str = BASE
) -> BratAnnotation:
    ann = Path(file).with_suffix(".ann")
    txt = Path(file).with_suffix(".txt")

    seq_file_text = (
        (loc.name, get_sample(base, str(loc))) for loc in [ann, txt]
    )

    with write_text_to_tempdir(seq_file_text) as dir_loc:
        gold_annotation = BratAnnotation(str(dir_loc / Path(ann).name))
        return gold_annotation


def glue_iob_label(iob: str, label: str) -> str:
    if len(label):
        return f"{iob}_{label}"
    else:
        return iob


def get_parameter_num(model_path: Path) -> int:
    """Obtain the number of parameter in a model

    Args:
        model_path (Path): Path to a SequenceTagger model

    Returns:
        int: Number of parameters of the model
    """
    model = SequenceTagger.load(model_path)
    parameters = model.parameters()
    return sum(map(torch.numel, parameters))


def make_pretty(styler):
    styler.background_gradient(axis=None, cmap="YlGnBu")
    styler.set_table_styles(
        [
            {
                "selector": "th.col_heading.level0",
                "props": "text-align: center; font-size: 1.5em;",
            },
            {"selector": "", "props": [("border", "1px solid grey")]},
            {"selector": "tbody td", "props": [("border", "1px solid grey")]},
            {"selector": "th", "props": [("border", "1px solid grey")]},
        ],
        overwrite=False,
    )
    styler.format(
        "{:.0f}",
    )
    return styler
