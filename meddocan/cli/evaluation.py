"""Module where the evaluation data are made for a given
``flair.models.SequenceTagger`` object.

The evaluation process can then be made using the command line available in the
`MEDDOCAN Evaluation Script`_ library.

.. _`MEDDOCAN Evaluation Script`:
   https://github.com/PlanTL-GOB-ES/MEDDOCAN-Evaluation-Script
"""
import logging
from pathlib import Path

from meddocan.data import ArchiveFolder
from meddocan.data.docs_iterators import GsDocs, SysDocs

from .utils import Arg, Opt, app

logger = logging.getLogger(__name__)

COMMAND_NAME = "generate-evaluation-data"


@app.command(
    name=COMMAND_NAME,
    no_args_is_help=False,
    help=(
        "Generate the files required by the MEDDOCAN Evaluation script to "
        "perform the model evaluation."
    ),
)
def generate_evaluation_data(
    model: str = Arg(..., help="Path to the Flair model to evaluate."),
    name: str = Arg(
        ...,
        help=(
            "Name of the folder that will holds the resuts produced by "
            "the ``Flair`` model."
        ),
    ),
    evaluation_root: Path = Arg(
        ...,
        help="Path to the root folder where the results will be stored.",
    ),
    force: bool = Opt(
        default=False,
        help="Force to create again the golds standard files.",
    ),
) -> None:
    """Create the files necessary for the ``evaluation.py`` script to produce
    the files that allow the MEDDOCAN team to compare the results obtained by
    the different participants.

    The function produce the following folder hierarchy:

    - evaluation_root
    - evaluation_root.golds.test.brat
    - evaluation_root.golds.test.brat.file-1.ann
    - evaluation_root.golds.test.brat.file-1.txt
    - ...
    - evaluation_root.golds.test.brat.file-n.ann
    - evaluation_root.golds.test.brat.file-n.ann
    - evaluation_root.name.test.brat
    - evaluation_root.name.test.brat.file-1.ann
    - evaluation_root.name.test.brat.file-1.txt
    - ...
    - evaluation_root.name.test.brat.file-n.ann
    - evaluation_root.name.test.brat.file-n.ann

    Example:

    TODO How to write doctest code that is fast enough with the necessity of
    loading a model?

    Args:
        model (Union[str, Path]): Path to the ``Flair`` model to evaluate.
        name (str): Name of the folder that will holds the resuts produced by
            the ``Flair`` model.
        evaluation_root (Union[str, Path]): Path to the root folder where the
            results will be stored.
        force (bool, optional): Force to create again the golds standard files.
            Defaults to False.
    """
    # Cf: Evaluation process on web.
    golds_loc = evaluation_root / "golds"
    sys_loc = evaluation_root / f"{name}"

    if not Path(golds_loc).exists() or force:
        gs_docs = GsDocs(ArchiveFolder.test)
        gs_docs.to_gold_standard(golds_loc)

    # TODO Is there a model name that can be extracted from the model to give
    # automatically a name to the folder?

    sys_docs = SysDocs(ArchiveFolder.test, model=model)
    sys_docs.to_evaluation_folder(sys_loc)
