"""Module where the evaluation data are made for a given
``flair.models.SequenceTagger`` object.

The evaluation process can then be made using the command line available in the
`MEDDOCAN Evaluation Script`_ library.

.. _`MEDDOCAN Evaluation Script`:
   https://github.com/PlanTL-GOB-ES/MEDDOCAN-Evaluation-Script
"""
from pathlib import Path
from typing import Union

from meddocan.data import ArchiveFolder
from meddocan.data.docs_iterators import GsDocs, SysDocs


def generate_evaluation_data(
    model: str,
    name: str,
    evaluation_root: Union[str, Path],
    force: bool = False,
) -> None:
    """Create the files necessary for the ``evaluation.py'' script to produce
    the files that allow the MEDDOCAN team to compare the results obtained by
    the different participants.

    The function produce the following folder hierarchy:

    - evaluation_root
    - evaluation_root.golds.test.brat
    - evaluation_root.golds.test.brat.file-1.ann
    - evaluation_root.golds.test.brat.file-1.txt
    - ...
    - evaluation_root.golds.test.brat.file-n.ann
    - evaluation_root.golds.test.brat.file-n.txt
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
        name (str): Name of the folder that will holds the results produced by
            the ``Flair`` model.
        evaluation_root (Union[str, Path]): Path to the root folder where the
            results will be stored.
        force (bool, optional): Force to create again the golds standard files.
            Defaults to False.
    """
    evaluation_root = Path(evaluation_root)

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
