"""Module where the evaluation data are made for a given
``flair.models.SequenceTagger`` object.

The evaluation process can then be made using the command line available in the
`MEDDOCAN Evaluation Script`_ library.

.. _`MEDDOCAN Evaluation Script`:
   https://github.com/PlanTL-GOB-ES/MEDDOCAN-Evaluation-Script
"""
import logging
import shutil
import sys
from enum import Enum
from pathlib import Path

import flair
import torch

from meddocan.cli.utils import Arg, Opt, app
from meddocan.data import ArchiveFolder
from meddocan.data.docs_iterators import GsDocs, SysDocs
from meddocan.evaluation.classes import (BratAnnotation, NER_Evaluation,
                                         Span_Evaluation, i2b2Annotation)
from meddocan.evaluation.evaluate import evaluate

flair.device = torch.device("cuda:1")

logger = logging.getLogger(__name__)

COMMAND_NAME = "generate-evaluation-data"


class Subtrack(str, Enum):
    NER: str = "ner"
    SPANS: str = "spans"


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
            "Name of the folder that will holds the results produced by "
            "the ``Flair`` model."
        ),
    ),
    evaluation_root: Path = Arg(
        ...,
        help="Path to the root folder where the results will be stored.",
    ),
    sentence_splitting: Path = Opt(
        default=None,
        help=(
            "The sub-directory `sentence_splitting` is mandatory to "
            "compute the `leak score` evaluation metric."
        ),
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
        name (str): Name of the folder that will holds the results produced by
            the ``Flair`` model.
        evaluation_root (Union[str, Path]): Path to the root folder where the
            results will be stored.
        sentence_splitting (Path): Path to the sub-directory
            `sentence_splitting`. This directory is mandatory to compute the
            `leak score` evaluation metric.
        force (bool, optional): Force to create again the golds standard files.
            Defaults to False.
    """
    # Cf: Evaluation process on web.
    golds_loc = evaluation_root / "golds"
    sys_loc = evaluation_root / f"{name}"

    for data in [ArchiveFolder.test, ArchiveFolder.dev]:

        # TODO Use the gold file of the downloaded folder?

        logger.info(f"Create gold data")
        gs_docs = GsDocs(data)
        gs_docs.to_gold_standard(golds_loc)

        # TODO Is there a model name that can be extracted from the model to give
        # automatically a name to the folder?

        # ---- Make inference on Gold with model

        logger.info("Create system data with model {model}")
        sys_docs = SysDocs(data, model=model)
        sys_docs.to_evaluation_folder(sys_loc)

        # ---- Evaluate the results on the test

        for subtrack in Subtrack:

            format = "brat"  # Format used in this implementation
            verbose = True
            remove_inference = True

            # Create file to write evaluation results

            output_base = evaluation_root / f"{name}" / data.value
            output_file: Path = output_base / subtrack.value
            if not output_base.exists():
                output_base.mkdir(parents=True, exist_ok=True)
            output_file.touch(mode=0o766, exist_ok=True)
            output_file.chmod(mode=0o766)

            original_stdout = sys.stdout  # Redirect print to stdout

            with output_file.open(mode="w") as f:

                eval_results = evaluate(
                    f"{str(golds_loc)}/{data.value}/brat/",
                    [f"{str(sys_loc)}/{data.value}/brat/"],
                    sentences_loc=sentence_splitting,  # TODO Create it by default when downloading datas
                    annotation_format=i2b2Annotation
                    if format == "i2b2"
                    else BratAnnotation,
                    subtrack=NER_Evaluation
                    if subtrack.value == "ner"
                    else Span_Evaluation,
                    verbose=verbose,
                )

                sys.stdout = (
                    f  # Change the standard output to the file we created.
                )
                eval_results.print_report()

                for e in eval_results.evaluations:
                    print(f"{e.sys_id}")
                    print(f"{e.micro_precision()=}")
                    print(f"{e.micro_recall()=}")

                sys.stdout = original_stdout

        # Path to ../test/brat or /dev/brat
        subfolder = Path(next(iter(sys_docs)).brat_files_pair.txt.at).parent
        if remove_inference:
            shutil.rmtree(golds_loc)
            shutil.rmtree(sys_loc / subfolder)
