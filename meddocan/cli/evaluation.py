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
from zipfile import Path as ZipPath

import flair
import torch

from meddocan.cli.utils import Arg, Opt, app
from meddocan.data import ArchiveFolder
from meddocan.data.docs_iterators import GsDocs, SysDocs
from meddocan.evaluation.classes import (
    BratAnnotation,
    NER_Evaluation,
    Span_Evaluation,
    i2b2Annotation,
)
from meddocan.evaluation.evaluate import evaluate

logger = logging.getLogger(__name__)

COMMAND_NAME = "eval"


class Subtrack(str, Enum):
    NER: str = "ner"
    SPANS: str = "spans"


"""help=(
    "Generate the files required by the MEDDOCAN Evaluation script to "
    "perform the model evaluation."
),"""


@app.command(
    name=COMMAND_NAME,
    no_args_is_help=False,
)
def eval(
    model: str = Arg(..., help="Path to the Flair model to evaluate."),
    name: str = Arg(
        ...,
        help=(
            "Name of the folder that will holds the results produced by "
            "the ``Flair`` model."
        ),
    ),
    evaluation_root: Path = Opt(
        default=None,
        help="Path to the root folder where the results will be stored.",
    ),
    sentence_splitting: Path = Opt(
        default=None,
        help=(
            "The sub-directory `sentence_splitting` is mandatory to "
            "compute the `leak score` evaluation metric."
        ),
    ),
    device: str = Opt(
        default="cuda:0",
        help="Device to use.",
    ),
) -> None:
    """Evaluate the model with the `meddocan` metrics.
    
    Calculate the f1 score for Ner (start, end, tag), Span (start, end) and 
    the merged span if there is no number or letter between consecutive spans.

    The function produce the following temporary folder hierarchy:

    evaluation_root
    ├── golds
    │   ├── dev
    |   |    └── brat
    |   |       ├── file-1.ann
    |   |       ├── file-1.txt
    |   |       ├── ...
    |   |       └── file-n.ann
    |   └── test
    |        └── brat
    |           ├── file-1.ann
    |           ├── file-1.txt
    |           ├── ...
    |           └── file-n.ann
    │       
    └── name
        ├── dev
        |    └── brat
        |       ├── file-1.ann
        |       ├── file-1.txt
        |       ├── ...
        |       └── file-n.ann
        └── test
             └── brat
                ├── file-1.ann
                ├── file-1.txt
                ├── ...
                └── file-n.ann

    Then the model is evaluate producing the following files:

    evaluation_root/name
    ├── dev
    │   ├── ner
    │   └── spans
    └── test
        ├── ner
        └── spans

    And the temporary folder are removed.

    Example:

    TODO How to write doctest code that is fast enough with the necessity of
    loading a model?

    Args:
        model (str): Path to the ``Flair`` model to evaluate.
        name (str): Name of the folder that will holds the results produced by\
            the ``Flair`` model.
        evaluation_root (str): Path to the root folder where the
            results will be stored.
        sentence_splitting (Path): Path to the sub-directory
            `sentence_splitting`. This directory is mandatory to compute the
            `leak score` evaluation metric.
        force (bool, optional): Force to create again the golds standard files.
            Defaults to False.
    """
    if device is not None:
        flair.device = torch.device(device)

    if evaluation_root is None:
        evaluation_root = Path(model).parent

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
                    sentences_loc=str(
                        sentence_splitting
                    ),  # TODO Create it by default when downloading datas
                    annotation_format=i2b2Annotation  # type: ignore  # https://github.com/python/mypy/issues/4717
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
        sys_doc = next(iter(sys_docs))

        txt_file = sys_doc.brat_files_pair.txt
        assert isinstance(txt_file, ZipPath)
        subfolder = Path(txt_file.at).parent  # type: ignore[attr-defined]
        # subfolder = Path(next(iter(sys_docs)).brat_files_pair.txt.at).parent  # type: ignore[attr-defined]
        if remove_inference:
            shutil.rmtree(golds_loc)
            shutil.rmtree(sys_loc / subfolder)
