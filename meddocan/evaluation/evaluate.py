###############################################################################
#
#   Copyright 2019 Secretaría de Estado para el Avance Digital (SEAD)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
#                        MEDDOCAN Evaluation Script
#
# This script is distributed as apart of the Medical Document Anonymization
# (MEDDOCAN) task. It is inspired on the evaluation script from the i2b2
# 2014 Cardiac Risk and Personal Health-care Information (PHI) tasks. It is
# intended to be used via command line:
#
# $> python evaluate.py [i2b2|brat] [ner|spans] GOLD SYSTEM
#
# It produces Precision, Recall and F1 (P/R/F1) and leak score measures for
# the NER subtrack and P/R/F1 for the SPAN subtrack. The latter includes a
# relaxed metric where the spans are merged if only non-alphanumerical
# characters are found between them.
#
# SYSTEM and GOLD may be individual files or also directories in which case
# all files in SYSTEM will be compared to files the GOLD directory based on
# their file names.
#
# Basic Examples:
#
# $> python evaluate.py i2b2 ner gold/01.xml system/run1/01.xml
#
# Evaluate the single system output file '01.xml' against the gold standard
# file '01.xml' NER subtrack. Input files in i2b2 format.
#
# $> python evaluate.py brat ner gold/01.ann system/run1/01.ann
#
# Evaluate the single system output file '01.ann' against the gold standard
# file '01.ann' NER subtrack. Input files in BRAT format.
#
# $> python evaluate.py i2b2 spans gold/ system/run1/
#
# Evaluate the set of system outputs in the folder system/run1 against the
# set of gold standard annotations in gold/ using the SPANS subtrack. Input
# files in i2b2 format.
#
# $> python evaluate.py brat ner gold/ system/run1/ system/run2/ system/run3/
#
# Evaluate the set of system outputs in the folder system/run1, system/run2
# and in the folder system/run3 against the set of gold standard annotations
# in gold/ using the NER subtrack. Input files in BRAT format.

import argparse
import os
from collections import defaultdict
from typing import Any, Dict, List, Union

from meddocan.evaluation.classes import (BratAnnotation, NER_Evaluation,
                                         Span_Evaluation, i2b2Annotation)


def get_document_dict_by_system_id(
    system_dirs: List[str],
    annotation_format: Union[BratAnnotation, i2b2Annotation],
):
    """Takes a list of directories and returns annotations."""

    documents = defaultdict(lambda: defaultdict(int))

    for d in system_dirs:
        for fn in os.listdir(d):
            if fn.endswith(".ann") or fn.endswith(".xml"):
                sa = annotation_format(os.path.join(d, fn))
                documents[sa.sys_id][sa.id] = sa

    return documents


def evaluate(
    gs: str,
    system: List[str],
    annotation_format: Union[BratAnnotation, i2b2Annotation],
    subtrack: Union[NER_Evaluation, Span_Evaluation],  # MeddocanEvaluation
    **kwargs: Dict[str, Any],
):
    """Evaluate the system by calling either NER_evaluation or Span_Evaluation.
    'system' can be a list containing either one file,  or one or more
    directories. 'gs' can be a file or a directory."""

    gold_ann = {}
    evaluations = []

    # Strip verbose keyword if it exists
    try:
        verbose = kwargs["verbose"]
        del kwargs["verbose"]
    except KeyError:
        verbose = False

    # Handle if two files were passed on the command line
    if os.path.isfile(system[0]) and os.path.isfile(gs):
        if (system[0].endswith(".ann") and gs.endswith(".ann")) or (
            system[0].endswith(".xml") or gs.endswith(".xml")
        ):
            gs = annotation_format(gs)
            sys = annotation_format(system[0])
            e = subtrack({sys.id: sys}, {gs.id: gs}, **kwargs)
            e.print_docs()
            evaluations.append(e)

    # Handle the case where 'gs' is a directory and 'system' is a list of directories.
    elif all([os.path.isdir(sys) for sys in system]) and os.path.isdir(gs):
        # Get a dict of gold annotations indexed by id

        for filename in os.listdir(gs):
            if filename.endswith(".ann") or filename.endswith(".xml"):
                annotations = annotation_format(os.path.join(gs, filename))
                gold_ann[annotations.id] = annotations

        for system_id, system_ann in sorted(
            get_document_dict_by_system_id(system, annotation_format).items()
        ):
            e = subtrack(system_ann, gold_ann, **kwargs)
            e.print_report(verbose=verbose)
            evaluations.append(e)

    else:
        raise Exception(
            "Must pass file file or [directory/]+ directory/"
            "on command line!"
        )

    return evaluations[0] if len(evaluations) == 1 else evaluations


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluation script for the MEDDOCAN track."
    )

    parser.add_argument("format", choices=["i2b2", "brat"], help="Format")
    parser.add_argument("subtrack", choices=["ner", "spans"], help="Subtrack")
    parser.add_argument(
        "-v",
        "--verbose",
        help="List also scores for each document",
        action="store_true",
    )
    parser.add_argument("gs_dir", help="Directory to load GS from")
    parser.add_argument(
        "sys_dir",
        help="Directories with system outputs (one or more)",
        nargs="+",
    )

    args = parser.parse_args()

    evaluate(
        args.gs_dir,
        args.sys_dir,
        i2b2Annotation if args.format == "i2b2" else BratAnnotation,
        NER_Evaluation if args.subtrack == "ner" else Span_Evaluation,
        verbose=args.verbose,
    )
