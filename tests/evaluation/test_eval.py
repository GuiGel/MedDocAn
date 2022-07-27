"""Module that partially test the :func:`meddocan.evaluation.eval` function.

    Working example:

    >>> from unittest.mock import Mock, patch
    >>> class Parent:
    ...     def child1(self, arg):
    ...             print(arg)
    ... 
    >>> def func():
    ...     parent = Parent()
    ...     parent.child1("coucou")
    ... 
    >>> with patch("__main__.Parent") as mock_parent:
    ...     mock_instance = mock_parent.return_value
    ...     child1 = Mock(return_value=None)
    ...     mock_instance.child1 = child1
    ...     func()
    ...     print(mock_instance.child1.call_args)
    ... 
    call('coucou')
"""
from pathlib import Path
from typing import Union
from unittest.mock import Mock, patch

import pytest

from meddocan.data import ArchiveFolder
from meddocan.evaluation import eval


@pytest.mark.parametrize(
    "model, name, evaluation_root, force",
    [
        ("evaluation_root", "result_folder_name", "model_loc", False),
        ("evaluation_root", "result_folder_name", "model_loc", True),
    ],
)
@patch("meddocan.evaluation.eval_func.GsDocs")
@patch("meddocan.evaluation.eval_func.SysDocs")
def test_eval(
    MockSysDocs,
    MockGsDocs,
    model: str,
    name: str,
    evaluation_root: Union[str, Path],
    force: bool,
) -> None:
    # 1 Prepare MagikMock instance for GsDocs and SysDocs
    mock_gs_docs = MockGsDocs.return_value
    to_gold_standard = Mock(return_value=None)
    mock_gs_docs.to_gold_standard = to_gold_standard

    mock_sys_docs = MockSysDocs.return_value
    to_evaluation_folder = Mock(return_value=None)
    mock_sys_docs.to_evaluation_folder = to_evaluation_folder

    # 2- Run the eval function that we are testing.
    eval(model, name, evaluation_root, force)

    # 3- Verify that all is called as expected
    MockGsDocs.assert_called_once_with(ArchiveFolder.test)

    golds_loc = Path(evaluation_root) / "golds"
    mock_gs_docs.to_gold_standard.assert_called_once_with(golds_loc)

    MockSysDocs.assert_called_once_with(ArchiveFolder.test, model=model)

    sys_loc = Path(evaluation_root) / f"{name}"
    mock_sys_docs.to_evaluation_folder.assert_called_once_with(sys_loc)
