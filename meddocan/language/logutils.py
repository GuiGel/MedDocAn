"""Copied from: https://gist.github.com/simon-weber/7853144"""
import logging
from contextlib import contextmanager


@contextmanager
def all_logging_disabled(highest_level: int = logging.CRITICAL):
    """A context manager that will prevent any logging messages
    triggered during the body from being processed.

    Args:
        highest_level (int, optional): the maximum logging level in use.
            This would only need to be changed if a custom level greater than
            CRITICAL is defined. Defaults to logging.CRITICAL.
    """
    # two kind-of hacks here:
    #   * can't get the highest logging level in effect => delegate to the user
    #   * can't get the current module-level override => use an undocumented
    #       (but non-private!) interface

    previous_level = logging.root.manager.disable

    logging.disable(highest_level)

    try:
        yield
    finally:
        logging.disable(previous_level)
