import logging

import pytest

from meddocan.language.logutils import all_logging_disabled

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


@pytest.mark.parametrize(argnames="disable_log,", argvalues=[False, True])
def test_all_logging_disabled(caplog, disable_log: bool):
    """Test that the context manager all_logging_disabled is working as
    expected.
    """
    # Cf: https://docs.pytest.org/en/7.1.x/how-to/logging.html
    if disable_log:
        with all_logging_disabled(highest_level=logging.INFO):
            logger.info("hello")
        assert caplog.record_tuples == []
    else:
        logger.info("hello")
        assert caplog.record_tuples == [
            ("test_logutils", logging.INFO, "hello")
        ]
