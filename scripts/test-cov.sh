#!/usr/bin/env bash

set -e
set -x

#pytest meddocan tests --cov=meddocan --cov-report=html --cov-report=xml --cov-context=test --doctest-modules -o console_output_style=progess --numprocesses=auto ${@}
pytest --cov --cov-report html --cov-config=.coveragerc --cov-report term-missing
# pytest --cov=meddocan tests --cov-report html -o console_output_style=progess --numprocesses=auto ${@}
# coverage html --omit="meddocan/evaluation/*" -d tests/coverage