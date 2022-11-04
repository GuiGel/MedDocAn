#!/usr/bin/env bash

set -e
set -x

export MYPYPATH=./stubs:./application
mypy --show-error-code meddocan application
black -l 79 meddocan tests application --check
isort -l 79 meddocan tests application --check-only