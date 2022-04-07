#!/usr/bin/env bash

set -e
set -x

export MYPYPATH=./stubs
mypy --show-error-code meddocan
black -l 79 meddocan tests --check
isort -l 79 meddocan tests --check-only