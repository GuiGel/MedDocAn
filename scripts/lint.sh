#!/usr/bin/env bash

set -e
set -x

export MYPYPATH=./stubs
mypy meddocan
black meddocan tests --check
isort meddocan tests --check-only