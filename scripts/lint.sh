#!/usr/bin/env bash

set -e
set -x

mypy meddocan
black meddocan tests --check
isort meddocan tests --check-only