#!/bin/bash

set -x
set -e

poetry run sphinx-apidoc -f -o docs/source meddocan
cp README.md docs/source/usage/introduction.md
cd docs
poetry run make html