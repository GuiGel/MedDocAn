#!/usr/bin/env bash

set -e
set -x

# cli to work on the presentation
poetry run jupyter lab  --no-browser --NotebookApp.password='' --NotebookApp.token=''