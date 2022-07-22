#!/bin/bash

set -x
set -e

cd docs/build/html
poetry run python -m http.server 8080