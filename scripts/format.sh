#!/bin/sh -e
set -x

autoflake --remove-all-unused-imports --recursive --remove-unused-variables --in-place meddocan tests --exclude=__init__.py
black -l 79 meddocan tests
isort -l 79 meddocan tests