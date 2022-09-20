#!/bin/sh -e
set -x

autoflake --remove-all-unused-imports --recursive --remove-unused-variables --in-place meddocan tests experiments --exclude=__init__.py
black -l 79 meddocan tests experiments
isort -l 79 meddocan tests experiments