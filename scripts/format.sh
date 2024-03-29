#!/bin/sh -e
set -x

autoflake --remove-all-unused-imports --recursive --remove-unused-variables --in-place meddocan tests experiments application presentation --exclude=__init__.py
black -l 79 meddocan tests experiments application presentation
isort -l 79 meddocan tests experiments application presentation