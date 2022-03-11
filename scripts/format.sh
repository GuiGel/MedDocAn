#!/bin/sh -e
set -x

autoflake --remove-all-unused-imports --recursive --remove-unused-variables --in-place docs_src typer tests --exclude=__init__.py
black meddocan tests
isort meddocan tests