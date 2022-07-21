#!/usr/bin/env bash

set -e
set -x

pytest -vv -s --cov=./ --cov-report html --cov-report xml --cov-config=.coveragerc --cov-report term-missing
