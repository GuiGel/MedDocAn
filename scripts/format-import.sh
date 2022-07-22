#!/bin/sh -e
set -x

# Sort imports one per line, so autoflake can remove unused imports
isort -l 79 --force-single-line-imports ./meddocan ./tests
sh ./scripts/format.sh