#!/usr/bin/env bash

set -e
set -x

# Check README.md is up to date
diff --brief docs/source/usage/introduction.md README.md