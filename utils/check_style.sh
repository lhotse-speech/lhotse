#!/usr/bin/env bash

IGNORE_LIST='F401,F403,E501,E741'
MAX_LINE_LENGTH=120

BASE=$(git rev-parse --show-toplevel)
ALLFILES=$(git ls-tree --full-tree --name-only -r HEAD | grep -e ".*\.py\$")
for FILE in ${ALLFILES}; do
  flake8 --ignore ${IGNORE_LIST} \
         --max-line-length ${MAX_LINE_LENGTH} \
         ${BASE}/${FILE}
done
