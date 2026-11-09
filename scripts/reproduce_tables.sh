#!/usr/bin/env bash
# Regenerate the LaTeX tables from the immutable raw results.
set -euo pipefail
cd "$(dirname "$0")/.."
RUN="${RUNNER:-uv run}"
$RUN memcheck verify-integrity --results results/raw
$RUN memcheck report --results results/raw --paper paper
echo
echo "Tables:"
find results/tables -name '*.tex' | sort
