#!/usr/bin/env bash
# Regenerate the figures from the immutable raw results.
set -euo pipefail
cd "$(dirname "$0")/.."
RUN="${RUNNER:-uv run}"
$RUN memcheck verify-integrity --results results/raw
$RUN memcheck report --results results/raw --paper paper
echo
echo "Figures (each with a .provenance.json sidecar naming its source runs):"
find results/figures -name '*.png' | sort
