#!/usr/bin/env bash
# Regenerate everything the paper depends on: processed tables, figures, LaTeX fragments,
# claim macros, the evidence matrix, and the audit sample. The paper's own LaTeX sources are
# not shipped with this artifact, so nothing is compiled here.
set -euo pipefail
cd "$(dirname "$0")/.."
RUN="${RUNNER:-uv run}"

$RUN memcheck verify-integrity --results results/raw
$RUN memcheck summarize --results results/raw --processed results/processed
$RUN memcheck report --results results/raw --paper paper
$RUN memcheck audit --results results/raw --output results/audit --sample 60
