#!/usr/bin/env bash
# Reproduce the smoke experiment. No API key, no cost, a few seconds.
set -euo pipefail
cd "$(dirname "$0")/.."

RUN="${RUNNER:-uv run}"

echo "==> environment"
$RUN memcheck doctor

echo "==> demo: ingest, update, retrieve, delete, restart, rebuild"
$RUN memcheck demo

echo "==> contracts against every backend (expected to FAIL for the baselines)"
$RUN memcheck validate --backend guarded_memory

echo "==> faults against every backend"
$RUN memcheck inject --fault all --backend all

echo "==> smoke experiment"
$RUN memcheck run --config configs/smoke.yaml

echo "==> RepoEvolutionBench"
$RUN memcheck run --config configs/repo_evolution.yaml

echo "==> integrity"
$RUN memcheck verify-integrity --results results/raw

echo
echo "Done. Raw results are under results/raw/. Next: bash scripts/reproduce_paper.sh"
