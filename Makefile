.DEFAULT_GOAL := help
RUN ?= uv run

.PHONY: help install lint types test check doctor demo validate inject smoke repo matrix \
        summarize report audit paper verify clean-derived lock

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

install: ## Create the environment and install the package with dev extras
	uv venv --python 3.12
	uv pip install -e ".[dev,parquet]"

lint: ## Ruff
	$(RUN) ruff check .

types: ## mypy
	$(RUN) mypy src

test: ## pytest
	$(RUN) pytest -q

check: lint types test ## All three quality gates

doctor: ## Report environment readiness
	$(RUN) memcheck doctor

demo: ## Ingest, update, retrieve, delete, restart, rebuild
	$(RUN) memcheck demo

validate: ## Run the six contracts against every backend
	$(RUN) memcheck validate --backend all || true

inject: ## Inject all six faults against every backend
	$(RUN) memcheck inject --fault all --backend all

smoke: ## Run the smoke experiment (mock provider, free)
	$(RUN) memcheck run --config configs/smoke.yaml

repo: ## Run RepoEvolutionBench
	$(RUN) memcheck run --config configs/repo_evolution.yaml

matrix: ## Run the full local matrix (slower, still free)
	$(RUN) memcheck run --config configs/experiment_matrix.yaml

summarize: ## Aggregate raw runs into processed tables
	$(RUN) memcheck summarize --results results/raw --processed results/processed

report: ## Regenerate figures, LaTeX tables, audit sample, and paper macros
	$(RUN) memcheck report --results results/raw --paper paper

audit: ## Export the human audit sample
	$(RUN) memcheck audit --results results/raw --output results/audit --sample 60

paper: ## Regenerate every table, figure, claim macro, and the audit sample
	bash scripts/reproduce_paper.sh

verify: ## Re-check every run directory against its checksums
	$(RUN) memcheck verify-integrity --results results/raw

lock: ## Regenerate the pinned dependency files
	uv lock || true
	uv pip compile pyproject.toml --extra dev --extra parquet -o requirements.lock.txt

clean-derived: ## Remove derived outputs. Raw results are never touched.
	rm -rf results/processed/* results/figures/* results/tables/* results/audit/* \
	       paper/generated/* paper/main.pdf
	@echo "derived outputs removed; results/raw/ is untouched by design"
