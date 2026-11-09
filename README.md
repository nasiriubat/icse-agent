# MemCheck

**Contract-driven testing, fault injection, and repair for persistent LLM agent memory.**

MemCheck is a reproducible research artifact for testing the *reliability*, *efficiency*, and
*longitudinal behavior* of persistent memory systems used by LLM agents.

The central claim under test:

> Persistent agent memory systems can silently violate lifecycle guarantees during duplicate
> writes, fact updates, deletion, restart, recovery, and store growth. Executable memory contracts
> and fault injection can detect these failures. A guarded memory layer can reduce them with
> measurable cost.

MemCheck contains two halves:

1. **An experiment and evaluation tool** — memory backends, executable contracts, deterministic
   fault injection, longitudinal workloads, and a resumable experiment runner.
2. **A paper generation and reporting pipeline** — metrics, figures, LaTeX tables, a claims
   registry, and an evidence matrix that traces every paper statement back to a stored artifact.

---

## 1. System purpose

Agent memory is usually evaluated on *task accuracy*: can the agent answer a question about
something that happened many sessions ago? That framing hides a class of defects that only show up
over a system's lifetime — a fact updated but not superseded, a deleted record resurfacing through
a stale index, a duplicated event silently amplifying a memory's weight, a crash between the write
and the index update.

MemCheck treats memory as a *stateful system with a lifecycle* and tests it the way you would test
a database: with executable invariants (contracts), injected faults, and longitudinal measurement
as the store grows.

## 2. Architecture

```
                  workloads                     experiment runner
        (synthetic / repo_evolution /   ──►   matrix · seeds · checkpoints
              longmemeval)                    fairness · cost · resume
                    │                                  │
                    ▼                                  ▼
  events ──► MemoryBackend ──► RetrievalResult ──► agent ──► answer ──► grader
             ├─ no_memory        (bounded evidence)   (provider)          │
             ├─ full_history                                              │
             ├─ transcript_rag                                            ▼
             └─ guarded_memory  ◄── faults injected            results/raw/<run_id>/
                    │                                            (immutable JSONL)
                    ▼                                                    │
              contracts (6)  ──► ContractViolation                       ▼
                                                              reporting → figures,
                                                              LaTeX tables, audit CSV
                                                                         │
                                                                         ▼
                                                              paper/ (claims registry,
                                                               evidence matrix, tables)
```

Key modules (`src/memcheck/`):

| Module | Role |
| --- | --- |
| `models.py` | Pydantic domain models (Event, MemoryRecord, Evidence, RetrievalResult, ContractViolation, ExperimentRunManifest) |
| `event_log.py` | Append-only, idempotent SQLite event log — the source of truth |
| `backends/` | `no_memory`, `full_history`, `transcript_rag`, `guarded_memory` |
| `contracts/` | Six executable memory contracts |
| `faults/` | Six deterministic fault injectors |
| `workloads/` | Synthetic longitudinal workload, RepoEvolutionBench, LongMemEval adapter |
| `agents/` | Answer agent and sandboxed patch-applying coding agent |
| `providers/` | Chat / extraction / embedding / tokenization / pricing interfaces + mock, Anthropic, OpenAI |
| `evaluation/` | Experiment runner, matrix expansion, fairness validator, grading, metrics |
| `reporting/` | Aggregation, figures, LaTeX tables, audit sampling, paper generation |

## 3. Installation

Requires Python 3.11+ and [`uv`](https://docs.astral.sh/uv/).

```bash
uv venv --python 3.12
uv pip install -e ".[dev,parquet]"
uv run memcheck doctor
```

Optional extras: `anthropic`, `openai`, `embeddings` (local sentence-transformers).
None are needed for the smoke experiment, the tests, or CI.

Without `uv`:

```bash
python3.12 -m venv .venv && . .venv/bin/activate
pip install -e ".[dev,parquet]"
memcheck doctor
```

## 4. Environment variables

| Variable | Meaning | Default |
| --- | --- | --- |
| `ALLOW_PAID_RUNS` | Must be exactly `true` to permit any paid provider call | `false` |
| `MAX_EXPERIMENT_COST_USD` | Hard ceiling on a run's *estimated* cost | `0` |
| `ANTHROPIC_API_KEY` | Anthropic credential (paid runs only) | unset |
| `OPENAI_API_KEY` | OpenAI credential (paid runs only) | unset |
| `MEMCHECK_DATA_DIR` | Benchmark data directory | `data/raw` |
| `MEMCHECK_ANONYMOUS` | `1` builds the paper without author identity | `0` |

Secrets are read from the environment only. They are never written to manifests, logs, or results.

Copy `.env.example` to `.env` and MemCheck will read it at start-up. A variable already set in
your shell always wins over the file, so a stale `.env` cannot quietly re-enable paid runs, and
credential *values* are never echoed back — only their names. `.env` is git-ignored. Point
elsewhere with `--env-file`, or skip it entirely with `--env-file /dev/null`.

## 5. Data acquisition

The default smoke and full local experiments need **no downloads** — the synthetic longitudinal
workload and RepoEvolutionBench are generated deterministically from code and fixtures in-repo.

LongMemEval is *not* redistributed here. See [`data/README.md`](data/README.md) for download
instructions, expected file names, and checksum registration:

```bash
uv run memcheck data verify --dataset longmemeval
```

## 6. Smoke experiment (no API key, no cost)

```bash
uv run memcheck demo                          # ingest → update → retrieve → delete → restart → rebuild
uv run memcheck validate --backend guarded_memory   # run the six contracts
uv run memcheck run --config configs/smoke.yaml     # full smoke matrix, mock provider
```

or:

```bash
bash scripts/reproduce_smoke.sh
```

## 7. Full experiment

```bash
uv run memcheck run --config configs/repo_evolution.yaml
uv run memcheck run --config configs/longmemeval.yaml     # requires downloaded data
uv run memcheck run --config configs/experiment_matrix.yaml
```

Interrupted runs resume without recomputation:

```bash
uv run memcheck resume --run_dir results/raw/<run_id>
```

## 8. Cost controls

Before any run touching a paid provider, MemCheck estimates request count, token usage, and
monetary cost from the dated pricing file (`configs/pricing.example.yaml`), prints the estimate,
and compares it against `MAX_EXPERIMENT_COST_USD`. The run is refused if the estimate exceeds the
limit, or if `ALLOW_PAID_RUNS` is not `true`. Mock-provider runs are always free and always
allowed. Provider responses are cached on an exact request hash so a completed request is never
re-billed.

## 9. Result processing

Each run writes an immutable directory `results/raw/<run_id>/` containing `manifest.json`,
`config.yaml`, and the JSONL streams (`events`, `writes`, `retrievals`, `answers`, `grades`,
`violations`, `costs`, `latency`, `store_statistics`), plus `run_status.json` and captured
stdout/stderr. A `checksums.json` file lets `memcheck verify-integrity` detect post-hoc edits.

```bash
uv run memcheck summarize --results results/raw   # → results/processed/*.csv|parquet
uv run memcheck report --results results/raw --paper paper
uv run memcheck audit --results results/raw --sample 60
```

## 10. Table and figure generation

```bash
uv run memcheck report --results results/raw --paper paper   # regenerate figures, tables, macros
```

Generated LaTeX fragments land in `paper/generated/`. The paper's own LaTeX sources are not part
of this artifact; `paper/` holds the claims registry and evidence matrix that bind every reported
number to a stored run. Every figure and table carries its source run ids,
generating script, configuration hash, and generation timestamp. Numbers that have no supporting
artifact are rendered as `[RESULT PENDING: ...]` — the generator refuses to print an unsupported
claim as a confirmed finding.

### Figure data as spreadsheets

A rasterised figure loses its numbers. Every report therefore also writes an `.xlsx` workbook
holding the data behind each figure, one per comparability group:

```
results/figures/<group>/figure_data.xlsx
paper/generated/figures/figure_data_<group>.xlsx   # copies, all groups
```

Each workbook opens on a `README` sheet carrying the same provenance stamp as the figures, then
one sheet per figure. Series drawn against store growth are pivoted **wide** — one row per
checkpoint, one column per condition — so selecting the block and inserting a line chart
reproduces the figure directly. Scatter plots stay **long**, because a wide layout would break
the pairing they exist to show. The aggregated metric frames and the raw contract violations are
exported whole alongside them, so a plot the paper never drew can still be built from the
workbook without re-running an experiment.

## 11. Known limitations

- The default provider is a **deterministic mock reader**, not a frontier LLM. It answers correctly
  if and only if the retrieved evidence supports the gold fact, and abstains otherwise. This
  isolates *memory-system* behavior from *language-model* ability, which is what the research
  questions ask about — but it means mock-provider results measure the memory layer, not
  end-to-end agent accuracy. See `DECISIONS.md` (D-006) and the threats-to-validity section.
- LongMemEval results require a separate licensed download and are not shipped.
- MemCheck is a research artifact. It is not production software, and no production-readiness
  claim is made.
- Statistical claims are limited by the repetition counts recorded in each run manifest.

## 12. Reproduction instructions

```bash
bash scripts/reproduce_smoke.sh     # run the mock-provider smoke matrix
bash scripts/reproduce_tables.sh    # regenerate LaTeX tables from results/raw
bash scripts/reproduce_figures.sh   # regenerate figures from results/raw
bash scripts/reproduce_paper.sh     # regenerate every table, figure, macro, and the audit sample
```

Quality gates:

```bash
uv run ruff check .
uv run mypy src
uv run pytest
```

See `DECISIONS.md` for design rationale and `DEVIATIONS.md` for every departure from the
original specification.
