# Data

MemCheck's default experiments need **no downloads**. The synthetic longitudinal workload and
RepoEvolutionBench are generated deterministically from code and fixtures in this repository, so
`scripts/reproduce_smoke.sh` runs on a clean checkout with no network access.

This directory exists for the one workload that *is* external.

## Layout

```
data/
├── raw/          # downloaded benchmark files (git-ignored)
├── processed/    # any derived form you create (git-ignored)
└── checksums/    # sha256 registry for downloads (committed)
```

Nothing under `raw/` or `processed/` is committed. `.gitignore` enforces this so a licensed
dataset cannot be redistributed by accident.

## LongMemEval

LongMemEval is **not redistributed with MemCheck**. It is released by its own authors under its
own terms, and this artifact deliberately ships the adapter without the data.

### 1. Obtain the dataset

Fetch it from the authors' official distribution and read their licence and terms of use before
using it. Do not take a copy from a mirror; the checksum registry below only means the file has
not changed since *you* registered it, not that it is authentic.

### 2. Place the files

| Subset | Expected filename | Config `subset` value |
| --- | --- | --- |
| Oracle / smoke | `longmemeval_oracle.json` | `smoke` |
| Small | `longmemeval_s.json` | `s` |
| Medium | `longmemeval_m.json` | `m` |

Put them directly in `data/raw/`, or point `workload.data_path` in the configuration elsewhere.

### 3. Register and verify checksums

```bash
# First time: record what you downloaded.
uv run memcheck data verify --dataset longmemeval --register

# Afterwards, and in any reproduction: confirm nothing has changed.
uv run memcheck data verify --dataset longmemeval
```

The registry is written to `data/checksums/longmemeval.json` and **is** committed, so a later run
can prove it used the same bytes. A mismatch is a hard error, not a warning.

### 4. Run

```bash
uv run memcheck run --config configs/longmemeval.yaml
```

Start with `subset: smoke` and `params.limit: 20`, as the shipped configuration does. Only widen
to `s` or `m` once a smoke run has completed and been reported.

### What the adapter does

It maps LongMemEval's question types onto MemCheck's categories:

| LongMemEval question type | MemCheck category |
| --- | --- |
| `single-session-user`, `single-session-assistant`, `single-session-preference` | `single_session_extraction` |
| `multi-session` | `cross_session_reasoning` |
| `temporal-reasoning` | `temporal_reasoning` |
| `knowledge-update` | `knowledge_update` |
| `abstention` | `abstention` |

Turns flagged as containing the answer become the expected evidence for that query, which is what
the evidence recall and precision metrics are computed against.

If the file is absent, `memcheck run` fails with a message pointing back to this page rather than
silently substituting a different workload.

## Gold labels

Gold labels are never modified. If a label appears wrong, record the discrepancy in
`DEVIATIONS.md` and report it upstream; do not edit the benchmark file. The synthetic workload's
labels come from the generator's own state and are exact by construction, so there is nothing to
adjust there either.

## Adding another benchmark

1. Implement a `Workload` subclass in `src/memcheck/workloads/`.
2. Register it in `build_workload`.
3. Document the download here, add its expected filenames to the checksum registry, and add a
   configuration under `configs/`.
4. Do not commit the data.
