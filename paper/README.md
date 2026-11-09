# paper/

This directory holds the traceability layer between the stored experiment runs and the reported
numbers. The paper's own LaTeX sources are not part of this artifact.

## Layout

```
paper/
├── claims_registry.yaml     # every empirical claim, bound to its run ids and source file
├── evidence_matrix.csv      # GENERATED from the registry: claim → artifact → section
└── generated/               # WRITTEN BY `memcheck report` — do not edit
    ├── claims.tex           # one LaTeX macro per registered claim
    ├── table*.tex           # the generated tables
    ├── figures/             # the generated figures (.png and .pdf) and their data workbooks
    └── report_index.json    # what was generated, what was skipped, and why
```

## Regenerate

```bash
uv run memcheck report --results results/raw --paper paper
```

This verifies each run's checksums, refuses to pool runs that are not comparable, recomputes the
metrics, redraws the figures, rewrites the tables, resolves every claim, and regenerates the
evidence matrix.

## The rule this directory enforces

**No fabricated number.** A claim in `claims_registry.yaml` prints a value only when its status is
`supported` or `partially_supported` *and* its `source_file` / `value_column` / `value_filter`
binding resolves to exactly one row of the processed results. Anything else renders as
`[RESULT PENDING: …]`. If an experiment is re-run, the run ids change and the registry must be
updated alongside it; that friction is intentional.

Each generated `.tex` file opens with a comment block recording its source run ids, configuration
hashes, generating script, and generation timestamp. Each figure has a `.provenance.json` sidecar
naming its source runs.
