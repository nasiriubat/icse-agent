# DECISIONS.md

Design decisions, with rationale and consequences. Append-only; supersede rather than delete.

---

### D-001 — Python 3.12 via `uv`, packaged with hatchling
**Context.** System Python on the development machine is 3.9; the specification requires 3.11+.
**Decision.** Create the environment with `uv venv --python 3.12`. Package with hatchling and a
`src/` layout so tests import the installed package rather than the working directory.
**Consequences.** `uv` must be present, or the documented `python3.12 -m venv` fallback is used.

### D-002 — SQLite as the single source of truth for the guarded backend
**Context.** The guarded backend needs transactional writes across an event log, versioned memory
records, and derived indexes.
**Decision.** One SQLite database per backend instance, with the event log, memory records,
tombstones, and the FTS5 index in the same file so a single transaction covers all of them.
**Consequences.** Crash simulation must therefore be *logical* (abort between the log append and
the derived-index update inside a controlled sequence) rather than a real `SIGKILL`; this is what
the interrupted-write fault does. Documented as a threat to validity.

### D-003 — Pure-Python BM25 is the ranker; FTS5 is only a candidate filter
**Context.** FTS5 availability depends on how the local SQLite was compiled, and `bm25()` scores
differ between builds. A reproducibility artifact cannot have machine-dependent ranking.
**Decision.** Rank with an in-repo BM25 (`bm25_py_v1`, k1=1.5, b=0.75, ties broken on document id
so the ordering is total). FTS5 is probed at startup and reported by `memcheck doctor`; where
present it is used only to narrow candidates, never to score.
**Consequences.** Retrieval scores are identical on every machine. The ranker identity is written
into every manifest so results are never silently compared across rankers.

### D-004 — Semantic retrieval is optional and off by default
**Context.** A pinned local embedding model is a heavy dependency for a CI-safe artifact.
**Decision.** Ship the embedding provider *interface* plus a deterministic hash-embedding mock;
`sentence-transformers` is an optional extra. The default configuration uses lexical retrieval.
**Consequences.** Reported retrieval numbers are lexical unless a config sets
`retrieval.semantic: true`, which is recorded in the manifest.

### D-005 — Backends return structured results, never printed text
Every backend method returns a Pydantic model. The CLI is the only layer that renders.

### D-006 — The default provider is a deterministic *oracle reader* mock
**Context.** The research questions ask about the memory layer, not about language-model ability.
Running a frontier LLM on every checkpoint is also expensive and non-deterministic.
**Decision.** The mock provider answers a query correctly **iff** the bounded evidence it receives
actually contains the gold fact, abstains when the evidence is missing/weak/conflicted, and
answers *incorrectly* when the evidence contains a stale or superseded value. It never uses
knowledge outside the injected evidence.
**Consequences.** Mock-provider results measure *memory-system quality with a perfect reader*.
They are an upper bound on end-to-end agent accuracy, not a substitute for it. Every table and
figure produced with the mock provider is labelled as such, and the limitation is stated in the
README and in the threats-to-validity section. Real-provider runs use the identical prompts and
budgets so the comparison stays fair.

### D-007 — Restart equivalence is defined on a canonical projection, not on bytes
**Context.** Byte equality of a SQLite file across restarts is neither achievable nor meaningful.
**Decision.** Two states are *equivalent* when (a) the multiset of `(tenant_id, canonical_key,
version, content, status, valid_from, valid_to)` over non-deleted records is equal, and (b) for
every probe query in the workload's probe set, the bounded retrieval result is equal on the
ordered list of `(memory_id, version)` and on the abstention decision. Latency, physical ids,
row order, and file size are excluded.
**Consequences.** This rule is implemented once in `contracts/restart_equivalence.py` and reused
by the fault harness so the reported "restart divergence" metric has exactly one definition.

### D-008 — Deletion is closure over every derived surface
A deletion request writes a tombstone in the same transaction that (a) marks matching records
`deleted`, (b) purges FTS/BM25 rows, (c) invalidates summaries derived from the deleted evidence,
and (d) evicts affected retrieval-cache entries. The deletion-closure contract probes all four
surfaces plus historical (`as_of_time`) queries, because a deleted fact must not be retrievable
even through a time-travel query.

### D-009 — Tenant isolation is enforced at the storage layer, not the prompt layer
Every query in the guarded backend carries `tenant_id` in the SQL `WHERE` clause and in the FTS
match; a missing tenant filter raises rather than defaulting to "all tenants". Cross-tenant
leakage is classified `critical` severity.

### D-010 — Faults are deterministic, seeded, and composable
Each fault is a context manager over a backend that records `expected_recovery_behavior` and
`observed_recovery_behavior`. Randomness comes from an explicit `random.Random(seed)` passed in;
no module-level RNG state is used anywhere in the fault package.

### D-011 — Raw results are immutable and content-addressed
`results/raw/<run_id>/` is written once. After the run, `checksums.json` records the SHA-256 of
every artifact; `memcheck verify-integrity` re-checks them. Processing writes only to
`results/processed/`. The reporting layer never opens a raw file for writing.

### D-012 — The paper generator refuses unsupported numeric claims
`paper/claims_registry.yaml` binds each empirical claim to run ids, a source file, rows, and an
analysis script. The generator emits a claim's number only when its status is `supported` or
`partially_supported` *and* the referenced artifact exists and hashes as expected. Otherwise it
emits `[RESULT PENDING: ...]`.

### D-013 — RepoEvolutionBench grading is deterministic unit tests, not an LLM judge
Coding tasks are graded by running an allowlisted `pytest` command inside a temporary sandbox
copy of the repository, with a timeout and no network. The model (or mock) may only return a
unified diff or a structured file edit; arbitrary shell commands are never executed.

### D-014 — LLM-judged grading always carries a human audit path
Where deterministic grading is impossible, the judge prompt is version-controlled and hashed into
the manifest, and `memcheck audit` exports a deterministic stratified sample (correct, incorrect,
abstained, disputed) as an editable CSV, then computes human/judge agreement.

### D-015 — Pricing is dated configuration, never a hard-coded constant
`configs/pricing.example.yaml` carries an `as_of` date and a source URL per model. Cost estimates
print the pricing file's date. Missing pricing for a model is an error on a paid run, not a
silent zero.

### D-016 — Idempotency lives in the append path, not in the schema
**Context.** The event log originally carried `UNIQUE (tenant_id, event_id)`. That made the
`abl_no_idempotency` ablation impossible to run: 21 of 210 conditions in the full matrix died
with an `IntegrityError` instead of exhibiting duplicate amplification.
**Decision.** The uniqueness constraint becomes a plain index. Duplicate detection is performed
solely by the append path, which the ablation disables.
**Consequences.** The guarded backend's idempotency is now exactly one mechanism rather than two
overlapping ones, and every guard in the system is ablatable by construction. A store running the
ablation genuinely stores duplicates, which is what the duplicate-replay contract then detects.
The lesson generalises: a property enforced in two places cannot be measured, because removing
one enforcement point only reveals the other.

### D-017 — Extraction is a policy behind an interface, not a method on the backend
**Context.** The guarded backend decided *what* to remember and *how to manage what it
remembered* in the same class. The LongMemEval result then showed that the first decision
dominates the second — the lifecycle passed all six contracts while the system scored below a
transcript baseline — and there was no way to vary one while holding the other fixed.
**Decision.** Extraction moves behind an `Extractor` interface with a rule-based and an
LLM implementation. The extractor's identity is recorded in the manifest and is a comparability
field, so the fairness validator refuses to pool runs whose write policy differs.
**Consequences.** The guarded backend becomes a controlled instrument: a lifecycle known to be
correct, against which write policies can be compared. Both extractors compute canonical keys
the same way, so a difference between them is a difference in *selection* rather than in how
identity is defined — without that, the comparison would measure the interface.

### D-018 — Contracts always run with the rule-based extractor
**Context.** A run configured with an LLM write policy could plausibly run its contracts with
that policy too.
**Decision.** It does not. Contracts construct their own scenarios with controlled phrasing and
always use the rule-based extractor.
**Consequences.** A contract result stays a statement about the lifecycle. Were contracts to run
under a model, a deletion-closure failure could mean the store leaked *or* that the extractor
never produced the fact to delete — and those are the two things the study exists to tell apart.
The cost is that contracts do not exercise the model-extraction path; that path is covered by
the workload runs instead.

### D-019 — Cross-scale comparison is a separate analysis, never a pooled report
**Context.** Store size is part of the workload parameter hash, so each size is its own
comparability group and the reporting layer will not pool them — correctly. But the question of
whether a benchmark's verdict depends on its size can only be asked across sizes.
**Decision.** `memcheck scale` reaches across groups without pooling them: each size keeps its
own accuracy, its own paired tests, and its own violation counts, and what is compared is the
conclusion each size supports. It selects the family of runs differing in nothing but size,
reports what it excluded and why, and flags a separation that appears, vanishes, and reappears
rather than reporting it as a detection.
**Consequences.** One analysis is permitted to do what the rest of the pipeline forbids, and it
is permitted only because it never averages across the boundary.

### D-021 — A length stop during extraction is retried once, then counted
**Context.** On the 100-instance LongMemEval run with the LLM write policy, one event's
extraction call exhausted the 4000-token output budget on reasoning and raised; the runner
treated that as a condition failure, so two of the three guarded seeds were lost after most of
their cost had been spent (2026-10-21).
**Decision.** The provider raises a typed `ProviderLengthError`. `LLMExtractor.extract` retries
that one call with four times the budget; if the retry also stops on length, the event
contributes no candidates and `truncated_responses` is incremented. The answer-side budget is
unchanged, so the pair of runs still differs only in the write policy.
**Consequences.** A single over-long event can no longer void a seed. The count is a per-run
quality figure; a run in which it is non-zero should say so.

### D-022 — Seed replicas on a deterministic benchmark are collapsed before pairing
**Context.** LongMemEval is a fixed dataset. The three seeds re-issue identical requests, the
run's exact-match cache answers them identically, and on every run so far every per-query
outcome was the same under every seed. A paired test keyed on (query, seed) therefore counted
each query three times: the 20-instance pair reported 27 discordant of 60, which is 9 of 20.
**Decision.** The write-policy analysis checks whether every condition's outcomes agree across
seeds for every query; if they do, it pairs on the query alone and says so in the table note. If
any query disagrees, the seeds carried information and the (query, seed) pairing is kept.
**Consequences.** Reported $n$, discordant counts, intervals, and $p$-values for LongMemEval are
per unique query. The per-group `paired_comparisons.csv` and `task_utility.csv` for LongMemEval
still count seeds (see DEVIATIONS V-015); the paper cites the write-policy table for those
quantities. ~~The synthetic workloads are unaffected: there the seed selects a different stream.~~
**Corrected 2026-11-03:** that sentence was false. The runner built every workload from the first
seed, so synthetic and repository seeds were replicas too (DEVIATIONS V-018, D-024).

### D-023 — The sweep figure tells conditions apart by colour and end labels only

**Context.** The first sweep figure used four marker shapes, four dash patterns, a filled/hollow
marker code for significance with an explanatory box, a second panel of grouped bars repeating
the same three violation counts at every size, and a provenance footer printed under the plot.
Read at column width it looked like chart furniture, and the user said so.
**Decision.** One panel. Solid lines in the palette colours, a translucent Wilson band per line,
and the condition's name and violation count written at the end of its line. Significance marks
live in the verdict table. The PDF that goes into the paper carries no footer; the PNG and the
provenance sidecar keep it, so traceability is unchanged.
**Consequences.** The figure no longer encodes significance; the caption points to Table 7 for
it. `SweepSeries` lost its marker and linestyle fields. `_finish` gained `pdf_footer`. The
write-policy figure follows the same rule: paired bars (light tint before, full colour after)
with the value at each bar end replaced the hollow/filled/half-filled marker code, and its PDF
also drops the footer.

### D-020 — Paired outcomes get a paired test
**Context.** No significance claim was made anywhere, because comparing two accuracy rates as
independent samples would overstate the evidence.
**Decision.** Conditions answer the same query set, so their per-query outcomes are paired and
compared with McNemar — exact binomial below 25 discordant pairs, chi-square with continuity
correction above — Holm-adjusted within each checkpoint or store size.
**Consequences.** The discordant-pair count is reported alongside every verdict, because it is
the comparison's real sample size: a "significant" result resting on six discordant pairs is
read differently from one resting on two hundred, and the tombstone ablation is exactly that
case.

### D-024 — Seeds are separate streams; ten of them, fixed before the re-run (2026-11-03)

**Context.** An audit found that `build_workload` always used `config.seeds[0]`, so the three
seeds of every synthetic and repository run replayed one stream. Every n, interval, discordant
count and p-value in the sweep and the matrix was inflated threefold. The v1 generator also
graded each current-value fact twice at the final checkpoint (`_cur` and `_xs`, identical
requests). Counted once, the earlier sweep separates the transcript baseline at 40 sessions, not
24, and never separates the tombstone ablation.
**Decision.** The runner builds one stream per seed for any workload that declares
`seed_sensitive`. The synthetic workload moves to v2, which drops `_xs`. The re-run uses **ten
seeds (1–10)** for the five sweep sizes and the 40-session matrix. The number is set here, before
any v2 result exists, for power: about 60–100 unique questions per seed, so roughly 600–1,000
per size. It will not be changed after the results are seen. RepoEvolutionBench and LongMemEval
are seed-insensitive and run once under the mock reader; their extra seeds were replicas.
Reports test on unique units and collapse seeds that reproduce each other exactly, and say so.
**Consequences.** The three-seed v1 runs move, unchanged and still checksummed, to
`results/raw_superseded/` so that no report pools them with v2. The paper reports the v2 numbers
and states that the first draft's numbers were inflated and why. Seeds are clusters (one
stream each), and queries within a stream share a store; intervals treat queries as independent
within a size, which the threats section states.

### D-025 — The LongMemEval judge is the benchmark's own (2026-11-03)

**Context.** `judge_v1` was a generic prompt on the reader model. LongMemEval specifies per-type
prompts (temporal off-by-one tolerance, knowledge-update leniency, a preference rubric, an
abstention prompt) and a separate metric model.
**Decision.** Grade with the official prompts, copied verbatim from `src/evaluation/evaluate_qa.py`
at commit d6dc8b5 and checked byte-for-byte against the upstream function, on
`gpt-4o-2024-08-06`, temperature 0, `max_tokens` 10, verdict `'yes' in reply`. Stored answers are
re-graded offline into a new run directory; the source run is never modified. The official judge
is primary because it is the benchmark's protocol, chosen before its numbers were seen; the
original judge's pair is reported beside it, not hidden.
**Consequences.** Prompt hashes are now a grouping and comparability field, so a re-graded run is
never pooled with its source. Grading calls are billed to the evaluation, not to a condition.

### D-026 — RepoEvolutionBench also runs with a real model writing the patch (2026-11-03)

**Context.** Under the oracle the patch is a one-line template filled with the resolved value, so
the benchmark tested memory, not an agent. The `coding_v1` prompt asked for a unified diff that
nothing parsed and promised file contents it was never sent.
**Decision.** `coding_v2` sends the one file the task may edit (never the tests, which hold the
gold) and asks for JSON edits under a strict schema; edits outside the task are rejected.
`configs/repo_evolution_openai.yaml` runs it with gpt-5-mini ($0.01 billed). The benchmark's
tests are unchanged: the one failing task (`"30 days"` vs. the pinned `"thirty-days"`) is
reported as a surface-form failure, not re-graded.

### D-027 — The human audit samples distinct answers by question type, blind (2026-11-03)

**Decision.** Unit: distinct (question, answer) pair across both re-graded runs. Strata: question
type, 12 each, seed 20261103. Excluded: abstentions on answerable questions (every grader marks
them incorrect). The rater sees question, date, gold, the official criterion in plain words, and
the answer; evidence, conditions and judge labels are in `audit_key.csv`. Agreement is reported
against both judges with bootstrap κ intervals.

### D-028 — Checkpoint-relative gold (workload v3) and the padding control (2026-11-04)

**Context.** The size sweep evaluates separately generated streams at their final size, and the
reviewer asked whether store size is really the only thing that changes with it (V-023).
**Decision.** v3 keeps v2's generator byte for byte (a pinned-digest test enforces it) and adds
only how queries are derived: one set per configured checkpoint, with gold taken from the
events delivered by that checkpoint, asked at exactly that checkpoint (`Query.asked_at_fraction`)
and never again, because a later checkpoint would grade a stale gold. Checkpoint fractions are
snapped to session ends so "16 sessions" means sixteen sessions. The padding control appends
sessions from an independently seeded generator so that the fact stream and query set are
identical at every padding level; padding sessions carry as many events as fact sessions.
Both the "topical" and "unrelated" styles are run, because they test different things: topical
padding shares query terms with the facts and can dilute lexical retrieval, unrelated padding
can only change corpus statistics. v3 also inherits a v2 quirk, left in place so the streams
match: after a contradiction, a "re-assert the same fact" draw reads the marker entry and emits
a fact whose value is the literal `__contradiction__` (816 of 14,944 events in the 64-session
sweep run; 11 of 932 graded golds at the full store are that string, through temporal queries).
A corrected generator would be v4 and would require re-running the sweep and matrix.
**Consequence.** The within-stream series and the padding series are reported beside the sweep,
never pooled with it; `memcheck scale` keeps them out of the sweep family because the workload
version and padding parameters are part of the sweep signature.

### D-029 — Contracts can run on a snapshot of the grown store (2026-11-04)

**Context.** Contracts build their scenario on an empty store, so "passed every contract at
every size" says nothing about a store that has absorbed 64 sessions of updates and deletions.
**Decision.** `contract_store: aged` makes the runner copy each condition's store after its
final checkpoint (SQLite online backup through a read-only connection, one copy per contract,
the live store untouched) and build the scenario on the copy; `ContractResult.store_state`
records which mode produced it. The default stays `fresh`, so every existing configuration and
result is unchanged. Five contracts write under their own tenant, which the workload never uses,
so on an aged store they meet the grown content only through the shared lexical index, the
retrieval cache and the write journal. Restart equivalence is different: it compares the full
non-deleted record set and probe results before and after a restart, so on an aged store it
checks the grown store itself. The aged results are reported as that, not as proof that the
other five invariants hold over the workload's own history; the stale and leak labels at the
final checkpoint and the fault-recovery matrix carry that evidence.

### D-030 — Workload v4 corrects two generator defects; v2 kept, v4 run as a robustness check (2026-11-04)

**Context.** D-028 recorded two defects inherited by v2 and v3: re-assertions after a
contradiction emitted the marker string as a fact value, and temporal gold was often the current
value (V-024).
**Decision.** Fix both behind `version: v4`, leaving v2 and v3 byte-identical so every reported
run stays reproducible. For the marker, read the last real assertion as the current value. This
is the least invasive fix: it adds no random draw, so the stream matches v2 up to the first
affected event. For temporal gold, use the value current just before the latest change of
value, at an instant strictly between its assertion and the change. Keys with no such instant
get no temporal query rather than an ambiguous one. The sweep and the 40-session matrix were
rerun under v4 with otherwise identical configurations, and `scripts/v4_robustness.py` compares
them with v2 through the same `scale.analyse` code the paper uses.
**Consequence.** v4 runs form their own report group (workload version is a grouping field)
and do not enter the paper's v2 tables. Which version the manuscript reports as primary is an
open decision for the author. The matrix and both ablation verdicts are unchanged. The
transcript-chunking verdict at 16 and 24 sessions is not.

### D-031 — The corrected labels become the primary synthetic evidence (2026-11-04)

**Context.** D-030 left open which workload version the paper reports. V-024 showed that the
v2 labels contain degenerate temporal questions and marker-valued golds, and that the
transcript-chunking verdict at 16 and 24 sessions changes when they are corrected.
**Decision.** Report the corrected labels. The sweep and the matrix use workload v4. The
longitudinal run and the padding control need v3's checkpoint-relative queries, so workload v5
combines those with v4's two fixes. Its final checkpoint asks exactly what v4 asks (pinned by
test), and v2, v3 and v4 output is byte-identical to the code before v5 existed.
`PRIMARY_WORKLOAD_VERSION = "v4"` selects the report's primary group, and `memcheck scale` now
selects the sweep of that version instead of breaking a tie between two equally wide sweeps.
**Consequence.** All synthetic numbers in the paper change. The v2 outputs remain in
`results/processed/_v2_archive/` and the v2 runs in `results/raw`, and
`scripts/v4_robustness.py` keeps the v2-against-v4 comparison, so the paper can state what moved.

