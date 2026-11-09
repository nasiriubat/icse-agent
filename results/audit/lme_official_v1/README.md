# Blinded audit of the LongMemEval judge

Open `audit_blind.csv` only. Do not open `audit_key.csv` until you have finished.

For each row:

1. Read `question`, `gold_answer` and `criterion`. The criterion is the benchmark's own rule for that question type.
2. Read `model_answer` and decide whether it meets the criterion **against the gold answer**. You are not shown the memory evidence on purpose: grade only the answer.
3. Write `correct` or `incorrect` in `human_label`. If you truly cannot decide, write `unsure` and say why in `human_notes`; such rows are reported, not scored.
4. Change no other column.

Then run `uv run memcheck audit --lme-agreement results/audit/lme_official_v1/audit_blind.csv --lme-key results/audit/lme_official_v1/audit_key.csv`.

The sample has 60 distinct (question, answer) pairs, 12 per question type, drawn with seed 20261103 from 193 distinct pairs in: longmemeval_openai_n100_20261021T100428Z_295868b7_rejudged_lme, longmemeval_openai_llm_extract_n100_20261022T094454Z_bc11e1d5_rejudged_lme. Answers that abstain on an answerable question are excluded, because every grader marks them incorrect.
