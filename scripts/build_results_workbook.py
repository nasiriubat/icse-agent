"""Gather every derived number the paper reports into one workbook for readers outside the repo.

Values are copied verbatim from results/processed/ (and the spend ledger), which are derived from
the checksummed raw runs. Run after `memcheck report` and the scripts under scripts/.
Writes results/memcheck_results.xlsx.

    uv run python scripts/build_results_workbook.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font
from openpyxl.utils import get_column_letter

PROCESSED = Path("results/processed")
OUTPUT = Path("results/memcheck_results.xlsx")
MATRIX_DIR = PROCESSED / (
    "synthetic_longitudinal_v4_195f8c8d_mock_oracle-reader-v1_rule_copula_v1_35b77a_30ac58_9554e0_fresh"
)
MATRIX_RUN = "experiment_matrix_v4_20261104T115543Z_e931c4ed"
REPO_RUNS = (
    ("repo_evolution_20261103T145036Z_e95bae10", "reference reader"),
    ("repo_evolution_openai_20261103T145029Z_2ee8bc08", "gpt-5-mini writes the patch, attempt 1"),
    ("repo_evolution_openai_attempt2_20261104T112800Z_b0a84b77", "gpt-5-mini, attempt 2"),
)
SWEEP_RUNS = "scale_v4_008/016/024/040/064 (20261104T1154-1155)"
LONGITUDINAL_RUN = "longitudinal_v5_064_20261104T122046Z_3d66d637"

FONT = Font(name="Arial", size=10)
BOLD = Font(name="Arial", size=10, bold=True)
TITLE = Font(name="Arial", size=11, bold=True)
ITALIC = Font(name="Arial", size=10, italic=True)


def convert(value: str | None) -> Any:
    if value is None or value == "":
        return None
    if value in ("True", "False"):
        return value == "True"
    try:
        return int(value) if value.lstrip("-").isdigit() else float(value)
    except ValueError:
        return value


def read_csv(path: Path, keep: Any = None) -> tuple[list[str], list[list[Any]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        records = list(csv.DictReader(handle))
    if keep is not None:
        records = [r for r in records if keep(r)]
    if not records:
        return [], []
    columns = list(records[0])
    return columns, [[convert(r.get(c)) for c in columns] for r in records]


def flatten(value: Any, prefix: str = "") -> list[tuple[str, Any]]:
    if isinstance(value, dict):
        out: list[tuple[str, Any]] = []
        for key, inner in value.items():
            out.extend(flatten(inner, f"{prefix}.{key}" if prefix else str(key)))
        return out
    if isinstance(value, list):
        if all(not isinstance(x, (dict, list)) for x in value):
            return [(prefix, ", ".join(str(x) for x in value))]
        out = []
        for index, inner in enumerate(value):
            out.extend(flatten(inner, f"{prefix}[{index}]"))
        return out
    return [(prefix, value)]


class Sheet:
    def __init__(self, workbook: Workbook, name: str) -> None:
        self.ws = workbook.create_sheet(name)
        self.row = 1
        self.header_rows: list[int] = []
        self.rows = 0

    def title(self, text: str) -> None:
        self.ws.cell(self.row, 1, text).font = TITLE
        self.row += 1

    def note(self, text: str) -> None:
        self.ws.cell(self.row, 1, text).font = ITALIC
        self.row += 1

    def blank(self) -> None:
        self.row += 1

    def table(self, columns: list[str], rows: list[list[Any]]) -> None:
        for j, header in enumerate(columns, 1):
            self.ws.cell(self.row, j, header).font = BOLD
        self.header_rows.append(self.row)
        self.row += 1
        for values in rows:
            for j, value in enumerate(values, 1):
                self.ws.cell(self.row, j, value).font = FONT
            self.row += 1
        self.rows += len(rows)

    def csv(self, path: Path, heading: str | None = None, keep: Any = None) -> None:
        if heading:
            self.title(heading)
        self.note(f"Source: {path}")
        self.table(*read_csv(path, keep))
        self.blank()

    def json(self, path: Path, heading: str | None = None) -> None:
        if heading:
            self.title(heading)
        self.note(f"Source: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.table(["field", "value"], [[k, v] for k, v in flatten(payload)])
        self.blank()

    def finish(self) -> None:
        if len(self.header_rows) == 1:
            self.ws.freeze_panes = self.ws.cell(self.header_rows[0] + 1, 1)
        widths: dict[int, int] = {}
        for line in self.ws.iter_rows():
            for cell in line:
                if cell.value is None or cell.font in (TITLE, ITALIC):
                    continue
                widths[cell.column] = max(widths.get(cell.column, 0), min(len(str(cell.value)), 60))
        for column, width in widths.items():
            self.ws.column_dimensions[get_column_letter(column)].width = width + 2


def repo_dir(run_id: str) -> Path:
    for directory in sorted(PROCESSED.glob("repo_evolution_v1_*_fresh")):
        _, rows = read_csv(directory / "manifests.csv")
        if rows and rows[0][0] == run_id:
            return directory
    raise SystemExit(f"no processed directory for {run_id}")


def main() -> None:
    workbook = Workbook()
    workbook.remove(workbook.active)
    readme = Sheet(workbook, "README")
    index: list[list[Any]] = []

    def add(name: str, sources: str, backs: str, runs: str, build: Any) -> None:
        sheet = Sheet(workbook, name)
        build(sheet)
        sheet.finish()
        index.append([name, sources, backs, runs, sheet.rows])

    scale = PROCESSED / "scale"
    add(
        "RQ1 sweep",
        "scale/scale_accuracy.csv",
        "Table VII, Fig. 2, Sec. V-A",
        SWEEP_RUNS,
        lambda s: s.csv(scale / "scale_accuracy.csv"),
    )
    add(
        "RQ1 stream-level",
        "scale/scale_accuracy_streams.csv",
        "Sec. IV statistics, V-A, VII",
        SWEEP_RUNS,
        lambda s: s.csv(scale / "scale_accuracy_streams.csv"),
    )
    add(
        "RQ1 question types",
        "scale/gap_by_category.csv; gap_standardised.csv",
        "Sec. V-A (question mix)",
        SWEEP_RUNS,
        lambda s: (
            s.csv(scale / "gap_by_category.csv", "Gap within each question type"),
            s.csv(scale / "gap_standardised.csv", "Type-standardised gap"),
        ),
    )
    add(
        "RQ1 answer labels",
        "scale/label_counts.csv",
        "Sec. V-B (stale and leaked answers)",
        SWEEP_RUNS,
        lambda s: s.csv(scale / "label_counts.csv"),
    )
    add(
        "RQ1 one growing store",
        "longitudinal/checkpoints.csv; gap_vs_guarded.csv",
        "Sec. V-A (one store as it grows)",
        LONGITUDINAL_RUN,
        lambda s: (
            s.csv(PROCESSED / "longitudinal/gap_vs_guarded.csv", "Gap to guarded per checkpoint"),
            s.csv(PROCESSED / "longitudinal/checkpoints.csv", "Metrics per checkpoint"),
        ),
    )
    add(
        "RQ1 padding control",
        "padding/gap_by_padding.csv",
        "Sec. V-A (updates drive the gap)",
        "padding_v5_{topical,unrelated}_{000..056} (20261104T1221-1222)",
        lambda s: s.csv(PROCESSED / "padding/gap_by_padding.csv"),
    )
    add(
        "RQ2 leading indicator",
        "scale/scale_violations.csv; scale_leading_indicator.csv",
        "Table VIII, Sec. V-B",
        SWEEP_RUNS,
        lambda s: (
            s.csv(scale / "scale_leading_indicator.csv", "First separation by accuracy"),
            s.csv(scale / "scale_violations.csv", "Contract violations per run, by size"),
        ),
    )
    add(
        "RQ2 matrix contracts",
        f"{MATRIX_DIR.name}/contract_violations.csv; reliability.csv",
        "Table II, Sec. V-B (faults)",
        MATRIX_RUN,
        lambda s: (
            s.note(
                "Violation counts are summed over the ten seeds; the paper prints them per run."
            ),
            s.csv(
                MATRIX_DIR / "contract_violations.csv", "Contract violations (sum over 10 seeds)"
            ),
            s.csv(MATRIX_DIR / "reliability.csv", "Fault recovery"),
        ),
    )
    add(
        "RQ2 matrix efficiency",
        f"{MATRIX_DIR.name}/efficiency.csv (fault none, cp100)",
        "Table V, Sec. V-B",
        MATRIX_RUN,
        lambda s: s.csv(
            MATRIX_DIR / "efficiency.csv",
            "Efficiency at the full store, no fault",
            keep=lambda r: r["fault"] == "none" and r["checkpoint"] == "cp100",
        ),
    )
    add(
        "RQ2 aged contracts",
        "contracts/aged_vs_fresh.csv",
        "Sec. V-B (grown store)",
        "scale_v4_064 (fresh) vs scale_v4_064_aged_contracts_20261104T123312Z_e2820662",
        lambda s: s.csv(PROCESSED / "contracts/aged_vs_fresh.csv"),
    )
    wp = PROCESSED / "write_policy"
    wp1 = PROCESSED / "write_policy_judge_v1"
    selection = json.loads((wp / "selection.json").read_text(encoding="utf-8"))
    add(
        "RQ3 write policy",
        "write_policy/{conditions,categories,saturation}.csv",
        "Table IX, Fig. 3, Sec. V-C",
        ", ".join(selection["paired_run_ids"]),
        lambda s: (
            s.csv(wp / "conditions.csv", "Conditions"),
            s.csv(wp / "categories.csv", "Question types"),
            s.csv(wp / "saturation.csv", "Reader saturation"),
        ),
    )
    add(
        "RQ3 paired tests",
        "write_policy/transcript_vs_guarded.json; extract_repeat.json",
        "Sec. V-C",
        "LongMemEval n100 runs (both extractors, repeat)",
        lambda s: (
            s.json(wp / "transcript_vs_guarded.json", "Transcript vs guarded (model-based run)"),
            s.json(wp / "extract_repeat.json", "Repeat of the model-based extraction run"),
        ),
    )
    add(
        "RQ3 full history",
        "write_policy/full_history_split.csv; full_history_transcript_vs_guarded.json",
        "Sec. V-C (irrelevant history)",
        "longmemeval_openai_s_n100_20261104T111853Z_2f3d473e_rejudged_lme",
        lambda s: (
            s.csv(wp / "full_history_split.csv", "Oracle split vs S split"),
            s.json(
                wp / "full_history_transcript_vs_guarded.json", "Transcript vs guarded, S split"
            ),
        ),
    )
    add(
        "RQ3 original judge",
        "write_policy_judge_v1/{conditions,categories,saturation}.csv",
        "Sec. VII (same conclusions under the first judge)",
        "same runs, original judge",
        lambda s: (
            s.csv(wp1 / "conditions.csv", "Conditions"),
            s.csv(wp1 / "categories.csv", "Question types"),
            s.csv(wp1 / "saturation.csv", "Reader saturation"),
        ),
    )
    audit = PROCESSED / "audit"
    add(
        "Judge audit",
        "audit/judge_vs_judge_lme.json; lme_human_agreement.json",
        "Sec. VII (the LongMemEval judge)",
        "both n100 re-graded runs",
        lambda s: (
            s.json(audit / "judge_vs_judge_lme.json", "Official vs original judge"),
            s.json(audit / "lme_human_agreement.json", "Human rater vs judges (60 blinded rows)"),
        ),
    )

    def repo(sheet: Sheet) -> None:
        for run_id, label in REPO_RUNS:
            directory = repo_dir(run_id)
            sheet.csv(directory / "coding_outcomes.csv", f"{run_id} ({label}): per condition")
            sheet.csv(directory / "raw_coding_tasks.csv", f"{run_id}: per task")

    add(
        "Repo evolution",
        "repo_evolution_v1_*_fresh/{coding_outcomes,raw_coding_tasks}.csv",
        "Sec. V-A (agent workload), VII",
        ", ".join(run for run, _ in REPO_RUNS),
        repo,
    )
    add(
        "v2 vs v4 labels",
        "v4/{sweep_comparison,first_separation,matrix_comparison,category_gap_064}.csv",
        "Sec. VI-A (degenerate gold labels)",
        "scale_v2_* vs scale_v4_*; experiment_matrix vs experiment_matrix_v4",
        lambda s: (
            s.csv(PROCESSED / "v4/sweep_comparison.csv", "Sweep, old vs corrected labels"),
            s.csv(PROCESSED / "v4/first_separation.csv", "First separation"),
            s.csv(PROCESSED / "v4/matrix_comparison.csv", "Matrix"),
            s.csv(PROCESSED / "v4/category_gap_064.csv", "Gap by type at 64 sessions"),
        ),
    )
    ledger = PROCESSED / "spend/ledger.csv"
    add(
        "Spend",
        "spend/ledger.csv",
        "Sec. IV (billed spend)",
        "every paid run, failed attempts included",
        lambda s: s.csv(ledger),
    )

    _, rows = read_csv(ledger)
    total = sum(float(r[-1]) for r in rows)
    readme.title("MemCheck results workbook")
    readme.note(
        "Every derived number the submission reports, copied verbatim from results/processed/,"
        " which is derived from the immutable, checksummed raw runs in results/raw/<run_id>/."
    )
    readme.note(
        "The sweep, matrix, one-growing-store and padding sheets use the corrected gold labels"
        " (workloads v4/v5, DEVIATIONS V-024 and V-025). The earlier v2 numbers are archived in"
        " results/processed/_v2_archive/ and compared in the 'v2 vs v4 labels' sheet."
    )
    readme.blank()
    readme.table(
        ["sheet", "source (under results/processed/)", "backs", "run ids", "data rows"], index
    )
    readme.blank()
    readme.note(f"Spend ledger total billed_usd = {total:.2f}.")
    readme.finish()
    for line in readme.ws.iter_rows():
        for cell in line:
            cell.alignment = Alignment(vertical="top")
    workbook.save(OUTPUT)
    print(OUTPUT)
    for name, *_, count in index:
        print(f"{name}: {count}")


if __name__ == "__main__":
    main()
