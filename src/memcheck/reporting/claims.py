"""Claims registry (Section 21) and generated paper macros.

The registry is the contract between the results and the prose. A claim may print a number only
when its status is ``supported`` or ``partially_supported`` **and** the artifact it names exists
and actually contains the value. Everything else renders as ``[RESULT PENDING: ...]``.

This is the mechanism that makes "no fabricated numbers" enforceable rather than aspirational.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import pandas as pd
import yaml
from pydantic import Field

from memcheck.models import StrictModel

VALID_STATUS = ("planned", "supported", "partially_supported", "unsupported", "pending")
PRINTABLE_STATUS = ("supported", "partially_supported")


class Claim(StrictModel):
    claim_id: str
    claim_text: str
    status: Literal["planned", "supported", "partially_supported", "unsupported", "pending"]
    research_question: str
    metric: str
    run_ids: list[str] = Field(default_factory=list)
    source_file: str | None = None
    source_rows: str | None = None
    table_or_figure: str | None = None
    analysis_script: str | None = None
    statistical_test: str | None = None
    notes: str = ""
    #: Optional value binding: which column of which processed file the macro reads.
    value_column: str | None = None
    value_filter: dict[str, Any] = Field(default_factory=dict)
    value_format: str = "{:.3f}"


class ClaimsRegistry(StrictModel):
    version: str = "1"
    claims: list[Claim] = Field(default_factory=list)


@dataclass
class ClaimResolution:
    claim: Claim
    macro: str
    rendered: str
    printable: bool
    reason: str


def load_registry(path: str | Path) -> ClaimsRegistry:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return ClaimsRegistry.model_validate(payload)


def _claim_dirs(claim: Claim, processed_dirs: dict[str, Path]) -> list[Path]:
    """The processed directories belonging to the runs this claim actually cites.

    Resolving a claim against *any* group's file is how a claim about one workload silently
    picks up another workload's number — the exact failure this registry exists to prevent. A
    claim is therefore resolved only within the runs it names.
    """
    directories: list[Path] = []
    for run_id in claim.run_ids:
        directory = processed_dirs.get(run_id)
        if directory is not None and directory not in directories:
            directories.append(directory)
    return directories


def _find_source(processed_dirs: list[Path], source_file: str) -> Path | None:
    for directory in processed_dirs:
        candidate = directory / source_file
        if candidate.exists():
            return candidate
    return None


def validate_registry(
    registry: ClaimsRegistry, processed_dirs: dict[str, Path], known_run_ids: set[str]
) -> list[str]:
    """Structural validation. Returns a list of problems; empty means the registry is sound."""
    problems: list[str] = []
    seen: set[str] = set()
    for claim in registry.claims:
        if claim.claim_id in seen:
            problems.append(f"{claim.claim_id}: duplicate claim_id")
        seen.add(claim.claim_id)
        if claim.status in PRINTABLE_STATUS:
            if not claim.run_ids:
                problems.append(
                    f"{claim.claim_id}: status is {claim.status} but no run_ids are recorded"
                )
            if not claim.source_file:
                problems.append(
                    f"{claim.claim_id}: status is {claim.status} but no source_file is recorded"
                )
            elif _find_source(_claim_dirs(claim, processed_dirs), claim.source_file) is None:
                problems.append(
                    f"{claim.claim_id}: source_file {claim.source_file!r} does not exist under "
                    f"the processed results of the run(s) it cites ({claim.run_ids})"
                )
            unknown = [run for run in claim.run_ids if run not in known_run_ids and run != "*"]
            if unknown:
                problems.append(
                    f"{claim.claim_id}: cites run ids that are not present under results/raw: "
                    f"{unknown}"
                )
            if not claim.analysis_script:
                problems.append(
                    f"{claim.claim_id}: status is {claim.status} but no analysis_script is named"
                )
        if "significant" in claim.claim_text.lower() and not claim.statistical_test:
            problems.append(
                f"{claim.claim_id}: the claim text says 'significant' but names no statistical_test"
            )
    return problems


def resolve_value(claim: Claim, processed_dirs: dict[str, Path]) -> tuple[str, bool, str]:
    """Read the claim's bound value out of the processed results.

    Returns ``(rendered, printable, reason)``. A claim never invents a value: if the binding
    does not resolve to exactly one row, the macro renders as pending.
    """
    if claim.status not in PRINTABLE_STATUS:
        return (
            f"[RESULT PENDING: claim {claim.claim_id} has status {claim.status}]",
            False,
            f"status={claim.status}",
        )
    if not claim.source_file or not claim.value_column:
        return (
            f"[RESULT PENDING: claim {claim.claim_id} has no value binding]",
            False,
            "no source_file/value_column binding",
        )
    directories = _claim_dirs(claim, processed_dirs)
    if not directories:
        return (
            f"[RESULT PENDING: claim {claim.claim_id} cites no run present under results/raw]",
            False,
            "no cited run is available",
        )
    path = _find_source(directories, claim.source_file)
    if path is None:
        return (
            f"[RESULT PENDING: {claim.source_file} has not been generated for {claim.run_ids}]",
            False,
            "missing source file",
        )
    frame = pd.read_csv(path)
    for column, value in claim.value_filter.items():
        if column not in frame.columns:
            return (
                f"[RESULT PENDING: {claim.source_file} has no column {column!r}]",
                False,
                f"missing filter column {column}",
            )
        frame = frame[frame[column].astype(str) == str(value)]
    if claim.value_column not in frame.columns:
        return (
            f"[RESULT PENDING: {claim.source_file} has no column {claim.value_column!r}]",
            False,
            "missing value column",
        )
    values = frame[claim.value_column].dropna()
    if len(values) != 1:
        return (
            f"[RESULT PENDING: claim {claim.claim_id} resolved to {len(values)} rows, expected 1]",
            False,
            f"resolved {len(values)} rows",
        )
    value = values.iloc[0]
    try:
        rendered = claim.value_format.format(float(value))
    except (TypeError, ValueError):
        rendered = str(value)
    return rendered, True, "resolved"


def macro_name(claim_id: str) -> str:
    """LaTeX macro names may only contain letters, so digits are spelled out."""
    digits = str.maketrans("0123456789", "ZOTHFVSXEN")
    return "\\memcheck" + claim_id.replace("-", "").replace("_", "").translate(digits)


def render_macros(
    registry: ClaimsRegistry, processed_dirs: dict[str, Path], output_path: Path
) -> list[ClaimResolution]:
    """Write ``paper/generated/claims.tex`` with one macro per claim."""
    resolutions: list[ClaimResolution] = []
    lines = [
        "% Generated by `memcheck report`. Do not edit by hand.",
        "% Each macro renders a claim's value, or an explicit pending marker when the claim is",
        "% not supported by an artifact. Editing this file by hand defeats the guarantee.",
        "",
    ]
    for claim in registry.claims:
        rendered, printable, reason = resolve_value(claim, processed_dirs)
        name = macro_name(claim.claim_id)
        safe = rendered.replace("[", "{[}").replace("]", "{]}") if not printable else rendered
        lines.append(f"% {claim.claim_id}: {claim.status} -- {reason}")
        lines.append(f"\\newcommand{{{name}}}{{{safe}}}")
        resolutions.append(
            ClaimResolution(
                claim=claim, macro=name, rendered=rendered, printable=printable, reason=reason
            )
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return resolutions


def write_evidence_matrix(
    registry: ClaimsRegistry, output_path: Path, resolutions: list[ClaimResolution]
) -> Path:
    """Regenerate ``paper/evidence_matrix.csv`` from the registry (Section 22)."""
    by_id = {resolution.claim.claim_id: resolution for resolution in resolutions}
    rows = []
    for claim in registry.claims:
        resolution = by_id.get(claim.claim_id)
        rows.append(
            {
                "research_question": claim.research_question,
                "claim_id": claim.claim_id,
                "experiment_condition": claim.value_filter.get("variant", "all"),
                "baseline": claim.value_filter.get("baseline", ""),
                "workload": claim.value_filter.get("workload", ""),
                "metric": claim.metric,
                "run_id": ";".join(claim.run_ids),
                "artifact": claim.source_file or "",
                "paper_section": claim.table_or_figure or "",
                "status": claim.status,
                "resolved_value": resolution.rendered if resolution else "",
                "printable": bool(resolution and resolution.printable),
                "analysis_script": claim.analysis_script or "",
                "statistical_test": claim.statistical_test or "",
            }
        )
    frame = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    return output_path
