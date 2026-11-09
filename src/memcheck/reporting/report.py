"""Reporting orchestration: raw runs in, paper-ready artifacts out."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from memcheck.evaluation.fairness import GROUPING_FIELDS, validate_across_runs
from memcheck.evaluation.runner import discover_runs, load_manifest, verify_integrity
from memcheck.models import ExperimentRunManifest, sha256_text, utcnow
from memcheck.reporting import aggregate, audit, claims, figures, tables, workbook, writepolicy
from memcheck.reporting.provenance import Provenance


@dataclass
class ReportResult:
    run_ids: list[str]
    group: str = "all"
    processed_files: list[Path] = field(default_factory=list)
    figure_results: list[figures.FigureResult] = field(default_factory=list)
    table_results: list[tables.TableResult] = field(default_factory=list)
    workbook_result: workbook.WorkbookResult | None = None
    claim_resolutions: list[claims.ClaimResolution] = field(default_factory=list)
    registry_problems: list[str] = field(default_factory=list)
    integrity_problems: list[str] = field(default_factory=list)
    fairness_problems: list[str] = field(default_factory=list)
    reader_warnings: list[str] = field(default_factory=list)
    audit_export: audit.AuditExport | None = None
    processed_dir: Path | None = None
    write_policy: writepolicy.WritePolicyResult | None = None

    @property
    def ok(self) -> bool:
        return not (self.integrity_problems or self.fairness_problems or self.registry_problems)

    def summary(self) -> str:
        drawn = sum(1 for figure in self.figure_results if not figure.skipped)
        pending_tables = sum(1 for table in self.table_results if table.pending)
        printable = sum(1 for claim in self.claim_resolutions if claim.printable)
        lines = [
            f"group                : {self.group}",
            f"runs pooled          : {len(self.run_ids)}",
            f"processed files      : {len(self.processed_files)}",
            f"figures drawn        : {drawn}/{len(self.figure_results)}",
            f"tables written       : {len(self.table_results)} ({pending_tables} pending)",
            f"claims printable     : {printable}/{len(self.claim_resolutions)}",
        ]
        if self.workbook_result is not None:
            written = sum(1 for sheet in self.workbook_result.sheets if not sheet.skipped)
            where = self.workbook_result.path or self.workbook_result.reason
            lines.append(f"workbook sheets      : {written} -> {where}")
        for label, problems in (
            ("integrity", self.integrity_problems),
            ("fairness", self.fairness_problems),
            ("claims registry", self.registry_problems),
            ("reader", self.reader_warnings),
        ):
            if problems:
                lines.append(f"{label} problems:")
                lines.extend(f"  - {problem}" for problem in problems)
        for figure in self.figure_results:
            if figure.skipped:
                lines.append(f"figure skipped: {figure.name} ({figure.reason})")
        if self.write_policy is not None:
            lines.append(self.write_policy.summary())
        return "\n".join(lines)


def group_slug(group: str) -> str:
    """A filename-safe form of a comparability-group key.

    Colons are legal on POSIX but not on Windows, and the key contains one in its
    ``provider:model`` segment.
    """
    return group.replace("@", "_").replace("/", "_").replace(":", "-")


def _copy_workbook(result: ReportResult, group: str, target_dir: Path) -> None:
    if result.workbook_result is None or result.workbook_result.path is None:
        return
    source = result.workbook_result.path
    target_dir.mkdir(parents=True, exist_ok=True)
    (target_dir / f"figure_data_{group_slug(group)}.xlsx").write_bytes(source.read_bytes())


def _group_key(manifest: ExperimentRunManifest) -> str:
    """The identity of the experiment a run belongs to.

    Derived from :data:`GROUPING_FIELDS` rather than hand-assembled, so that adding a
    comparability field cannot silently leave the grouping behind. It did once: the extractor
    became a comparability field and this key did not change, which pooled two runs that differed
    in their write policy.
    """
    parts: list[str] = []
    for field_name in GROUPING_FIELDS:
        raw = getattr(manifest, field_name, "")
        if isinstance(raw, dict):
            value = sha256_text(json.dumps(raw, sort_keys=True))[:6] if raw else ""
        elif isinstance(raw, list):
            # A condition set is not readable inline; a short digest of it distinguishes runs
            # without making the key unusable as a directory name.
            value = sha256_text("|".join(sorted(str(item) for item in raw)))[:6] if raw else ""
        else:
            value = str(raw or "")
            # Hashes are long and carry no meaning to a reader; a prefix distinguishes them.
            if field_name.endswith("_hash"):
                value = value[:8]
        if value:
            parts.append(value)
    return "/".join(parts)


def comparability_groups(results_root: Path) -> dict[str, list[Path]]:
    """Partition runs into groups that may legitimately be pooled.

    Two runs of different workloads answer different questions at different checkpoints; pooling
    them would average incomparable numbers. Grouping by benchmark name and version keeps each
    report internally fair, and the cross-run fairness check then applies *within* a group where
    it is meaningful.
    """
    groups: dict[str, list[Path]] = {}
    for run_dir in discover_runs(results_root):
        manifest = load_manifest(run_dir)
        groups.setdefault(_group_key(manifest), []).append(run_dir)
    return dict(sorted(groups.items()))


#: The synthetic workload version whose matrix and size sweep supply the paper's main results.
#: v4 corrects two label defects in v2 (DEVIATIONS V-024); the primary evidence moved to it in
#: D-031. Runs of other versions explore exactly as many conditions, so without this pin the
#: choice would fall to a tie-break; which version is primary is a decision.
PRIMARY_WORKLOAD_VERSION = "v4"


def _primary_group(groups: dict[str, list[Path]]) -> str:
    """Choose the group whose tables become the paper's main results.

    The richest synthetic-longitudinal group of :data:`PRIMARY_WORKLOAD_VERSION` wins, measured
    by how many conditions it explores. Selecting the first by name would hand the paper a single
    point of the size sweep, whose fault table is empty by construction -- the sweep deliberately
    injects none.
    """

    def breadth(run_dirs: list[Path]) -> int:
        total = 0
        for run_dir in run_dirs:
            manifest = load_manifest(run_dir)
            total += (len(manifest.ablations) + 1) * max(1, len(manifest.faults))
        return total

    synthetic = [key for key in groups if key.startswith("synthetic_longitudinal")]
    pinned = [
        key
        for key in synthetic
        if key.startswith(f"synthetic_longitudinal/{PRIMARY_WORKLOAD_VERSION}/")
    ]
    candidates = pinned or synthetic or list(groups)
    return max(candidates, key=lambda key: (breadth(groups[key]), key))


def build_reports(
    results_root: Path,
    paper_dir: Path,
    *,
    primary: str | None = None,
    strict: bool = False,
    audit_sample_size: int = 60,
) -> list[ReportResult]:
    """Generate one report per comparability group.

    The *primary* group's tables and figures are the ones copied into ``paper/generated`` as the
    paper's main results; every other group is still fully generated under its own subdirectory
    so nothing is silently dropped.
    """
    groups = comparability_groups(results_root)
    if not groups:
        raise FileNotFoundError(
            f"no runs found under {results_root}; run `memcheck run --config configs/smoke.yaml`"
        )
    primary_key = primary or _primary_group(groups)
    all_run_ids = {run.name for dirs in groups.values() for run in dirs}
    results: list[ReportResult] = []
    for key, run_dirs in groups.items():
        slug = group_slug(key)
        results.append(
            build_report(
                results_root,
                paper_dir,
                group=key,
                run_dirs=run_dirs,
                processed_dir=Path("results/processed") / slug,
                figures_dir=Path("results/figures") / slug,
                tables_dir=Path("results/tables") / slug,
                audit_dir=Path("results/audit") / slug,
                copy_to_paper=(key == primary_key),
                audit_sample_size=audit_sample_size,
                strict=strict,
            )
        )

    # The write-policy comparison (D-017) is, with the scale sweep, the one analysis allowed to
    # reach across groups. It pairs two runs rather than pooling them, so it runs once, after
    # every group, and its outputs are attached to the primary group's report.
    for result in results:
        if result.group == primary_key:
            result.write_policy = writepolicy.build(results_root, paper_dir)

    # The registry cites runs and artifacts from every group, so it is validated and rendered
    # once, against the union of them, rather than per group.
    registry_path = paper_dir / "claims_registry.yaml"
    if registry_path.exists():
        registry = claims.load_registry(registry_path)
        # Map each run id to the processed directory of its own comparability group, so a
        # claim resolves against the runs it cites and never against another workload's file.
        processed_dirs = {
            run_id: result.processed_dir
            for result in results
            if result.processed_dir is not None
            for run_id in result.run_ids
        }
        problems = claims.validate_registry(registry, processed_dirs, all_run_ids)
        resolutions = claims.render_macros(
            registry, processed_dirs, paper_dir / "generated" / "claims.tex"
        )
        claims.write_evidence_matrix(registry, paper_dir / "evidence_matrix.csv", resolutions)
        for result in results:
            result.registry_problems = problems
            result.claim_resolutions = resolutions
    return results


def build_report(
    results_root: Path,
    paper_dir: Path,
    *,
    group: str = "all",
    run_dirs: list[Path] | None = None,
    processed_dir: Path | None = None,
    figures_dir: Path | None = None,
    tables_dir: Path | None = None,
    audit_dir: Path | None = None,
    audit_sample_size: int = 60,
    copy_to_paper: bool = True,
    strict: bool = False,
) -> ReportResult:
    """Regenerate every derived artifact from the immutable raw runs."""
    processed_dir = processed_dir or Path("results/processed")
    figures_dir = figures_dir or Path("results/figures")
    tables_dir = tables_dir or Path("results/tables")
    audit_dir = audit_dir or Path("results/audit")

    run_dirs = run_dirs if run_dirs is not None else discover_runs(results_root)
    result = ReportResult(run_ids=[run.name for run in run_dirs], group=group)
    if not run_dirs:
        raise FileNotFoundError(
            f"no runs found under {results_root}; run `memcheck run --config configs/smoke.yaml`"
        )

    # 1. Raw results must be unmodified before anything is derived from them.
    for run_dir in run_dirs:
        ok, problems = verify_integrity(run_dir)
        if not ok:
            result.integrity_problems.extend(problems)

    # 2. Runs must be comparable before they are pooled into one table.
    manifests = [load_manifest(run_dir) for run_dir in run_dirs]
    fairness = validate_across_runs(manifests)
    result.fairness_problems.extend(fairness.problems)
    if strict and not result.ok:
        return result

    # 3. Aggregate and compute metrics.
    frames = aggregate.load_all(results_root, run_dirs=run_dirs)
    from memcheck.reporting import metrics as metrics_module

    metric_frames = metrics_module.compute_all(frames)
    result.processed_files = aggregate.write_processed(
        {**{f"raw_{name}": frame for name, frame in frames.items()}, **metric_frames},
        processed_dir,
    )
    aggregate.manifests_frame(results_root, run_dirs=run_dirs).to_csv(
        processed_dir / "manifests.csv", index=False
    )
    result.processed_files.append(processed_dir / "manifests.csv")

    provenance = Provenance.create(
        artifact="memcheck report",
        script="src/memcheck/reporting/report.py",
        run_ids=result.run_ids,
        configuration_hashes=[manifest.configuration_hash for manifest in manifests],
    )

    # 3b. Warn when the reader, not the memory layer, is the binding constraint. Without this a
    # free-text benchmark produces four identical scores that look like a finding.
    saturation = metric_frames.get("reader_saturation")
    if saturation is not None and not saturation.empty:
        final = saturation[saturation["checkpoint"] == saturation["checkpoint"].max()]
        # Only backends that actually retrieve can tell us anything about the reader: one that
        # supplies evidence for almost nothing trivially never "declined despite evidence".
        # Among the backends with real coverage, the best one is the diagnostic — a weak
        # baseline abstaining often is a finding about that baseline, whereas the reader is
        # implicated only when even the strongest condition cannot use what it was handed.
        covering = final[
            final["answerable_with_evidence"] >= 0.5 * final["answerable_queries"].clip(lower=1)
        ]
        best = float(covering["reader_saturation"].min()) if not covering.empty else 0.0
        if best >= 0.5:
            result.reader_warnings.append(
                f"reader saturation is {best:.2f} at the full store even for the best backend: "
                "the reader wrongly declined on that fraction of answerable queries for which "
                "evidence was retrieved. Differences between backends are not measurable under "
                "this reader — these runs need a real model, not a templated one."
            )

    # 4. Figures, tables, and the spreadsheet carrying the figures' own data.
    result.figure_results = figures.generate_all(frames, metric_frames, figures_dir, provenance)
    result.table_results = tables.generate_all(frames, metric_frames, tables_dir, provenance)
    result.workbook_result = workbook.write_workbook(
        frames,
        metric_frames,
        figures_dir / "figure_data.xlsx",
        provenance,
        group=group,
    )

    # 5. Audit export.
    result.audit_export = audit.export(frames, audit_dir, sample_size=audit_sample_size)

    # 6. The claims registry spans every group, so it is resolved once by build_reports()
    # after all groups have been processed rather than per group here.
    result.processed_dir = processed_dir

    # 7. Copy figures and tables into the paper's generated directory.
    generated = paper_dir / "generated"
    generated.mkdir(parents=True, exist_ok=True)
    # Every group's workbook is copied out, not only the primary one: the whole point of the
    # spreadsheet export is that a reader can re-plot any comparison, including the ones the
    # paper only discusses in prose.
    _copy_workbook(result, group, generated / "figures")
    if not copy_to_paper:
        # A non-primary group still contributes its coding-outcome table, which no other
        # workload can produce.
        for table in result.table_results:
            if "coding" in table.name:
                (generated / table.path.name).write_text(
                    table.path.read_text(encoding="utf-8"), encoding="utf-8"
                )
        return result
    for table in result.table_results:
        (generated / table.path.name).write_text(
            table.path.read_text(encoding="utf-8"), encoding="utf-8"
        )
    figures_target = generated / "figures"
    figures_target.mkdir(parents=True, exist_ok=True)
    for figure in result.figure_results:
        if figure.path is None:
            continue
        for suffix in (".png", ".pdf"):
            source = figure.path.with_suffix(suffix)
            if source.exists():
                (figures_target / source.name).write_bytes(source.read_bytes())

    # 8. A machine-readable index of what this report generated.
    (generated / "report_index.json").write_text(
        json.dumps(
            {
                "generated_at": utcnow().isoformat(),
                "run_ids": result.run_ids,
                "figures": [
                    {"name": figure.name, "skipped": figure.skipped, "reason": figure.reason}
                    for figure in result.figure_results
                ],
                "tables": [
                    {"name": table.name, "pending": table.pending, "reason": table.reason}
                    for table in result.table_results
                ],
                "workbook": (
                    {
                        "path": str(result.workbook_result.path)
                        if result.workbook_result.path
                        else None,
                        "sheets": [
                            {"name": sheet.name, "rows": sheet.rows, "skipped": sheet.skipped}
                            for sheet in result.workbook_result.sheets
                        ],
                    }
                    if result.workbook_result is not None
                    else None
                ),
                "claims": [
                    {
                        "claim_id": resolution.claim.claim_id,
                        "status": resolution.claim.status,
                        "printable": resolution.printable,
                        "value": resolution.rendered,
                    }
                    for resolution in result.claim_resolutions
                ],
                "integrity_problems": result.integrity_problems,
                "fairness_problems": result.fairness_problems,
                "registry_problems": result.registry_problems,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return result
