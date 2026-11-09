"""The ``memcheck`` command line interface.

Every command returns a non-zero exit code on failure, so scripts and CI can rely on it.
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from memcheck import __version__
from memcheck.utilities import dotenv

app = typer.Typer(
    name="memcheck",
    help="Contract-driven testing, fault injection, and repair for persistent agent memory.",
    no_args_is_help=True,
    add_completion=False,
)
data_app = typer.Typer(help="Benchmark data acquisition and verification.")
app.add_typer(data_app, name="data")

console = Console()
error_console = Console(stderr=True, style="bold red")

EPOCH = datetime(2026, 1, 1, tzinfo=UTC)


def _fail(message: str, code: int = 1) -> None:
    error_console.print(message)
    raise typer.Exit(code)


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    version: bool = typer.Option(False, "--version", help="Print the version and exit."),
    env_file: Path = typer.Option(
        Path(".env"), "--env-file", help="Read environment variables from this file if present."
    ),
) -> None:
    # A real environment variable always wins over the file, so an exported ALLOW_PAID_RUNS
    # cannot be silently overridden by a stale .env in the working directory.
    applied = dotenv.load(env_file)
    if applied:
        # stderr, so a redirected JSON or CSV output is not prefixed with this notice.
        error_console.print(f"[dim]{dotenv.describe(applied)}[/dim]")
    if version:
        console.print(f"memcheck {__version__}")
        raise typer.Exit(0)
    if ctx.invoked_subcommand is None:
        console.print(ctx.get_help())
        raise typer.Exit(0)


# --------------------------------------------------------------------------------------
# doctor / init / demo
# --------------------------------------------------------------------------------------


@app.command()
def doctor() -> None:
    """Report environment readiness: interpreter, SQLite, LaTeX, credentials, cost controls."""
    from memcheck.config import paid_run_authorization
    from memcheck.utilities.environment import (
        cpu_information,
        latex_engine,
        operating_system,
        python_version,
        sqlite_has_fts5,
        sqlite_version,
    )

    table = Table(title="memcheck doctor", show_lines=False)
    table.add_column("check")
    table.add_column("status")
    table.add_column("detail")

    def row(name: str, ok: bool | None, detail: str) -> None:
        if ok:
            mark = "[green]ok[/green]"
        elif ok is None:
            mark = "[yellow]note[/yellow]"
        else:
            mark = "[red]missing[/red]"
        table.add_row(name, mark, detail)

    interpreter_ok = sys.version_info >= (3, 11)
    row("python >= 3.11", interpreter_ok, python_version())
    row("operating system", None, operating_system())
    row("cpu", None, cpu_information())
    row("sqlite", True, sqlite_version())
    row(
        "sqlite fts5",
        None,
        "available (used as a candidate filter)"
        if sqlite_has_fts5()
        else "absent (BM25 ranking is pure Python, so results are unaffected)",
    )
    engine = latex_engine()
    row(
        "latex engine",
        None if engine is None else True,
        engine or "absent: use `memcheck paper --validate` instead of `--build`",
    )

    authorization = paid_run_authorization()
    row("paid runs", None, "authorized" if authorization.allowed else authorization.reason)
    row(
        ".env file",
        None,
        "present (loaded; shell variables take precedence)"
        if Path(".env").exists()
        else "absent (copy .env.example to .env to use one)",
    )
    for variable in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY"):
        import os

        present = bool(os.environ.get(variable))
        row(variable, None, "set (value never logged)" if present else "unset (mock provider only)")

    optional = []
    for module in ("anthropic", "openai", "pyarrow", "hypothesis"):
        try:
            __import__(module)
            optional.append(f"{module}: installed")
        except ImportError:
            optional.append(f"{module}: not installed")
    row("optional extras", None, "; ".join(optional))

    console.print(table)
    if not interpreter_ok:
        _fail("Python 3.11 or newer is required.")
    console.print("\nEnvironment is ready for mock-provider experiments.")


@app.command()
def init(
    directory: Path = typer.Option(Path("."), "--dir", help="Where to create the layout."),
) -> None:
    """Create the results/ and data/ directory layout if it is missing."""
    created = []
    for relative in (
        "results/raw",
        "results/processed",
        "results/figures",
        "results/tables",
        "results/audit",
        "data/raw",
        "data/processed",
        "data/checksums",
        "paper/generated",
    ):
        path = directory / relative
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            (path / ".gitkeep").touch()
            created.append(str(path))
    console.print(f"created {len(created)} director(ies)" if created else "layout already present")
    for name in created:
        console.print(f"  {name}")


@app.command()
def demo(
    keep: bool = typer.Option(False, "--keep", help="Keep the demo database instead of deleting."),
) -> None:
    """Demonstrate ingestion, update, retrieval, deletion, restart, and index rebuild."""
    import tempfile

    from memcheck.backends import build_backend
    from memcheck.config import RetrievalConfig
    from memcheck.models import Event, EventType

    workspace = Path(tempfile.mkdtemp(prefix="memcheck_demo_"))
    backend = build_backend("guarded_memory", workspace / "demo.db", RetrievalConfig())
    query = "What is the deployment target?"

    def event(index: int, content: str, **kwargs: object) -> Event:
        return Event(
            event_id=f"demo_{index:03d}",
            tenant_id="acme",
            session_id="demo_session",
            event_type=kwargs.pop("event_type", EventType.user_message),  # type: ignore[arg-type]
            event_time=EPOCH + timedelta(days=index),
            content=content,
            metadata=kwargs.pop("metadata", {}),  # type: ignore[arg-type]
        )

    def show(step: str, detail: str) -> None:
        console.print(f"[bold cyan]{step:<22}[/bold cyan] {detail}")

    try:
        result = backend.ingest_event(event(1, "The deployment target is docker-swarm."))
        show("1. ingest", f"outcome={result.outcome.value} memories={result.created_memory_ids}")

        duplicate = backend.ingest_event(event(1, "The deployment target is docker-swarm."))
        show("2. duplicate replay", f"outcome={duplicate.outcome.value} ({duplicate.notes})")

        update = backend.ingest_event(event(5, "The deployment target is kubernetes."))
        show(
            "3. update",
            f"outcome={update.outcome.value} superseded={update.superseded_memory_ids}",
        )

        current = backend.retrieve(query, "acme")
        show(
            "4. retrieve (now)",
            f"{[item.content for item in current.evidence]} abstained={current.abstained}",
        )

        historical = backend.retrieve(query, "acme", as_of_time=EPOCH + timedelta(days=3))
        show(
            "5. retrieve (as-of)",
            f"{[item.content for item in historical.evidence]} (history preserved)",
        )

        deletion = backend.ingest_event(
            event(
                7,
                "Delete everything about the deployment target.",
                event_type=EventType.deletion_request,
                metadata={"deletion_target": "deployment target"},
            )
        )
        show("6. delete", f"outcome={deletion.outcome.value} {deletion.metadata}")

        after = backend.retrieve(query, "acme")
        show(
            "7. retrieve after delete",
            f"abstained={after.abstained} reason={after.abstention_reason}",
        )

        backend.restart()
        restarted = backend.retrieve(query, "acme")
        show("8. restart", f"abstained={restarted.abstained} (deletion survived the restart)")

        rebuilt = backend.rebuild_indexes()
        final = backend.retrieve(query, "acme")
        show(
            "9. rebuild indexes",
            f"{rebuilt} document(s) rebuilt from the event log; abstained={final.abstained}",
        )

        statistics = backend.get_statistics()
        show(
            "10. statistics",
            f"events={statistics.event_count} active={statistics.active_memory_count} "
            f"superseded={statistics.superseded_record_count} "
            f"deleted={statistics.deleted_record_count}",
        )
    finally:
        backend.close()
        if not keep:
            import shutil

            shutil.rmtree(workspace, ignore_errors=True)
        else:
            console.print(f"\ndemo store kept at {workspace}")


# --------------------------------------------------------------------------------------
# ingest / replay / validate / inject
# --------------------------------------------------------------------------------------


@app.command()
def ingest(
    backend: str = typer.Option("guarded_memory", "--backend"),
    store: Path = typer.Option(Path("results/demo_store.db"), "--store"),
    content: str = typer.Option(..., "--content", help="Event content to ingest."),
    tenant: str = typer.Option("default", "--tenant"),
    session: str = typer.Option("session_01", "--session"),
) -> None:
    """Ingest a single event into a store and print the structured write result."""
    from memcheck.backends import build_backend
    from memcheck.config import RetrievalConfig
    from memcheck.models import Event, EventType
    from memcheck.models import utcnow as now
    from memcheck.utilities.hashing import stable_id

    instance = build_backend(backend, store, RetrievalConfig())
    try:
        result = instance.ingest_event(
            Event(
                event_id=stable_id("cli", tenant, session, content),
                tenant_id=tenant,
                session_id=session,
                event_type=EventType.user_message,
                event_time=now(),
                content=content,
            )
        )
        console.print_json(json.dumps(result.model_dump(mode="json"), default=str))
    finally:
        instance.close()


@app.command()
def replay(
    config: Path = typer.Option(..., "--config", help="Experiment configuration."),
    limit: int = typer.Option(20, "--limit", help="How many events to print."),
) -> None:
    """Print a workload's chronological event stream without running an experiment."""
    from memcheck.config import load_config
    from memcheck.workloads import build_workload

    workload = build_workload(load_config(config))
    workload.load()
    events = workload.events()
    table = Table(title=f"{workload.name}@{workload.version}: {len(events)} events")
    for column in ("#", "time", "tenant", "session", "type", "content"):
        table.add_column(column, overflow="fold")
    for index, event in enumerate(events[:limit]):
        table.add_row(
            str(index),
            event.event_time.date().isoformat(),
            event.tenant_id,
            event.session_id,
            event.event_type.value,
            event.content[:70],
        )
    console.print(table)
    console.print(json.dumps(workload.describe(), indent=2, default=str))


@app.command()
def validate(
    backend: str = typer.Option("guarded_memory", "--backend", help="Backend to test, or 'all'."),
    contract: str = typer.Option("all", "--contract"),
    seed: int = typer.Option(1, "--seed"),
) -> None:
    """Run the executable memory contracts against a backend. Exits non-zero on a violation."""
    import tempfile

    from memcheck.backends import build_backend
    from memcheck.config import CORE_BACKENDS, RetrievalConfig
    from memcheck.contracts import ContractContext, resolve_contracts

    backends = list(CORE_BACKENDS) if backend == "all" else [backend]
    retrieval = RetrievalConfig()
    workspace = Path(tempfile.mkdtemp(prefix="memcheck_validate_"))
    failures = 0

    table = Table(title="memory contracts")
    for column in ("backend", "contract", "result", "violations", "detail"):
        table.add_column(column, overflow="fold")

    for name in backends:

        def factory(suffix: str, name: str = name) -> object:
            return build_backend(name, workspace / f"{name}_{suffix}.db", retrieval)

        context = ContractContext(
            run_id="cli-validate",
            backend_name=name,
            backend_factory=factory,  # type: ignore[arg-type]
            seed=seed,
            evidence_budget=retrieval.evidence_budget,
            token_budget=retrieval.token_budget,
        )
        try:
            contracts = resolve_contracts([contract] if contract != "all" else ["all"])
        except KeyError as error:
            _fail(str(error))
        for item in contracts:
            result = item.run(context)
            if result.skipped:
                status = "[yellow]skipped[/yellow]"
            elif result.passed:
                status = "[green]pass[/green]"
            else:
                status = "[red]FAIL[/red]"
                failures += len(result.violations)
            detail = result.skip_reason or (
                result.violations[0].observed_behavior[:80] if result.violations else ""
            )
            table.add_row(name, item.name, status, str(len(result.violations)), detail)

    console.print(table)
    import shutil

    shutil.rmtree(workspace, ignore_errors=True)
    if failures:
        _fail(f"{failures} contract violation(s) detected")
    console.print("[green]all contracts held[/green]")


@app.command()
def inject(
    fault: str = typer.Option("all", "--fault", help="Fault name, or 'all'."),
    backend: str = typer.Option("guarded_memory", "--backend", help="Backend, or 'all'."),
    seed: int = typer.Option(7, "--seed"),
) -> None:
    """Inject faults deterministically and report the recovery observed."""
    import tempfile

    from memcheck.backends import build_backend
    from memcheck.config import CORE_BACKENDS, RetrievalConfig
    from memcheck.contracts.base import make_event
    from memcheck.faults import ALL_FAULTS, resolve_fault, run_fault

    faults = [item() for item in ALL_FAULTS] if fault == "all" else [resolve_fault(fault)]
    backends = list(CORE_BACKENDS) if backend == "all" else [backend]
    retrieval = RetrievalConfig()
    workspace = Path(tempfile.mkdtemp(prefix="memcheck_inject_"))

    events = [
        make_event(1, "The primary database is postgres."),
        make_event(2, "The deployment target is docker-swarm."),
        make_event(3, "The retention period is thirty days."),
        make_event(4, "The deployment target is kubernetes."),
    ]

    table = Table(title="fault injection")
    for column in ("backend", "fault", "recovered", "observed"):
        table.add_column(column, overflow="fold")
    unrecovered = 0
    for name in backends:
        for item in faults:
            store = build_backend(name, workspace / f"{name}_{item.name}.db", retrieval)
            try:
                report = run_fault(item, store, events, seed=seed, backend_name=name)
            finally:
                store.close()
            marker = "[green]yes[/green]" if report.recovered else "[red]no[/red]"
            if not report.recovered:
                unrecovered += 1
            table.add_row(name, item.name, marker, report.observed_recovery_behavior[:90])
    console.print(table)
    import shutil

    shutil.rmtree(workspace, ignore_errors=True)
    console.print(f"\n{unrecovered} unrecovered fault(s) — see faults.jsonl in a full run.")


# --------------------------------------------------------------------------------------
# run / resume / summarize / audit / report / paper
# --------------------------------------------------------------------------------------


@app.command()
def run(
    config: Path = typer.Option(..., "--config", help="Experiment configuration file."),
    run_id: str | None = typer.Option(None, "--run-id"),
    results: Path | None = typer.Option(None, "--results", help="Override results root."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Estimate cost and exit."),
) -> None:
    """Run an experiment configuration into an immutable results directory."""
    from memcheck.config import load_config
    from memcheck.evaluation import ExperimentRunner, expand
    from memcheck.evaluation import cost as cost_module
    from memcheck.workloads import build_workload

    try:
        configuration = load_config(config)
    except (FileNotFoundError, ValueError) as error:
        _fail(f"configuration error: {error}")

    if dry_run:
        conditions = expand(configuration)
        workload = build_workload(configuration)
        workload.load()
        estimate = cost_module.estimate_for_workload(configuration, conditions, workload)
        console.print(f"conditions: {len(conditions)}")
        console.print(estimate.render())
        raise typer.Exit(0 if estimate.within_limit else 2)

    runner = ExperimentRunner(
        configuration,
        run_id=run_id,
        results_root=results,
        on_progress=lambda message: console.print(f"[dim]{message}[/dim]"),
    )
    try:
        status = runner.run()
    except Exception as error:
        _fail(f"run failed: {error}")
    console.print(
        f"\n[bold]{status.state}[/bold]: {status.conditions_completed} completed, "
        f"{status.conditions_failed} failed"
    )
    console.print(f"results: {runner.paths.root}")
    if status.conditions_failed:
        _fail("some conditions failed; see stderr.log in the run directory", code=3)


@app.command()
def resume(
    run_dir: Path = typer.Option(..., "--run_dir", help="Run directory to resume."),
) -> None:
    """Resume an interrupted run, skipping conditions already recorded as complete."""
    import yaml

    from memcheck.config import ExperimentConfig
    from memcheck.evaluation import ExperimentRunner

    config_path = run_dir / "config.yaml"
    if not config_path.exists():
        _fail(f"{run_dir} has no config.yaml; it is not a MemCheck run directory")
    configuration = ExperimentConfig.model_validate(
        yaml.safe_load(config_path.read_text(encoding="utf-8"))
    )
    runner = ExperimentRunner(
        configuration,
        run_id=run_dir.name,
        results_root=run_dir.parent,
        on_progress=lambda message: console.print(f"[dim]{message}[/dim]"),
    )
    status = runner.run(resume=True)
    console.print(
        f"{status.state}: {status.conditions_completed} completed, "
        f"{status.conditions_failed} failed"
    )
    if status.conditions_failed:
        _fail("some conditions failed", code=3)


@app.command()
def summarize(
    results: Path = typer.Option(Path("results/raw"), "--results"),
    processed: Path = typer.Option(Path("results/processed"), "--processed"),
) -> None:
    """Aggregate raw runs into processed CSV/Parquet tables."""
    from memcheck.reporting import aggregate, comparability_groups
    from memcheck.reporting import metrics as metrics_module

    try:
        groups = comparability_groups(results)
    except FileNotFoundError as error:
        _fail(str(error))

    # Aggregate per comparability group, exactly as `report` does. Pooling two workloads into one
    # table would average numbers that answer different questions at different checkpoints.
    utility = None
    total = 0
    for key, run_dirs in groups.items():
        slug = key.replace("@", "_").replace("/", "_")
        frames = aggregate.load_all(results, run_dirs=run_dirs)
        metric_frames = metrics_module.compute_all(frames)
        written = aggregate.write_processed(
            {**{f"raw_{name}": frame for name, frame in frames.items()}, **metric_frames},
            processed / slug,
        )
        total += len(written)
        console.print(f"{key}: {len(written)} file(s) -> {processed / slug}")
        if utility is None or utility.empty:
            utility = metric_frames.get("task_utility")
    console.print(f"wrote {total} file(s) across {len(groups)} comparability group(s)")
    if utility is not None and not utility.empty:
        final = utility[utility["checkpoint"] == utility["checkpoint"].max()]
        table = Table(title=f"accuracy at the full store ({next(iter(groups))})")
        for column in ("variant", "fault", "n", "accuracy"):
            table.add_column(column)
        for _, row in final.sort_values("accuracy", ascending=False).iterrows():
            table.add_row(
                str(row["variant"]), str(row["fault"]), str(int(row["n"])), f"{row['accuracy']:.3f}"
            )
        console.print(table)


@app.command()
def scale(
    results: Path = typer.Option(Path("results/raw"), "--results"),
    benchmark: str = typer.Option("synthetic_longitudinal", "--benchmark"),
    reference: str = typer.Option("guarded_memory", "--reference"),
    output: Path = typer.Option(Path("results/processed/scale"), "--output"),
    figures: Path = typer.Option(
        Path("results/figures/scale"), "--figures", help="Where to draw the sweep figure."
    ),
    paper: Path = typer.Option(Path("paper"), "--paper", help="Where to write the LaTeX tables."),
    workload_version: str = typer.Option(
        "", "--workload-version", help="Sweep of this workload version (default: the primary)."
    ),
) -> None:
    """Compare the verdict each store size produces, across a size sweep.

    This is the only analysis that deliberately reaches across comparability groups. It never
    pools them: each store size keeps its own accuracy, paired tests, and violation counts, and
    what is compared is the conclusion each size would have supported.
    """
    from memcheck.reporting import scale as scale_module
    from memcheck.reporting.report import PRIMARY_WORKLOAD_VERSION

    analysis = scale_module.analyse(
        results,
        benchmark=benchmark,
        reference=reference,
        workload_version=workload_version or PRIMARY_WORKLOAD_VERSION,
    )
    for problem in analysis.problems:
        console.print(f"[yellow]problem:[/yellow] {problem}")
    if analysis.accuracy.empty:
        _fail(f"no runs of benchmark {benchmark!r} with reference condition {reference!r}")

    sizes = sorted(analysis.accuracy["sessions"].unique())
    accuracy = Table(title=f"accuracy at the full store, by store size (reference: {reference})")
    accuracy.add_column("variant", overflow="fold")
    for size in sizes:
        accuracy.add_column(f"{size}s", justify="right")
    for variant, group in analysis.accuracy.groupby("variant"):
        by_size = group.set_index("sessions")
        cells = []
        for size in sizes:
            if size not in by_size.index:
                cells.append("--")
                continue
            row = by_size.loc[size]
            value = f"{row['accuracy']:.3f}"
            # Mark only what the correction actually supports.
            if bool(row["separated_from_reference"]):
                value = f"[red]{value}*[/red]"
            cells.append(value)
        accuracy.add_row(str(variant), *cells)
    console.print(accuracy)
    console.print("[dim]* differs from the reference, Holm-adjusted p < 0.05[/dim]\n")

    if not analysis.leading_indicator.empty:
        leading = Table(title="when contracts fired, against when accuracy noticed")
        for column in (
            "variant",
            "violations at smallest store",
            "first size that separates",
            "discordant pairs there",
            "stable",
            "gap at largest store",
        ):
            leading.add_column(column, overflow="fold")
        for _, row in analysis.leading_indicator.iterrows():
            stable = "yes" if row["separation_is_stable"] else "[yellow]no[/yellow]"
            leading.add_row(
                str(row["variant"]),
                str(row["violations_at_smallest_store"]),
                str(row["first_separating_sessions"]),
                str(row["discordant_pairs_at_first_separation"]),
                stable,
                f"{row['gap_at_largest_store']:.3f}",
            )
        console.print(leading)

    output.mkdir(parents=True, exist_ok=True)
    analysis.accuracy.to_csv(output / "scale_accuracy.csv", index=False)
    analysis.leading_indicator.to_csv(output / "scale_leading_indicator.csv", index=False)
    analysis.violations.to_csv(output / "scale_violations.csv", index=False)
    analysis.streams.to_csv(output / "scale_accuracy_streams.csv", index=False)

    from memcheck.reporting.provenance import Provenance

    provenance = Provenance.create(
        artifact="memcheck scale",
        script="src/memcheck/reporting/scale.py",
        run_ids=analysis.run_ids,
        configuration_hashes=[],
    )
    latex = scale_module.write_latex(analysis, paper / "generated", provenance)
    # The sweep figure and its workbook are drawn once under results/figures and copied beside
    # the paper's other figures, exactly as `memcheck report` does for the per-group ones.
    published = scale_module.publish_figure(analysis, figures, paper / "generated", provenance)
    console.print(
        f"\nwrote {4 + len(latex)} file(s) from {len(analysis.run_ids)} run(s): "
        f"{output} and {paper / 'generated'}; figure and workbook: "
        f"{', '.join(path.name for path in published) or 'none'} -> {paper / 'generated/figures'}"
    )


@app.command("write-policy")
def write_policy(
    results: Path = typer.Option(Path("results/raw"), "--results"),
    benchmark: str = typer.Option("longmemeval", "--benchmark"),
    output: Path = typer.Option(Path("results/processed/write_policy"), "--output"),
    paper: Path = typer.Option(Path("paper"), "--paper", help="Where to write the outputs."),
) -> None:
    """Pair the two runs that differ only in the guarded backend's write policy.

    Like `scale`, this reaches across comparability groups without pooling them: the same query
    is paired with itself across the two runs, and the paired difference is what is reported.
    """
    from memcheck.reporting import writepolicy

    outcome = writepolicy.build(results, paper, benchmark=benchmark, processed_dir=output)
    for problem in outcome.analysis.problems:
        console.print(f"[yellow]problem:[/yellow] {problem}")
    if not outcome.analysis.paired:
        _fail(f"no pair of {benchmark!r} runs differing only in the write policy was found")

    table = Table(title="accuracy at the full store under each write policy (paired)")
    for column in ("condition", "rule", "LLM", "discordant (gained/lost)", "p (Holm)"):
        table.add_column(column, justify="right" if column != "condition" else "left")
    for _, row in outcome.analysis.conditions.iterrows():
        table.add_row(
            str(row["label"]),
            f"{row['accuracy_rule']:.3f}",
            f"{row['accuracy_llm']:.3f}",
            f"{int(row['discordant_pairs'])} ({int(row['gained'])}/{int(row['lost'])})",
            f"{row['p_value_holm']:.4f}",
        )
    console.print(table)
    console.print(outcome.summary())


@app.command()
def audit(
    results: Path = typer.Option(Path("results/raw"), "--results"),
    output: Path = typer.Option(Path("results/audit"), "--output"),
    sample: int = typer.Option(60, "--sample", help="Rows to export."),
    agreement: Path | None = typer.Option(None, "--agreement", help="Score a filled-in CSV."),
    lme: list[Path] = typer.Option(
        [], "--lme", help="Re-judged LongMemEval run(s): export the blinded judge audit (D-027)."
    ),
    lme_agreement: Path | None = typer.Option(
        None, "--lme-agreement", help="Score a filled-in audit_blind.csv."
    ),
    lme_key: Path | None = typer.Option(None, "--lme-key", help="The matching audit_key.csv."),
    references: Path = typer.Option(Path("data/raw/longmemeval_oracle.json"), "--references"),
    json_output: Path | None = typer.Option(
        None, "--json-output", help="Also write the agreement JSON to this file."
    ),
) -> None:
    """Export a deterministic stratified audit sample, or score a filled-in one."""
    from memcheck.reporting import aggregate, lme_audit
    from memcheck.reporting import audit as audit_module

    if lme_agreement is not None:
        if lme_key is None:
            _fail("--lme-agreement needs --lme-key")
        scored = lme_audit.agreement(lme_agreement, lme_key)  # type: ignore[arg-type]
        text = json.dumps(scored, indent=2, default=str)
        if json_output is not None:
            json_output.parent.mkdir(parents=True, exist_ok=True)
            json_output.write_text(text + "\n", encoding="utf-8")
        console.print_json(text)
        return
    if lme and json_output is not None:
        scored = lme_audit.judge_agreement(lme, references)
        json_output.parent.mkdir(parents=True, exist_ok=True)
        json_output.write_text(json.dumps(scored, indent=2, default=str) + "\n", encoding="utf-8")
        console.print_json(json.dumps(scored, default=str))
        return
    if lme:
        lme_export = lme_audit.export(lme, references, output, sample_size=sample)
        console.print(
            f"wrote {lme_export.rows} row(s) to {lme_export.blind_path} "
            f"(key: {lme_export.key_path}); per type: {lme_export.per_type}; "
            f"population {lme_export.population} distinct pairs"
        )
        return
    if agreement is not None:
        if not agreement.exists():
            _fail(f"{agreement} does not exist")
        console.print_json(json.dumps(audit_module.agreement(agreement), default=str))
        return
    try:
        frames = aggregate.load_all(results)
    except FileNotFoundError as error:
        _fail(str(error))
    export = audit_module.export(frames, output, sample_size=sample)
    console.print(f"wrote {export.rows} row(s) to {export.path}")
    console.print(f"strata: {export.strata_counts}")
    console.print(f"judge prompt hash: {export.judge_prompt_hash}")


@app.command()
def rejudge(
    run: Path = typer.Option(..., "--run", help="A stored LongMemEval run directory."),
    references: Path = typer.Option(
        Path("data/raw/longmemeval_oracle.json"), "--references", help="LongMemEval data file."
    ),
    judge_model: str = typer.Option("gpt-4o-2024-08-06", "--judge-model"),
    pricing_file: Path = typer.Option(Path("configs/pricing.example.yaml"), "--pricing"),
    output_root: Path = typer.Option(Path("results/raw"), "--output-root"),
) -> None:
    """Re-grade a stored LongMemEval run with the official per-type judge (D-025).

    The source run is read, never written; the re-graded copy is a new run directory.
    """
    from memcheck.config import load_pricing, paid_run_authorization
    from memcheck.evaluation import rejudge as rejudge_module
    from memcheck.evaluation.runner import verify_integrity
    from memcheck.models import ProviderCall
    from memcheck.providers.base import ChatMessage, ChatRequest
    from memcheck.providers.openai_provider import OpenAIProvider

    intact, problems = verify_integrity(run)
    if not intact:
        _fail("; ".join(problems))
    authorization = paid_run_authorization()
    if not authorization.allowed:
        _fail(f"refusing a paid re-grade: {authorization.reason}")
    pricing = load_pricing(pricing_file)
    pairs = rejudge_module.unique_pairs(run)
    # Conservative: 1,500 prompt tokens per call bounds the longest preference rubric and answer.
    estimate = pricing.cost_for(f"openai:{judge_model}", 1_500 * pairs, 10 * pairs)
    console.print(f"{pairs} judge call(s), estimated at most ${estimate:.3f} ({pricing.as_of})")
    if estimate > authorization.max_cost_usd:
        _fail(f"estimate ${estimate:.3f} exceeds MAX_EXPERIMENT_COST_USD")
    provider = OpenAIProvider(pricing=pricing)

    def judge(prompt: str) -> tuple[str, ProviderCall]:
        response = provider.chat(
            ChatRequest(
                model_identifier=judge_model,
                system_prompt="",
                messages=[ChatMessage(role="user", content=prompt)],
                temperature=0.0,
                max_output_tokens=rejudge_module.OFFICIAL_MAX_TOKENS,
                purpose="judge",
            )
        )
        return response.text, response.call

    try:
        result = rejudge_module.rejudge(
            run, references, output_root, judge, judge_model=judge_model
        )
    finally:
        provider.close()
    console.print(
        f"wrote {result.run_dir}: {result.rows} row(s), {result.judged_pairs} judged pair(s), "
        f"{result.changed_rows} label(s) changed, ${result.cost_usd:.4f}"
    )


@app.command()
def report(
    results: Path = typer.Option(Path("results/raw"), "--results"),
    paper: Path = typer.Option(Path("paper"), "--paper"),
    run_dir: Path | None = typer.Option(None, "--run_dir", help="Report on one run only."),
    strict: bool = typer.Option(False, "--strict", help="Exit non-zero on any problem."),
) -> None:
    """Regenerate processed tables, figures, LaTeX tables, audit sample, and paper macros."""
    from memcheck.reporting import build_report, build_reports

    try:
        if run_dir is not None:
            outcomes = [build_report(run_dir.parent, paper, run_dirs=[run_dir], strict=strict)]
        else:
            outcomes = build_reports(results, paper, strict=strict)
    except FileNotFoundError as error:
        _fail(str(error))
    problems = 0
    for outcome in outcomes:
        console.print(outcome.summary())
        console.print("")
        if not outcome.ok:
            problems += 1
    if strict and problems:
        _fail(f"{problems} report group(s) generated with problems (strict mode)", code=4)


@app.command()
def paper(
    build: bool = typer.Option(False, "--build", help="Compile the PDF (needs a LaTeX engine)."),
    validate_only: bool = typer.Option(False, "--validate", help="Validate the source only."),
    paper_dir: Path = typer.Option(Path("paper"), "--paper"),
) -> None:
    """Build or validate the paper."""
    from memcheck.paper import build_pdf, validate_sources

    if build:
        ok, message = build_pdf(paper_dir)
        console.print(message)
        if not ok:
            _fail("paper build failed", code=5)
        return

    report_ = validate_sources(paper_dir)
    console.print(report_.render())
    if not validate_only and not report_.ok:
        _fail("paper source validation failed", code=5)
    if validate_only and not report_.ok:
        _fail("paper source validation failed", code=5)


@app.command("verify-integrity")
def verify_integrity_command(
    results: Path = typer.Option(Path("results/raw"), "--results"),
) -> None:
    """Re-check every run directory against its recorded checksums."""
    from memcheck.evaluation import discover_runs, verify_integrity

    runs = discover_runs(results)
    if not runs:
        _fail(f"no runs found under {results}")
    problems: list[str] = []
    for run_directory in runs:
        ok, run_problems = verify_integrity(run_directory)
        console.print(
            f"{'[green]ok[/green]' if ok else '[red]MODIFIED[/red]'}  {run_directory.name}"
        )
        problems.extend(run_problems)
    if problems:
        for problem in problems:
            error_console.print(f"  - {problem}")
        _fail(f"{len(problems)} integrity problem(s)")


# --------------------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------------------


@data_app.command("verify")
def data_verify(
    dataset: str = typer.Option("longmemeval", "--dataset"),
    register: bool = typer.Option(False, "--register", help="Record the current checksum."),
    data_dir: Path = typer.Option(Path("data/raw"), "--data-dir"),
) -> None:
    """Verify (or register) checksums for a downloaded benchmark."""
    from memcheck.utilities.hashing import file_sha256
    from memcheck.workloads.longmemeval import EXPECTED_FILES

    if dataset != "longmemeval":
        _fail(f"unknown dataset {dataset!r}")
    registry_path = Path("data/checksums/longmemeval.json")
    registry = (
        json.loads(registry_path.read_text(encoding="utf-8")) if registry_path.exists() else {}
    )
    found = 0
    for subset, filenames in EXPECTED_FILES.items():
        path = next((data_dir / name for name in filenames if (data_dir / name).exists()), None)
        if path is None:
            console.print(
                f"[yellow]missing[/yellow] {subset:<6} "
                f"{' or '.join(str(data_dir / name) for name in filenames)}"
            )
            continue
        filename = path.name
        found += 1
        digest = file_sha256(path)
        if register:
            registry[filename] = digest
            console.print(f"[green]registered[/green] {filename} {digest[:16]}...")
        elif registry.get(filename) == digest:
            console.print(f"[green]ok[/green] {filename}")
        elif filename in registry:
            _fail(f"checksum mismatch for {filename}: expected {registry[filename]}, got {digest}")
        else:
            console.print(f"[yellow]unregistered[/yellow] {filename} {digest[:16]}...")
    if register:
        registry_path.parent.mkdir(parents=True, exist_ok=True)
        registry_path.write_text(json.dumps(registry, indent=2, sort_keys=True), encoding="utf-8")
    if found == 0:
        console.print(
            "\nNo LongMemEval files found. LongMemEval is not redistributed with MemCheck; "
            "see data/README.md for download instructions."
        )


if __name__ == "__main__":  # pragma: no cover
    app()
