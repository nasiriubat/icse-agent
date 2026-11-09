"""The paper's main tables come from the pinned workload version, not the alphabetical tie-break."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from memcheck.reporting import report


def test_equally_broad_rerun_under_a_later_version_does_not_become_primary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    matrix = SimpleNamespace(ablations=["a", "b"], faults=["none", "x"])
    monkeypatch.setattr(report, "load_manifest", lambda _: matrix)
    groups = {
        f"synthetic_longitudinal/{report.PRIMARY_WORKLOAD_VERSION}/h/mock/r/e/p": [Path("old")],
        "synthetic_longitudinal/v9/h/mock/r/e/p": [Path("new")],
    }
    chosen = report._primary_group(groups)
    assert chosen.startswith(f"synthetic_longitudinal/{report.PRIMARY_WORKLOAD_VERSION}/")


def test_falls_back_to_any_synthetic_group_when_the_pinned_version_is_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(report, "load_manifest", lambda _: SimpleNamespace(ablations=[], faults=[]))
    groups = {"synthetic_longitudinal/v9/h": [Path("only")], "longmemeval/v1/h": [Path("x")]}
    assert report._primary_group(groups) == "synthetic_longitudinal/v9/h"
