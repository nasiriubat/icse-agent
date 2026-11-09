"""Provenance stamps.

Section 19 requires every generated table and figure to carry its source run ids, generating
script, configuration hash, and generation timestamp. That stamp is produced here once and
attached in the format each artifact allows: a LaTeX comment block, a figure footer, and a
sidecar JSON file so the stamp is machine-readable too.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from memcheck.models import utcnow


@dataclass
class Provenance:
    artifact: str
    script: str
    run_ids: list[str]
    configuration_hashes: list[str]
    generated_at: str

    @classmethod
    def create(
        cls,
        artifact: str,
        script: str,
        run_ids: list[str],
        configuration_hashes: list[str],
    ) -> Provenance:
        return cls(
            artifact=artifact,
            script=script,
            run_ids=sorted(set(run_ids)),
            configuration_hashes=sorted(set(configuration_hashes)),
            generated_at=utcnow().isoformat(),
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "artifact": self.artifact,
            "generating_script": self.script,
            "source_run_ids": self.run_ids,
            "configuration_hashes": self.configuration_hashes,
            "generated_at": self.generated_at,
        }

    def latex_comment(self) -> str:
        return "\n".join(
            [
                "% ---------------------------------------------------------------------",
                f"% artifact            : {self.artifact}",
                f"% generating script   : {self.script}",
                f"% source run ids      : {', '.join(self.run_ids) or 'none'}",
                f"% configuration hashes: {', '.join(self.configuration_hashes) or 'none'}",
                f"% generated at        : {self.generated_at}",
                "% Regenerate with: uv run memcheck report --results results/raw --paper paper",
                "% ---------------------------------------------------------------------",
            ]
        )

    def figure_footer(self) -> str:
        runs = ", ".join(self.run_ids) or "none"
        return (
            f"{self.artifact} | runs: {runs} | config: "
            f"{', '.join(hash_[:8] for hash_ in self.configuration_hashes) or 'none'} | "
            f"{self.generated_at}"
        )

    def write_sidecar(self, path: Path) -> Path:
        sidecar = path.with_suffix(path.suffix + ".provenance.json")
        sidecar.write_text(json.dumps(self.as_dict(), indent=2, sort_keys=True), encoding="utf-8")
        return sidecar
