"""Read only discovery and normalized views of persisted runs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RunRecord:
    path: Path
    payload: dict[str, Any]

    @property
    def context(self) -> dict[str, Any]:
        value = self.payload.get("final_context") or self.payload.get("diagnostic_context", {})
        return value if isinstance(value, dict) else {}

    @property
    def kind(self) -> str:
        return "Failure" if "diagnostic_context" in self.payload else "Result"

    @property
    def status(self) -> str:
        fallback = "completed" if self.kind == "Result" else "failed"
        return str(self.payload.get("status") or fallback)

    @property
    def outcome(self) -> str:
        run_info = self.payload.get("run_info", {})
        if not isinstance(run_info, dict):
            run_info = {}
        return str(
            self.payload.get("outcome")
            or run_info.get("outcome")
            or self.context.get("_outcome")
            or self.status
        )

    @property
    def resumable(self) -> bool:
        return bool(self.payload.get("resumable"))

    @property
    def error(self) -> str:
        return str(self.payload.get("error") or "No error detail recorded")

    @property
    def topic(self) -> str:
        run_info = self.payload.get("run_info", {})
        if not isinstance(run_info, dict):
            run_info = {}
        return str(run_info.get("topic") or self.context.get("topic") or "Unknown topic")

    @property
    def timestamp(self) -> str:
        run_info = self.payload.get("run_info", {})
        if not isinstance(run_info, dict):
            run_info = {}
        return str(run_info.get("timestamp_iso") or "Unknown").replace("T", " ")[:19]

    @property
    def novelty_status(self) -> str:
        novelty = self.context.get("novelty_verification", {})
        value = str(novelty.get("status") or "") if isinstance(novelty, dict) else ""
        explicit_outcome = self.payload.get("outcome") or self.context.get("_outcome")
        return value or (str(explicit_outcome) if explicit_outcome else "N/A")

    @property
    def evidence_coverage(self) -> str:
        audit = self.context.get("evidence_audit", {})
        value = (
            audit.get("claim_grounding_coverage", audit.get("evidence_coverage"))
            if isinstance(audit, dict)
            else None
        )
        if isinstance(value, int | float):
            return f"{float(value) * 100:.0f}%"
        return "N/A"

    @property
    def citation_coverage(self) -> str:
        """Explicit alias that does not imply cited claims are directly supported."""
        return self.evidence_coverage

    @property
    def support_coverage(self) -> str:
        audit = self.context.get("evidence_audit", {})
        value = audit.get("claim_support_coverage") if isinstance(audit, dict) else None
        if isinstance(value, int | float):
            return f"{float(value) * 100:.0f}%"
        return "N/A"


def discover_runs(output_dir: Path) -> tuple[list[RunRecord], list[str]]:
    records: list[RunRecord] = []
    warnings = []
    paths = []
    if output_dir.exists():
        paths = list(output_dir.glob("result_*.json")) + list(output_dir.glob("failed_*.json"))
    for path in paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("root value is not an object")
            records.append(RunRecord(path=path, payload=payload))
        except (OSError, json.JSONDecodeError, ValueError) as error:
            warnings.append(f"Skipped {path.name}: {error}")

    def modified(record: RunRecord) -> float:
        try:
            return record.path.stat().st_mtime
        except OSError:
            return 0

    records.sort(key=modified, reverse=True)
    return records, warnings
