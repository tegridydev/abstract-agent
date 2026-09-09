"""Typed terminal and per stage outcomes for the v6 pipeline runtime."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Generic, TypeVar


class RunOutcome(StrEnum):
    """Truthful terminal states exposed by the CLI and persisted result format."""

    PROPOSAL_COMPLETE = "proposal_complete"
    EVIDENCE_ONLY = "evidence_only"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    INTERRUPTED = "interrupted"
    TRANSPORT_FAILURE = "transport_failure"
    CONFIGURATION_ERROR = "configuration_error"


class StageStatus(StrEnum):
    SUCCEEDED = "succeeded"
    DEGRADED = "degraded"
    SKIPPED = "skipped"
    TERMINAL = "terminal"
    FAILED = "failed"


T = TypeVar("T")


@dataclass(frozen=True)
class StageOutcome(Generic[T]):
    """One committed unit of work; invalid model content is data, not an exception."""

    status: StageStatus
    value: T | None = None
    category: str | None = None
    reason: str | None = None
    retryable: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def usable(self) -> bool:
        return (
            self.status in {StageStatus.SUCCEEDED, StageStatus.DEGRADED} and self.value is not None
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "value": self.value,
            "category": self.category,
            "reason": self.reason,
            "retryable": self.retryable,
            "metadata": self.metadata,
        }


def terminal_result(outcome: RunOutcome, reason: str, **metadata: Any) -> dict[str, Any]:
    """Create the stable application owned terminal payload used by reports and UI."""
    return {
        "status": outcome.value,
        "reason": " ".join(reason.split()),
        "metadata": metadata,
    }


__all__ = ["RunOutcome", "StageOutcome", "StageStatus", "terminal_result"]
