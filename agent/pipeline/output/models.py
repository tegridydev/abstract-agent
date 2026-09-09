"""Typed artifacts emitted by the model output compiler."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

OutputRoute = Literal[
    "content_schema",
    "content_json",
    "tagged_text",
    "deterministic_fallback",
]
CompilationOutcome = Literal["exact", "normalized", "projected", "fallback"]
RecoverySource = Literal[
    "none",
    "local_normalization",
    "focused_model_repair",
    "deterministic_fallback",
    "semantic_projection",
    "stage_projection",
]


@dataclass(frozen=True)
class CompilationIssue:
    """One bounded, non sensitive description of an output defect or repair."""

    code: str
    message: str
    path: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ModelOutputArtifact:
    """Raw output plus transport facts needed for deterministic compilation."""

    content: str
    route: OutputRoute = "content_schema"
    done_reason: str | None = None
    output_limit_reached: bool = False
    model: str | None = None
    request_count: int = 1


@dataclass
class CompilationTrace:
    """Auditable trace without prompts, source passages, or generated content."""

    route: OutputRoute
    outcome: CompilationOutcome
    parser: str
    attempts: int = 1
    policy_id: str | None = None
    prompt_id: str | None = None
    prompt_version: str | None = None
    prompt_hash: str | None = None
    contract_version: str | None = None
    human_review_required: bool = False
    recovery_source: RecoverySource = "none"
    repair_attempted: bool = False
    issues: list[CompilationIssue] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["issues"] = [issue.as_dict() for issue in self.issues]
        return value


@dataclass
class CompiledOutput:
    """A validated application value and the route used to create it."""

    value: Any
    trace: CompilationTrace


__all__ = [
    "CompilationIssue",
    "CompilationOutcome",
    "CompilationTrace",
    "CompiledOutput",
    "ModelOutputArtifact",
    "OutputRoute",
    "RecoverySource",
]
