"""Read only migration of legacy diagnostics into the v6 execution graph."""

from __future__ import annotations

from typing import Any

from ..contracts.pipeline import PipelineConfig
from ..contracts.research import HypothesisCandidate

LEGACY_GENERATIVE_STAGES = {
    "Generate Independent Hypotheses",
    "Audit Hypothesis Candidates",
    "Evidence and Novelty Critique",
    "Find Missing Evidence Critiques",
    "Recover Missing Evidence Critiques",
    "Merge Evidence Critiques",
    "Methodology Critique",
    "Find Missing Methodology Critiques",
    "Recover Missing Methodology Critiques",
    "Merge Methodology Critiques",
    "Select Candidate",
    "Revise Winning Candidate",
    "Audit Initial Revision",
    "Prepare Revision Repair",
    "Repair Revised Candidate",
    "Gate Revised Candidate",
    "Build Novelty Queries",
    "Retrieve Closest Prior Art",
    "Assess Prior Art Relevance",
    "Merge Evidence Pools",
    "Format Expanded Evidence",
    "Verify Candidate Novelty",
    "Find Missing Novelty Verifications",
    "Recover Missing Novelty Verifications",
    "Merge Novelty Verifications",
    "Audit Novelty Evidence Breadth",
    "Choose Verified Candidate",
    "Compose Research Proposal",
    "Audit Research Proposal",
}


def _importable_candidates(context: dict[str, Any]) -> list[dict[str, Any]]:
    source = context.get("hypothesis_candidates", [])
    candidates = []
    for value in source if isinstance(source, list) else []:
        if not isinstance(value, dict):
            continue
        provenance = value.get("generation_provenance", {})
        if provenance.get("mode") not in {"structured_exact", "structured_normalized"}:
            continue
        if not isinstance(value.get("scientific_spec"), dict):
            continue
        try:
            candidates.append(HypothesisCandidate.model_validate(value).model_dump())
        except (TypeError, ValueError):
            continue
    identifiers = [item["candidate_id"] for item in candidates]
    return candidates if len(identifiers) == len(set(identifiers)) else []


def migrate_legacy_context(
    context: dict[str, Any],
    *,
    schema_version: str,
    config: PipelineConfig,
) -> tuple[dict[str, Any], str]:
    """Return a filtered v6 context and safe resume node without mutating the source file."""
    active = str(context.get("_active_step") or "")
    current_names = {step.name for step in config.steps}
    imported = _importable_candidates(context)
    if active in LEGACY_GENERATIVE_STAGES:
        resume_step = "Audit Candidate Pool" if imported else "Prepare Candidate Kernels"
    elif active in current_names:
        resume_step = active
    else:
        resume_step = "Prepare Candidate Kernels"

    allowed = set(config.context_init)
    for step in config.steps:
        allowed.update(step.outputs)
    migrated = {
        key: value for key, value in context.items() if key.startswith("_") or key in allowed
    }
    discarded = sorted(key for key in context if not key.startswith("_") and key not in allowed)
    if imported:
        migrated["hypothesis_candidates"] = imported
    elif resume_step == "Prepare Candidate Kernels":
        migrated.pop("hypothesis_candidates", None)
    migrated["_active_step"] = resume_step
    migrated["_migrated_from"] = {
        "schema_version": schema_version,
        "pipeline_version": str(context.get("_pipeline_version") or schema_version),
        "original_step": active,
        "discarded_keys": discarded,
        "imported_exact_candidates": len(imported),
    }
    migrated.pop("_failure_diagnostic", None)
    migrated.pop("_pending_step_outputs", None)
    return migrated, resume_step


__all__ = ["migrate_legacy_context"]
