"""Runtime bindings for built in scientific and literature tool stages."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ...contracts.pipeline import ToolStep


def tool_overrides(
    step: ToolStep,
    context: dict[str, Any],
    *,
    search_depth: str,
    knowledge_enabled: bool,
    knowledge_refresh: str,
    minimum_papers: int,
    knowledge_store_factory: Callable[[], Any],
    embedding_service: Any,
) -> dict[str, Any]:
    """Bind run settings without coupling the generic executor to scientific policy."""
    overrides: dict[str, Any] = {}
    if step.function.endswith("aggregate_sources") and search_depth == "deep":
        overrides["max_results"] = 60
        overrides["external_expansion_limit"] = 2
    if step.function.endswith("aggregate_sources") and "Prior Art" in step.name:
        overrides["disabled_providers"] = [
            str(status.get("provider"))
            for status in context.get("source_statuses", [])
            if status.get("status") == "error"
            or status.get("circuit_state") == "open"
            or status.get("error_category")
            in {"http_401", "http_403", "http_429", "provider_error"}
        ]
    if step.function.endswith("aggregate_sources") and knowledge_enabled:
        overrides["knowledge_store"] = knowledge_store_factory()
        if knowledge_refresh == "always":
            overrides["use_cache"] = False
        elif knowledge_refresh == "offline":
            overrides["offline"] = True
        else:
            overrides["cache_ttl_seconds"] = 86_400 if "Prior Art" in step.name else 7 * 86_400
    if step.function.endswith("summarize_evidence_admission"):
        overrides["minimum_total"] = minimum_papers
        overrides["minimum_core"] = min(3, minimum_papers)
    if step.function == "knowledge.build_evidence_ledger":
        overrides["knowledge_store"] = knowledge_store_factory()
        overrides["embedding_service"] = embedding_service
    if step.function == "agent_helpers.assess_relevance":
        overrides["embedding_service"] = embedding_service
    if step.function in {
        "pipeline.stages.audit_hypothesis_candidates",
        "pipeline.stages.audit_candidate_pool",
        "pipeline.stages.gate_revised_candidate",
        "pipeline.stages.finalize_revision_candidates",
    }:
        overrides["embedding_service"] = embedding_service
    return overrides
