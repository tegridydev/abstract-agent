"""Application owned model capability summaries."""

from __future__ import annotations

from typing import Any


def build_capability_reports(
    model_metadata: dict[str, dict[str, Any]],
    *,
    compilation_events: list[dict[str, Any]] | None = None,
    stage_outcomes: list[dict[str, Any]] | None = None,
) -> dict[str, dict[str, Any]]:
    """Separate transport/context capabilities from observed contract reliability."""
    events = compilation_events or []
    outcomes = stage_outcomes or []
    reports: dict[str, dict[str, Any]] = {}
    for model, metadata in model_metadata.items():
        policies: dict[str, dict[str, Any]] = {}
        for item in (event for event in events if event.get("model") == model):
            policy_id = str(item.get("policy_id") or "unknown")
            outcome = str(item.get("outcome") or "unknown")
            policy = policies.setdefault(
                policy_id,
                {
                    "attempts": 0,
                    "exact": 0,
                    "normalized": 0,
                    "projected": 0,
                    "fallback": 0,
                },
            )
            policy["attempts"] += 1
            if outcome in policy:
                policy[outcome] += 1
        reports[model] = {
            "transport": "available" if metadata else "not_preflighted",
            "completion": "completion" in metadata.get("capabilities", []),
            "reasoning_mode": metadata.get("reasoning_mode"),
            "structured_output_mode": metadata.get("structured_output_mode", "unknown"),
            "contract_tier": metadata.get("contract_tier", "unknown"),
            "advertised_context_length": metadata.get("context_length"),
            "allocated_context_length": metadata.get("allocated_context_length"),
            "observed_policies": policies,
            "degraded_units": sum(
                item.get("status") == "degraded" and item.get("metadata", {}).get("model") == model
                for item in outcomes
                if isinstance(item, dict)
            ),
        }
    return reports


__all__ = ["build_capability_reports"]
