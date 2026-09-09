"""Non sensitive classification of model contract transport outcomes."""

from __future__ import annotations

from typing import Any

_STRUCTURAL_ISSUES = {
    "invalid_json",
    "schema_echo",
    "truncated_output",
    "duplicate_key",
}
_STRUCTURAL_MARKERS = (
    "duplicate json key",
    "json schema definition instead of a data instance",
    "invalid json",
    "truncated json",
)


def is_structural_compilation_failure(metrics: dict[str, Any]) -> bool:
    """Distinguish route failures from domain content normalization."""
    issue_codes = {
        str(issue.get("code"))
        for issue in metrics.get("compilation_trace", {}).get("issues", [])
        if isinstance(issue, dict)
    }
    validation_text = str(metrics.get("validation_error") or "").casefold()
    return (
        bool(metrics.get("grammar_fallback_count"))
        or bool(issue_codes & _STRUCTURAL_ISSUES)
        or any(marker in validation_text for marker in _STRUCTURAL_MARKERS)
    )


__all__ = ["is_structural_compilation_failure"]
