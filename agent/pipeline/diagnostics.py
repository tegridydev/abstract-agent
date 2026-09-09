"""Small helpers for recording recovery and generation provenance."""

from __future__ import annotations

import hashlib
from typing import Any


def bounded_output_diagnostic(content: str, *, tail_chars: int = 500) -> dict[str, Any]:
    """Record enough information to reproduce a failure without persisting full prompts."""
    encoded = content.encode("utf-8", errors="replace")
    return {
        "response_characters": len(content),
        "response_sha256": hashlib.sha256(encoded).hexdigest(),
        "response_tail": content[-tail_chars:],
    }


def generation_summary(context: dict[str, Any]) -> dict[str, Any]:
    modes: dict[str, int] = {}
    for event in context.get("_generation_events", []):
        mode = str(event.get("mode", "structured_exact"))
        modes[mode] = modes.get(mode, 0) + 1
    fallback_count = sum(count for mode, count in modes.items() if mode != "structured_exact")
    return {
        "modes": modes,
        "fallback_count": fallback_count,
        "human_review_required": bool(fallback_count),
    }


__all__ = ["bounded_output_diagnostic", "generation_summary"]
