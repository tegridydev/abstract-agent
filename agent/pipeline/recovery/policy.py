"""Application owned limits for focused model output recovery."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BoundedRecoveryPolicy:
    """Hard limits that keep repair work smaller than the original generation."""

    attempts_per_identity: int = 1
    max_output_tokens: int = 512
    max_prompt_characters: int = 6_000
    max_evidence_items: int = 6
    inactivity_timeout_seconds: float = 120.0


DEFAULT_BOUNDED_RECOVERY = BoundedRecoveryPolicy()


__all__ = ["BoundedRecoveryPolicy", "DEFAULT_BOUNDED_RECOVERY"]
