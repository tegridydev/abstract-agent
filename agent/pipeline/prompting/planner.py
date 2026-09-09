"""Plan complete model requests before rendering or transport.

Character limits are useful as serialization guards, but they are not context limits.  This
module accounts for every request component in estimated tokens and always returns a decision
instead of raising when work cannot fit.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class PromptFit(StrEnum):
    FIT = "fit"
    SPLIT = "split"
    DEGRADED = "degraded"


@dataclass(frozen=True)
class BudgetPolicy:
    """Conservative context policy suitable for unknown local Ollama allocations."""

    # Ollama may allocate only 4k on lower-VRAM systems. Unknown allocations fail safe.
    fallback_context_tokens: int = 4_096
    safety_fraction: float = 0.20
    minimum_safety_tokens: int = 512
    thinking_reserve_tokens: int = 0


@dataclass
class TokenEstimator:
    """Conservative UTF-8 estimator that can learn from Ollama prompt telemetry."""

    bytes_per_token: float = 2.5
    samples: int = 0

    def estimate(self, text: str) -> int:
        return max(1, math.ceil(len(text.encode("utf-8")) / max(1.0, self.bytes_per_token)))

    def observe(self, text: str, prompt_tokens: int) -> None:
        if not text or prompt_tokens <= 0:
            return
        observed = len(text.encode("utf-8")) / prompt_tokens
        # Bias toward the safer (smaller bytes/token) estimate and adapt slowly.
        self.bytes_per_token = max(1.0, min(self.bytes_per_token, observed * 0.9))
        self.samples += 1


@dataclass(frozen=True)
class PromptPlan:
    fit: PromptFit
    estimated_input_tokens: int
    available_input_tokens: int
    context_tokens: int
    output_tokens: int
    safety_tokens: int
    component_tokens: dict[str, int] = field(default_factory=dict)
    included_ids: tuple[str, ...] = ()
    omitted_ids: tuple[str, ...] = ()
    reason: str | None = None

    @property
    def can_execute(self) -> bool:
        return self.fit is PromptFit.FIT

    def as_dict(self) -> dict[str, Any]:
        return {
            "fit": self.fit.value,
            "estimated_input_tokens": self.estimated_input_tokens,
            "available_input_tokens": self.available_input_tokens,
            "context_tokens": self.context_tokens,
            "output_tokens": self.output_tokens,
            "safety_tokens": self.safety_tokens,
            "component_tokens": self.component_tokens,
            "included_ids": list(self.included_ids),
            "omitted_ids": list(self.omitted_ids),
            "reason": self.reason,
        }


def plan_prompt(
    *,
    components: dict[str, str],
    output_tokens: int,
    allocated_context_tokens: int | None,
    advertised_context_tokens: int | None = None,
    splittable: bool = False,
    included_ids: tuple[str, ...] = (),
    omitted_ids: tuple[str, ...] = (),
    estimator: TokenEstimator | None = None,
    policy: BudgetPolicy | None = None,
) -> PromptPlan:
    """Return a fit/split/degraded decision without throwing a budget exception."""
    policy = policy or BudgetPolicy()
    estimator = estimator or TokenEstimator()
    advertised = int(advertised_context_tokens or 0)
    allocated = int(allocated_context_tokens or 0)
    context_tokens = allocated or min(
        advertised or policy.fallback_context_tokens,
        policy.fallback_context_tokens,
    )
    context_tokens = max(1_024, context_tokens)
    safety_tokens = max(
        policy.minimum_safety_tokens,
        math.ceil(context_tokens * policy.safety_fraction),
    )
    reserved = output_tokens + policy.thinking_reserve_tokens + safety_tokens
    available = max(0, context_tokens - reserved)
    by_component = {name: estimator.estimate(text) for name, text in components.items()}
    required = sum(by_component.values())
    if required <= available:
        fit = PromptFit.FIT
        reason = None
    elif splittable:
        fit = PromptFit.SPLIT
        reason = "the complete batch exceeds the request budget and must be split by identity"
    else:
        fit = PromptFit.DEGRADED
        reason = "one atomic model request cannot fit the allocated context safely"
    return PromptPlan(
        fit=fit,
        estimated_input_tokens=required,
        available_input_tokens=available,
        context_tokens=context_tokens,
        output_tokens=output_tokens,
        safety_tokens=safety_tokens,
        component_tokens=by_component,
        included_ids=included_ids,
        omitted_ids=omitted_ids,
        reason=reason,
    )


__all__ = ["BudgetPolicy", "PromptFit", "PromptPlan", "TokenEstimator", "plan_prompt"]
