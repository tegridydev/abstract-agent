"""Versioned identifiers and default draft routes for model output policies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class StageOutputPolicy:
    policy_id: str
    contract_version: str
    baseline: str = "basic_schema"
    compact_schema: dict[str, Any] | None = None
    model_owned_contract: bool = False


POLICIES: dict[str, StageOutputPolicy] = {
    "planning.research": StageOutputPolicy("planning.research", "1"),
    "scopes.identify": StageOutputPolicy("scopes.identify", "1"),
    "gaps.map": StageOutputPolicy("gaps.map", "1"),
    "hypotheses.kernel": StageOutputPolicy("hypotheses.kernel", "1"),
    "hypotheses.protocol": StageOutputPolicy("hypotheses.protocol", "1"),
    "hypotheses.generate": StageOutputPolicy("hypotheses.generate", "8", model_owned_contract=True),
    "critiques.evidence_unit": StageOutputPolicy("critiques.evidence_unit", "1"),
    "critiques.methodology_unit": StageOutputPolicy("critiques.methodology_unit", "1"),
    "critiques.evidence": StageOutputPolicy("critiques.evidence", "3"),
    "critiques.recover": StageOutputPolicy("critiques.recover", "3"),
    "critiques.methodology": StageOutputPolicy("critiques.methodology", "3"),
    "hypotheses.revise": StageOutputPolicy("hypotheses.revise", "8", model_owned_contract=True),
    "hypotheses.repair": StageOutputPolicy("hypotheses.repair", "8", model_owned_contract=True),
    "novelty.verify": StageOutputPolicy("novelty.verify", "1"),
    "novelty.recover": StageOutputPolicy("novelty.recover", "1"),
    "novelty.unit": StageOutputPolicy("novelty.unit", "1"),
    "proposal.compose": StageOutputPolicy("proposal.compose", "5", model_owned_contract=True),
}


def get_output_policy(policy_id: str | None, *, fallback: str) -> StageOutputPolicy:
    identifier = policy_id or fallback
    return POLICIES.get(identifier) or StageOutputPolicy(identifier, "1")


__all__ = ["POLICIES", "StageOutputPolicy", "get_output_policy"]
