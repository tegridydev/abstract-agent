"""Application owned hypothesis identity, scope, and evidence grounding."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError

from ...contracts.research import HypothesisCandidate, ScientificHypothesisSpec

_WORD = re.compile(r"[a-z0-9][a-z0-9_-]*", re.IGNORECASE)
_MAX_SUPPORTING_PAPERS = 2
_BOUNDARY_AXIS = "boundary-condition-or-failure-mode"


@dataclass(frozen=True)
class HypothesisAxisResolution:
    """Application owned effective axis and the evidence sets that make it admissible."""

    configured_axis: str
    effective_axis: str
    substitution_reason: str | None = None
    primary_scope_ids: tuple[str, ...] = ()
    secondary_scope_ids: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "configured_axis": self.configured_axis,
            "effective_axis": self.effective_axis,
            "axis_substitution_reason": self.substitution_reason,
        }


def resolve_hypothesis_axis(
    configured_axis: str,
    research_scope: dict[str, Any],
    evidence_scope_options: list[dict[str, Any]],
    core_paper_ids: set[str],
) -> HypothesisAxisResolution:
    """Resolve cross domain intent only when scopes have independently attributable evidence."""
    configured = str(configured_axis or "candidate")
    selected = set(research_scope.get("supporting_paper_ids", [])) & core_paper_ids
    if "cross-domain" not in configured.casefold():
        return HypothesisAxisResolution(
            configured,
            configured,
            primary_scope_ids=tuple(sorted(selected)),
        )

    scope_sets = [
        set(option.get("paper_ids", [])) & core_paper_ids
        for option in evidence_scope_options
        if isinstance(option, dict)
    ]
    scope_sets = [paper_ids for paper_ids in scope_sets if paper_ids]
    if research_scope.get("mode") == "domain":
        alternatives = [
            (index, paper_ids - selected)
            for index, paper_ids in enumerate(scope_sets)
            if paper_ids - selected
        ]
        if selected and alternatives:
            _, secondary = min(alternatives, key=lambda item: (-len(item[1]), item[0]))
            return HypothesisAxisResolution(
                configured,
                configured,
                primary_scope_ids=tuple(sorted(selected)),
                secondary_scope_ids=tuple(sorted(secondary)),
            )
        reason = "No validated evidence scope contains core evidence outside the selected scope."
    else:
        pairs: list[tuple[int, int, set[str], set[str]]] = []
        for left_index, left in enumerate(scope_sets):
            for right_index, right in enumerate(scope_sets[left_index + 1 :], left_index + 1):
                left_exclusive = left - right
                right_exclusive = right - left
                if left_exclusive and right_exclusive:
                    pairs.append((left_index, right_index, left_exclusive, right_exclusive))
        if pairs:
            _, _, primary, secondary = min(
                pairs,
                key=lambda item: (
                    -min(len(item[2]), len(item[3])),
                    -(len(item[2]) + len(item[3])),
                    item[0],
                    item[1],
                ),
            )
            return HypothesisAxisResolution(
                configured,
                configured,
                primary_scope_ids=tuple(sorted(primary)),
                secondary_scope_ids=tuple(sorted(secondary)),
            )
        reason = "No two validated evidence scopes contain mutually exclusive core evidence."
    return HypothesisAxisResolution(
        configured,
        _BOUNDARY_AXIS,
        substitution_reason=reason,
        primary_scope_ids=tuple(sorted(selected)),
    )


def _clean(value: Any, maximum: int, fallback: str) -> str:
    text = " ".join(str(value or "").split()) or fallback
    if len(text) <= maximum:
        return text
    shortened = text[: maximum - 3]
    boundary = shortened.rfind(" ")
    if boundary > maximum // 2:
        shortened = shortened[:boundary]
    return shortened.rstrip() + "..."


def _items(value: Any, maximum: int, fallback: str) -> list[str]:
    raw = value if isinstance(value, list) else []
    items = []
    for item in raw:
        text = " ".join(str(item or "").split())
        if text and text not in items:
            items.append(text)
        if len(items) == maximum:
            break
    return items or [fallback]


def _bounded_text(value: Any, maximum: int) -> str:
    """Return compact model authored text without inventing a required value."""
    text = " ".join(str(value or "").split())
    if len(text) <= maximum:
        return text
    shortened = text[: maximum - 3]
    boundary = shortened.rfind(" ")
    if boundary > maximum // 2:
        shortened = shortened[:boundary]
    return shortened.rstrip() + "..."


def _bounded_items(value: Any, maximum: int, item_maximum: int | None = None) -> list[str]:
    raw = value if isinstance(value, list) else []
    items: list[str] = []
    for item in raw:
        text = (
            _bounded_text(item, item_maximum) if item_maximum else " ".join(str(item or "").split())
        )
        if text and text not in items:
            items.append(text)
        if len(items) >= maximum:
            break
    return items


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _normalize_operation(value: Any) -> dict[str, Any]:
    source = _mapping(value)
    return {
        "name": _bounded_text(source.get("name"), 160),
        "operational_definition": _bounded_text(source.get("operational_definition"), 360),
    }


def _normalize_outcome(value: Any) -> dict[str, Any]:
    source = _mapping(value)
    return {
        "name": _bounded_text(source.get("name"), 160),
        "metric": _bounded_text(source.get("metric"), 160),
        "expected_relation": source.get("expected_relation"),
        "decision_rule": _bounded_text(source.get("decision_rule"), 300),
    }


def _normalize_scientific_spec(
    supplied: dict[str, Any],
) -> tuple[dict[str, Any] | None, bool]:
    """Validate or safely bound the optional v5 scientific intermediate representation."""
    try:
        return ScientificHypothesisSpec.model_validate(supplied).model_dump(), False
    except ValidationError:
        pass

    experiment = _mapping(supplied.get("experiment"))
    mechanism = _mapping(supplied.get("mechanism"))
    secondary = supplied.get("secondary_outcomes")
    normalized = {
        "title": _bounded_text(supplied.get("title"), 160),
        "study_design": supplied.get("study_design"),
        "system": _bounded_text(supplied.get("system"), 240),
        "exposure_or_intervention": _normalize_operation(supplied.get("exposure_or_intervention")),
        "comparator": _normalize_operation(supplied.get("comparator")),
        "primary_outcome": _normalize_outcome(supplied.get("primary_outcome")),
        "secondary_outcomes": [
            _normalize_outcome(item)
            for item in (secondary if isinstance(secondary, list) else [])[:2]
            if isinstance(item, dict)
        ],
        "mechanism": {
            "cause": _bounded_text(mechanism.get("cause"), 240),
            "mediator": _bounded_text(mechanism.get("mediator"), 240),
            "measurable_consequence": _bounded_text(mechanism.get("measurable_consequence"), 240),
        },
        "novelty_baseline": _bounded_text(supplied.get("novelty_baseline"), 240),
        "novelty_difference": _bounded_text(supplied.get("novelty_difference"), 300),
        "experiment": {
            "units": _bounded_text(experiment.get("units"), 240),
            "procedure": _bounded_text(experiment.get("procedure"), 600),
            "controls": _bounded_items(experiment.get("controls"), 4),
        },
        "evidence_slots": _bounded_items(supplied.get("evidence_slots"), 3),
        "confounders": _bounded_items(supplied.get("confounders"), 4),
        "risks": _bounded_items(supplied.get("risks"), 4),
        "potential_insights": _bounded_items(supplied.get("potential_insights"), 4),
        "addressed_finding_ids": _bounded_items(supplied.get("addressed_finding_ids"), 12),
    }
    try:
        return ScientificHypothesisSpec.model_validate(normalized).model_dump(), True
    except ValidationError:
        return None, True


def _append_supporting(
    supporting: list[str], candidates: Any, allowed: set[str], maximum: int
) -> None:
    for value in candidates if isinstance(candidates, (list, tuple)) else []:
        if len(supporting) >= maximum:
            return
        paper_id = str(value)
        if paper_id in allowed and paper_id not in supporting:
            supporting.append(paper_id)


def _terms(value: Any) -> set[str]:
    return {match.group(0).casefold() for match in _WORD.finditer(str(value or ""))}


def _card_passage(card: dict[str, Any]) -> str:
    findings = card.get("findings", [])
    for finding in findings if isinstance(findings, list) else []:
        if isinstance(finding, dict):
            passage = str(finding.get("source_passage") or finding.get("claim") or "").strip()
            if passage:
                return passage[:360]
    return str(card.get("relevance") or "Source backed evidence card")[:360]


def _jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 0.0


_RELATION_PHRASES = {
    "increases": "will be higher with {exposure} than with {comparator}",
    "decreases": "will be lower with {exposure} than with {comparator}",
    "differs": "will differ between {exposure} and {comparator}",
    "noninferior": "will be noninferior with {exposure} relative to {comparator}",
    "positive_association": "will be positively associated with {exposure}",
    "negative_association": "will be negatively associated with {exposure}",
}


def _render_spec(spec: dict[str, Any]) -> dict[str, Any]:
    """Render compact scientific fields into the stable public candidate contract."""
    if isinstance(spec.get("primary_outcome"), dict):
        exposure_spec = _mapping(spec.get("exposure_or_intervention"))
        comparator_spec = _mapping(spec.get("comparator"))
        outcome_spec = spec["primary_outcome"]
        exposure = _clean(exposure_spec.get("name"), 160, "the proposed exposure")
        comparator = _clean(comparator_spec.get("name"), 160, "the comparator")
        outcome_name = _clean(outcome_spec.get("name"), 160, "the primary outcome")
        metric = _clean(outcome_spec.get("metric"), 160, "the preregistered metric")
        outcome = (
            outcome_name
            if metric.casefold() in outcome_name.casefold()
            else (f"{outcome_name} ({metric})")
        )
        relation_name = str(outcome_spec.get("expected_relation") or "differs")
        relation = _RELATION_PHRASES.get(relation_name, _RELATION_PHRASES["differs"]).format(
            exposure=exposure, comparator=comparator
        )
        system = _clean(spec.get("system"), 240, "the studied system")
        decision_rule = _clean(
            outcome_spec.get("decision_rule"), 300, "the preregistered decision rule is met"
        )
        predictions = [f"{metric} will satisfy this decision rule: {decision_rule}."]
        secondary_outcomes = spec.get("secondary_outcomes")
        for item in (secondary_outcomes if isinstance(secondary_outcomes, list) else [])[:2]:
            if not isinstance(item, dict):
                continue
            secondary_relation = _RELATION_PHRASES.get(
                str(item.get("expected_relation")), _RELATION_PHRASES["differs"]
            ).format(exposure=exposure, comparator=comparator)
            predictions.append(
                f"{_clean(item.get('name'), 160, 'The secondary outcome')} measured by "
                f"{_clean(item.get('metric'), 160, 'the secondary metric')} "
                f"{secondary_relation}; decision rule: "
                f"{_clean(item.get('decision_rule'), 300, 'preregistered')}."
            )
        predictions = list(
            dict.fromkeys(" ".join(item.split()).replace("..", ".") for item in predictions)
        )
        relation_failure = {
            "increases": "does not exceed the comparator or reverses direction",
            "decreases": "does not fall below the comparator or reverses direction",
            "differs": "does not meet the preregistered minimum two sided difference",
            "noninferior": "crosses the preregistered noninferiority margin",
            "positive_association": "is non positive or reverses direction",
            "negative_association": "is non negative or reverses direction",
        }.get(relation_name, "does not meet the preregistered decision rule")
        experiment = _mapping(spec.get("experiment"))
        controls = _items(experiment.get("controls"), 4, "the preregistered comparator")
        mechanism = _mapping(spec.get("mechanism"))
        return {
            "hypothesis": f"In {system}, {outcome} {relation}.",
            "mechanism": (
                f"{_clean(mechanism.get('cause'), 240, 'The intervention')} changes "
                f"{_clean(mechanism.get('mediator'), 240, 'the proposed mediator')}, producing "
                f"{_clean(mechanism.get('measurable_consequence'), 240, outcome_name)}."
            ),
            "novelty_delta": (
                f"Relative to {_clean(spec.get('novelty_baseline'), 240, 'the closest baseline')}, "
                f"the proposed difference is "
                f"{_clean(spec.get('novelty_difference'), 300, 'unverified')}."
            ),
            "predictions": predictions,
            "falsifier": (
                f"Falsify the claim if {metric} {relation_failure} under this decision rule: "
                f"{decision_rule}."
            ),
            "proposed_experiment": (
                f"On {_clean(experiment.get('units'), 240, 'the preregistered study units')}, "
                f"{_clean(experiment.get('procedure'), 600, 'run the preregistered comparison')}. "
                f"Hold constant: {', '.join(controls)}."
            ),
            "novel_insights": spec.get("potential_insights"),
        }
    exposure = _clean(spec.get("exposure_or_intervention"), 300, "the proposed exposure")
    comparator = _clean(spec.get("comparator"), 240, "the preregistered comparator")
    outcome = _clean(spec.get("primary_outcome"), 240, "the primary outcome")
    relation = _RELATION_PHRASES.get(
        str(spec.get("expected_relation")), _RELATION_PHRASES["differs"]
    ).format(exposure=exposure, comparator=comparator)
    system = _clean(spec.get("system"), 240, "the studied system")
    hypothesis = f"In {system}, {outcome} {relation}."
    predictions = [hypothesis]
    secondary_predictions = spec.get("secondary_predictions")
    for item in (secondary_predictions if isinstance(secondary_predictions, list) else [])[:3]:
        if not isinstance(item, dict):
            continue
        secondary_outcome = _clean(item.get("outcome"), 240, "the secondary outcome")
        secondary_relation = _RELATION_PHRASES.get(
            str(item.get("expected_relation")), _RELATION_PHRASES["differs"]
        ).format(exposure=exposure, comparator=comparator)
        predictions.append(f"{secondary_outcome} {secondary_relation}.")
    falsifier = (
        f"Falsify the claim if the preregistered contrast between {exposure} and {comparator} "
        f"for {outcome} is absent, fails its noninferiority margin, or reverses direction."
    )
    return {
        "hypothesis": hypothesis,
        "predictions": predictions,
        "falsifier": falsifier,
        "proposed_experiment": spec.get("experiment"),
        "novel_insights": spec.get("potential_insights"),
    }


def project_hypothesis_candidate(
    raw: Any,
    *,
    topic: str,
    candidate_id: str,
    research_scope: dict[str, Any],
    evidence_scope_options: list[dict[str, Any]],
    evidence_ledger: dict[str, Any],
    evidence_papers: list[dict[str, Any]],
    model: str | None = None,
    semantic_draft: str | None = None,
    generation_mode: str | None = None,
    validation_attempts: int = 1,
    axis_resolution: HypothesisAxisResolution | None = None,
) -> dict[str, Any]:
    """Project imperfect model prose into a conservative, source grounded candidate."""
    supplied = dict(raw) if isinstance(raw, dict) else {}
    v5_required = {
        "system",
        "exposure_or_intervention",
        "comparator",
        "primary_outcome",
        "mechanism",
        "experiment",
    }
    has_v5_intent = v5_required <= set(supplied)
    has_v5_spec = has_v5_intent and all(
        isinstance(supplied.get(key), dict)
        for key in (
            "exposure_or_intervention",
            "comparator",
            "primary_outcome",
            "mechanism",
            "experiment",
        )
    )
    has_v4_spec = {
        "system",
        "exposure_or_intervention",
        "comparator",
        "primary_outcome",
        "expected_relation",
    } <= set(supplied)
    scientific_spec, scientific_spec_changed = (
        _normalize_scientific_spec(supplied) if has_v5_intent else (None, False)
    )
    if has_v5_spec or has_v4_spec:
        supplied = {**supplied, **_render_spec(supplied)}
    draft = " ".join(str(semantic_draft or (raw if isinstance(raw, str) else "")).split())
    content = " ".join(
        str(supplied.get(key) or "")
        for key in ("title", "hypothesis", "mechanism", "proposed_experiment", "novelty_delta")
    )
    alignment_text = f"{draft} {content}".strip() or f"{topic} {candidate_id}"
    alignment_terms = _terms(alignment_text)

    roles = {
        str(paper.get("paper_id")): paper.get("relevance_judgement", {}).get("role")
        for paper in evidence_papers
        if isinstance(paper, dict) and paper.get("paper_id")
    }
    cards = {
        str(card.get("paper_id")): card
        for card in evidence_ledger.get("cards", [])
        if isinstance(card, dict) and card.get("paper_id")
    }
    evidence_slots = {f"E{index}": paper_id for index, paper_id in enumerate(cards, start=1)}
    core_ids = [
        paper_id
        for paper_id in cards
        if roles.get(paper_id) in {"direct_evidence", "transferable_method"}
    ]
    core_id_set = set(core_ids)
    resolution = axis_resolution or resolve_hypothesis_axis(
        candidate_id,
        research_scope,
        evidence_scope_options,
        core_id_set,
    )
    if "cross-domain" in resolution.effective_axis.casefold() and (
        not core_id_set.intersection(resolution.primary_scope_ids)
        or not core_id_set.intersection(resolution.secondary_scope_ids)
    ):
        # A caller may have resolved against papers before the ledger was reconciled.
        # Re-resolve against the cards that can actually ground this candidate.
        resolution = resolve_hypothesis_axis(
            resolution.configured_axis,
            research_scope,
            evidence_scope_options,
            core_id_set,
        )
    requested_ids = [
        evidence_slots[slot]
        for slot in supplied.get("evidence_slots", [])
        if slot in evidence_slots and evidence_slots[slot] in core_id_set
    ]
    ranked = sorted(
        core_ids,
        key=lambda paper_id: (
            -_jaccard(alignment_terms, _terms(_card_passage(cards[paper_id]))),
            paper_id,
        ),
    )
    selected_scope_ids = set(research_scope.get("supporting_paper_ids", [])) & core_id_set
    wants_cross_domain = "cross-domain" in resolution.effective_axis.casefold()
    supporting: list[str] = []
    applicability = "broad"
    target_domain = None
    if wants_cross_domain:
        primary = set(resolution.primary_scope_ids)
        secondary = set(resolution.secondary_scope_ids)
        primary_ranked = [paper_id for paper_id in [*requested_ids, *ranked] if paper_id in primary]
        secondary_ranked = [
            paper_id for paper_id in [*requested_ids, *ranked] if paper_id in secondary
        ]
        _append_supporting(supporting, primary_ranked[:1], core_id_set, _MAX_SUPPORTING_PAPERS)
        _append_supporting(supporting, secondary_ranked[:1], core_id_set, _MAX_SUPPORTING_PAPERS)
        applicability = "cross_domain"
    elif research_scope.get("mode") == "domain":
        scoped = [
            paper_id for paper_id in [*requested_ids, *ranked] if paper_id in selected_scope_ids
        ]
        _append_supporting(supporting, scoped, core_id_set, _MAX_SUPPORTING_PAPERS)
        applicability = "domain_specific"
        target_domain = _clean(research_scope.get("label"), 100, "Selected domain")
    else:
        _append_supporting(
            supporting, [*requested_ids, *ranked], core_id_set, _MAX_SUPPORTING_PAPERS
        )
    if len(supporting) < _MAX_SUPPORTING_PAPERS:
        _append_supporting(supporting, ranked, core_id_set, _MAX_SUPPORTING_PAPERS)
    supporting = list(dict.fromkeys(supporting))[:_MAX_SUPPORTING_PAPERS]
    if not supporting:
        raise ValueError("cannot project a hypothesis without core evidence cards")

    fallback_hypothesis = (
        f"Investigate whether a measurable change in {topic} is associated with a reproducible "
        "change in the selected evaluation outcomes."
    )
    fallback_mechanism = (
        "The mechanism was not reliably specified by the model and remains a proposed inference."
    )
    fallback_experiment = (
        "Compare a preregistered intervention and control on the same datasets and evaluation "
        "protocol, reporting uncertainty and failure cases."
    )
    used_model_content = bool(content or draft)
    mode = generation_mode or ("draft_projected" if used_model_content else "conservative_fallback")
    if scientific_spec_changed and mode == "structured_exact":
        mode = "structured_normalized" if scientific_spec is not None else "draft_projected"
    contract_valid = mode in {"structured_exact", "structured_normalized"}
    limitations = (
        []
        if contract_valid
        else [
            "The model did not satisfy the complete hypothesis contract.",
            "Application owned projection supplied metadata and evidence grounding.",
        ]
    )
    if has_v5_intent and scientific_spec is None:
        limitations.append(
            "The incomplete compact scientific specification was omitted after safe projection."
        )
    if resolution.substitution_reason:
        limitations.append(
            f"Configured axis '{resolution.configured_axis}' was replaced with "
            f"'{resolution.effective_axis}': {resolution.substitution_reason}"
        )
    result = {
        "candidate_id": resolution.effective_axis,
        "title": _clean(
            supplied.get("title"),
            240,
            f"{topic.title()}: {resolution.effective_axis.replace('-', ' ')}",
        ),
        "hypothesis": _clean(supplied.get("hypothesis") or draft, 1000, fallback_hypothesis),
        "mechanism": _clean(supplied.get("mechanism"), 1000, fallback_mechanism),
        "novelty_delta": _clean(
            supplied.get("novelty_delta"),
            600,
            "The difference from closest prior art is unverified and requires human review.",
        ),
        "supporting_paper_ids": supporting,
        "conflicting_paper_ids": [
            paper_id
            for paper_id in dict.fromkeys(supplied.get("conflicting_paper_ids", []))
            if paper_id in cards and paper_id not in supporting
        ][:8],
        "predictions": _items(
            supplied.get("predictions"),
            6,
            "The preregistered intervention should produce a reproducible directional difference.",
        ),
        "falsifier": _clean(
            supplied.get("falsifier"),
            800,
            "Reject the hypothesis if the preregistered effect is absent or reverses across "
            "independent evaluations.",
        ),
        "proposed_experiment": _clean(
            supplied.get("proposed_experiment"), 1200, fallback_experiment
        ),
        "risks": _items(
            supplied.get("risks"),
            8,
            "The projected contract may omit domain specific confounders and requires review.",
        ),
        "novel_insights": _items(
            supplied.get("novel_insights"),
            6,
            "This is an unverified research direction rather than an established novel finding.",
        ),
        "applicability": applicability,
        "target_domain": target_domain,
        "grounding_links": [
            {
                "paper_id": paper_id,
                "evidence_role": roles[paper_id],
                "alignment_method": "lexical" if used_model_content else "application_fallback",
                "alignment_score": round(
                    _jaccard(alignment_terms, _terms(_card_passage(cards[paper_id]))), 3
                ),
                "source_passage": _card_passage(cards[paper_id]),
            }
            for paper_id in supporting
        ],
        "generation_provenance": {
            "mode": mode,
            "model": model,
            "validation_attempts": max(0, validation_attempts),
            "human_review_required": not contract_valid,
            "application_enriched": True,
            "contract_integrity": {
                "structured_exact": "exact",
                "structured_normalized": "normalized",
                "draft_projected": "projected",
                "conservative_fallback": "fallback",
            }.get(mode, "projected"),
            "scientific_integrity": "not_audited",
            "configured_axis": resolution.configured_axis,
            "effective_axis": resolution.effective_axis,
            "axis_substitution_reason": resolution.substitution_reason,
            "limitations": limitations,
        },
    }
    if scientific_spec is not None:
        result["scientific_spec"] = scientific_spec
    return HypothesisCandidate.model_validate(result).model_dump()


__all__ = [
    "HypothesisAxisResolution",
    "project_hypothesis_candidate",
    "resolve_hypothesis_axis",
]
