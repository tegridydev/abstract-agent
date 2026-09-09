"""Application owned v6 hypothesis and critique work units."""

from __future__ import annotations

import json
import re
from typing import Any, Literal

from ...agent_helpers import select_candidate
from .critiques import normalize_critique_item, recover_critiques
from .hypotheses import project_hypothesis_candidate, resolve_hypothesis_axis
from .validity import audit_hypothesis_candidates

DEFAULT_AXES = ("mechanism", "method-or-measurement", "cross-domain-application")
_WORD = re.compile(r"[a-z0-9][a-z0-9_-]*", re.IGNORECASE)


def _terms(value: Any) -> set[str]:
    return {match.group(0).casefold() for match in _WORD.finditer(str(value or ""))}


def _text(value: Any, limit: int) -> str:
    compact = " ".join(str(value or "").split())
    if len(compact) <= limit:
        return compact
    shortened = compact[: limit - 1]
    boundary = shortened.rfind(" ")
    if boundary >= limit // 2:
        shortened = shortened[:boundary]
    return shortened.rstrip() + "…"


def _outcome_value(value: Any) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    if isinstance(value, dict) and value.get("status") in {
        "succeeded",
        "degraded",
        "skipped",
        "terminal",
        "failed",
    }:
        payload = value.get("value")
        return (payload if isinstance(payload, dict) else None), value
    return (value if isinstance(value, dict) else None), {
        "status": "succeeded",
        "metadata": {"integrity": "exact"},
    }


def _cards(evidence_ledger: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        card
        for card in evidence_ledger.get("cards", [])
        if isinstance(card, dict) and card.get("paper_id")
    ]


def _card_view(card: dict[str, Any], slot: str) -> dict[str, Any]:
    findings = []
    for finding in card.get("findings", []) if isinstance(card.get("findings"), list) else []:
        if not isinstance(finding, dict):
            continue
        claim = _text(finding.get("claim"), 180)
        passage = _text(finding.get("source_passage"), 160)
        if claim or passage:
            findings.append({"claim": claim, "source_passage": passage})
        if len(findings) == 1:
            break
    return {
        "slot": slot,
        "findings": findings,
        "method": _text(card.get("method"), 120),
        "limitations": [_text(item, 90) for item in card.get("limitations", [])[:1]],
        "relevance": _text(card.get("relevance"), 100),
    }


def _operation_view(value: Any) -> dict[str, str]:
    source = value if isinstance(value, dict) else {}
    return {
        "name": _text(source.get("name"), 100),
        "operational_definition": _text(source.get("operational_definition"), 160),
    }


def _outcome_view(value: Any) -> dict[str, Any]:
    source = value if isinstance(value, dict) else {}
    return {
        "name": _text(source.get("name"), 100),
        "metric": _text(source.get("metric"), 100),
        "expected_relation": source.get("expected_relation"),
        "decision_rule": _text(source.get("decision_rule"), 160),
    }


def _spec_view(value: Any) -> dict[str, Any]:
    source = value if isinstance(value, dict) else {}
    mechanism = source.get("mechanism") if isinstance(source.get("mechanism"), dict) else {}
    experiment = source.get("experiment") if isinstance(source.get("experiment"), dict) else {}
    return {
        "title": _text(source.get("title"), 120),
        "study_design": source.get("study_design"),
        "system": _text(source.get("system"), 140),
        "exposure_or_intervention": _operation_view(source.get("exposure_or_intervention")),
        "comparator": _operation_view(source.get("comparator")),
        "primary_outcome": _outcome_view(source.get("primary_outcome")),
        "mechanism": {
            "cause": _text(mechanism.get("cause"), 100),
            "mediator": _text(mechanism.get("mediator"), 100),
            "measurable_consequence": _text(mechanism.get("measurable_consequence"), 100),
        },
        "experiment": {
            "units": _text(experiment.get("units"), 100),
            "procedure": _text(experiment.get("procedure"), 160),
            "controls": [_text(item, 70) for item in experiment.get("controls", [])[:2]],
        },
        "confounders": [_text(item, 70) for item in source.get("confounders", [])[:2]],
        "risks": [_text(item, 70) for item in source.get("risks", [])[:2]],
    }


def _audit_view(value: Any) -> dict[str, Any]:
    source = value if isinstance(value, dict) else {}
    return {
        "status": source.get("status"),
        "grounding_tier": source.get("grounding_tier"),
        "findings": [
            {
                "category": item.get("category"),
                "severity": item.get("severity"),
                "field_paths": item.get("field_paths", [])[:3],
                "issue": _text(item.get("issue"), 100),
                "required_revision": _text(item.get("required_revision"), 100),
            }
            for item in source.get("findings", [])[:2]
            if isinstance(item, dict)
        ],
    }


def prepare_candidate_kernel_packets(
    topic: str,
    research_scope: dict[str, Any],
    evidence_scope_options: dict[str, Any],
    evidence_ledger: dict[str, Any],
    gap_map: dict[str, Any],
    top_papers: list[dict[str, Any]],
    *,
    axes: tuple[str, ...] = DEFAULT_AXES,
    max_evidence_items: int = 3,
) -> list[dict[str, Any]]:
    """Create one compact, stable packet per feasible innovation axis."""
    cards = _cards(evidence_ledger)
    card_slots = {str(card["paper_id"]): f"E{index}" for index, card in enumerate(cards, start=1)}
    roles = {
        str(paper.get("paper_id")): paper.get("relevance_judgement", {}).get("role")
        for paper in top_papers
        if isinstance(paper, dict) and paper.get("paper_id")
    }
    core_ids = {
        paper_id
        for paper_id in card_slots
        if roles.get(paper_id) in {"direct_evidence", "transferable_method"}
    }
    scope_options = evidence_scope_options.get("options", [])
    gap_items = [
        item
        for key in ("contradictions", "unresolved_questions")
        for item in gap_map.get(key, [])
        if isinstance(item, dict)
    ]
    packets: list[dict[str, Any]] = []
    for configured_axis in axes:
        resolution = resolve_hypothesis_axis(
            configured_axis,
            research_scope,
            scope_options,
            core_ids,
        )
        focus = _terms(f"{topic} {resolution.effective_axis} {research_scope.get('label', '')}")
        required = set(resolution.primary_scope_ids) | set(resolution.secondary_scope_ids)
        ordered = sorted(
            (card for card in cards if str(card["paper_id"]) in core_ids),
            key=lambda card: (
                str(card["paper_id"]) not in required if required else False,
                -len(focus & _terms(json.dumps(card, ensure_ascii=False))),
                str(card["paper_id"]),
            ),
        )
        selected = ordered[:max_evidence_items]
        selected_ids = {str(card["paper_id"]) for card in selected}
        gaps = []
        for item in gap_items:
            paper_ids = [str(value) for value in item.get("paper_ids", [])]
            if paper_ids and not selected_ids.intersection(paper_ids):
                continue
            gaps.append(
                {
                    "statement": _text(item.get("statement"), 180),
                    "evidence_slots": [
                        card_slots[paper_id] for paper_id in paper_ids if paper_id in selected_ids
                    ],
                }
            )
            if len(gaps) == 3:
                break
        packets.append(
            {
                "unit_id": resolution.effective_axis,
                "configured_axis": resolution.configured_axis,
                "axis_resolution": resolution.as_dict(),
                "input": {
                    "topic": _text(topic, 180),
                    "innovation_axis": resolution.effective_axis,
                    "scope": {
                        "mode": research_scope.get("mode"),
                        "label": _text(research_scope.get("label"), 100),
                    },
                    "evidence_cards": [
                        _card_view(card, card_slots[str(card["paper_id"])]) for card in selected
                    ],
                    "evidence_gaps": gaps,
                },
            }
        )
    return packets


def prepare_candidate_protocol_packets(
    candidate_kernel_packets: list[dict[str, Any]],
    candidate_kernels: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Carry forward only kernels that validated as complete model owned contracts."""
    packets = []
    for packet, outcome in zip(candidate_kernel_packets, candidate_kernels, strict=False):
        kernel, trace = _outcome_value(outcome)
        if kernel is None or trace.get("status") != "succeeded":
            continue
        packets.append(
            {
                "unit_id": packet["unit_id"],
                "configured_axis": packet["configured_axis"],
                "axis_resolution": packet["axis_resolution"],
                "kernel": kernel,
                "input": {
                    "topic": packet["input"]["topic"],
                    "innovation_axis": packet["input"]["innovation_axis"],
                    "candidate_kernel": kernel,
                    "evidence_cards": packet["input"]["evidence_cards"],
                    "evidence_gaps": packet["input"]["evidence_gaps"],
                },
                "kernel_integrity": trace.get("metadata", {}).get("integrity", "exact"),
            }
        )
    return packets


def compile_candidate_units(
    candidate_protocol_packets: list[dict[str, Any]],
    candidate_protocols: list[dict[str, Any]],
    topic: str,
    research_scope: dict[str, Any],
    evidence_scope_options: dict[str, Any],
    evidence_ledger: dict[str, Any],
    top_papers: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Assemble complete scientific IR; incomplete units remain absent and cannot be selected."""
    candidates = []
    for packet, outcome in zip(candidate_protocol_packets, candidate_protocols, strict=False):
        protocol, trace = _outcome_value(outcome)
        if protocol is None or trace.get("status") != "succeeded":
            continue
        scientific_spec = {**packet["kernel"], **protocol}
        integrity = (
            "structured_normalized"
            if "normalized"
            in {
                packet.get("kernel_integrity"),
                trace.get("metadata", {}).get("integrity"),
            }
            else "structured_exact"
        )
        resolution = resolve_hypothesis_axis(
            packet["configured_axis"],
            research_scope,
            evidence_scope_options.get("options", []),
            {
                str(paper.get("paper_id"))
                for paper in top_papers
                if paper.get("relevance_judgement", {}).get("role")
                in {"direct_evidence", "transferable_method"}
            },
        )
        try:
            candidate = project_hypothesis_candidate(
                scientific_spec,
                topic=topic,
                candidate_id=packet["unit_id"],
                research_scope=research_scope,
                evidence_scope_options=evidence_scope_options.get("options", []),
                evidence_ledger=evidence_ledger,
                evidence_papers=top_papers,
                generation_mode=integrity,
                axis_resolution=resolution,
            )
        except ValueError:
            continue
        candidates.append(candidate)
    return candidates


def audit_candidate_pool(
    hypothesis_candidates: list[dict[str, Any]],
    evidence_ledger: dict[str, Any],
    topic: str,
    *,
    embedding_service: Any | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    audited, audits = audit_hypothesis_candidates(
        hypothesis_candidates,
        evidence_ledger,
        topic,
        embedding_service=embedding_service,
    )
    complete = [
        candidate
        for candidate in audited
        if candidate.get("scientific_spec")
        and candidate.get("generation_provenance", {}).get("mode")
        in {"structured_exact", "structured_normalized"}
    ]
    outcome = {
        "status": "continue" if complete else "evidence_only",
        "reason": (
            None
            if complete
            else "No model generated candidate completed the bounded scientific contracts."
        ),
    }
    return complete, audits, outcome


def prepare_all_candidate_critique_packets(
    hypothesis_candidates: list[dict[str, Any]],
    candidate_validity_audits: list[dict[str, Any]],
    evidence_ledger: dict[str, Any],
) -> list[dict[str, Any]]:
    """Build one compact critique packet per candidate; batch review is intentionally absent."""
    cards = {str(card.get("paper_id")): card for card in _cards(evidence_ledger)}
    audits = {
        str(audit.get("candidate_id")): audit
        for audit in candidate_validity_audits
        if isinstance(audit, dict)
    }
    packets = []
    for candidate in hypothesis_candidates:
        candidate_id = str(candidate["candidate_id"])
        packets.append(
            {
                "unit_id": candidate_id,
                "input": {
                    "scientific_spec": _spec_view(candidate.get("scientific_spec", {})),
                    "deterministic_audit": _audit_view(
                        audits.get(candidate_id, {"status": "missing"})
                    ),
                    "evidence_cards": [
                        _card_view(cards[paper_id], f"E{index}")
                        for index, paper_id in enumerate(
                            candidate.get("supporting_paper_ids", []), start=1
                        )
                        if paper_id in cards
                    ],
                },
            }
        )
    return packets


def compile_candidate_critiques(
    critique_packets: list[dict[str, Any]],
    critique_drafts: list[dict[str, Any]],
    *,
    critic_source: Literal["evidence_critic", "methodology_critic"],
) -> dict[str, Any]:
    """Attach application owned identities and conservative fallback coverage."""
    critiques = []
    for packet, outcome in zip(critique_packets, critique_drafts, strict=False):
        candidate_id = str(packet["unit_id"])
        draft, trace = _outcome_value(outcome)
        if draft is None or trace.get("status") != "succeeded":
            critiques.extend(recover_critiques([candidate_id])["critiques"])
            continue
        item, _ = normalize_critique_item(
            {**draft, "candidate_id": candidate_id, "novelty": 1},
            expected_candidate_ids={candidate_id},
            critic_source=critic_source,
        )
        if item is None:
            critiques.extend(recover_critiques([candidate_id])["critiques"])
        else:
            critiques.append(item)
    known = {str(item["candidate_id"]) for item in critiques}
    for packet in critique_packets:
        candidate_id = str(packet["unit_id"])
        if candidate_id not in known:
            critiques.extend(recover_critiques([candidate_id])["critiques"])
    return {"critiques": critiques}


def select_candidate_v6(
    hypothesis_candidates: list[dict[str, Any]],
    evidence_critiques: dict[str, Any],
    methodology_critiques: dict[str, Any],
    candidate_validity_audits: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    selected, alternate, scorecard = select_candidate(
        hypothesis_candidates,
        evidence_critiques,
        methodology_critiques,
        candidate_validity_audits,
        grounding_weight=0.45,
        novelty_weight=0.0,
        falsifiability_weight=0.25,
        feasibility_weight=0.25,
        clarity_weight=0.05,
    )
    eligible = any(item.get("eligible") for item in scorecard)
    return (
        selected,
        alternate,
        scorecard,
        {
            "status": "continue" if eligible else "evidence_only",
            "reason": (
                None
                if eligible
                else "No complete candidate passed the scientific gate with usable critic scores."
            ),
        },
    )


__all__ = [
    "audit_candidate_pool",
    "compile_candidate_critiques",
    "compile_candidate_units",
    "prepare_all_candidate_critique_packets",
    "prepare_candidate_kernel_packets",
    "prepare_candidate_protocol_packets",
    "select_candidate_v6",
]
