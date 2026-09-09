"""Application owned bounded prompt views for scientific context."""

from __future__ import annotations

import hashlib
import html
import json
import re
from dataclasses import dataclass
from typing import Any

DEFAULT_VIEW_BUDGET = 6_000
REGISTERED_VIEWS = frozenset(
    {
        "compact_v1",
        "scope_v5",
        "hypothesis_v5",
        "critique_v5",
        "revision_v5",
        "novelty_v5",
        "proposal_v5",
    }
)


@dataclass(frozen=True)
class PromptView:
    text: str
    content_hash: str
    character_count: int
    included_ids: tuple[str, ...]
    omitted_items: int


def _short_text(value: Any, limit: int) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    prefix = text[: limit - 1]
    boundary = prefix.rfind(" ")
    if boundary >= limit // 2:
        prefix = prefix[:boundary]
    return prefix.rstrip() + "…"


def _terms(value: Any) -> set[str]:
    return set(re.findall(r"[a-z0-9][a-z0-9_-]*", str(value or "").casefold()))


def _similarity(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def _identifier(value: dict[str, Any]) -> str | None:
    for key in ("candidate_id", "paper_id", "scope_id", "query_id", "claim_id"):
        if value.get(key):
            return str(value[key])
    return None


def _compact(value: Any, *, depth: int = 0) -> Any:
    """Compact values structurally; serialized JSON is never sliced."""
    if hasattr(value, "model_dump"):
        value = value.model_dump()
    if depth >= 5:
        return _short_text(value, 160)
    if isinstance(value, str):
        return _short_text(value, 360)
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, dict):
        return {
            str(key): _compact(item, depth=depth + 1)
            for key, item in value.items()
            if not str(key).startswith("_")
        }
    if isinstance(value, (list, tuple)):
        return [_compact(item, depth=depth + 1) for item in value]
    return _short_text(value, 360)


def _scientific_spec_view(value: Any, *, concise: bool = False) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {"contract_status": "missing"}
    if not concise:
        compacted = _compact(
            {key: item for key, item in value.items() if key != "addressed_finding_ids"}
        )
        return compacted if isinstance(compacted, dict) else {}

    def operation(key: str) -> dict[str, str]:
        item = value.get(key, {})
        if not isinstance(item, dict):
            return {"name": _short_text(item, 100), "operational_definition": "missing"}
        return {
            "name": _short_text(item.get("name"), 100),
            "operational_definition": _short_text(item.get("operational_definition"), 140),
        }

    def outcome(item: Any) -> dict[str, Any]:
        if not isinstance(item, dict):
            return {"name": _short_text(item, 100), "contract_status": "missing"}
        return {
            "name": _short_text(item.get("name"), 100),
            "metric": _short_text(item.get("metric"), 100),
            "expected_relation": item.get("expected_relation"),
            "decision_rule": _short_text(item.get("decision_rule"), 140),
        }

    mechanism = value.get("mechanism", {})
    experiment = value.get("experiment", {})
    if not isinstance(mechanism, dict):
        mechanism = {}
    if not isinstance(experiment, dict):
        experiment = {}
    return {
        "title": _short_text(value.get("title"), 120),
        "study_design": value.get("study_design"),
        "system": _short_text(value.get("system"), 120),
        "exposure_or_intervention": operation("exposure_or_intervention"),
        "comparator": operation("comparator"),
        "primary_outcome": outcome(value.get("primary_outcome", {})),
        "secondary_outcomes": [outcome(item) for item in value.get("secondary_outcomes", [])[:2]],
        "mechanism": {
            "cause": _short_text(mechanism.get("cause"), 100),
            "mediator": _short_text(mechanism.get("mediator"), 100),
            "measurable_consequence": _short_text(mechanism.get("measurable_consequence"), 100),
        },
        "experiment": {
            "units": _short_text(experiment.get("units"), 100),
            "procedure": _short_text(experiment.get("procedure"), 160),
            "controls": [_short_text(item, 80) for item in experiment.get("controls", [])[:2]],
        },
        "novelty_baseline": _short_text(value.get("novelty_baseline"), 120),
        "novelty_difference": _short_text(value.get("novelty_difference"), 140),
        "evidence_slots": list(value.get("evidence_slots", []))[:2],
        "confounders": [_short_text(item, 80) for item in value.get("confounders", [])[:3]],
        "risks": [_short_text(item, 80) for item in value.get("risks", [])[:3]],
        "potential_insights": [
            _short_text(item, 100) for item in value.get("potential_insights", [])[:3]
        ],
    }


def _collect_ids(value: Any, output: list[str]) -> None:
    if isinstance(value, dict):
        identifier = _identifier(value)
        if identifier and identifier not in output:
            output.append(identifier)
        for item in value.values():
            _collect_ids(item, output)
    elif isinstance(value, list):
        for item in value:
            _collect_ids(item, output)


def _collect_slot_ids(value: Any, output: list[str]) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if key in {"citation_slot_map", "evidence_slot_map"} and isinstance(item, dict):
                for identifier in item.values():
                    text = str(identifier)
                    if text and text not in output:
                        output.append(text)
            _collect_slot_ids(item, output)
    elif isinstance(value, list):
        for item in value:
            _collect_slot_ids(item, output)


def _contains_scalar(value: Any, target: str) -> bool:
    if isinstance(value, dict):
        return any(_contains_scalar(item, target) for item in value.values())
    if isinstance(value, list):
        return any(_contains_scalar(item, target) for item in value)
    return value == target


def _trim_one(value: Any) -> tuple[Any, int, bool]:
    """Remove one complete, lowest priority item or field."""
    if isinstance(value, dict):
        candidates = [
            (key, item)
            for key, item in value.items()
            if key not in {"topic", "candidate_id", "paper_id", "scope_id", "title"}
        ]
        for key, item in reversed(candidates):
            trimmed, omitted, changed = _trim_one(item)
            if changed:
                copy = dict(value)
                copy[key] = trimmed
                return copy, omitted, True
        if candidates:
            copy = dict(value)
            copy.pop(candidates[-1][0])
            return copy, 1, True
    elif isinstance(value, list) and value:
        for index in range(len(value) - 1, -1, -1):
            trimmed, omitted, changed = _trim_one(value[index])
            if changed:
                copy = list(value)
                copy[index] = trimmed
                return copy, omitted, True
        if len(value) > 1:
            return list(value[:-1]), 1, True
    return value, 0, False


def build_prompt_view(
    view_name: str,
    input_keys: list[str],
    context: dict[str, Any],
    *,
    budget: int = DEFAULT_VIEW_BUDGET,
) -> PromptView:
    """Build a schema valid bounded view with IDs, omission count, size, and hash."""
    if view_name not in REGISTERED_VIEWS:
        raise ValueError(f"unknown prompt view: {view_name}")
    if budget < 1_024:
        raise ValueError("prompt view budget must be at least 1024 characters")
    evidence_slot_map: dict[str, str] = {}
    omissions_by_section: dict[str, int] = {}

    paper_titles = {
        str(paper.get("paper_id")): _short_text(paper.get("title"), 180)
        for paper in context.get("top_papers", [])
        if isinstance(paper, dict) and paper.get("paper_id")
    }

    def compact_cards(maximum: int = 8, paper_ids: list[str] | None = None) -> list[dict[str, Any]]:
        ledger = context.get("evidence_ledger", {})
        raw_cards = ledger.get("cards", []) if isinstance(ledger, dict) else []
        indexed = {
            str(card.get("paper_id")): (index, card)
            for index, card in enumerate(raw_cards, start=1)
            if isinstance(card, dict) and card.get("paper_id")
        }
        ordered = (
            [indexed[paper_id] for paper_id in paper_ids if paper_id in indexed]
            if paper_ids is not None
            else list(indexed.values())
        )
        cards = []
        for index, card in ordered[:maximum]:
            paper_id = str(card.get("paper_id") or "")
            slot = f"E{index}"
            if paper_id:
                evidence_slot_map[slot] = paper_id
            finding = next(
                (item for item in card.get("findings", []) if isinstance(item, dict)), {}
            )
            cards.append(
                {
                    "slot": slot,
                    "title": paper_titles.get(paper_id, ""),
                    "claim": _short_text(finding.get("claim"), 180),
                    "source_passage": _short_text(finding.get("source_passage"), 220),
                    "method": _short_text(card.get("method"), 120),
                    "limitations": [
                        _short_text(item, 120) for item in card.get("limitations", [])[:1]
                    ],
                }
            )
        return cards

    def ranked_card_ids(
        query: Any,
        *,
        preferred: list[str] | None = None,
        allowed: set[str] | None = None,
        maximum: int = 6,
        focus: Any = None,
    ) -> list[str]:
        ledger = context.get("evidence_ledger", {})
        raw_cards = ledger.get("cards", []) if isinstance(ledger, dict) else []
        query_terms = _terms(query)
        focus_terms = _terms(focus)
        preferred_ids = [
            paper_id
            for paper_id in dict.fromkeys(preferred or [])
            if allowed is None or paper_id in allowed
        ]
        candidates = []
        for index, card in enumerate(raw_cards):
            if not isinstance(card, dict) or not card.get("paper_id"):
                continue
            paper_id = str(card["paper_id"])
            if allowed is not None and paper_id not in allowed:
                continue
            finding = next(
                (item for item in card.get("findings", []) if isinstance(item, dict)), {}
            )
            card_text = " ".join(
                (
                    paper_titles.get(paper_id, ""),
                    str(finding.get("claim") or ""),
                    str(finding.get("source_passage") or ""),
                )
            )
            card_terms = _terms(card_text)
            candidates.append(
                (
                    paper_id,
                    len(query_terms & card_terms) + 5 * len(focus_terms & card_terms),
                    _similarity(query_terms, card_terms),
                    index,
                )
            )
        ranked = [
            paper_id
            for paper_id, _, _, _ in sorted(
                candidates, key=lambda item: (-item[1], -item[2], item[3])
            )
        ]
        return list(dict.fromkeys([*preferred_ids, *ranked]))[:maximum]

    if view_name == "scope_v5":
        papers = context.get("top_papers", [])
        payload = {
            "topic": context.get("topic"),
            "papers": [
                {
                    "paper_id": paper.get("paper_id"),
                    "title": _short_text(paper.get("title"), 180),
                    "summary": _short_text(paper.get("summary"), 320),
                    "role": paper.get("relevance_judgement", {}).get("role"),
                }
                for paper in papers[:12]
                if isinstance(paper, dict)
            ],
        }
    elif view_name == "hypothesis_v5":
        raw_gap_map = context.get("gap_map", {})
        gap_items = [
            item
            for key in ("established", "contradictions", "unresolved_questions")
            for item in (raw_gap_map.get(key, []) if isinstance(raw_gap_map, dict) else [])
            if isinstance(item, dict)
        ]
        gap_paper_ids = [
            str(paper_id) for item in gap_items for paper_id in item.get("paper_ids", [])
        ]
        research_scope = context.get("research_scope", {})
        allowed = (
            set(research_scope.get("supporting_paper_ids", []))
            if isinstance(research_scope, dict) and research_scope.get("mode") == "domain"
            else None
        )
        maximum_cards = 6 if budget >= 4_500 else 2
        selected_card_ids = ranked_card_ids(
            " ".join(
                (
                    str(context.get("topic") or ""),
                    str(research_scope.get("label") or "")
                    if isinstance(research_scope, dict)
                    else "",
                    " ".join(str(item.get("statement") or "") for item in gap_items),
                )
            ),
            preferred=gap_paper_ids,
            allowed=allowed,
            maximum=maximum_cards,
        )
        cards = compact_cards(maximum_cards, selected_card_ids)
        raw_card_count = len(context.get("evidence_ledger", {}).get("cards", []))
        omissions_by_section["evidence_cards"] = max(0, raw_card_count - len(cards))
        slot_by_paper = {paper_id: slot for slot, paper_id in evidence_slot_map.items()}
        compact_gaps = []
        seen_statements: set[str] = set()
        for item in gap_items:
            statement = _short_text(item.get("statement"), 180)
            normalized_statement = statement.casefold()
            if not statement or normalized_statement in seen_statements:
                continue
            slots = [
                slot_by_paper[str(paper_id)]
                for paper_id in item.get("paper_ids", [])
                if str(paper_id) in slot_by_paper
            ]
            if item.get("status") != "unknown" and not slots:
                continue
            seen_statements.add(normalized_statement)
            compact_gaps.append(
                {
                    "status": item.get("status"),
                    "statement": statement,
                    "evidence_slots": slots,
                }
            )
            if len(compact_gaps) == (4 if budget >= 4_500 else 2):
                break
        omissions_by_section["evidence_gaps"] = max(0, len(gap_items) - len(compact_gaps))
        payload = {
            "topic": context.get("topic"),
            "research_scope": {
                "mode": research_scope.get("mode"),
                "label": research_scope.get("label"),
            }
            if isinstance(research_scope, dict)
            else {},
            "evidence_gaps": compact_gaps,
            "evidence_cards": cards,
        }
    elif view_name == "critique_v5":
        candidates = context.get(
            "audited_hypothesis_candidates", context.get("hypothesis_candidates", [])
        )
        candidate_items = [item for item in candidates if isinstance(item, dict)]
        supporting_ids = list(
            dict.fromkeys(
                str(paper_id)
                for candidate in candidate_items
                for paper_id in candidate.get("supporting_paper_ids", [])
            )
        )
        audits = {
            str(item.get("candidate_id")): item
            for item in context.get("candidate_validity_audits", [])
            if isinstance(item, dict) and item.get("candidate_id")
        }
        payload = {
            "candidates": [
                {
                    "candidate_id": candidate.get("candidate_id"),
                    "scientific_spec": _scientific_spec_view(
                        candidate.get("scientific_spec", {}), concise=True
                    ),
                    "supporting_evidence_slots": [
                        f"E{index}"
                        for index, card in enumerate(
                            context.get("evidence_ledger", {}).get("cards", []), start=1
                        )
                        if isinstance(card, dict)
                        and card.get("paper_id") in candidate.get("supporting_paper_ids", [])
                    ],
                    "deterministic_audit": _compact(
                        audits.get(str(candidate.get("candidate_id"))) or {"status": "missing"}
                    ),
                }
                for candidate in candidate_items
            ],
            "evidence_cards": compact_cards(6, supporting_ids),
        }
        omissions_by_section["evidence_cards"] = max(0, len(supporting_ids) - 6)
    elif view_name == "revision_v5":
        selected_candidate = context.get("selected_candidate", {})
        selected_score = next(
            (
                item
                for item in context.get("candidate_scorecard", [])
                if item.get("candidate_id") == selected_candidate.get("candidate_id")
            ),
            {},
        )
        blocking_findings = selected_score.get("blocking_findings", [])
        if not blocking_findings:
            blocking_findings = [
                {
                    "finding_id": finding_id,
                    "severity": "major",
                    "issue": revision,
                    "required_revision": revision,
                }
                for finding_id, revision in zip(
                    selected_score.get("unresolved_finding_ids", []),
                    selected_score.get("unresolved_major_revisions", []),
                    strict=False,
                )
            ]
        selected_provenance = selected_candidate.get("generation_provenance", {})
        selected_spec = selected_candidate.get("scientific_spec", {})
        intervention = (
            selected_spec.get("exposure_or_intervention", {})
            if isinstance(selected_spec, dict)
            else {}
        )
        comparator = selected_spec.get("comparator", {}) if isinstance(selected_spec, dict) else {}
        focus = " ".join(
            (
                str(intervention.get("name") or "") if isinstance(intervention, dict) else "",
                str(comparator.get("name") or "") if isinstance(comparator, dict) else "",
                str(selected_spec.get("novelty_baseline") or "")
                if isinstance(selected_spec, dict)
                else "",
            )
        )
        revision_scope = context.get("research_scope", {})
        revision_allowed = (
            set(revision_scope.get("supporting_paper_ids", []))
            if isinstance(revision_scope, dict) and revision_scope.get("mode") == "domain"
            else None
        )
        maximum_cards = 6 if budget >= 4_500 else 2
        selected_card_ids = ranked_card_ids(
            json.dumps(selected_candidate.get("scientific_spec", {}), ensure_ascii=False)
            + " "
            + " ".join(str(item.get("required_revision") or "") for item in blocking_findings),
            preferred=[str(item) for item in selected_candidate.get("supporting_paper_ids", [])],
            allowed=revision_allowed,
            maximum=maximum_cards,
            focus=focus,
        )
        payload = {
            "selected_candidate": {
                "candidate_id": selected_candidate.get("candidate_id"),
                "configured_axis": selected_provenance.get("configured_axis"),
                "effective_axis": selected_provenance.get("effective_axis"),
                "scientific_spec": _scientific_spec_view(
                    selected_candidate.get("scientific_spec", {}), concise=budget < 4_500
                ),
            },
            "blocking_findings": _compact(blocking_findings),
            "evidence_cards": compact_cards(maximum_cards, selected_card_ids),
        }
        omissions_by_section["evidence_cards"] = max(
            0,
            len(context.get("evidence_ledger", {}).get("cards", [])) - maximum_cards,
        )
    elif view_name == "proposal_v5":
        payload = {
            "topic": context.get("topic"),
            "research_scope": _compact(context.get("research_scope", {})),
            "final_candidate": _compact(context.get("final_candidate", {})),
            "novelty_verification": _compact(context.get("novelty_verification", {})),
        }
    elif view_name == "compact_v1" and "evidence_ledger" in context:
        payload = {
            "topic": context.get("topic"),
            "research_scope": _compact(context.get("research_scope", {})),
            "evidence_cards": compact_cards(),
        }
    elif view_name == "novelty_v5":
        packets = []
        novelty_papers = context.get("novelty_papers", [])
        for candidate_key in ("revised_candidate", "alternate_candidate"):
            candidate = (
                context.get("validated_revised_candidate", {})
                if candidate_key == "revised_candidate"
                else context.get(candidate_key, {})
            ) or context.get(candidate_key, {})
            candidate_id = str(candidate.get("candidate_id") or "")
            matched = [
                paper
                for paper in novelty_papers
                if candidate_id in paper.get("matched_candidate_ids", [])
            ][:8]
            slot_map = {
                f"P{index}": str(paper.get("paper_id"))
                for index, paper in enumerate(matched, start=1)
            }
            packets.append(
                {
                    "candidate": _compact(candidate),
                    "citation_slot_map": slot_map,
                    "closest_prior_art": [
                        {
                            "slot": slot,
                            "title": _short_text(paper.get("title"), 240),
                            "summary": _short_text(paper.get("summary"), 700),
                            "year": paper.get("year"),
                        }
                        for slot, paper in zip(slot_map, matched, strict=True)
                    ],
                }
            )
        payload = {"candidate_packets": packets}
    else:
        payload = {key: _compact(context[key]) for key in input_keys}
    ledger = payload.get("evidence_ledger")
    if (
        view_name in {"hypothesis_v5", "revision_v5"}
        and isinstance(ledger, dict)
        and isinstance(ledger.get("cards"), list)
    ):
        slotted_cards = []
        for index, card in enumerate(ledger["cards"], start=1):
            if not isinstance(card, dict):
                continue
            paper_id = str(card.get("paper_id") or "")
            slot = f"E{index}"
            if paper_id:
                evidence_slot_map[slot] = paper_id
            slotted_cards.append({**card, "paper_id": slot})
        ledger["cards"] = slotted_cards
    omitted = sum(omissions_by_section.values())
    while True:
        active_slot_map = {
            slot: paper_id
            for slot, paper_id in evidence_slot_map.items()
            if _contains_scalar(payload, slot)
        }
        ids: list[str] = list(active_slot_map.values())
        _collect_slot_ids(payload, ids)
        _collect_ids(payload, ids)
        core = {
            "view": view_name,
            "data": payload,
            "evidence_slot_map": active_slot_map,
            "included_ids": ids,
            "omitted_items": omitted,
            "omissions_by_section": omissions_by_section,
        }
        canonical = json.dumps(core, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        digest = hashlib.sha256(canonical.encode()).hexdigest()
        document = {**core, "content_hash": digest}
        serialized = json.dumps(document, ensure_ascii=False, separators=(",", ":"))
        wrapped = f"<input_view>{html.escape(serialized, quote=False)}</input_view>"
        if len(wrapped) <= budget:
            return PromptView(wrapped, digest, len(wrapped), tuple(ids), omitted)
        if view_name in {"hypothesis_v5", "critique_v5", "revision_v5"}:
            raise ValueError(
                f"protected prompt view '{view_name}' cannot fit required scientific fields "
                f"within its {budget}-character budget"
            )
        payload, removed, changed = _trim_one(payload)
        omitted += removed
        if removed:
            omissions_by_section["generic_trimming"] = (
                omissions_by_section.get("generic_trimming", 0) + removed
            )
        if not changed:
            raise ValueError("prompt view cannot fit its identity fields within budget")


__all__ = ["DEFAULT_VIEW_BUDGET", "PromptView", "REGISTERED_VIEWS", "build_prompt_view"]
