"""Build compact recovery packets without resending an entire research corpus."""

from __future__ import annotations

import json
from typing import Any

from .policy import DEFAULT_BOUNDED_RECOVERY


def _clip(value: Any, *, string_limit: int = 600) -> Any:
    if isinstance(value, str):
        compact = " ".join(value.split())
        return compact if len(compact) <= string_limit else compact[: string_limit - 3] + "..."
    if isinstance(value, list):
        return [_clip(item, string_limit=string_limit) for item in value]
    if isinstance(value, dict):
        return {key: _clip(item, string_limit=string_limit) for key, item in value.items()}
    return value


def _serialized_size(value: Any) -> int:
    return len(json.dumps(value, ensure_ascii=False, separators=(",", ":")))


def _limit_lists(value: Any, *, item_limit: int) -> Any:
    if isinstance(value, list):
        return [_limit_lists(item, item_limit=item_limit) for item in value[:item_limit]]
    if isinstance(value, dict):
        return {key: _limit_lists(item, item_limit=item_limit) for key, item in value.items()}
    return value


def _bound_packet(packet: dict[str, Any], max_characters: int) -> dict[str, Any]:
    """Bound complete fields only; never slice serialized JSON."""
    bounded = _clip(packet)
    for key in ("related_gaps", "evidence_cards", "prior_art"):
        items = bounded.get(key)
        while (
            isinstance(items, list)
            and len(items) > 1
            and _serialized_size(bounded) > max_characters
        ):
            items.pop()
    if _serialized_size(bounded) > max_characters:
        bounded = _clip(bounded, string_limit=240)
    for item_limit, string_limit in ((3, 160), (2, 120), (1, 80)):
        if _serialized_size(bounded) <= max_characters:
            break
        bounded = _limit_lists(_clip(packet, string_limit=string_limit), item_limit=item_limit)
    if _serialized_size(bounded) > max_characters:
        candidate = packet.get("candidate", {})
        bounded = _clip(
            {
                "candidate_id": packet.get("candidate_id"),
                "candidate": {
                    key: candidate.get(key)
                    for key in ("candidate_id", "title", "hypothesis", "mechanism", "falsifier")
                    if isinstance(candidate, dict) and candidate.get(key) is not None
                },
            },
            string_limit=80,
        )
    if _serialized_size(bounded) > max_characters:
        raise ValueError("focused recovery identity cannot fit within the prompt budget")
    return bounded


def _missing_ids(
    candidates: list[dict[str, Any]], initial: dict[str, Any], collection_key: str
) -> list[str]:
    candidate_ids = [str(candidate["candidate_id"]) for candidate in candidates]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("candidate IDs must be unique")
    items = initial.get(collection_key, [])
    if not isinstance(items, list):
        items = []
    observed = [str(item.get("candidate_id", "")) for item in items if isinstance(item, dict)]
    if len(observed) != len(set(observed)):
        raise ValueError(f"initial {collection_key} contain duplicate candidate IDs")
    unknown = set(observed) - set(candidate_ids)
    if unknown:
        raise ValueError(f"initial {collection_key} contain unknown IDs: {sorted(unknown)}")
    return [candidate_id for candidate_id in candidate_ids if candidate_id not in observed]


def prepare_candidate_critique_packets(
    hypothesis_candidates: list[dict[str, Any]],
    initial_critiques: dict[str, Any],
    evidence_ledger: dict[str, Any],
    gap_map: dict[str, Any] | None = None,
    max_prompt_characters: int = DEFAULT_BOUNDED_RECOVERY.max_prompt_characters,
    max_evidence_items: int = DEFAULT_BOUNDED_RECOVERY.max_evidence_items,
) -> list[dict[str, Any]]:
    """Return one bounded packet per critique identity omitted by the initial batch."""
    by_id = {str(candidate["candidate_id"]): candidate for candidate in hypothesis_candidates}
    missing = _missing_ids(hypothesis_candidates, initial_critiques, "critiques")
    cards = evidence_ledger.get("cards", []) if isinstance(evidence_ledger, dict) else []
    gap_items = [
        item
        for key in ("established", "contradictions", "unresolved_questions")
        for item in ((gap_map or {}).get(key, []) if isinstance(gap_map, dict) else [])
        if isinstance(item, dict)
    ]
    packets = []
    for candidate_id in missing:
        candidate = by_id[candidate_id]
        supporting = set(candidate.get("supporting_paper_ids", []))
        relevant_cards = [
            card for card in cards if isinstance(card, dict) and card.get("paper_id") in supporting
        ][:max_evidence_items]
        relevant_gaps = [
            item
            for item in gap_items
            if not item.get("paper_ids") or supporting.intersection(item.get("paper_ids", []))
        ][:max_evidence_items]
        packets.append(
            _bound_packet(
                {
                    "candidate_id": candidate_id,
                    "candidate": candidate,
                    "evidence_cards": relevant_cards,
                    "related_gaps": relevant_gaps,
                },
                max_prompt_characters,
            )
        )
    return packets


def prepare_novelty_recovery_packets(
    revised_candidate: dict[str, Any],
    alternate_candidate: dict[str, Any],
    initial_verifications: dict[str, Any],
    novelty_papers: list[dict[str, Any]],
    max_prompt_characters: int = DEFAULT_BOUNDED_RECOVERY.max_prompt_characters,
    max_evidence_items: int = DEFAULT_BOUNDED_RECOVERY.max_evidence_items,
) -> list[dict[str, Any]]:
    """Return candidate specific prior art packets for omitted novelty decisions."""
    candidates = list(
        {
            str(revised_candidate["candidate_id"]): revised_candidate,
            str(alternate_candidate["candidate_id"]): alternate_candidate,
        }.values()
    )
    by_id = {str(candidate["candidate_id"]): candidate for candidate in candidates}
    missing = _missing_ids(candidates, initial_verifications, "verifications")
    packets = []
    for candidate_id in missing:
        candidate = by_id[candidate_id]
        supporting = set(candidate.get("supporting_paper_ids", []))
        ordered = sorted(
            (paper for paper in novelty_papers if isinstance(paper, dict)),
            key=lambda paper: (
                candidate_id not in paper.get("matched_candidate_ids", []),
                paper.get("paper_id") not in supporting,
                -float(paper.get("composite_score", 0) or 0),
                str(paper.get("paper_id", "")),
            ),
        )
        prior_art = [
            {
                "paper_id": paper.get("paper_id"),
                "title": paper.get("title"),
                "summary": paper.get("summary"),
                "sources": paper.get("sources", [paper.get("source")]),
                "relevance_judgement": paper.get("relevance_judgement"),
            }
            for paper in ordered[:max_evidence_items]
        ]
        packets.append(
            _bound_packet(
                {
                    "candidate_id": candidate_id,
                    "candidate": candidate,
                    "prior_art": prior_art,
                },
                max_prompt_characters,
            )
        )
    return packets


def prepare_novelty_packets(
    revised_candidate: dict[str, Any],
    alternate_candidate: dict[str, Any],
    novelty_papers: list[dict[str, Any]],
    max_prompt_characters: int = DEFAULT_BOUNDED_RECOVERY.max_prompt_characters,
    max_evidence_items: int = DEFAULT_BOUNDED_RECOVERY.max_evidence_items,
) -> list[dict[str, Any]]:
    """Return one bounded prior art packet for every distinct candidate."""
    return prepare_novelty_recovery_packets(
        revised_candidate,
        alternate_candidate,
        {"verifications": []},
        novelty_papers,
        max_prompt_characters,
        max_evidence_items,
    )


__all__ = [
    "prepare_candidate_critique_packets",
    "prepare_novelty_packets",
    "prepare_novelty_recovery_packets",
]
