"""Conservative novelty verification fallback policy."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from ...contracts.research import NoveltyDecisionDraft, NoveltyVerifications


def compile_novelty_decisions(
    novelty_packets: list[dict[str, Any]],
    novelty_decision_drafts: list[dict[str, Any]],
) -> dict[str, Any]:
    """Attach application owned candidate IDs and conservatively fill failed units."""
    verifications: list[dict[str, Any]] = []
    for packet, outcome in zip(novelty_packets, novelty_decision_drafts, strict=True):
        candidate_id = str(packet.get("candidate_id") or "")
        known_ids = {
            str(item.get("paper_id"))
            for item in packet.get("prior_art", [])
            if isinstance(item, dict) and item.get("paper_id")
        }
        draft = outcome.get("value") if outcome.get("status") == "succeeded" else None
        try:
            decision = NoveltyDecisionDraft.model_validate(draft).model_dump()
            cited = set(decision["closest_prior_art_paper_ids"]) | set(
                decision["materially_matching_paper_ids"]
            )
            if not cited <= known_ids:
                raise ValueError("novelty unit cited prior art outside its packet")
        except (ValueError, TypeError):
            decision = {
                "status": "uncertain",
                "closest_prior_art_paper_ids": [],
                "materially_matching_paper_ids": [],
                "reasoning": (
                    "The bounded novelty unit did not produce a valid evidence backed "
                    "assessment, so novelty is not claimed."
                ),
                "limitations": ["Human review of the closest prior art is required."],
            }
        verifications.append({"candidate_id": candidate_id, **decision})
    return NoveltyVerifications.model_validate({"verifications": verifications}).model_dump()


def normalize_novelty_item(
    raw_item: Any,
    *,
    expected_candidate_ids: set[str],
    allowed_paper_ids: set[str],
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    """Conservatively retain usable novelty content without inventing a conclusion."""
    if not isinstance(raw_item, dict):
        return None, [{"operation": "drop", "reason": "novelty decision is not an object"}]
    candidate_id = str(raw_item.get("candidate_id") or "")
    if not candidate_id or candidate_id not in expected_candidate_ids:
        return None, [
            {
                "operation": "drop",
                "reason": f"unknown novelty candidate_id '{candidate_id or 'missing'}'",
            }
        ]
    operations: list[dict[str, Any]] = []

    def references(key: str) -> list[str]:
        value = raw_item.get(key, [])
        items = value if isinstance(value, list) else []
        normalized = list(
            dict.fromkeys(
                reference
                for reference in items
                if isinstance(reference, str) and reference in allowed_paper_ids
            )
        )[:8]
        if normalized != items:
            operations.append(
                {
                    "path": key,
                    "operation": "filter",
                    "reason": "removed unknown, duplicate, or excessive paper IDs",
                }
            )
        return normalized

    closest = references("closest_prior_art_paper_ids")
    matching = [
        paper_id for paper_id in references("materially_matching_paper_ids") if paper_id in closest
    ]
    status = raw_item.get("status")
    valid_statuses = {"novelty_supported", "prior_art_overlap", "uncertain"}
    if status not in valid_statuses:
        status = "uncertain"
        operations.append(
            {
                "path": "status",
                "operation": "derive",
                "reason": "unsupported novelty status was downgraded to uncertain",
            }
        )
    if (
        (status != "uncertain" and not closest)
        or (status == "prior_art_overlap" and not matching)
        or (status == "novelty_supported" and matching)
    ):
        status = "uncertain"
        operations.append(
            {
                "path": "status",
                "operation": "derive",
                "reason": "citation evidence did not support the supplied novelty status",
            }
        )
    reasoning = str(raw_item.get("reasoning") or "").strip()
    if not reasoning:
        reasoning = "The model did not provide a usable novelty rationale, so novelty is uncertain."
        operations.append(
            {
                "path": "reasoning",
                "operation": "fallback",
                "reason": "inserted a conservative missing rationale notice",
            }
        )
    limitations = raw_item.get("limitations", [])
    limitations = (
        list(dict.fromkeys(str(item).strip() for item in limitations if str(item).strip()))[:6]
        if isinstance(limitations, list)
        else []
    )
    review_notice = "Human review is required for the recovered novelty decision."
    if operations and review_notice not in limitations:
        limitations = [
            *limitations[:5],
            review_notice,
        ]
    item = {
        "candidate_id": candidate_id,
        "status": status,
        "closest_prior_art_paper_ids": closest,
        "materially_matching_paper_ids": matching,
        "reasoning": reasoning[:1000],
        "limitations": limitations,
    }
    return item, operations


def recover_novelty(candidate_ids: Iterable[str]) -> dict:
    return NoveltyVerifications.model_validate(
        {
            "verifications": [
                {
                    "candidate_id": candidate_id,
                    "status": "uncertain",
                    "closest_prior_art_paper_ids": [],
                    "materially_matching_paper_ids": [],
                    "reasoning": (
                        "The verification model did not return a valid assessment, so novelty "
                        "is not claimed."
                    ),
                    "limitations": ["A human must review the closest prior art."],
                }
                for candidate_id in sorted(set(candidate_ids))
            ]
        }
    ).model_dump()


__all__ = ["compile_novelty_decisions", "normalize_novelty_item", "recover_novelty"]
