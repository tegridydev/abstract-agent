"""Pure semantic reconciliation helpers for structured model responses."""

from typing import Any


def normalize_keyed_response_items(
    value: dict[str, Any],
    collection_key: str,
    identity_key: str,
    *,
    equivalent_keys: tuple[str, ...] | None = None,
) -> tuple[dict[str, Any], list[str]]:
    """Collapse identical keyed model items and reject conflicting duplicates."""
    normalized = dict(value)
    items = list(normalized.get(collection_key, []))
    by_id: dict[str, dict[str, Any]] = {}
    duplicate_ids: list[str] = []
    for item in items:
        identity = item[identity_key]
        existing = by_id.get(identity)
        if existing is None:
            by_id[identity] = item
            continue
        keys = equivalent_keys or tuple(sorted(set(existing) | set(item)))
        if any(existing.get(key) != item.get(key) for key in keys):
            raise ValueError(f"conflicting duplicate '{identity}' in {collection_key}")
        duplicate_ids.append(identity)
    normalized[collection_key] = list(by_id.values())
    return normalized, duplicate_ids


def reconcile_evidence_card_duplicates(
    value: dict[str, Any], *, exact: bool
) -> tuple[dict[str, Any], dict[str, list[str]]]:
    """Deduplicate ledger cards while isolating conflicts for focused recovery."""
    normalized = dict(value)
    groups: dict[str, list[dict[str, Any]]] = {}
    for card in normalized.get("cards", []):
        groups.setdefault(card["paper_id"], []).append(card)

    identical: list[str] = []
    conflicting: list[str] = []
    for paper_id, cards in groups.items():
        if len(cards) > 1 and all(card == cards[0] for card in cards[1:]):
            identical.append(paper_id)
        elif len(cards) > 1:
            conflicting.append(paper_id)

    if exact and conflicting:
        raise ValueError(
            "conflicting duplicate evidence cards for: " + ", ".join(sorted(conflicting))
        )
    normalized["cards"] = [
        cards[0] for paper_id, cards in groups.items() if paper_id not in conflicting
    ]
    return normalized, {
        "identical_paper_ids": identical,
        "recovery_paper_ids": conflicting,
        "conflicting_paper_ids": conflicting,
    }
