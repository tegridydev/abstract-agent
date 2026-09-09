"""Scope and gap map recovery policies."""

from __future__ import annotations


def recover_scopes() -> dict:
    return {"options": []}


def recover_gap_map() -> dict:
    return {
        "established": [],
        "contradictions": [],
        "unresolved_questions": [
            {
                "statement": (
                    "Evidence gaps remain unresolved because the selected model did not return "
                    "a valid gap map."
                ),
                "paper_ids": [],
                "status": "unknown",
            }
        ],
    }


__all__ = ["recover_gap_map", "recover_scopes"]
