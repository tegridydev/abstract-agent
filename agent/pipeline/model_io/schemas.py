"""Ollama compatible transport schemas kept separate from domain orchestration."""

from __future__ import annotations

import copy
from typing import Any

from ...contracts.research import RESPONSE_SCHEMAS

_OLLAMA_SCHEMA_KEYS = {
    "$defs",
    "$ref",
    "additionalProperties",
    "anyOf",
    "enum",
    "items",
    "properties",
    "required",
    "type",
}


def ollama_format_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Retain structural JSON Schema while omitting fragile repetition bounds."""

    def clean(value: Any, *, named_children: bool = False) -> Any:
        if isinstance(value, list):
            return [clean(item) for item in value]
        if not isinstance(value, dict):
            return value
        if named_children:
            return {key: clean(item) for key, item in value.items()}
        return {
            key: clean(item, named_children=key in {"$defs", "properties"})
            for key, item in value.items()
            if key in _OLLAMA_SCHEMA_KEYS
        }

    return clean(schema)


def slotted_relevance_format_schema(slots: list[str]) -> dict[str, Any]:
    """Require each application owned relevance batch slot exactly once."""
    base = RESPONSE_SCHEMAS["slotted_paper_relevance_assessments"].model_json_schema()
    definition = base["$defs"]["PaperRelevanceJudgement"]
    return {
        "$defs": {"PaperRelevanceJudgement": definition},
        "additionalProperties": False,
        "properties": {
            "assessments": {
                "additionalProperties": False,
                "properties": {slot: {"$ref": "#/$defs/PaperRelevanceJudgement"} for slot in slots},
                "required": slots,
                "type": "object",
            }
        },
        "required": ["assessments"],
        "type": "object",
    }


def identity_scoped_format_schema(schema_name: str, candidate_id: str) -> dict[str, Any]:
    """Constrain a legacy focused contract to one application owned identity."""
    schema = copy.deepcopy(RESPONSE_SCHEMAS[schema_name].model_json_schema())
    definition_name = {
        "candidate_critiques": "CandidateScore",
        "novelty_verifications": "NoveltyVerification",
    }[schema_name]
    candidate_schema = schema["$defs"][definition_name]["properties"]["candidate_id"]
    candidate_schema.clear()
    candidate_schema.update({"enum": [candidate_id], "type": "string"})
    return schema


__all__ = [
    "identity_scoped_format_schema",
    "ollama_format_schema",
    "slotted_relevance_format_schema",
]
