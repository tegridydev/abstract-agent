"""Stage specific generation contracts and deterministic response normalization.

Generation contracts intentionally describe only values the model owns.  Strict domain
contracts remain in :mod:`agent.schemas` and are applied after these policies run.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ResponseNormalization:
    """One deterministic, provenance safe change to generated data."""

    path: str
    operation: str
    reason: str

    def as_dict(self) -> dict[str, str]:
        return {
            "path": self.path,
            "operation": self.operation,
            "reason": self.reason,
        }


def _shorten_display_text(value: str, maximum: int) -> str:
    """Shorten display metadata at a useful boundary without changing research prose."""
    normalized = " ".join(value.split())
    if len(normalized) <= maximum:
        return normalized
    candidate = normalized[: maximum - 3]
    boundary = candidate.rfind(" ")
    if boundary >= max(1, len(candidate) // 2):
        candidate = candidate[:boundary]
    return candidate.rstrip() + "..."


def scope_generation_schema() -> dict[str, Any]:
    """Return a compact scope schema without application owned identifiers or bounds."""
    option = {
        "additionalProperties": False,
        "properties": {
            "label": {"type": "string"},
            "description": {"type": "string"},
            "paper_ids": {"items": {"type": "string"}, "type": "array"},
        },
        "required": ["label", "description", "paper_ids"],
        "type": "object",
    }
    return {
        "additionalProperties": False,
        "properties": {"options": {"items": option, "type": "array"}},
        "required": ["options"],
        "type": "object",
    }


def gap_generation_schema() -> dict[str, Any]:
    """Return a gap map schema whose containing collection owns the item status."""
    item = {
        "additionalProperties": False,
        "properties": {
            "statement": {"type": "string"},
            "paper_ids": {"items": {"type": "string"}, "type": "array"},
        },
        "required": ["statement", "paper_ids"],
        "type": "object",
    }
    return {
        "additionalProperties": False,
        "properties": {
            key: {"items": item, "type": "array"}
            for key in ("established", "contradictions", "unresolved_questions")
        },
        "required": ["established", "contradictions", "unresolved_questions"],
        "type": "object",
    }


def hypothesis_generation_schema(*, include_addressed_findings: bool = False) -> dict[str, Any]:
    """Describe the v5 scientific IR; Python renders canonical prose and evidence IDs."""

    def text(*, minimum: int = 1, maximum: int) -> dict[str, Any]:
        return {"type": "string", "minLength": minimum, "maxLength": maximum}

    def strings(*, minimum: int = 0, maximum: int, item_maximum: int = 240) -> dict[str, Any]:
        return {
            "type": "array",
            "items": text(maximum=item_maximum),
            "minItems": minimum,
            "maxItems": maximum,
        }

    relation = {
        "enum": [
            "increases",
            "decreases",
            "differs",
            "noninferior",
            "positive_association",
            "negative_association",
        ],
        "type": "string",
    }
    operation = {
        "additionalProperties": False,
        "type": "object",
        "properties": {
            "name": text(maximum=160),
            "operational_definition": {
                **text(minimum=8, maximum=360),
                "description": "A reproducible definition of exactly how this condition is run.",
            },
        },
        "required": ["name", "operational_definition"],
    }
    outcome = {
        "additionalProperties": False,
        "type": "object",
        "properties": {
            "name": text(maximum=160),
            "metric": text(maximum=160),
            "expected_relation": relation,
            "decision_rule": {
                **text(minimum=8, maximum=300),
                "description": "A measurable threshold that can pass or fail.",
            },
        },
        "required": ["name", "metric", "expected_relation", "decision_rule"],
    }
    properties: dict[str, Any] = {
        "title": text(maximum=160),
        "system": text(maximum=240),
        "exposure_or_intervention": operation,
        "comparator": operation,
        "primary_outcome": outcome,
        "secondary_outcomes": {
            "type": "array",
            "items": outcome,
            "maxItems": 2,
        },
        "mechanism": {
            "additionalProperties": False,
            "type": "object",
            "properties": {
                "cause": {
                    **text(maximum=240),
                    "description": "The proposed causal change, not a topic label.",
                },
                "mediator": {
                    **text(maximum=240),
                    "description": "The intermediate process connecting cause and outcome.",
                },
                "measurable_consequence": {
                    **text(maximum=240),
                    "description": "The observable consequence of the proposed mediator.",
                },
            },
            "required": ["cause", "mediator", "measurable_consequence"],
        },
        "novelty_baseline": text(maximum=240),
        "novelty_difference": text(maximum=300),
        "experiment": {
            "additionalProperties": False,
            "type": "object",
            "properties": {
                "units": text(maximum=240),
                "procedure": text(minimum=8, maximum=600),
                "controls": strings(minimum=1, maximum=4),
            },
            "required": ["units", "procedure", "controls"],
        },
    }
    properties["study_design"] = {
        "enum": ["experimental", "observational", "measurement"],
        "type": "string",
    }
    properties["evidence_slots"] = {
        **strings(minimum=1, maximum=2, item_maximum=24),
        "description": "One or two E slots present in the supplied evidence packet.",
    }
    properties["confounders"] = strings(minimum=1, maximum=4)
    properties["risks"] = strings(minimum=1, maximum=4)
    properties["potential_insights"] = strings(minimum=1, maximum=3)
    if include_addressed_findings:
        properties["addressed_finding_ids"] = {
            **strings(maximum=8, item_maximum=120),
            "description": "Exact blocking finding IDs actually resolved by this revision.",
        }
    return {
        "additionalProperties": False,
        "type": "object",
        "properties": properties,
        "required": [key for key in properties if key != "addressed_finding_ids"],
    }


def candidate_kernel_schema() -> dict[str, Any]:
    """Return only the causal comparison fields owned by the model."""
    complete = hypothesis_generation_schema()
    keys = (
        "title",
        "study_design",
        "system",
        "exposure_or_intervention",
        "comparator",
        "primary_outcome",
        "evidence_slots",
    )
    properties = {key: complete["properties"][key] for key in keys}
    return {
        "additionalProperties": False,
        "type": "object",
        "properties": properties,
        "required": list(keys),
    }


def candidate_protocol_schema() -> dict[str, Any]:
    """Return only the mechanism and study protocol fields owned by the model."""
    complete = hypothesis_generation_schema()
    keys = (
        "secondary_outcomes",
        "mechanism",
        "novelty_baseline",
        "novelty_difference",
        "experiment",
        "confounders",
        "risks",
        "potential_insights",
    )
    properties = {key: complete["properties"][key] for key in keys}
    return {
        "additionalProperties": False,
        "type": "object",
        "properties": properties,
        "required": list(keys),
    }


def candidate_critique_draft_schema() -> dict[str, Any]:
    """Compact single candidate critic output without model owned identity or novelty claims."""
    finding = {
        "additionalProperties": False,
        "type": "object",
        "properties": {
            "category": {
                "enum": [
                    "evidence",
                    "internal_consistency",
                    "intervention",
                    "outcome",
                    "falsifiability",
                    "mechanism",
                    "confounding",
                    "feasibility",
                    "clarity",
                ],
                "type": "string",
            },
            "severity": {"enum": ["minor", "major", "fatal"], "type": "string"},
            "issue": {"type": "string", "maxLength": 360},
            "required_revision": {"type": "string", "maxLength": 360},
        },
        "required": ["category", "severity", "issue", "required_revision"],
    }
    return {
        "additionalProperties": False,
        "type": "object",
        "properties": {
            key: {"type": "integer", "minimum": 1, "maximum": 5}
            for key in ("grounding", "falsifiability", "feasibility", "clarity")
        }
        | {
            "verdict": {"enum": ["pass", "revise", "exclude"], "type": "string"},
            "findings": {"type": "array", "items": finding, "maxItems": 4},
        },
        "required": [
            "grounding",
            "falsifiability",
            "feasibility",
            "clarity",
            "verdict",
            "findings",
        ],
    }


def novelty_decision_draft_schema() -> dict[str, Any]:
    """Compact single candidate novelty output with application owned identity."""
    reference_list = {
        "type": "array",
        "items": {"type": "string"},
        "maxItems": 6,
    }
    return {
        "additionalProperties": False,
        "type": "object",
        "properties": {
            "status": {
                "enum": ["novelty_supported", "prior_art_overlap", "uncertain"],
                "type": "string",
            },
            "closest_prior_art_paper_ids": reference_list,
            "materially_matching_paper_ids": reference_list,
            "reasoning": {"type": "string", "maxLength": 800},
            "limitations": {
                "type": "array",
                "items": {"type": "string"},
                "maxItems": 4,
            },
        },
        "required": [
            "status",
            "closest_prior_art_paper_ids",
            "materially_matching_paper_ids",
            "reasoning",
            "limitations",
        ],
    }


def critique_generation_schema(candidate_ids: list[str] | None = None) -> dict[str, Any]:
    """Compact non redundant critic contract; application derives legacy fields and flags."""
    finding = {
        "additionalProperties": False,
        "type": "object",
        "properties": {
            "category": {
                "enum": [
                    "evidence",
                    "novelty",
                    "internal_consistency",
                    "intervention",
                    "outcome",
                    "falsifiability",
                    "mechanism",
                    "confounding",
                    "feasibility",
                    "clarity",
                ],
                "type": "string",
            },
            "severity": {"enum": ["minor", "major", "fatal"], "type": "string"},
            "issue": {"type": "string"},
            "required_revision": {"type": "string"},
        },
        "required": ["category", "severity", "issue", "required_revision"],
    }
    candidate_id: dict[str, Any] = {"type": "string"}
    if candidate_ids:
        candidate_id["enum"] = candidate_ids
    critique = {
        "additionalProperties": False,
        "type": "object",
        "properties": {
            "candidate_id": candidate_id,
            "grounding": {"type": "integer"},
            "novelty": {"type": "integer"},
            "falsifiability": {"type": "integer"},
            "feasibility": {"type": "integer"},
            "clarity": {"type": "integer"},
            "verdict": {"enum": ["pass", "revise", "exclude"], "type": "string"},
            "findings": {"type": "array", "items": finding},
        },
        "required": [
            "candidate_id",
            "grounding",
            "novelty",
            "falsifiability",
            "feasibility",
            "clarity",
            "verdict",
            "findings",
        ],
    }
    return {
        "additionalProperties": False,
        "type": "object",
        "properties": {"critiques": {"type": "array", "items": critique}},
        "required": ["critiques"],
    }


def proposal_generation_schema() -> dict[str, Any]:
    """Describe commentary only; audited candidate fields remain owned by the application."""
    strings = {"rationale": {"type": "string"}}
    arrays = {
        key: {"type": "array", "items": {"type": "string"}} for key in ("limitations_and_risks",)
    }
    properties = strings | arrays
    return {
        "additionalProperties": False,
        "type": "object",
        "properties": properties,
        "required": list(properties),
    }


def normalize_scope_response(
    value: Any, allowed_paper_ids: set[str]
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    """Keep useful evidence clusters and assign stable application owned scope IDs."""
    raw_options = value.get("options", []) if isinstance(value, dict) else []
    if not isinstance(raw_options, list):
        raw_options = []
    events: list[ResponseNormalization] = []
    options: list[dict[str, Any]] = []
    labels: set[str] = set()
    for index, raw in enumerate(raw_options):
        path = f"options[{index}]"
        if not isinstance(raw, dict):
            events.append(ResponseNormalization(path, "drop", "scope option is not an object"))
            continue
        label = " ".join(str(raw.get("label", "")).split())
        description = " ".join(str(raw.get("description", "")).split())
        supplied_ids = raw.get("paper_ids", [])
        if not isinstance(supplied_ids, list):
            supplied_ids = []
        paper_ids = [
            paper_id
            for paper_id in dict.fromkeys(str(item) for item in supplied_ids)
            if paper_id in allowed_paper_ids
        ]
        label_key = label.casefold()
        if len(label) < 2 or len(description) < 3:
            events.append(ResponseNormalization(path, "drop", "scope display text is incomplete"))
            continue
        if len(paper_ids) < 2:
            events.append(
                ResponseNormalization(path, "drop", "scope has fewer than two known core papers")
            )
            continue
        if label_key in labels:
            events.append(ResponseNormalization(path, "drop", "scope label is duplicated"))
            continue
        paper_set = set(paper_ids)
        if any(
            len(paper_set & set(existing["paper_ids"]))
            / len(paper_set | set(existing["paper_ids"]))
            >= 0.80
            for existing in options
        ):
            events.append(
                ResponseNormalization(
                    path,
                    "drop",
                    "scope evidence overlaps an existing scope by at least 80%",
                )
            )
            continue
        short_label = _shorten_display_text(label, 80)
        short_description = _shorten_display_text(description, 240)
        if short_label != label:
            events.append(ResponseNormalization(f"{path}.label", "shorten", "display limit"))
        if short_description != description:
            events.append(ResponseNormalization(f"{path}.description", "shorten", "display limit"))
        if paper_ids != supplied_ids:
            events.append(
                ResponseNormalization(
                    f"{path}.paper_ids", "filter", "unknown or duplicate paper IDs"
                )
            )
        labels.add(label_key)
        options.append(
            {
                "scope_id": f"scope_{len(options) + 1}",
                "label": short_label,
                "description": short_description,
                "paper_ids": paper_ids[:12],
            }
        )
        if len(options) == 4:
            if index + 1 < len(raw_options):
                events.append(
                    ResponseNormalization("options", "limit", "at most four scopes are displayed")
                )
            break
    return {"options": options}, [event.as_dict() for event in events]


def normalize_gap_response(
    value: Any,
    allowed_paper_ids: set[str],
    *,
    topic: str,
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    """Derive redundant statuses, isolate invalid items, and preserve valid gap content."""
    source = value if isinstance(value, dict) else {}
    limits = {"established": 5, "contradictions": 3, "unresolved_questions": 5}
    statuses = {
        "established": "established",
        "contradictions": "contradictory",
        "unresolved_questions": "unknown",
    }
    result: dict[str, list[dict[str, Any]]] = {key: [] for key in limits}
    events: list[ResponseNormalization] = []
    for key, maximum in limits.items():
        raw_items = source.get(key, [])
        if not isinstance(raw_items, list):
            events.append(ResponseNormalization(key, "replace", "gap collection is not a list"))
            continue
        for index, raw in enumerate(raw_items):
            path = f"{key}[{index}]"
            if not isinstance(raw, dict):
                events.append(ResponseNormalization(path, "drop", "gap item is not an object"))
                continue
            statement = " ".join(str(raw.get("statement", "")).split())
            if not statement or len(statement) > 240:
                events.append(
                    ResponseNormalization(path, "drop", "gap statement is empty or overlong")
                )
                continue
            supplied_ids = raw.get("paper_ids", [])
            if not isinstance(supplied_ids, list):
                supplied_ids = []
            paper_ids = [
                paper_id
                for paper_id in dict.fromkeys(str(item) for item in supplied_ids)
                if paper_id in allowed_paper_ids
            ][:4]
            if key != "unresolved_questions" and not paper_ids:
                events.append(
                    ResponseNormalization(path, "drop", "evidence backed gap has no known citation")
                )
                continue
            if raw.get("status") != statuses[key]:
                events.append(
                    ResponseNormalization(
                        f"{path}.status", "derive", f"status is owned by the '{key}' collection"
                    )
                )
            if paper_ids != supplied_ids:
                events.append(
                    ResponseNormalization(
                        f"{path}.paper_ids", "filter", "unknown, duplicate, or excess paper IDs"
                    )
                )
            result[key].append(
                {"statement": statement, "paper_ids": paper_ids, "status": statuses[key]}
            )
            if len(result[key]) == maximum:
                if index + 1 < len(raw_items):
                    events.append(ResponseNormalization(key, "limit", "collection display limit"))
                break
    if not any(result.values()):
        topic_text = _shorten_display_text(" ".join(topic.split()), 180) or "the research topic"
        result["unresolved_questions"].append(
            {
                "statement": (
                    f"What remains unresolved about {topic_text} in the retrieved evidence?"
                ),
                "paper_ids": [],
                "status": "unknown",
            }
        )
        events.append(
            ResponseNormalization(
                "unresolved_questions[0]",
                "fallback",
                "no valid generated gap item remained",
            )
        )
    return result, [event.as_dict() for event in events]


__all__ = [
    "ResponseNormalization",
    "candidate_critique_draft_schema",
    "candidate_kernel_schema",
    "candidate_protocol_schema",
    "critique_generation_schema",
    "gap_generation_schema",
    "hypothesis_generation_schema",
    "normalize_gap_response",
    "normalize_scope_response",
    "novelty_decision_draft_schema",
    "proposal_generation_schema",
    "scope_generation_schema",
]
