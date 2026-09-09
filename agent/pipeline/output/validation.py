"""Response validation and application owned metadata normalization.

Scientific projection and transport execution remain separate from these policies.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from ollama import ResponseError
from pydantic import ValidationError

from ... import agent_helpers
from ...contracts.research import GapItem


def _is_truncated_json(error: Exception, content: str, done_reason: str | None) -> bool:
    """Identify incomplete structured output without treating other invalid JSON as truncation."""
    if done_reason == "length":
        return True
    if isinstance(error, json.JSONDecodeError) and error.pos >= len(content.rstrip()):
        return True
    message = str(error).casefold()
    return bool(content.strip()) and any(
        marker in message
        for marker in (
            "eof while parsing",
            "unexpected end of json",
            "unexpected eof",
            "unterminated string",
        )
    )


def _output_limit_reached(
    error: Exception,
    content: str,
    response_metrics: dict[str, Any],
    requested_tokens: int,
) -> bool:
    """Confirm truncation from Ollama completion metadata instead of parser errors alone."""
    if response_metrics.get("done_reason") == "length":
        return True
    try:
        eval_count = int(response_metrics.get("eval_count", -1))
    except (TypeError, ValueError):
        return False
    return eval_count >= requested_tokens and _is_truncated_json(error, content, None)


def _json_start_offset(content: str) -> int | None:
    positions = [index for index in (content.find("{"), content.find("[")) if index >= 0]
    return min(positions) if positions else None


def _discard_incomplete_gap_tail(value: Any) -> Any:
    """Drop only a final, incomplete GapItem produced by partial JSON parsing."""
    if not isinstance(value, dict):
        return value
    cleaned = dict(value)
    for key in ("established", "contradictions", "unresolved_questions"):
        items = cleaned.get(key)
        if not isinstance(items, list) or not items:
            continue
        try:
            GapItem.model_validate(items[-1])
        except ValidationError:
            cleaned[key] = items[:-1]
    return cleaned


def _is_format_initialization_error(error: ResponseError) -> bool:
    message = str(error.error).casefold()
    return error.status_code == 400 and any(
        marker in message for marker in ("grammar", "json schema", "structured output", "format")
    )


def _is_structural_output_error(error: Exception) -> bool:
    """Identify failures that show a requested output grammar was not enforced."""
    if isinstance(error, json.JSONDecodeError):
        return True
    if not isinstance(error, ValidationError):
        return False
    structural_types = {
        "dict_type",
        "extra_forbidden",
        "json_invalid",
        "list_type",
        "missing",
        "model_type",
    }
    return any(issue.get("type") in structural_types for issue in error.errors())


def _response_value(response: Any, key: str, default: Any = None) -> Any:
    if hasattr(response, key):
        return getattr(response, key)
    if isinstance(response, dict):
        return response.get(key, default)
    return default


def _referenced_paper_ids(value: Any) -> set[str]:
    """Collect only fields whose schema meaning is an evidence paper identifier."""
    found: set[str] = set()
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "paper_id" and isinstance(item, str):
                found.add(item)
            elif (key.endswith("paper_ids") or key == "references") and isinstance(item, list):
                found.update(identifier for identifier in item if isinstance(identifier, str))
            else:
                found.update(_referenced_paper_ids(item))
    elif isinstance(value, list):
        for item in value:
            found.update(_referenced_paper_ids(item))
    return found


def _paper_reference_validator(
    evidence_papers: list[dict[str, Any]], corrections: list[dict[str, str]]
) -> Callable[[Any], Any]:
    allowed = ", ".join(sorted(paper["paper_id"] for paper in evidence_papers))

    def validate(value: Any) -> Any:
        normalized, found_corrections, unknown = agent_helpers.normalize_paper_references(
            value, evidence_papers
        )
        if unknown:
            raise ValueError(
                "referenced unknown paper IDs: "
                + ", ".join(sorted(unknown))
                + f". Allowed paper IDs: {allowed}"
            )
        if isinstance(normalized, dict) and {
            "references",
            "claim_evidence",
        }.issubset(normalized):
            declared = set(normalized.get("references", []))
            claim_ids = {
                paper_id
                for link in normalized.get("claim_evidence", [])
                for paper_id in link.get("paper_ids", [])
            }
            missing = claim_ids - declared
            if missing:
                raise ValueError(
                    "claim linked paper IDs must be included in references: "
                    + ", ".join(sorted(missing))
                )
        corrections[:] = found_corrections
        return normalized

    return validate


def _candidate_critique_validator(
    expected_ids: set[str],
    *,
    exact: bool,
    normalize: Callable[[Any], Any] | None = None,
) -> Callable[[Any], Any]:
    """Validate critic identity coverage early enough for bounded model repair."""

    def validate(value: Any) -> Any:
        normalized = normalize(value) if normalize else value
        critiques = normalized.get("critiques", [])
        ids = [critique["candidate_id"] for critique in critiques]
        actual = set(ids)
        if len(ids) != len(actual):
            raise ValueError("candidate critiques contain duplicate candidate IDs")
        extra = actual - expected_ids
        missing = expected_ids - actual if exact else set()
        if extra or missing:
            raise ValueError(
                "candidate critique IDs do not match the requested candidates; "
                f"missing={sorted(missing)}, extra={sorted(extra)}"
            )
        return normalized

    return validate


def _shorten_bounded_text(value: str, maximum: int) -> str:
    """Shorten generated text without cutting mid word when a useful boundary exists."""
    if len(value) <= maximum:
        return value
    if maximum <= 3:
        return value[:maximum]
    candidate = value[: maximum - 3]
    boundary = max(candidate.rfind(" "), candidate.rfind("\n"), candidate.rfind("\t"))
    if boundary >= max(1, len(candidate) // 2):
        candidate = candidate[:boundary]
    return candidate.rstrip() + "..."


def _normalize_schema_bounds(
    value: Any, error: ValidationError
) -> tuple[Any, list[dict[str, Any]]]:
    """Repair only strings and lists that exceed declared Pydantic bounds."""
    normalizations: list[dict[str, Any]] = []
    for issue in error.errors(include_url=False):
        maximum = issue.get("ctx", {}).get("max_length")
        location = issue.get("loc", ())
        if not isinstance(maximum, int) or maximum < 0 or not location:
            continue
        parent = value
        try:
            for part in location[:-1]:
                parent = parent[part]
            key = location[-1]
            current = parent[key]
        except (KeyError, IndexError, TypeError):
            continue
        kind = None
        if issue.get("type") == "string_too_long" and isinstance(current, str):
            parent[key] = _shorten_bounded_text(current, maximum)
            kind = "string"
        elif issue.get("type") == "too_long" and isinstance(current, list):
            parent[key] = current[:maximum]
            kind = "list"
        if kind is None or len(current) <= maximum:
            continue
        path = ""
        for part in location:
            path += f"[{part}]" if isinstance(part, int) else ("." if path else "") + str(part)
        normalizations.append(
            {
                "path": path,
                "kind": kind,
                "original_length": len(current),
                "limit": maximum,
            }
        )
    return value, normalizations


def _normalize_app_owned_metadata(
    value: Any,
    response_schema: str | None,
    *,
    expected_candidate_id: str | None = None,
) -> tuple[Any, list[dict[str, Any]]]:
    """Repair metadata whose meaning is defined by this application, not by the model.

    This deliberately does not rewrite scientific content.  It only makes nullable/derived
    discriminator fields internally consistent and restores IDs supplied by the application.
    """
    if response_schema not in {"hypothesis_candidate", "research_proposal"} or not isinstance(
        value, dict
    ):
        return value, []
    normalized = dict(value)
    events: list[dict[str, Any]] = []
    if "generation_provenance" in normalized:
        events.append(
            {
                "path": "generation_provenance",
                "kind": "application_metadata",
                "action": "restored_application_owned_value",
            }
        )
    normalized["generation_provenance"] = {
        "mode": "structured_exact",
        "model": None,
        "validation_attempts": 1,
        "human_review_required": False,
        "limitations": [],
    }
    if response_schema == "research_proposal":
        return normalized, events
    applicability = normalized.get("applicability")
    if applicability in {"broad", "cross_domain"} and normalized.get("target_domain"):
        events.append(
            {
                "path": "target_domain",
                "kind": "application_metadata",
                "action": "cleared_for_non_domain_candidate",
            }
        )
        normalized["target_domain"] = None
    if expected_candidate_id and normalized.get("candidate_id") != expected_candidate_id:
        events.append(
            {
                "path": "candidate_id",
                "kind": "application_metadata",
                "action": "restored_expected_value",
            }
        )
        normalized["candidate_id"] = expected_candidate_id
    return normalized, events


def _strip_application_owned_wire_fields(
    value: Any, response_schema: str | None
) -> tuple[Any, list[dict[str, Any]]]:
    """Remove canonical fields a model emitted outside its compact wire contract."""
    if not isinstance(value, dict):
        return value, []
    owned = {
        "hypothesis_candidate": {
            "candidate_id",
            "supporting_paper_ids",
            "conflicting_paper_ids",
            "applicability",
            "target_domain",
            "grounding_links",
            "generation_provenance",
        },
        "research_proposal": {
            "proposal_status",
            "claim_evidence",
            "references",
            "generation_provenance",
        },
    }.get(response_schema, set())
    normalized = dict(value)
    events = []
    for key in sorted(owned & set(normalized)):
        normalized.pop(key)
        events.append(
            {
                "path": key,
                "kind": "application_metadata",
                "action": "discarded_model_owned_copy",
            }
        )
    return normalized, events
