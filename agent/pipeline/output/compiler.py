"""Schema agnostic model output compilation with bounded deterministic recovery."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, ValidationError

from ..errors import PipelineInvariantError
from .models import (
    CompilationIssue,
    CompilationTrace,
    CompiledOutput,
    ModelOutputArtifact,
)
from .parser import parse_output

Normalizer = Callable[[Any], tuple[Any, list[dict[str, Any]]]]
Validator = Callable[[Any], Any]
Projector = Callable[[Any | None, Exception], Any]


def validate_compact_instance(value: Any, schema: dict[str, Any], path: str = "$") -> Any:
    """Validate the small JSON Schema subset used by model owned draft contracts."""
    expected_type = schema.get("type")
    type_checks = {
        "object": lambda item: isinstance(item, dict),
        "array": lambda item: isinstance(item, list),
        "string": lambda item: isinstance(item, str),
        "integer": lambda item: isinstance(item, int) and not isinstance(item, bool),
        "number": lambda item: isinstance(item, int | float) and not isinstance(item, bool),
        "boolean": lambda item: isinstance(item, bool),
        "null": lambda item: item is None,
    }
    if expected_type in type_checks and not type_checks[expected_type](value):
        raise ValueError(f"{path} must be {expected_type}")
    if "enum" in schema and value not in schema["enum"]:
        raise ValueError(f"{path} is not an allowed value")
    if expected_type == "object":
        properties = schema.get("properties", {})
        missing = [key for key in schema.get("required", []) if key not in value]
        if missing:
            raise ValueError(f"{path} is missing required fields: {missing}")
        if schema.get("additionalProperties") is False:
            extra = sorted(set(value) - set(properties))
            if extra:
                raise ValueError(f"{path} contains application owned or unknown fields: {extra}")
        for key, child in properties.items():
            if key in value:
                validate_compact_instance(value[key], child, f"{path}.{key}")
    elif expected_type == "array":
        child = schema.get("items", {})
        for index, item in enumerate(value):
            validate_compact_instance(item, child, f"{path}[{index}]")
    return value


def _issue_from_event(event: dict[str, Any]) -> CompilationIssue:
    return CompilationIssue(
        code=str(event.get("operation") or event.get("kind") or "normalized"),
        message=str(event.get("reason") or event.get("action") or "normalized output"),
        path=str(event["path"]) if event.get("path") is not None else None,
    )


def compile_model_output(
    artifact: ModelOutputArtifact,
    *,
    schema: type[BaseModel] | None = None,
    normalizer: Normalizer | None = None,
    validator: Validator | None = None,
    projector: Projector | None = None,
    fallback: Projector | None = None,
    allow_partial: bool = False,
    policy_id: str | None = None,
    prompt_id: str | None = None,
    prompt_version: str | None = None,
    prompt_hash: str | None = None,
    contract_version: str | None = None,
) -> CompiledOutput:
    """Compile model content to an application contract or a conservative fallback.

    Parsing and representation repair are generic. Semantic projection and fallback remain
    stage owned callables, so this module never invents domain facts.
    """
    issues: list[CompilationIssue] = []
    parser = "none"
    value: Any | None = None
    failure: Exception | None = None
    normalized = False
    try:
        output_limit_reached = artifact.output_limit_reached or artifact.done_reason == "length"
        value, parser, parser_issues = parse_output(
            artifact.content,
            allow_partial=allow_partial,
            truncated=output_limit_reached,
        )
        issues.extend(parser_issues)
        if normalizer:
            value, events = normalizer(value)
            normalized = bool(events)
            issues.extend(_issue_from_event(event) for event in events)
        if schema:
            value = schema.model_validate(value).model_dump()
        if validator:
            value = validator(value)
        outcome = "normalized" if normalized or parser != "exact_json" else "exact"
        return CompiledOutput(
            value,
            CompilationTrace(
                route=artifact.route,
                outcome=outcome,
                parser=parser,
                attempts=artifact.request_count,
                policy_id=policy_id,
                prompt_id=prompt_id,
                prompt_version=prompt_version,
                prompt_hash=prompt_hash,
                contract_version=contract_version,
                recovery_source=("local_normalization" if outcome == "normalized" else "none"),
                issues=issues,
            ),
        )
    except (ValidationError, ValueError, TypeError, KeyError) as error:
        failure = error
        issues.append(CompilationIssue("contract_invalid", str(error)[:500]))

    for mode, recovery in (("projected", projector), ("fallback", fallback)):
        if recovery is None:
            continue
        try:
            recovered = recovery(value, failure or ValueError("invalid output"))
            if schema:
                recovered = schema.model_validate(recovered).model_dump()
            if validator:
                recovered = validator(recovered)
            return CompiledOutput(
                recovered,
                CompilationTrace(
                    route=("deterministic_fallback" if mode == "fallback" else artifact.route),
                    outcome=mode,
                    parser=parser,
                    attempts=artifact.request_count,
                    policy_id=policy_id,
                    prompt_id=prompt_id,
                    prompt_version=prompt_version,
                    prompt_hash=prompt_hash,
                    contract_version=contract_version,
                    human_review_required=True,
                    recovery_source=(
                        "deterministic_fallback" if mode == "fallback" else "local_normalization"
                    ),
                    issues=issues,
                ),
            )
        except (ValidationError, ValueError, TypeError, KeyError) as error:
            issues.append(CompilationIssue(f"{mode}_failed", str(error)[:500]))

    details = "; ".join(f"{issue.code}: {issue.message}" for issue in issues[-3:])
    raise PipelineInvariantError(
        "model output could not be compiled and no valid deterministic fallback remained"
        + (f" ({details})" if details else "")
    ) from failure


__all__ = ["compile_model_output", "validate_compact_instance"]
