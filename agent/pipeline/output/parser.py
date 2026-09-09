"""Conservative parsers for common completion model output shapes."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from typing import Any

from pydantic_core import from_json

from .models import CompilationIssue

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)
_TAG = re.compile(
    r"<(?P<name>[A-Za-z_][A-Za-z0-9_.-]*)>(?P<value>.*?)</(?P=name)>",
    re.DOTALL,
)
_SCHEMA_KEYS = {"$defs", "$schema", "properties", "required", "title", "type"}
_MAX_PARTIAL_PREFIX = 512


class DuplicateKeyError(ValueError):
    """JSON object contained a duplicate key and is therefore ambiguous."""


def _unique_object(pairs: Iterable[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DuplicateKeyError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _loads(value: str) -> Any:
    return json.loads(value, object_pairs_hook=_unique_object)


def is_schema_echo(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    score = len(_SCHEMA_KEYS & set(value))
    return "$defs" in value or "properties" in value or score >= 3


def _balanced_json(text: str) -> str | None:
    """Return the first balanced JSON object/array while respecting quoted strings."""
    positions = [(text.find("{"), "{"), (text.find("["), "[")]
    starts = [(index, char) for index, char in positions if index >= 0]
    if not starts:
        return None
    start, opener = min(starts)
    closer = "}" if opener == "{" else "]"
    stack = [closer]
    quoted = False
    escaped = False
    for index in range(start + 1, len(text)):
        char = text[index]
        if quoted:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
            continue
        if char == '"':
            quoted = True
        elif char in "{[":
            stack.append("}" if char == "{" else "]")
        elif char in "}]":
            if not stack or char != stack[-1]:
                return None
            stack.pop()
            if not stack:
                return text[start : index + 1]
    return None


def _partial_root(text: str) -> tuple[str, int]:
    positions = [index for index, char in enumerate(text) if char in "{["]
    if not positions:
        raise ValueError("truncated output contains no JSON object or array")
    for offset in positions:
        if offset > _MAX_PARTIAL_PREFIX:
            break
        lookahead = offset + 1
        while lookahead < len(text) and text[lookahead].isspace():
            lookahead += 1
        following = text[lookahead : lookahead + 1]
        if text[offset] == "{" and following not in {'"', "}", ""}:
            continue
        if (
            text[offset] == "["
            and following not in {'"', "{", "[", "]", "-", "t", "f", "n", ""}
            and not following.isdigit()
        ):
            continue
        return text[offset:], offset
    if min(positions) > _MAX_PARTIAL_PREFIX or any(
        offset > _MAX_PARTIAL_PREFIX for offset in positions
    ):
        raise ValueError("JSON root appears too far into the truncated response")
    raise ValueError("truncated output contains no plausible JSON object or array")


def _reject_partial_duplicate_keys(text: str) -> None:
    """Reject keys repeated in any object completed far enough to expose both names."""
    containers: list[set[str] | None] = []
    index = 0
    while index < len(text):
        char = text[index]
        if char == '"':
            end = index + 1
            escaped = False
            while end < len(text):
                current = text[end]
                if escaped:
                    escaped = False
                elif current == "\\":
                    escaped = True
                elif current == '"':
                    break
                end += 1
            if end >= len(text):
                return
            lookahead = end + 1
            while lookahead < len(text) and text[lookahead].isspace():
                lookahead += 1
            if containers and containers[-1] is not None and text[lookahead : lookahead + 1] == ":":
                key = json.loads(text[index : end + 1])
                keys = containers[-1]
                if key in keys:
                    raise DuplicateKeyError(f"duplicate JSON key: {key}")
                keys.add(key)
            index = end + 1
            continue
        if char == "{":
            containers.append(set())
        elif char == "[":
            containers.append(None)
        elif char in "}]" and containers:
            containers.pop()
        index += 1


def _tagged_value(text: str) -> Any:
    tags = list(_TAG.finditer(text))
    if not tags:
        raise ValueError("output contains neither JSON nor tagged fields")
    value: dict[str, Any] = {}
    for match in tags:
        name = match.group("name")
        if name in value:
            raise DuplicateKeyError(f"duplicate tagged field: {name}")
        raw = match.group("value").strip()
        if raw.casefold() in {"null", "none", "n/a"}:
            parsed: Any = None
        else:
            try:
                parsed = _loads(raw)
            except (json.JSONDecodeError, DuplicateKeyError):
                parsed = raw
        value[name] = parsed
    return value


def parse_output(
    content: str,
    *,
    allow_partial: bool = False,
    truncated: bool = False,
) -> tuple[Any, str, list[CompilationIssue]]:
    """Parse output in strict to permissive order without guessing missing content."""
    text = str(content or "").strip()
    if not text:
        raise ValueError("model output is empty")
    errors: list[CompilationIssue] = []
    original_error: json.JSONDecodeError | None = None
    try:
        value = _loads(text)
        parser = "exact_json"
    except DuplicateKeyError:
        raise
    except json.JSONDecodeError as error:
        original_error = error
        errors.append(CompilationIssue("invalid_json", str(error)[:300]))
        value = None
        parser = ""
        for match in _FENCE.finditer(text):
            try:
                value = _loads(match.group(1).strip())
                parser = "fenced_json"
                break
            except DuplicateKeyError:
                raise
            except json.JSONDecodeError:
                continue
        if value is None and not text.startswith("<"):
            candidate = _balanced_json(text)
            if candidate:
                try:
                    value = _loads(candidate)
                    parser = "balanced_json"
                except DuplicateKeyError:
                    raise
                except json.JSONDecodeError:
                    pass
        if value is None and allow_partial and truncated:
            partial, prefix_length = _partial_root(text)
            _reject_partial_duplicate_keys(partial)
            value = from_json(partial, allow_partial=True)
            parser = "partial_json" if prefix_length == 0 else "prefixed_partial_json"
            if prefix_length:
                errors.append(
                    CompilationIssue(
                        "discarded_prefix",
                        f"ignored {prefix_length} characters before the partial JSON root",
                    )
                )
            errors.append(
                CompilationIssue(
                    "truncated_output",
                    "used only complete values from a transport confirmed truncated response",
                )
            )
        if value is None:
            try:
                value = _tagged_value(text)
                parser = "tagged_text"
            except ValueError:
                if original_error is not None:
                    raise original_error from None
                raise
    if is_schema_echo(value):
        raise ValueError("model returned a JSON Schema definition instead of data")
    return value, parser, errors


__all__ = ["DuplicateKeyError", "is_schema_echo", "parse_output"]
