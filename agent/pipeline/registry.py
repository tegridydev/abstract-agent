"""Explicit allow list for built in pipeline tools."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any

from .. import agent_helpers, knowledge, multi_source
from ..contracts.pipeline import PipelineConfig, ToolStep
from . import recovery, stages

_MODULES: dict[str, Any] = {
    "agent_helpers": agent_helpers,
    "knowledge": knowledge,
    "multi_source": multi_source,
    "pipeline.recovery": recovery,
    "pipeline.stages": stages,
}


def resolve_tool(function_path: str) -> Callable[..., Any]:
    """Resolve only explicitly trusted in package tool namespaces."""
    module_name, separator, function_name = function_path.rpartition(".")
    module = _MODULES.get(module_name) if separator else None
    function = getattr(module, function_name, None) if module else None
    if not callable(function) or function_name.startswith("_"):
        raise ValueError(f"unknown built in pipeline tool '{function_path}'")
    return function


def validate_tool_contracts(
    config: PipelineConfig, *, resolver: Callable[[str], Callable[..., Any]] = resolve_tool
) -> None:
    """Fail fast when configured tool arguments drift from their callables."""
    for step in config.steps:
        if not isinstance(step, ToolStep):
            continue
        signature = inspect.signature(resolver(step.function))
        arguments = set(step.argument_sources()) | set(step.params)
        accepted = {
            name
            for name, parameter in signature.parameters.items()
            if parameter.kind
            in {inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY}
        }
        required = {
            name
            for name, parameter in signature.parameters.items()
            if parameter.default is inspect.Parameter.empty
            and parameter.kind
            in {inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY}
        }
        accepts_extras = any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        )
        missing = required - arguments
        unexpected = set() if accepts_extras else arguments - accepted
        if missing or unexpected:
            details = []
            if missing:
                details.append(f"missing required arguments {sorted(missing)}")
            if unexpected:
                details.append(f"unexpected arguments {sorted(unexpected)}")
            raise ValueError(f"step '{step.name}' has " + " and ".join(details))


__all__ = ["resolve_tool", "validate_tool_contracts"]
