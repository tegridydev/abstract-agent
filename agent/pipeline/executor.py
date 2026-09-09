"""Schema agnostic primitives for executing configured pipeline steps."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ..contracts.pipeline import ToolStep


def map_step_result(result: Any, outputs: list[str], handler: str) -> dict[str, Any]:
    if not outputs:
        return {}
    if isinstance(result, tuple):
        if len(result) != len(outputs):
            raise ValueError(
                f"tool '{handler}' returned {len(result)} values for {len(outputs)} outputs"
            )
        return dict(zip(outputs, result, strict=True))
    if len(outputs) == 1:
        return {outputs[0]: result}
    if isinstance(result, dict) and all(key in result for key in outputs):
        return {key: result[key] for key in outputs}
    raise ValueError(f"tool '{handler}' did not return the declared outputs {outputs}")


def execute_tool_step(
    step: ToolStep,
    context: dict[str, Any],
    *,
    resolver: Callable[[str], Callable[..., Any]],
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Invoke one trusted tool with explicit context to argument bindings."""
    function = resolver(step.function)
    arguments = {
        argument: context[context_key] for argument, context_key in step.argument_sources().items()
    }
    arguments.update(step.params)
    arguments.update(overrides or {})
    return map_step_result(function(**arguments), step.outputs, step.function)


__all__ = ["execute_tool_step", "map_step_result"]
