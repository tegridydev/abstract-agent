"""Atomic run context and bounded resume lifecycle."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from ..contracts.pipeline import PipelineStep
from .state import PipelineState


def prepare_pipeline_context(
    initial_context: dict[str, Any],
    configured_steps: list[PipelineStep],
    start_step: str | None,
) -> tuple[dict[str, Any], list[PipelineStep]]:
    """Create isolated execution state and apply bounded resume semantics."""
    context = dict(initial_context)
    steps = list(configured_steps)
    if start_step is None:
        return context, steps

    matching = [index for index, step in enumerate(steps) if step.name == start_step]
    if not matching:
        raise ValueError(f"resume step is not present in the current pipeline: {start_step}")
    start_index = matching[0]
    missing = sorted(set(steps[start_index].inputs) - set(context))
    if missing:
        raise ValueError(f"resume step '{start_step}' is missing inputs: {missing}")
    for step in steps[start_index:]:
        for output in step.outputs:
            context.pop(output, None)
    try:
        resume_count = int(context.get("_resume_count", 0)) + 1
    except (TypeError, ValueError) as error:
        raise ValueError("diagnostic resume count is invalid") from error
    if not 1 <= resume_count <= 3:
        raise ValueError("diagnostic resume limit of 3 has been reached")
    context["_resume_count"] = resume_count
    context["_resumed_from_step"] = start_step
    context["_resume_timestamp_iso"] = datetime.now().isoformat()
    return context, steps[start_index:]


__all__ = ["PipelineState", "prepare_pipeline_context"]
