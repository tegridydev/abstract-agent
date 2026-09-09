"""Versioned results, diagnostics, and trusted resume checkpoint validation."""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import TypeAdapter, ValidationError

from .. import agent_helpers
from ..application.io import atomic_write
from ..checkpoints import context_checksum, normalize_context, read_checkpoint_object
from ..contracts.pipeline import LlmStep, PipelineConfig
from ..contracts.research import RESPONSE_SCHEMAS
from ..contracts.sources import SourceStatus
from ..literature_types import PaperRecord
from ..ollama_models import parse_ollama_host
from .migrations import migrate_legacy_context


def slugify(text: str) -> str:
    value = "_".join(
        "".join(character for character in part if character.isalnum())
        for part in text.casefold().split()
    )
    return value[:60].strip("_") or "untitled"


def render_text_result(context: dict[str, Any], config: PipelineConfig, topic: str) -> str:
    lines = [
        f"Pipeline: {config.pipeline_name} v{config.version}",
        f"Topic: {topic}",
        f"Timestamp: {datetime.now().isoformat()}",
        "=" * 50,
        "",
    ]
    for key in config.final_outputs:
        lines.append(f"--- {key.replace('_', ' ').title()} ---")
        value = context.get(key)
        if key == "top_papers" and isinstance(value, list):
            lines.append(agent_helpers.format_citations_rich(value))
        elif value is None:
            lines.append("[Not generated]")
        elif isinstance(value, (dict, list)):
            lines.append(json.dumps(value, indent=2, ensure_ascii=False, default=str))
        else:
            lines.append(str(value))
        lines.append("")
    return "\n".join(lines)


def save_results(
    context: dict[str, Any],
    config: PipelineConfig,
    topic: str,
    output_dir: str | Path,
) -> tuple[Path, Path]:
    directory = Path(output_dir)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = f"result_{slugify(topic)}_{stamp}_{uuid.uuid4().hex[:8]}"
    text_path = directory / f"{base}.txt"
    json_path = directory / f"{base}.json"
    outcome = str(context.get("_outcome", "proposal_complete"))
    payload = {
        "schema_version": "6.0",
        "status": (
            "complete"
            if outcome
            in {
                "proposal_complete",
                "evidence_only",
                "insufficient_evidence",
                "insufficient_valid_candidates",
            }
            else outcome
        ),
        "outcome": outcome,
        "resumable": False,
        "pipeline": {"name": config.pipeline_name, "version": str(config.version)},
        "run_info": {
            "topic": topic,
            "timestamp_iso": datetime.now().isoformat(),
            "ollama_host_used": parse_ollama_host(str(context.get("_pipeline_ollama_host")))[0]
            if context.get("_pipeline_ollama_host")
            else None,
            "insufficient_literature_override": context.get(
                "_insufficient_literature_override", False
            ),
            "search_depth": context.get("_search_depth", "standard"),
            "research_scope": context.get("research_scope"),
            "model_assignments": context.get("_pipeline_model_assignments", {}),
            "resolved_models_by_step": context.get("_resolved_models_by_step", {}),
            "resolved_model_metadata": context.get("_resolved_model_metadata", {}),
            "resumed_from_step": context.get("_resumed_from_step"),
            "resume_count": context.get("_resume_count", 0),
            "outcome": outcome,
            "knowledge": context.get("_knowledge", {}),
            "model_capabilities": context.get("_model_capabilities", {}),
        },
        "source_statuses": context.get("source_statuses", []),
        "novelty_source_statuses": context.get("novelty_source_statuses", []),
        "llm_metrics": context.get("_llm_metrics", {}),
        "generation_events": context.get("_generation_events", []),
        "compilation_events": context.get("_compilation_events", []),
        "warnings": context.get("_warnings", []),
        "final_context": context,
    }
    atomic_write(text_path, render_text_result(context, config, topic))
    atomic_write(json_path, json.dumps(payload, ensure_ascii=False, indent=2, default=str))
    return text_path, json_path


def is_resumable_context(context: dict[str, Any], config: PipelineConfig) -> bool:
    active_step = context.get("_active_step")
    step = next((item for item in config.steps if item.name == active_step), None)
    try:
        resume_count = int(context.get("_resume_count", 0))
    except (TypeError, ValueError):
        return False
    return bool(step and set(step.inputs).issubset(context) and 0 <= resume_count < 3)


def save_failure_diagnostic(
    context: dict[str, Any],
    config: PipelineConfig,
    topic: str,
    output_dir: str | Path,
    error: str,
    *,
    status: str = "insufficient_evidence",
) -> Path:
    directory = Path(output_dir)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = directory / f"failed_{slugify(topic)}_{stamp}_{uuid.uuid4().hex[:8]}.json"
    diagnostic_context = normalize_context(context)
    active_step = diagnostic_context.get("_active_step")
    resumable = is_resumable_context(diagnostic_context, config)
    payload = {
        "schema_version": "6.0",
        "status": status,
        "error": error,
        "resumable": resumable,
        "resume_step": active_step if resumable else None,
        "context_checksum": context_checksum(diagnostic_context),
        "pipeline": {"name": config.pipeline_name, "version": str(config.version)},
        "run_info": {
            "topic": topic,
            "timestamp_iso": datetime.now().isoformat(),
            "search_depth": context.get("_search_depth", "standard"),
            "research_scope": context.get("research_scope"),
        },
        "diagnostic_context": diagnostic_context,
    }
    atomic_write(path, json.dumps(payload, ensure_ascii=False, indent=2, default=str))
    return path


def load_resume_checkpoint(
    path: str | Path, config: PipelineConfig
) -> tuple[str, dict[str, Any], str]:
    payload = read_checkpoint_object(path)
    schema_version = str(payload.get("schema_version") or "")
    resumable_versions = {"4.7", "4.8", "4.9", "5.0", "6.0"}
    if schema_version not in resumable_versions:
        raise ValueError(
            "only v4.7 through v6.0 diagnostics can resume the built in v6.0 pipeline; "
            "older results remain available in history"
        )
    manifest = payload.get("pipeline")
    compatible_version = isinstance(manifest, dict) and (
        str(manifest.get("version")) == str(config.version)
        or (
            schema_version in {"4.7", "4.8", "4.9", "5.0"}
            and str(manifest.get("version")) == schema_version
            and str(config.version) == "6.0"
        )
    )
    if not compatible_version or manifest.get("name") != config.pipeline_name:
        raise ValueError("resume diagnostic belongs to a different pipeline")
    context = payload.get("diagnostic_context")
    if not isinstance(context, dict):
        raise ValueError("resume diagnostic has no valid diagnostic context")
    if payload.get("context_checksum") != context_checksum(context):
        raise ValueError("resume diagnostic context checksum does not match")
    if not payload.get("resumable"):
        raise ValueError("diagnostic is not resumable")
    active_step = context.get("_active_step")
    if payload.get("resume_step") != active_step:
        raise ValueError("resume diagnostic step metadata does not match its context")
    if schema_version != "6.0" and str(config.version) == "6.0":
        context, active_step = migrate_legacy_context(
            context,
            schema_version=schema_version,
            config=config,
        )
    step = next((item for item in config.steps if item.name == active_step), None)
    if step is None:
        raise ValueError("resume diagnostic does not identify a current pipeline step")
    missing = sorted(set(step.inputs) - set(context))
    if missing:
        raise ValueError(f"resume checkpoint is missing required inputs: {missing}")
    try:
        resume_count = int(context.get("_resume_count", 0))
    except (TypeError, ValueError) as error:
        raise ValueError("resume diagnostic has an invalid resume count") from error
    if not 0 <= resume_count < 3:
        raise ValueError("diagnostic resume limit of 3 has been reached")
    allowed = set(config.context_init)
    for configured_step in config.steps:
        allowed.update(configured_step.outputs)
    unknown = sorted(key for key in context if not key.startswith("_") and key not in allowed)
    if unknown:
        raise ValueError(f"resume diagnostic has unknown context keys: {unknown}")
    output_schemas = {
        output: (
            RESPONSE_SCHEMAS[item.response_schema],
            bool(item.map_input or item.variants),
            item.item_failure == "degrade",
        )
        for item in config.steps
        if isinstance(item, LlmStep) and item.response_schema
        for output in item.outputs
    }
    for key, (schema, mapped, outcome_wrapped) in output_schemas.items():
        if key not in context:
            continue
        values = context[key] if mapped else [context[key]]
        if not isinstance(values, list):
            raise ValueError("mapped output must be a list")
        try:
            for value in values:
                if outcome_wrapped and isinstance(value, dict) and "status" in value:
                    if value.get("status") != "succeeded":
                        continue
                    value = value.get("value")
                schema.model_validate(value)
        except ValidationError as error:
            raise ValueError(f"resume diagnostic has invalid '{key}': {error}") from error
    paper_adapter = TypeAdapter(list[PaperRecord])
    for key in (
        "literature_candidates",
        "top_papers",
        "novelty_candidates",
        "novelty_papers",
        "evidence_papers",
    ):
        if key in context:
            try:
                paper_adapter.validate_python(context[key])
            except ValidationError as error:
                raise ValueError(f"resume diagnostic has invalid '{key}': {error}") from error
    status_adapter = TypeAdapter(list[SourceStatus])
    for key in ("source_statuses", "novelty_source_statuses"):
        if key in context:
            try:
                status_adapter.validate_python(context[key])
            except ValidationError as error:
                raise ValueError(f"resume diagnostic has invalid '{key}': {error}") from error
    run_info = payload.get("run_info") if isinstance(payload.get("run_info"), dict) else {}
    topic = str(run_info.get("topic") or context.get("topic") or "").strip()
    if not topic:
        raise ValueError("resume diagnostic has no research topic")
    return topic, context, str(active_step)


__all__ = [
    "is_resumable_context",
    "load_resume_checkpoint",
    "render_text_result",
    "save_failure_diagnostic",
    "save_results",
    "slugify",
]
