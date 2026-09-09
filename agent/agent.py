"""CLI and built in research pipeline orchestration."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import time
import traceback
from collections.abc import Callable
from contextlib import suppress
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from ollama import Client, ResponseError
from pydantic import ValidationError
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

# Preserve the historical file launcher while loading internals as a package.
if __name__ == "__main__" and not __package__:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from agent.agent import main

    raise SystemExit(main())

from . import agent_helpers, knowledge, multi_source
from .llm.response_policies import (
    candidate_critique_draft_schema,
    candidate_kernel_schema,
    candidate_protocol_schema,
    critique_generation_schema,
    gap_generation_schema,
    hypothesis_generation_schema,
    normalize_gap_response,
    normalize_scope_response,
    novelty_decision_draft_schema,
    proposal_generation_schema,
    scope_generation_schema,
)
from .ollama_models import (
    OllamaModelService,
    is_cloud_model,
    ollama_http_timeout,
    parse_ollama_host,
)
from .pipeline import (
    ModelContractProfileStore,
    ModelOutputArtifact,
    PipelineState,
    StageOutcome,
    StageStatus,
    build_capability_reports,
    build_prompt_packet,
    build_prompt_view,
    compile_model_output,
    consume_chat_stream,
    escape_prompt_data,
    execute_tool_step,
    get_output_policy,
    hydrate_prompt_files,
    is_structural_compilation_failure,
    parse_output,
    plan_prompt,
    prepare_pipeline_context,
    resolve_tool,
    validate_compact_instance,
    validate_tool_contracts,
)
from .pipeline import identity_scoped_format_schema as _identity_scoped_format_schema
from .pipeline import ollama_format_schema as _ollama_format_schema
from .pipeline import slotted_relevance_format_schema as _slotted_relevance_format_schema
from .pipeline.errors import (
    InsufficientEvidenceError,
    ModelOutputError,
    ModelTransportError,
    OllamaRuntimeError,
)
from .pipeline.output.parser import is_schema_echo as _is_schema_echo
from .pipeline.output.validation import (
    _candidate_critique_validator as _candidate_critique_validator,
)
from .pipeline.output.validation import (
    _discard_incomplete_gap_tail as _discard_incomplete_gap_tail,
)
from .pipeline.output.validation import (
    _is_format_initialization_error as _is_format_initialization_error,
)
from .pipeline.output.validation import (
    _is_structural_output_error as _is_structural_output_error,
)
from .pipeline.output.validation import (
    _is_truncated_json as _is_truncated_json,
)
from .pipeline.output.validation import (
    _json_start_offset as _json_start_offset,
)
from .pipeline.output.validation import (
    _normalize_app_owned_metadata as _normalize_app_owned_metadata,
)
from .pipeline.output.validation import (
    _normalize_schema_bounds as _normalize_schema_bounds,
)
from .pipeline.output.validation import (
    _output_limit_reached as _output_limit_reached,
)
from .pipeline.output.validation import (
    _paper_reference_validator as _paper_reference_validator,
)
from .pipeline.output.validation import (
    _referenced_paper_ids as _referenced_paper_ids,
)
from .pipeline.output.validation import (
    _response_value as _response_value,
)
from .pipeline.output.validation import (
    _shorten_bounded_text as _shorten_bounded_text,
)
from .pipeline.output.validation import (
    _strip_application_owned_wire_fields as _strip_application_owned_wire_fields,
)
from .pipeline.persistence import (
    is_resumable_context as _pipeline_is_resumable_context,
)
from .pipeline.persistence import (
    load_resume_checkpoint as _pipeline_load_resume_checkpoint,
)
from .pipeline.persistence import (
    save_failure_diagnostic as _pipeline_save_failure_diagnostic,
)
from .pipeline.persistence import (
    save_results as _pipeline_save_results,
)
from .pipeline.persistence import (
    slugify as _pipeline_slugify,
)
from .pipeline.stages import (
    HypothesisAxisResolution,
    build_conservative_proposal,
    normalize_critique_item,
    normalize_novelty_item,
    normalize_research_plan,
    project_hypothesis_candidate,
    resolve_hypothesis_axis,
)
from .pipeline.stages import (
    recover_critiques as recover_candidate_critiques,
)
from .pipeline.stages import (
    recover_gap_map as conservative_gap_map,
)
from .pipeline.stages import (
    recover_novelty as recover_novelty_verifications,
)
from .pipeline.stages import (
    recover_research_plan as conservative_research_plan,
)
from .pipeline.stages import (
    recover_scopes as conservative_scopes,
)
from .pipeline.stages.runtime import tool_overrides
from .reporting import ConsoleReporter, NullReporter, PipelineEvent, PipelineReporter
from .response_validation import (
    normalize_keyed_response_items as _normalize_keyed_response_items,
)
from .response_validation import (
    reconcile_evidence_card_duplicates as _reconcile_evidence_card_duplicates,
)
from .schemas import (
    RESPONSE_SCHEMAS,
    EvidenceCard,
    LlmStep,
    ModelAssignments,
    PaperRelevanceJudgement,
    PipelineConfig,
    ResearchScope,
    ToolStep,
)

console = Console()
DEFAULT_CONFIG_PATH = Path(__file__).with_name("pipeline") / "default.yaml"

EXIT_SUCCESS = 0
EXIT_CONFIG = 2
EXIT_SOURCE = 3
EXIT_EVIDENCE = 4
EXIT_OLLAMA = 5
EXIT_OUTPUT = 6
EXIT_CANCELLED = 130


def _resolve_tool(function_path: str) -> Callable[..., Any]:
    return resolve_tool(function_path)


def _validate_tool_contracts(config: PipelineConfig) -> None:
    validate_tool_contracts(config, resolver=_resolve_tool)


def load_pipeline_config() -> PipelineConfig:
    """Load the single built in, versioned research pipeline."""
    try:
        raw = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
        raw = hydrate_prompt_files(raw, DEFAULT_CONFIG_PATH)
        config = PipelineConfig.model_validate(raw)
        _validate_tool_contracts(config)
        return config
    except FileNotFoundError as error:
        raise ValueError("built in pipeline definition is missing") from error
    except yaml.YAMLError as error:
        raise ValueError(f"invalid built in pipeline definition: {error}") from error
    except ValidationError as error:
        raise ValueError(f"invalid built in pipeline definition:\n{error}") from error
    except ValueError as error:
        raise ValueError(f"invalid built in pipeline definition: {error}") from error


def _raise_post_generation_error(
    error: Exception,
    *,
    context: dict[str, Any],
    step: LlmStep,
    model: str,
    metrics: dict[str, Any],
    generated_values: list[Any],
    generated_metrics: list[dict[str, Any]],
    pending_steps: dict[str, Any],
    configured_variant: Any,
    axis_resolution: HypothesisAxisResolution | None,
) -> None:
    """Checkpoint completed variants before surfacing canonicalization errors."""
    diagnostic = {
        "category": "post_generation_validation_error",
        "model": model,
        "attempts": int(metrics.get("request_count", 1)),
        "configured_variant": (
            axis_resolution.configured_axis if axis_resolution else configured_variant
        ),
        "effective_variant": (
            axis_resolution.effective_axis if axis_resolution else configured_variant
        ),
        "validation_error": str(error).splitlines()[0],
    }
    failure_metric = {
        "model": model,
        "request_count": diagnostic["attempts"],
        "outcome": "invalid_output",
        **diagnostic,
    }
    context["_llm_metrics"][step.name] = [*generated_metrics, failure_metric]
    if step.variants or step.map_input:
        pending_steps[step.name] = {
            "values": generated_values,
            "metrics": generated_metrics,
            "completed_count": len(generated_values),
            "failed_variant": configured_variant,
            "failed_effective_variant": diagnostic["effective_variant"],
        }
    context["_failure_diagnostic"] = {"step": step.name, **diagnostic}
    raise ModelOutputError(
        f"{step.name}: model response failed post generation validation: "
        f"{diagnostic['validation_error']}",
        diagnostic=diagnostic,
    ) from error


class AbstractAgent:
    """A validated interaction point for an Ollama model."""

    def __init__(self, model: str, host: str, timeout: float) -> None:
        if not model:
            raise ValueError("Ollama model name cannot be empty")
        self.model = model
        self.host = host
        self.inactivity_timeout = timeout
        self.client = Client(host=host, timeout=ollama_http_timeout(timeout))

    def ask(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        temperature: float = 0.7,
        response_schema: str | None = None,
        response_validator: Callable[[Any], Any] | None = None,
        response_normalizer: Callable[[Any], tuple[Any, list[dict[str, Any]]]] | None = None,
        allow_schema_bound_normalization: bool = False,
        format_schema_override: dict[str, Any] | None = None,
        prompt_schema_override: dict[str, Any] | None = None,
        final_recovery: Callable[[Any | None, Exception], Any] | None = None,
        max_output_tokens: int = 2048,
        repair_output_tokens: int | None = None,
        allow_partial_recovery: bool = False,
        progress_callback: Callable[[int, int], None] | None = None,
        reasoning_mode: str | None = None,
        structured_output_mode: str | None = None,
        generation_token_limit: int | None = None,
        expected_candidate_id: str | None = None,
        validation_attempts: int | None = None,
        output_route: str = "content_schema",
        policy_id: str | None = None,
        prompt_id: str | None = None,
        prompt_version: str | None = None,
        prompt_hash: str | None = None,
        contract_version: str | None = None,
        defer_domain_validation: bool = False,
    ) -> tuple[Any, dict[str, Any]]:
        if not prompt:
            raise ValueError("user prompt cannot be empty")
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        schema_class = RESPONSE_SCHEMAS.get(response_schema) if response_schema else None
        if response_schema and schema_class is None:
            raise ValueError(f"unknown response schema: {response_schema}")
        format_schema = None
        if format_schema_override is not None:
            if schema_class is None:
                raise ValueError("format_schema_override requires response_schema")
            format_schema = _ollama_format_schema(format_schema_override)
        elif schema_class:
            format_schema = _ollama_format_schema(schema_class.model_json_schema())
        # Ollama receives a grammar-safe schema, while the model sees the complete contract.
        # Bounds and descriptions materially improve compliance even when the local grammar
        # implementation cannot compile every JSON Schema keyword.
        if prompt_schema_override is not None and schema_class is None:
            raise ValueError("prompt_schema_override requires response_schema")
        prompt_schema = (
            prompt_schema_override
            if prompt_schema_override is not None
            else format_schema
            if schema_class
            else None
        )
        schema_json = (
            json.dumps(prompt_schema, ensure_ascii=False, separators=(",", ":"))
            if prompt_schema
            else ""
        )
        grounded_prompt = prompt
        if schema_json:
            grounded_prompt += f"\n\nReturn only JSON matching this schema:\n{schema_json}"
        messages.append({"role": "user", "content": grounded_prompt})
        attempts = validation_attempts or (2 if schema_class or response_validator else 1)
        if attempts not in {1, 2}:
            raise ValueError("validation_attempts must be 1 or 2")
        last_error: Exception | None = None
        invalid_content: str | None = None
        last_was_truncated = False
        total_streamed_characters = 0
        total_thinking_characters = 0
        total_request_count = 0
        grammar_fallback_count = 0
        grammar_fallback_error: str | None = None
        format_overrides = getattr(self, "_runtime_format_mode_overrides", {})
        contract_key = response_schema or "__unstructured__"
        active_format_mode = (
            format_overrides.get(contract_key) or structured_output_mode or "schema"
        )
        if active_format_mode not in {"schema", "json", "prompt"}:
            raise ValueError(f"unknown structured output mode: {active_format_mode}")
        initial_format_mode = active_format_mode
        think_value: bool | str | None = {
            "disabled": False,
            "low": "low",
            "enabled": True,
        }.get(reasoning_mode)
        thinking_required = reasoning_mode in {"low", "enabled"}
        last_schema_value: Any | None = None
        last_response_metrics: dict[str, Any] = {}
        last_attempt_tokens = max_output_tokens
        last_request_generation_tokens = max_output_tokens
        last_was_thinking_exhausted = False
        last_output_limit_reached = False
        last_was_schema_echo = False
        last_first_token_seconds: float | None = None
        for attempt in range(attempts):
            content = ""
            schema_normalizations: list[dict[str, Any]] = []
            metadata_normalizations: list[dict[str, Any]] = []
            policy_normalizations: list[dict[str, Any]] = []
            request_messages = list(messages)
            if attempt and last_was_schema_echo:
                root_keys = sorted((format_schema or {}).get("properties", {}))
                request_messages = messages[:-1]
                request_messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"{prompt}\n\nReturn the requested DATA INSTANCE as JSON, not a JSON "
                            "Schema definition. Do not return $defs, properties, required, title, "
                            f"or schema metadata. Required top level keys: {', '.join(root_keys)}."
                        ),
                    }
                )
            elif attempt and last_was_truncated:
                request_messages.append(
                    {
                        "role": "user",
                        "content": (
                            "The previous JSON was truncated. Regenerate the complete object from "
                            "the original request. Be substantially more concise, omit optional "
                            "detail, and return JSON only. Do not continue the previous text."
                        ),
                    }
                )
            elif attempt:
                request_messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Your previous response did not validate. Return only corrected JSON "
                            "matching the supplied schema.\n"
                            f"Validation error: {last_error}\n"
                            f"Previous response: {(invalid_content or '')[:1500]}"
                        ),
                    }
                )
            try:
                attempt_tokens = max_output_tokens
                if attempt and last_was_truncated:
                    attempt_tokens = repair_output_tokens or max_output_tokens
                response_metrics: dict[str, Any] = {}
                reasoning_budget_retry = False
                generation_tokens_override: int | None = None
                while True:
                    try:
                        request_tokens = attempt_tokens
                        if thinking_required:
                            request_tokens += max(2048, attempt_tokens)
                        token_ceiling = generation_token_limit or 32768
                        request_tokens = min(request_tokens, token_ceiling)
                        if generation_tokens_override is not None:
                            request_tokens = generation_tokens_override
                        last_request_generation_tokens = request_tokens
                        total_request_count += 1
                        request_format: dict[str, Any] | str | None = None
                        if format_schema and active_format_mode == "schema":
                            request_format = format_schema
                        elif format_schema and active_format_mode == "json":
                            request_format = "json"
                        request_kwargs: dict[str, Any] = {
                            "model": self.model,
                            "messages": request_messages,
                            "options": {
                                "temperature": 0 if attempt else temperature,
                                "num_predict": request_tokens,
                            },
                            "stream": True,
                        }
                        if request_format is not None:
                            request_kwargs["format"] = request_format
                        if think_value is not None:
                            request_kwargs["think"] = think_value
                        request_started_at = time.monotonic()
                        stream_result = consume_chat_stream(
                            self.client.chat(**request_kwargs),
                            attempt=attempt + 1,
                            progress_callback=progress_callback,
                            started_at=request_started_at,
                        )
                        content += stream_result.content
                        thinking = stream_result.thinking
                        total_streamed_characters += len(stream_result.content)
                        total_thinking_characters += len(thinking)
                        response_metrics = stream_result.metrics
                        last_first_token_seconds = stream_result.first_token_seconds
                        content_is_complete_json = False
                        if content.strip():
                            try:
                                json.loads(content)
                            except (json.JSONDecodeError, TypeError):
                                pass
                            else:
                                content_is_complete_json = True
                        if (
                            thinking_required
                            and response_metrics.get("done_reason") == "length"
                            and thinking.strip()
                            and not content_is_complete_json
                            and not reasoning_budget_retry
                        ):
                            expanded_tokens = min(
                                token_ceiling,
                                max(
                                    request_tokens * 2,
                                    attempt_tokens + math.ceil(len(thinking) / 3) + 1024,
                                ),
                            )
                            if expanded_tokens > request_tokens:
                                reasoning_budget_retry = True
                                generation_tokens_override = expanded_tokens
                                content = ""
                                response_metrics = {}
                                continue
                        break
                    except ResponseError as error:
                        if (
                            format_schema
                            and active_format_mode in {"schema", "json"}
                            and _is_format_initialization_error(error)
                        ):
                            active_format_mode = (
                                "json" if active_format_mode == "schema" else "prompt"
                            )
                            grammar_fallback_count += 1
                            grammar_fallback_error = str(error.error)[:500]
                            content = ""
                            response_metrics = {}
                            continue
                        raise
                if not content.strip():
                    raise ValueError("Ollama response contained no message content")
                value: Any = content.strip()
                if schema_class:
                    parsed, parser_name, parser_events = parse_output(
                        content,
                        allow_partial=False,
                        truncated=False,
                    )
                    if _is_schema_echo(parsed):
                        raise ValueError(
                            "model returned the JSON Schema definition instead of a data instance"
                        )
                    if defer_domain_validation:
                        parsed, metadata_normalizations = _strip_application_owned_wire_fields(
                            parsed, response_schema
                        )
                        validate_compact_instance(parsed, format_schema or {})
                    else:
                        parsed, metadata_normalizations = _normalize_app_owned_metadata(
                            parsed,
                            response_schema,
                            expected_candidate_id=expected_candidate_id,
                        )
                    if response_normalizer:
                        parsed, policy_normalizations = response_normalizer(parsed)
                    policy_normalizations = [
                        *[
                            {
                                "path": "output",
                                "operation": event.code,
                                "reason": event.message,
                            }
                            for event in parser_events
                        ],
                        *policy_normalizations,
                    ]
                    if defer_domain_validation:
                        value = parsed
                    else:
                        try:
                            value = schema_class.model_validate(parsed).model_dump()
                        except ValidationError as schema_error:
                            if not allow_schema_bound_normalization:
                                raise
                            parsed, schema_normalizations = _normalize_schema_bounds(
                                parsed, schema_error
                            )
                            if not schema_normalizations:
                                raise
                            value = schema_class.model_validate(parsed).model_dump()
                    last_schema_value = value
                if response_validator and not defer_domain_validation:
                    value = response_validator(value)
                metrics = response_metrics
                metrics["request_count"] = total_request_count
                metrics["streamed_characters"] = total_streamed_characters
                metrics["response_characters"] = len(content)
                metrics["thinking_characters"] = total_thinking_characters
                metrics["generation_characters"] = (
                    total_streamed_characters + total_thinking_characters
                )
                metrics["first_token_seconds"] = last_first_token_seconds
                metrics["max_output_tokens"] = max_output_tokens
                metrics["attempt_output_tokens"] = attempt_tokens
                metrics["request_generation_tokens"] = request_tokens
                metrics["reasoning_mode"] = reasoning_mode or "automatic"
                metrics["recovery_mode"] = (
                    "normalized"
                    if schema_normalizations or metadata_normalizations or policy_normalizations
                    else "regenerated"
                    if attempt
                    else "none"
                )
                if schema_normalizations:
                    metrics["schema_normalization_count"] = len(schema_normalizations)
                    metrics["schema_normalized_paths"] = [
                        item["path"] for item in schema_normalizations
                    ]
                    metrics["schema_normalizations"] = schema_normalizations
                if metadata_normalizations:
                    metrics["metadata_normalization_count"] = len(metadata_normalizations)
                    metrics["metadata_normalizations"] = metadata_normalizations
                if policy_normalizations:
                    metrics["policy_normalization_count"] = len(policy_normalizations)
                    metrics["policy_normalizations"] = policy_normalizations
                    metrics["recovery_outcome"] = (
                        "conservative_fallback"
                        if any(
                            item.get("operation") == "fallback" for item in policy_normalizations
                        )
                        else "normalized"
                    )
                else:
                    metrics["recovery_outcome"] = "regenerated" if attempt else "exact"
                metrics["format_mode"] = (
                    "schema"
                    if active_format_mode == "schema"
                    else "json_fallback"
                    if active_format_mode == "json" and initial_format_mode == "schema"
                    else "prompt_fallback"
                    if active_format_mode == "prompt" and initial_format_mode != "prompt"
                    else active_format_mode
                )
                metrics["grammar_fallback_count"] = grammar_fallback_count
                metrics["output_parser"] = locals().get("parser_name", "exact_json")
                metrics["compilation_outcome"] = (
                    "normalized"
                    if schema_normalizations or metadata_normalizations or policy_normalizations
                    else "exact"
                )
                metrics["repair_attempted"] = attempt > 0
                metrics["output_route"] = output_route
                if grammar_fallback_error:
                    metrics["grammar_fallback_error"] = grammar_fallback_error
                return value, metrics
            except (ValidationError, ValueError) as error:
                last_error = error
                last_response_metrics = response_metrics
                last_attempt_tokens = attempt_tokens
                invalid_content = locals().get("content")
                if invalid_content:
                    with suppress(json.JSONDecodeError, TypeError):
                        last_schema_value = json.loads(invalid_content)
                last_was_truncated = _is_truncated_json(
                    error,
                    invalid_content or "",
                    response_metrics.get("done_reason"),
                )
                last_output_limit_reached = _output_limit_reached(
                    error,
                    invalid_content or "",
                    response_metrics,
                    last_request_generation_tokens,
                )
                last_was_schema_echo = False
                if (invalid_content or "").strip():
                    with suppress(json.JSONDecodeError, TypeError):
                        last_was_schema_echo = _is_schema_echo(json.loads(invalid_content or ""))
                invalid_content_is_complete_json = False
                if (invalid_content or "").strip():
                    try:
                        json.loads(invalid_content or "")
                    except (json.JSONDecodeError, TypeError):
                        pass
                    else:
                        invalid_content_is_complete_json = True
                last_was_thinking_exhausted = bool(
                    response_metrics.get("done_reason") == "length"
                    and not invalid_content_is_complete_json
                    and total_thinking_characters
                )
                if (
                    attempt == 0
                    and active_format_mode == "schema"
                    and (
                        _is_structural_output_error(error)
                        or last_was_schema_echo
                        or not invalid_content_is_complete_json
                    )
                ):
                    active_format_mode = "json"
                    self._runtime_format_mode_override = "json"
                    self._runtime_format_mode_overrides = {
                        **format_overrides,
                        contract_key: "json",
                    }
                    grammar_fallback_count += 1
                    grammar_fallback_error = (
                        "schema mode returned structurally nonconforming output"
                    )
                if (
                    attempt == attempts - 1
                    and last_output_limit_reached
                    and allow_partial_recovery
                    and schema_class
                    and not defer_domain_validation
                ):
                    try:
                        partial, partial_parser, partial_parser_issues = parse_output(
                            invalid_content or "",
                            allow_partial=True,
                            truncated=True,
                        )
                        if response_schema == "gap_map":
                            partial = _discard_incomplete_gap_tail(partial)
                        partial_events: list[dict[str, Any]] = []
                        if response_normalizer:
                            partial, partial_events = response_normalizer(partial)
                        value = schema_class.model_validate(partial).model_dump()
                        if response_validator:
                            value = response_validator(value)
                        populated = sum(
                            len(item) for item in value.values() if isinstance(item, list)
                        )
                        if populated < 1:
                            raise ValueError(
                                "partial structured output contained no complete items"
                            )
                        metrics = response_metrics
                        metrics["request_count"] = total_request_count
                        metrics["streamed_characters"] = total_streamed_characters
                        metrics["response_characters"] = len(invalid_content or "")
                        metrics["thinking_characters"] = total_thinking_characters
                        metrics["generation_characters"] = (
                            total_streamed_characters + total_thinking_characters
                        )
                        metrics["max_output_tokens"] = max_output_tokens
                        metrics["attempt_output_tokens"] = attempt_tokens
                        metrics["recovery_mode"] = "validated_partial"
                        metrics["recovery_outcome"] = "truncation_salvaged"
                        metrics["output_limit_reached"] = True
                        metrics["output_parser"] = partial_parser
                        metrics["partial_parser_issues"] = [
                            issue.as_dict() for issue in partial_parser_issues
                        ]
                        if partial_events:
                            metrics["policy_normalization_count"] = len(partial_events)
                            metrics["policy_normalizations"] = partial_events
                        metrics["validation_error"] = str(error)[:500]
                        metrics["format_mode"] = active_format_mode
                        metrics["grammar_fallback_count"] = grammar_fallback_count
                        if grammar_fallback_error:
                            metrics["grammar_fallback_error"] = grammar_fallback_error
                        return value, metrics
                    except (ValidationError, ValueError, TypeError):
                        pass
                continue
            except ResponseError as error:
                if error.status_code == 404:
                    raise ModelTransportError(
                        f"model '{self.model}' is not available on {self.host}"
                    ) from error
                raise ModelTransportError(
                    f"Ollama returned HTTP {error.status_code}: {error.error}"
                ) from error
            except TimeoutError as error:
                raise ModelTransportError(
                    f"Ollama produced no response data for "
                    f"{self.inactivity_timeout:g} seconds while running model '{self.model}'. "
                    "Increase Preferences > Ollama inactivity timeout or use "
                    "--ollama-inactivity-timeout."
                ) from error
            except ConnectionError as error:
                raise ModelTransportError(
                    f"cannot connect to Ollama at {self.host}: {error}"
                ) from error
            except Exception as error:
                name = type(error).__name__
                if "timeout" in name.casefold():
                    raise ModelTransportError(
                        f"Ollama produced no response data for "
                        f"{getattr(self, 'inactivity_timeout', 600):g} seconds while running "
                        f"model '{self.model}'. Increase Preferences > Ollama inactivity timeout "
                        "or use --ollama-inactivity-timeout."
                    ) from error
                raise ModelTransportError(f"Ollama request failed: {name}: {error}") from error
        recovery_error: Exception | None = None
        if final_recovery and last_error:
            try:
                compiled = compile_model_output(
                    ModelOutputArtifact(
                        content=invalid_content or "",
                        route=output_route,  # type: ignore[arg-type]
                        done_reason=last_response_metrics.get("done_reason"),
                        output_limit_reached=last_output_limit_reached,
                        model=self.model,
                        request_count=total_request_count,
                    ),
                    schema=schema_class,
                    normalizer=response_normalizer,
                    validator=response_validator,
                    projector=final_recovery,
                    allow_partial=allow_partial_recovery,
                    policy_id=policy_id,
                    prompt_id=prompt_id,
                    prompt_version=prompt_version,
                    prompt_hash=prompt_hash,
                    contract_version=contract_version,
                )
                value = compiled.value
            except Exception as error:
                recovery_error = error
            else:
                metrics = last_response_metrics
                metrics["request_count"] = total_request_count
                metrics["streamed_characters"] = total_streamed_characters
                metrics["response_characters"] = len(invalid_content or "")
                metrics["thinking_characters"] = total_thinking_characters
                metrics["generation_characters"] = (
                    total_streamed_characters + total_thinking_characters
                )
                metrics["output_limit_reached"] = last_output_limit_reached
                metrics["first_token_seconds"] = last_first_token_seconds
                metrics["max_output_tokens"] = max_output_tokens
                metrics["attempt_output_tokens"] = last_attempt_tokens
                metrics["recovery_mode"] = "conservative_fallback"
                metrics["recovery_outcome"] = compiled.trace.outcome
                metrics["compilation_outcome"] = compiled.trace.outcome
                metrics["compilation_trace"] = compiled.trace.as_dict()
                metrics["repair_attempted"] = attempts > 1
                metrics["compilation_trace"]["repair_attempted"] = attempts > 1
                metrics["output_route"] = compiled.trace.route
                metrics["validation_error"] = str(last_error)[:500]
                metrics["format_mode"] = active_format_mode
                metrics["grammar_fallback_count"] = grammar_fallback_count
                if grammar_fallback_error:
                    metrics["grammar_fallback_error"] = grammar_fallback_error
                return value, metrics
        reason = (
            "reasoning exhausted the generation budget before producing an answer"
            if last_was_thinking_exhausted
            else "the output token limit was reached before a complete JSON contract"
            if last_output_limit_reached
            else "incomplete JSON"
            if last_was_truncated
            else "invalid structured output"
        )
        guidance = (
            " Increase the model context or reduce the prompt/output size."
            if last_was_thinking_exhausted
            else ""
        )
        raise ModelOutputError(
            f"model '{self.model}' returned {reason} after {attempts} attempts: "
            f"{last_error}.{guidance}",
            diagnostic={
                "model": self.model,
                "attempts": attempts,
                "category": (
                    "reasoning_budget_exhausted"
                    if last_was_thinking_exhausted
                    else "truncated_json"
                    if last_was_truncated
                    else "validation_error"
                ),
                "initial_output_tokens": max_output_tokens,
                "repair_output_tokens": repair_output_tokens or max_output_tokens,
                "streamed_characters": total_streamed_characters,
                "thinking_characters": total_thinking_characters,
                "reasoning_mode": reasoning_mode or "automatic",
                "request_generation_tokens": last_request_generation_tokens,
                "done_reason": last_response_metrics.get("done_reason"),
                "eval_count": last_response_metrics.get("eval_count"),
                "output_limit_reached": last_output_limit_reached,
                "response_characters": len(invalid_content or ""),
                "json_start_offset": _json_start_offset(invalid_content or ""),
                "validation_error": str(last_error)[:500],
                "response_tail": (invalid_content or "")[-500:],
                "format_mode": active_format_mode,
                "grammar_fallback_count": grammar_fallback_count,
                "grammar_fallback_error": grammar_fallback_error,
                "recovery_error": str(recovery_error)[:500] if recovery_error else None,
            },
        )

    def draft(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        temperature: float = 0,
        max_output_tokens: int = 1536,
        progress_callback: Callable[[int, int], None] | None = None,
        reasoning_mode: str | None = None,
        generation_token_limit: int | None = None,
    ) -> tuple[str, dict[str, Any]]:
        """Generate an unconstrained semantic draft before structural realization."""
        value, metrics = self.ask(
            prompt
            + "\n\nWrite a concise semantic draft in plain text. Focus on scientific meaning; "
            "do not emit JSON, schemas, paper IDs, or application metadata.",
            system_prompt=system_prompt,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            progress_callback=progress_callback,
            reasoning_mode=reasoning_mode,
            generation_token_limit=generation_token_limit,
        )
        if not isinstance(value, str) or not value.strip():
            raise ModelOutputError(f"model '{self.model}' returned no usable semantic draft")
        return value.strip(), metrics

    def close(self) -> None:
        close = getattr(self.client, "close", None)
        if callable(close):
            close()


class PipelineManager:
    """Execute validated steps and track provenance."""

    def __init__(
        self,
        config: PipelineConfig,
        *,
        ollama_host: str | None = None,
        ollama_timeout: float | None = None,
        minimum_papers: int | None = None,
        search_depth: str = "standard",
        allow_insufficient_literature: bool = False,
        agent_factory: Callable[[str, str, float], AbstractAgent] = AbstractAgent,
        reporter: PipelineReporter | None = None,
        model_assignments: ModelAssignments | dict[str, Any] | None = None,
        model_service_factory: Callable[[str, float], OllamaModelService] = OllamaModelService,
        requested_scope: str = "broad",
        scope_selector: Callable[[str, list[dict[str, Any]]], dict[str, Any]] | None = None,
        knowledge_enabled: bool = False,
        knowledge_dir: str | Path = "knowledge",
        embedding_model: str | None = None,
        knowledge_refresh: str = "balanced",
        output_profile_store: ModelContractProfileStore | None = None,
    ) -> None:
        self.config = config
        self.ollama_host = (ollama_host or os.getenv("OLLAMA_HOST") or config.ollama_host).rstrip(
            "/"
        )
        configured_timeout = os.getenv("OLLAMA_TIMEOUT_SECONDS")
        self.ollama_timeout = float(
            ollama_timeout
            if ollama_timeout is not None
            else configured_timeout or config.ollama_timeout_seconds
        )
        if not 0 < self.ollama_timeout <= 3600:
            raise ValueError("Ollama inactivity timeout must be between 1 and 3600 seconds")
        self.allow_insufficient_literature = allow_insufficient_literature
        self.minimum_papers = (
            minimum_papers if minimum_papers is not None else config.minimum_unique_papers
        )
        if self.minimum_papers < 1:
            raise ValueError("minimum papers must be at least 1")
        if search_depth not in {"standard", "deep"}:
            raise ValueError("search depth must be 'standard' or 'deep'")
        self.search_depth = search_depth
        self.last_context: dict[str, Any] = {}
        self.agent_factory = agent_factory
        self.reporter = reporter or NullReporter()
        self.model_assignments = ModelAssignments.model_validate(model_assignments or {})
        self.model_service_factory = model_service_factory
        self.requested_scope = requested_scope.strip() or "broad"
        self.scope_selector = scope_selector
        self.knowledge_enabled = knowledge_enabled
        self.knowledge_dir = Path(knowledge_dir)
        self.embedding_model = embedding_model
        if knowledge_refresh not in {"balanced", "always", "offline"}:
            raise ValueError("knowledge refresh must be 'balanced', 'always', or 'offline'")
        if knowledge_refresh == "offline" and not knowledge_enabled:
            raise ValueError("offline literature refresh requires local knowledge to be enabled")
        self.knowledge_refresh = knowledge_refresh
        self._knowledge_store: Any | None = None
        self._embedding_service: Any | None = None
        self._embedding_model_metadata: dict[str, Any] | None = None
        self._agents: dict[tuple[str, float], AbstractAgent] = {}
        self._resolved_models = {
            step.name: self._resolve_model(step)
            for step in self.config.steps
            if isinstance(step, LlmStep)
        }
        self._validated_model_metadata: dict[str, dict[str, Any]] = {}
        self.output_profile_store = output_profile_store or ModelContractProfileStore()

    def _emit(self, kind: str, message: str = "", **kwargs: Any) -> None:
        self.reporter.emit(PipelineEvent(kind=kind, message=message, **kwargs))

    def _get_agent(
        self, model: str, inactivity_timeout_seconds: float | None = None
    ) -> AbstractAgent:
        timeout = float(inactivity_timeout_seconds or self.ollama_timeout)
        key = (model, timeout)
        if key not in self._agents:
            self._agents[key] = self.agent_factory(model, self.ollama_host, timeout)
        return self._agents[key]

    def _resolve_model(self, step: LlmStep) -> str:
        return self.model_assignments.resolve(step.model_role, step.model)

    def _ensure_knowledge_store(self) -> Any | None:
        if not self.knowledge_enabled:
            return None
        if self._knowledge_store is None:
            try:
                self._knowledge_store = knowledge.KnowledgeStore(self.knowledge_dir)
            except (OSError, ValueError) as error:
                self.knowledge_enabled = False
                self._emit(
                    "warning",
                    "Local knowledge storage is unavailable; continuing without persistence "
                    f"({type(error).__name__}).",
                )
        return self._knowledge_store

    def _stream_progress_callback(self, step: LlmStep) -> Callable[[int, int], None]:
        def report(attempt: int, characters: int) -> None:
            self._emit(
                "model_progress",
                step_name=step.name,
                data={"attempt": attempt, "streamed_characters": characters},
            )

        return report

    def _prompt_metrics(
        self,
        step: LlmStep,
        prompt: str,
        model: str,
        format_schema_override: dict[str, Any] | None = None,
        *,
        include_prompt_schema: bool = True,
    ) -> dict[str, Any]:
        schema = RESPONSE_SCHEMAS.get(step.response_schema) if step.response_schema else None
        schema_text = (
            json.dumps(
                _ollama_format_schema(format_schema_override or schema.model_json_schema()),
                ensure_ascii=False,
                separators=(",", ":"),
            )
            if schema
            else ""
        )
        system_text = step.persona or step.system_prompt or ""
        schema_component = schema_text * (2 if include_prompt_schema else 1)
        prompt_chars = len(prompt) + len(system_text) + len(schema_component)
        metadata = self._validated_model_metadata.get(model, {})
        context_length = int(metadata.get("context_length") or 0)
        allocated_context = int(metadata.get("allocated_context_length") or 0)
        answer_tokens = step.repair_output_tokens or step.max_output_tokens
        reasoning_mode = metadata.get("reasoning_mode")
        generation_tokens = answer_tokens
        if reasoning_mode in {"low", "enabled"}:
            generation_tokens += max(2048, answer_tokens)
        request_plan = plan_prompt(
            components={"system": system_text, "prompt": prompt, "schema": schema_component},
            output_tokens=generation_tokens,
            allocated_context_tokens=allocated_context or None,
            advertised_context_tokens=context_length or None,
            splittable=False,
        )
        safe_generation_tokens = max(
            0,
            request_plan.context_tokens
            - request_plan.estimated_input_tokens
            - request_plan.safety_tokens,
        )
        return {
            "prompt_characters": prompt_chars,
            "estimated_prompt_tokens": request_plan.estimated_input_tokens,
            "context_length": context_length,
            "allocated_context_length": allocated_context,
            "max_output_tokens": step.max_output_tokens,
            "safe_generation_tokens": safe_generation_tokens,
            "prompt_can_execute": request_plan.can_execute,
            "prompt_plan": request_plan.as_dict(),
            "schema_in_prompt": include_prompt_schema,
        }

    @property
    def resolved_models(self) -> set[str]:
        return set(self._resolved_models.values())

    def preflight(self) -> None:
        """Verify the Ollama host and every configured model before expensive work."""
        service = self.model_service_factory(self.ollama_host, self.ollama_timeout)
        try:
            validated = service.validate_models(set(self._resolved_models.values()), probe=True)
            self._validated_model_metadata = {
                name: model.model_dump() for name, model in validated.items()
            }
            for name, metadata in self._validated_model_metadata.items():
                if metadata.get("probe_warning"):
                    self._emit("warning", f"{name}: {metadata['probe_warning']}")
            if self.knowledge_enabled:
                selector = getattr(service, "select_embedding_model", None)
                selected = selector(self.embedding_model) if callable(selector) else None
                if selected is None:
                    self._emit(
                        "warning",
                        "No installed embedding model is available; evidence and knowledge "
                        "retrieval will use lexical ranking.",
                    )
                else:
                    self._embedding_model_metadata = selected.model_dump()
                    self._embedding_service = knowledge.OllamaEmbeddingService(
                        selected.name,
                        self.ollama_host,
                        self.ollama_timeout,
                        digest=selected.digest,
                    )
        except OllamaRuntimeError:
            raise
        except ValueError as error:
            raise ModelTransportError(str(error)) from error
        except Exception as error:
            raise ModelTransportError(
                f"Ollama preflight failed at {self.ollama_host}: {error}"
            ) from error
        finally:
            service.close()

    @staticmethod
    def _map_tool_result(result: Any, outputs: list[str], function_path: str) -> dict[str, Any]:
        try:
            from .pipeline.executor import map_step_result
        except ImportError:  # pragma: no cover - direct script execution
            from pipeline.executor import map_step_result

        return map_step_result(result, outputs, function_path)

    def run_pipeline(
        self, initial_context: dict[str, Any], *, start_step: str | None = None
    ) -> dict[str, Any]:
        started = time.monotonic()
        profile_run_id = datetime.now().isoformat()
        context, steps = prepare_pipeline_context(initial_context, self.config.steps, start_step)
        state = PipelineState(context)
        context.setdefault("_llm_metrics", {})
        context.setdefault("_reference_corrections", [])
        context["_pipeline_ollama_host"] = self.ollama_host
        context["_pipeline_version"] = str(self.config.version)
        context["_minimum_papers"] = self.minimum_papers
        context["_insufficient_literature_override"] = self.allow_insufficient_literature
        context["_search_depth"] = self.search_depth
        context["_requested_scope"] = self.requested_scope
        self.last_context = context
        context["_pipeline_model_assignments"] = self.model_assignments.model_dump()
        context["_resolved_models_by_step"] = dict(self._resolved_models)
        context["_resolved_model_metadata"] = dict(self._validated_model_metadata)
        context["_model_capabilities"] = build_capability_reports(self._validated_model_metadata)
        context["_knowledge"] = {
            "enabled": self.knowledge_enabled,
            "directory": str(self.knowledge_dir),
            "embedding_model": (
                self._embedding_model_metadata.get("name")
                if self._embedding_model_metadata
                else None
            ),
            "refresh": self.knowledge_refresh,
        }
        self._emit(
            "pipeline_started",
            f"Starting {self.config.pipeline_name} v{self.config.version}",
            total_steps=len(steps),
        )
        for index, step in enumerate(steps, start=1):
            step_started_at = time.monotonic()
            context["_active_step"] = step.name
            self._emit(
                "step_started",
                step_name=step.name,
                step_index=index,
                total_steps=len(steps),
                data={"model_step": isinstance(step, LlmStep)},
            )
            if isinstance(step, ToolStep):
                overrides = tool_overrides(
                    step,
                    context,
                    search_depth=self.search_depth,
                    knowledge_enabled=self.knowledge_enabled,
                    knowledge_refresh=self.knowledge_refresh,
                    minimum_papers=self.minimum_papers,
                    knowledge_store_factory=self._ensure_knowledge_store,
                    embedding_service=self._embedding_service,
                )
                result_data = execute_tool_step(
                    step, context, resolver=_resolve_tool, overrides=overrides
                )
            else:
                packet_keys = [key for key in step.inputs if key != step.map_input]
                prompt_packet = build_prompt_packet(
                    packet_keys,
                    context,
                    dynamic_budget=(
                        9_000 if step.generation_strategy == "draft_conditioned" else 14_000
                    ),
                )
                values = dict(prompt_packet.values)
                prompt_view = None
                if step.prompt_view:
                    prompt_view = build_prompt_view(
                        step.prompt_view,
                        packet_keys,
                        context,
                        budget=step.prompt_view_budget,
                    )
                    values["input_view"] = prompt_view.text
                values.update({"name": step.name, "persona": step.persona or ""})
                mapped_items = context[step.map_input] if step.map_input else None
                if step.map_input and not isinstance(mapped_items, list):
                    raise ValueError(f"{step.name} map_input must contain a list")
                variants = mapped_items if step.map_input else (step.variants or [None])
                generated_values: list[Any] = []
                generated_metrics: list[dict[str, Any]] = []
                pending_steps = context.setdefault("_pending_step_outputs", {})
                pending = pending_steps.get(step.name, {})
                if (step.variants or step.map_input) and isinstance(pending, dict):
                    pending_values = pending.get("values", [])
                    pending_metrics = pending.get("metrics", [])
                    if (
                        isinstance(pending_values, list)
                        and isinstance(pending_metrics, list)
                        and len(pending_values) == len(pending_metrics) <= len(variants)
                    ):
                        generated_values.extend(pending_values)
                        generated_metrics.extend(pending_metrics)
                        variants = variants[len(pending_values) :]
                stage_output_route: str | None = None
                profile_observations: list[dict[str, Any]] = []
                total_units = len(generated_values) + len(variants)
                for unit_index, variant in enumerate(variants, start=len(generated_values) + 1):
                    configured_variant = variant
                    hypothesis_axis_resolution: HypothesisAxisResolution | None = None
                    if (
                        step.response_schema == "slotted_paper_relevance_assessments"
                        and isinstance(variant, dict)
                        and not isinstance(variant.get("slot_map"), dict)
                    ):
                        paper_ids = variant.get("paper_ids", [])
                        if not isinstance(paper_ids, list) or not paper_ids:
                            raise ValueError(f"{step.name} requires paper_ids or a batch slot_map")
                        variant = {
                            **variant,
                            "slot_map": {
                                f"p{slot_index:02d}": paper_id
                                for slot_index, paper_id in enumerate(paper_ids, start=1)
                            },
                        }
                    variant_values = dict(values)
                    if step.map_input:
                        variant_values[step.map_item_key] = escape_prompt_data(variant)
                    elif variant is not None:
                        effective_variant = variant
                        if step.response_schema == "hypothesis_candidate":
                            selected_scope = context.get(
                                "research_scope", {"mode": "broad", "label": "Broad topic"}
                            )
                            scope_options = context.get("evidence_scope_options", {}).get(
                                "options", []
                            )
                            candidate_pools = (
                                context.get("top_papers", []),
                                context.get("evidence_papers", []),
                            )
                            core_ids = {
                                str(paper["paper_id"])
                                for pool in candidate_pools
                                for paper in pool
                                if isinstance(paper, dict)
                                and paper.get("paper_id")
                                and paper.get("relevance_judgement", {}).get("role")
                                in {"direct_evidence", "transferable_method"}
                            }
                            hypothesis_axis_resolution = resolve_hypothesis_axis(
                                str(variant), selected_scope, scope_options, core_ids
                            )
                            effective_variant = hypothesis_axis_resolution.effective_axis
                        variant_values[step.variant_key] = effective_variant
                    elif step.response_schema == "hypothesis_candidate" and context.get(
                        "selected_candidate"
                    ):
                        selected_candidate = context["selected_candidate"]
                        selected_provenance = selected_candidate.get("generation_provenance", {})
                        configured_axis = str(
                            selected_provenance.get("configured_axis")
                            or selected_candidate.get("candidate_id")
                            or "candidate"
                        )
                        candidate_pools = (
                            context.get("top_papers", []),
                            context.get("evidence_papers", []),
                        )
                        core_ids = {
                            str(paper["paper_id"])
                            for pool in candidate_pools
                            for paper in pool
                            if isinstance(paper, dict)
                            and paper.get("paper_id")
                            and paper.get("relevance_judgement", {}).get("role")
                            in {"direct_evidence", "transferable_method"}
                        }
                        hypothesis_axis_resolution = resolve_hypothesis_axis(
                            configured_axis,
                            context.get(
                                "research_scope", {"mode": "broad", "label": "Broad topic"}
                            ),
                            context.get("evidence_scope_options", {}).get("options", []),
                            core_ids,
                        )
                    prompt = step.prompt_template.format(**variant_values)
                    resolved_model = self._resolved_models[step.name]
                    prompt_resource_hash = hashlib.sha256(
                        "\0".join(
                            (
                                step.persona or step.system_prompt or "",
                                step.prompt_template,
                                step.prompt_version,
                            )
                        ).encode("utf-8")
                    ).hexdigest()
                    output_policy = get_output_policy(
                        step.policy,
                        fallback=(step.prompt_id or step.name.casefold().replace(" ", ".")),
                    )
                    model_metadata = self._validated_model_metadata.get(resolved_model, {})
                    model_digest = str(model_metadata.get("digest") or "")
                    configured_mode = str(model_metadata.get("structured_output_mode") or "schema")
                    baseline = {
                        "schema": "basic_schema",
                        "json": "basic_json",
                        "prompt": "text_projection",
                    }.get(configured_mode, "basic_schema")
                    learned_output_route = (
                        self.output_profile_store.route_for(
                            model_digest=model_digest,
                            policy_id=output_policy.policy_id,
                            prompt_hash=prompt_resource_hash,
                            contract_version=output_policy.contract_version,
                            baseline=baseline,  # type: ignore[arg-type]
                        )
                        if model_digest
                        else {
                            "basic_schema": "content_schema",
                            "basic_json": "content_json",
                            "text_projection": "tagged_text",
                        }[baseline]
                    )
                    if stage_output_route is None:
                        stage_output_route = learned_output_route
                    output_route = stage_output_route
                    format_schema_override = None
                    prompt_schema_override = None
                    if step.response_schema == "slotted_paper_relevance_assessments":
                        if not isinstance(variant, dict) or not isinstance(
                            variant.get("slot_map"), dict
                        ):
                            raise ValueError(f"{step.name} requires a batch slot_map")
                        format_schema_override = _slotted_relevance_format_schema(
                            list(variant["slot_map"])
                        )
                    elif step.response_schema == "evidence_scope_options":
                        format_schema_override = scope_generation_schema()
                        prompt_schema_override = format_schema_override
                    elif step.response_schema == "gap_map":
                        format_schema_override = gap_generation_schema()
                        prompt_schema_override = format_schema_override
                    elif step.response_schema == "hypothesis_candidate":
                        format_schema_override = hypothesis_generation_schema(
                            include_addressed_findings=step.name
                            in {"Revise Winning Candidate", "Repair Revised Candidate"}
                        )
                        prompt_schema_override = format_schema_override
                    elif step.response_schema == "candidate_kernel":
                        format_schema_override = candidate_kernel_schema()
                        prompt_schema_override = format_schema_override
                    elif step.response_schema == "candidate_protocol":
                        format_schema_override = candidate_protocol_schema()
                        prompt_schema_override = format_schema_override
                    elif step.response_schema == "candidate_critique_draft":
                        format_schema_override = candidate_critique_draft_schema()
                        prompt_schema_override = format_schema_override
                    elif step.response_schema == "novelty_decision_draft":
                        format_schema_override = novelty_decision_draft_schema()
                        prompt_schema_override = format_schema_override
                    elif step.response_schema == "candidate_critiques":
                        critique_ids = (
                            [str(variant.get("candidate_id", ""))]
                            if step.map_input and isinstance(variant, dict)
                            else [
                                str(item.get("candidate_id", ""))
                                for item in context.get(
                                    "audited_hypothesis_candidates",
                                    context.get("hypothesis_candidates", []),
                                )
                            ]
                        )
                        format_schema_override = critique_generation_schema(
                            [item for item in critique_ids if item]
                        )
                        prompt_schema_override = format_schema_override
                    elif step.response_schema == "research_proposal":
                        format_schema_override = proposal_generation_schema()
                        prompt_schema_override = format_schema_override
                    if step.map_input and step.response_schema == "novelty_verifications":
                        focused_candidate_id = str(variant.get("candidate_id", ""))
                        format_schema_override = _identity_scoped_format_schema(
                            step.response_schema, focused_candidate_id
                        )
                        prompt_schema_override = format_schema_override
                    prompt_metrics = self._prompt_metrics(
                        step, prompt, resolved_model, format_schema_override
                    )
                    if (
                        not prompt_metrics.get("prompt_can_execute", True)
                        and prompt_schema_override is not None
                    ):
                        # Keep the transport grammar, but avoid duplicating it in the model prompt
                        # when the allocated context cannot safely hold both copies.
                        prompt_schema_override = {}
                        prompt_metrics = self._prompt_metrics(
                            step,
                            prompt,
                            resolved_model,
                            format_schema_override,
                            include_prompt_schema=False,
                        )
                        prompt_metrics["schema_grounding_omitted_for_budget"] = True
                    prompt_metrics.update(
                        {
                            "dynamic_prompt_characters": prompt_packet.dynamic_characters,
                            "prompt_resource_hash": prompt_resource_hash,
                            "prompt_packet_hash": prompt_packet.prompt_hash,
                            "prompt_truncated_fields": list(prompt_packet.truncated_fields),
                            "prompt_view": step.prompt_view,
                            "prompt_view_hash": prompt_view.content_hash if prompt_view else None,
                            "prompt_view_characters": (
                                prompt_view.character_count if prompt_view else 0
                            ),
                            "prompt_view_omitted_items": (
                                prompt_view.omitted_items if prompt_view else 0
                            ),
                        }
                    )
                    evidence_papers = list(
                        {
                            paper["paper_id"]: paper
                            for pool in (
                                context.get("literature_candidates", []),
                                context.get("top_papers", []),
                                context.get("novelty_candidates", []),
                                context.get("novelty_papers", []),
                                context.get("evidence_papers", []),
                            )
                            for paper in pool
                            if isinstance(paper, dict) and paper.get("paper_id")
                        }.values()
                    )
                    response_corrections: list[dict[str, str]] = []
                    normalization_events: list[str] = []
                    relevance_recovery_events: list[dict[str, Any]] = []
                    evidence_card_events: list[dict[str, list[str]]] = []
                    semantic_draft: str | None = None
                    semantic_draft_holder: dict[str, str | None] = {"value": None}
                    draft_metrics: dict[str, Any] | None = None
                    final_recovery: Callable[[Any | None, Exception], Any] | None = None
                    response_normalizer: (
                        Callable[[Any], tuple[Any, list[dict[str, Any]]]] | None
                    ) = None

                    reference_validator = (
                        _paper_reference_validator(evidence_papers, response_corrections)
                        if step.response_schema and evidence_papers
                        else None
                    )
                    if step.response_schema == "research_plan":

                        def normalize_plan(
                            response: Any,
                            raw_topic: str = str(context["topic"]),
                        ) -> tuple[Any, list[dict[str, str]]]:
                            return normalize_research_plan(response, raw_topic)

                        response_normalizer = normalize_plan
                        topic_terms = set(
                            re.findall(r"[a-z0-9]+", str(context["topic"]).casefold())
                        ) - {"novel", "new", "method", "methods", "technique", "techniques"}

                        def validate_research_plan(
                            response: Any, required_topic_terms: set[str] = topic_terms
                        ) -> Any:
                            concepts = response.get("required_concepts", [])
                            if not concepts or not any(
                                (
                                    set(re.findall(r"[a-z0-9]+", str(concept).casefold()))
                                    & required_topic_terms
                                )
                                for concept in concepts
                            ):
                                raise ValueError(
                                    "required_concepts must contain distinctive terms from the "
                                    "topic"
                                )
                            return response

                        reference_validator = validate_research_plan

                        def recover_research_plan(
                            _response: Any | None,
                            _error: Exception,
                            raw_topic: str = str(context["topic"]),
                        ) -> Any:
                            return conservative_research_plan(raw_topic)

                        final_recovery = recover_research_plan
                    if step.response_schema == "evidence_scope_options":
                        core_paper_ids = {
                            str(paper["paper_id"])
                            for paper in context.get("top_papers", [])
                            if paper.get("relevance_judgement", {}).get("role")
                            in {"direct_evidence", "transferable_method"}
                        }

                        def normalize_scopes(
                            response: Any,
                            allowed: set[str] = core_paper_ids,
                        ) -> tuple[Any, list[dict[str, Any]]]:
                            return normalize_scope_response(response, allowed)

                        response_normalizer = normalize_scopes

                        def recover_scopes(_response: Any | None, _error: Exception) -> Any:
                            return conservative_scopes()

                        final_recovery = recover_scopes
                    if step.response_schema == "gap_map":

                        def recover_gap_map(_response: Any | None, _error: Exception) -> Any:
                            return conservative_gap_map()

                        final_recovery = recover_gap_map
                    if step.response_schema == "paper_relevance_assessments" and step.map_input:
                        expected_ids = set(variant.get("paper_ids", []))
                        base_validator = reference_validator

                        def validate_relevance_batch(
                            response: Any,
                            expected: set[str] = expected_ids,
                            normalize: Callable[[Any], Any] | None = base_validator,
                            events: list[str] = normalization_events,
                        ) -> Any:
                            normalized = normalize(response) if normalize else response
                            normalized, duplicates = _normalize_keyed_response_items(
                                normalized,
                                "assessments",
                                "paper_id",
                                equivalent_keys=("score", "role"),
                            )
                            events.extend(duplicates)
                            ids = [item["paper_id"] for item in normalized.get("assessments", [])]
                            actual = set(ids)
                            if actual != expected:
                                missing = sorted(expected - actual)
                                extra = sorted(actual - expected)
                                raise ValueError(
                                    f"relevance batch must assess every paper exactly once; "
                                    f"missing={missing}, extra={extra}"
                                )
                            return normalized

                        reference_validator = validate_relevance_batch
                    if (
                        step.response_schema == "slotted_paper_relevance_assessments"
                        and step.map_input
                    ):
                        slot_map = {
                            str(slot): str(paper_id)
                            for slot, paper_id in variant["slot_map"].items()
                        }

                        def canonicalize_slotted_relevance(
                            response: Any,
                            expected: dict[str, str] = slot_map,
                        ) -> Any:
                            assessments = response.get("assessments", {})
                            actual = set(assessments)
                            required = set(expected)
                            if actual != required:
                                raise ValueError(
                                    "relevance slots do not match the requested batch; "
                                    f"missing={sorted(required - actual)}, "
                                    f"extra={sorted(actual - required)}"
                                )
                            return {
                                "assessments": [
                                    {"paper_id": paper_id, **assessments[slot]}
                                    for slot, paper_id in expected.items()
                                ]
                            }

                        def recover_slotted_relevance(
                            response: Any | None,
                            error: Exception,
                            expected: dict[str, str] = slot_map,
                            events: list[dict[str, Any]] = relevance_recovery_events,
                        ) -> Any:
                            supplied = (
                                response.get("assessments", {})
                                if isinstance(response, dict)
                                and isinstance(response.get("assessments"), dict)
                                else {}
                            )
                            missing_slots = [slot for slot in expected if slot not in supplied]
                            extra_slots = [slot for slot in supplied if slot not in expected]
                            normalized_slots = []
                            invalid_slots = []
                            recovered: dict[str, dict[str, Any]] = {}
                            for slot in expected:
                                judgement = supplied.get(slot)
                                try:
                                    recovered[slot] = PaperRelevanceJudgement.model_validate(
                                        judgement
                                    ).model_dump()
                                    continue
                                except (ValidationError, TypeError):
                                    pass
                                if (
                                    isinstance(judgement, dict)
                                    and isinstance(judgement.get("score"), int)
                                    and 1 <= judgement["score"] <= 5
                                    and judgement.get("role")
                                    in {
                                        "direct_evidence",
                                        "transferable_method",
                                        "background",
                                        "irrelevant",
                                    }
                                    and str(judgement.get("rationale", "")).strip()
                                ):
                                    rationale = " ".join(str(judgement["rationale"]).split())
                                    if len(rationale) > 240:
                                        rationale = rationale[:237].rsplit(" ", 1)[0] + "..."
                                    recovered[slot] = {
                                        "score": judgement["score"],
                                        "role": judgement["role"],
                                        "rationale": rationale,
                                    }
                                    normalized_slots.append(slot)
                                    continue
                                invalid_slots.append(slot)
                                recovered[slot] = {
                                    "score": 1,
                                    "role": "irrelevant",
                                    "rationale": (
                                        "The relevance judge did not return a valid decision; "
                                        "the paper was conservatively excluded."
                                    ),
                                }
                            events.append(
                                {
                                    "policy": "per_slot_validation",
                                    "missing_slots": missing_slots,
                                    "missing_paper_ids": [expected[slot] for slot in missing_slots],
                                    "invalid_slots": invalid_slots,
                                    "normalized_slots": normalized_slots,
                                    "discarded_slots": extra_slots,
                                    "validation_error": str(error)[:500],
                                }
                            )
                            return {"assessments": recovered}

                        reference_validator = canonicalize_slotted_relevance
                        final_recovery = recover_slotted_relevance
                    if step.response_schema == "gap_map":
                        allowed_gap_paper_ids = {
                            str(paper["paper_id"])
                            for paper in evidence_papers
                            if paper.get("paper_id")
                        }
                        gap_topic = str(context.get("topic", "the research topic"))

                        def normalize_gaps(
                            response: Any,
                            allowed: set[str] = allowed_gap_paper_ids,
                            topic: str = gap_topic,
                        ) -> tuple[Any, list[dict[str, Any]]]:
                            return normalize_gap_response(response, allowed, topic=topic)

                        response_normalizer = normalize_gaps
                        base_validator = reference_validator

                        def validate_gap_map(
                            response: Any,
                            normalize: Callable[[Any], Any] | None = base_validator,
                        ) -> Any:
                            normalized = normalize(response) if normalize else response
                            item_count = sum(
                                len(normalized.get(key, []))
                                for key in (
                                    "established",
                                    "contradictions",
                                    "unresolved_questions",
                                )
                            )
                            if item_count < 1:
                                raise ValueError("gap map must contain at least one complete item")
                            for key, allowed_statuses in (
                                ("established", {"established", "inferred"}),
                                ("contradictions", {"contradictory"}),
                                ("unresolved_questions", {"unknown"}),
                            ):
                                for item in normalized.get(key, []):
                                    if item.get("status") not in allowed_statuses:
                                        raise ValueError(
                                            "gap map item status "
                                            f"'{item.get('status')}' is invalid "
                                            f"inside '{key}'"
                                        )
                                    if item.get("status") != "unknown" and not item.get(
                                        "paper_ids"
                                    ):
                                        raise ValueError(
                                            "evidence backed gap map items must cite paper IDs"
                                        )
                            return normalized

                        reference_validator = validate_gap_map
                    if step.response_schema == "candidate_critiques":
                        critic_source = (
                            "evidence_critic"
                            if "Evidence" in step.name
                            else "methodology_critic"
                            if "Methodology" in step.name
                            else "legacy"
                        )
                        if step.map_input:
                            expected_candidate_ids = {str(variant.get("candidate_id", ""))}
                        else:
                            expected_candidate_ids = {
                                candidate["candidate_id"]
                                for candidate in context.get(
                                    "audited_hypothesis_candidates",
                                    context.get("hypothesis_candidates", []),
                                )
                            }
                        base_validator = reference_validator

                        def normalize_critique_items(
                            response: Any,
                            expected: set[str] = expected_candidate_ids,
                            source_name: str = critic_source,
                        ) -> tuple[Any, list[dict[str, Any]]]:
                            source = response if isinstance(response, dict) else {}
                            raw_items = source.get("critiques", [])
                            if not isinstance(raw_items, list):
                                raw_items = []
                            valid: list[dict[str, Any]] = []
                            operations: list[dict[str, Any]] = []
                            seen: set[str] = set()
                            for item_index, raw_item in enumerate(raw_items):
                                candidate_id = (
                                    str(raw_item.get("candidate_id") or "")
                                    if isinstance(raw_item, dict)
                                    else ""
                                )
                                item, item_operations = normalize_critique_item(
                                    raw_item,
                                    expected_candidate_ids=expected,
                                    critic_source=source_name,
                                )
                                operations.extend(
                                    {
                                        **event,
                                        "path": (
                                            f"critiques[{item_index}].{event.get('path', '')}"
                                        ).rstrip("."),
                                    }
                                    for event in item_operations
                                )
                                if item is None:
                                    operations.append(
                                        {
                                            "path": f"critiques[{item_index}]",
                                            "operation": "drop",
                                            "reason": (
                                                f"invalid critique for {candidate_id or 'unknown'}"
                                            ),
                                        }
                                    )
                                    continue
                                if candidate_id in seen:
                                    operations.append(
                                        {
                                            "path": f"critiques[{item_index}]",
                                            "operation": "drop",
                                            "reason": f"duplicate critique for {candidate_id}",
                                        }
                                    )
                                    continue
                                seen.add(candidate_id)
                                valid.append(item)
                            return {"critiques": valid}, operations

                        response_normalizer = normalize_critique_items

                        def normalize_critiques(
                            response: Any,
                            normalize: Callable[[Any], Any] | None = base_validator,
                            events: list[str] = normalization_events,
                        ) -> Any:
                            normalized = normalize(response) if normalize else response
                            normalized, duplicates = _normalize_keyed_response_items(
                                normalized, "critiques", "candidate_id"
                            )
                            events.extend(duplicates)
                            return normalized

                        reference_validator = _candidate_critique_validator(
                            expected_candidate_ids,
                            exact=bool(step.map_input),
                            normalize=normalize_critiques,
                        )

                        def recover_critiques(
                            _response: Any | None,
                            _error: Exception,
                            expected: set[str] = expected_candidate_ids,
                        ) -> Any:
                            return recover_candidate_critiques(expected)

                        final_recovery = recover_critiques
                    if step.response_schema == "evidence_ledger":
                        expected_paper_ids = (
                            {str(variant.get("paper_id", ""))}
                            if step.map_input
                            else {paper["paper_id"] for paper in context.get("top_papers", [])}
                        )
                        base_validator = reference_validator

                        def normalize_evidence_items(
                            response: Any,
                        ) -> tuple[Any, list[dict[str, Any]]]:
                            source = response if isinstance(response, dict) else {}
                            raw_cards = source.get("cards", [])
                            if not isinstance(raw_cards, list):
                                raw_cards = []
                            cards: list[dict[str, Any]] = []
                            operations: list[dict[str, Any]] = []
                            seen: set[str] = set()
                            for card_index, raw_card in enumerate(raw_cards):
                                paper_id = (
                                    str(raw_card.get("paper_id") or "")
                                    if isinstance(raw_card, dict)
                                    else ""
                                )
                                try:
                                    card = EvidenceCard.model_validate(raw_card).model_dump()
                                except (ValidationError, ValueError, TypeError) as error:
                                    operations.append(
                                        {
                                            "path": f"cards[{card_index}]",
                                            "operation": "drop",
                                            "reason": (
                                                "invalid evidence card for "
                                                f"{paper_id or 'unknown'}: "
                                                f"{str(error).splitlines()[0]}"
                                            ),
                                        }
                                    )
                                    continue
                                if paper_id in seen:
                                    operations.append(
                                        {
                                            "path": f"cards[{card_index}]",
                                            "operation": "drop",
                                            "reason": f"duplicate evidence card for {paper_id}",
                                        }
                                    )
                                    continue
                                seen.add(paper_id)
                                cards.append(card)
                            return {"cards": cards}, operations

                        response_normalizer = normalize_evidence_items

                        def validate_evidence_ledger(
                            response: Any,
                            expected: set[str] = expected_paper_ids,
                            exact: bool = bool(step.map_input),
                            normalize: Callable[[Any], Any] | None = base_validator,
                            events: list[dict[str, list[str]]] = evidence_card_events,
                        ) -> Any:
                            normalized = normalize(response) if normalize else response
                            normalized, event = _reconcile_evidence_card_duplicates(
                                normalized, exact=exact
                            )
                            actual = {card["paper_id"] for card in normalized.get("cards", [])}
                            extra = actual - expected
                            missing = expected - actual if exact else set()
                            if extra or missing:
                                raise ValueError(
                                    "evidence ledger paper IDs do not match the requested papers; "
                                    f"missing={sorted(missing)}, extra={sorted(extra)}"
                                )
                            if any(event.values()):
                                events.append(event)
                            return normalized

                        reference_validator = validate_evidence_ledger
                    if step.response_schema == "hypothesis_candidate":
                        expected_candidate_id = None
                        if step.variants:
                            expected_candidate_id = (
                                hypothesis_axis_resolution.effective_axis
                                if hypothesis_axis_resolution
                                else str(variant)
                            )
                        elif "selected_candidate" in context:
                            expected_candidate_id = context["selected_candidate"]["candidate_id"]
                        base_validator = reference_validator
                        roles = {
                            paper["paper_id"]: paper.get("relevance_judgement", {}).get("role")
                            for paper in evidence_papers
                        }
                        selected_research_scope = context.get(
                            "research_scope", {"mode": "broad", "label": "Broad topic"}
                        )
                        available_scope_options = context.get("evidence_scope_options", {}).get(
                            "options", []
                        )
                        if hypothesis_axis_resolution is None:
                            selected_provenance = context.get("selected_candidate", {}).get(
                                "generation_provenance", {}
                            )
                            hypothesis_axis_resolution = resolve_hypothesis_axis(
                                str(
                                    selected_provenance.get("configured_axis")
                                    or expected_candidate_id
                                    or "candidate"
                                ),
                                selected_research_scope,
                                available_scope_options,
                                {
                                    paper_id
                                    for paper_id, role in roles.items()
                                    if role in {"direct_evidence", "transferable_method"}
                                },
                            )

                        def validate_hypothesis(
                            response: Any,
                            expected: str | None = expected_candidate_id,
                            normalize: Callable[[Any], Any] | None = base_validator,
                            evidence_roles: dict[str, Any] = roles,
                            selected_scope: dict[str, Any] = selected_research_scope,
                            axis: HypothesisAxisResolution = hypothesis_axis_resolution,
                        ) -> Any:
                            normalized = normalize(response) if normalize else response
                            if expected and normalized.get("candidate_id") != expected:
                                raise ValueError(
                                    f"candidate_id must be exactly '{expected}', got "
                                    f"'{normalized.get('candidate_id')}'"
                                )
                            invalid = [
                                paper_id
                                for paper_id in normalized.get("supporting_paper_ids", [])
                                if evidence_roles.get(paper_id)
                                not in {"direct_evidence", "transferable_method"}
                            ]
                            if invalid:
                                raise ValueError(
                                    "supporting_paper_ids require core evidence: "
                                    + ", ".join(sorted(invalid))
                                )
                            applicability = normalized.get("applicability", "broad")
                            supporting = set(normalized.get("supporting_paper_ids", []))
                            if selected_scope.get("mode") == "broad":
                                if applicability == "domain_specific":
                                    raise ValueError(
                                        "broad research scope cannot select a domain specific "
                                        "hypothesis"
                                    )
                                if applicability == "cross_domain" and (
                                    "cross-domain" not in axis.effective_axis.casefold()
                                    or not supporting & set(axis.primary_scope_ids)
                                    or not supporting & set(axis.secondary_scope_ids)
                                ):
                                    raise ValueError(
                                        "cross domain hypotheses require independently "
                                        "attributable evidence from two validated scopes"
                                    )
                            else:
                                selected_ids = set(selected_scope.get("supporting_paper_ids", []))
                                if not supporting & selected_ids:
                                    raise ValueError(
                                        "hypotheses must cite evidence from the selected scope"
                                    )
                                if (
                                    applicability == "domain_specific"
                                    and str(normalized.get("target_domain", "")).casefold()
                                    != str(selected_scope.get("label", "")).casefold()
                                ):
                                    raise ValueError(
                                        "target_domain must match the selected research scope"
                                    )
                                if applicability == "cross_domain" and (
                                    "cross-domain" not in axis.effective_axis.casefold()
                                    or not supporting & set(axis.primary_scope_ids)
                                    or not supporting & set(axis.secondary_scope_ids)
                                ):
                                    raise ValueError(
                                        "cross domain hypotheses require selected scope evidence "
                                        "and independently attributable evidence from another "
                                        "validated scope"
                                    )
                            return normalized

                        reference_validator = validate_hypothesis

                        def recover_hypothesis(
                            response: Any | None,
                            _error: Exception,
                            expected: str | None = expected_candidate_id,
                            selected_scope: dict[str, Any] = selected_research_scope,
                            scope_options: list[dict[str, Any]] = available_scope_options,
                            papers: list[dict[str, Any]] = evidence_papers,
                            selected_model: str = resolved_model,
                            draft_holder: dict[str, str | None] = semantic_draft_holder,
                            axis: HypothesisAxisResolution = hypothesis_axis_resolution,
                        ) -> Any:
                            return project_hypothesis_candidate(
                                response,
                                topic=str(context.get("topic", "research topic")),
                                candidate_id=expected or "candidate",
                                research_scope=selected_scope,
                                evidence_scope_options=scope_options,
                                evidence_ledger=context.get("evidence_ledger", {}),
                                evidence_papers=papers,
                                model=selected_model,
                                semantic_draft=draft_holder["value"],
                                axis_resolution=axis,
                            )

                        final_recovery = recover_hypothesis
                    if step.response_schema == "novelty_verifications":
                        expected_candidate_ids = (
                            {str(variant.get("candidate_id", ""))}
                            if step.map_input
                            else {
                                context.get(
                                    "validated_revised_candidate",
                                    context["revised_candidate"],
                                )["candidate_id"],
                                context["alternate_candidate"]["candidate_id"],
                            }
                        )
                        base_validator = reference_validator
                        roles = {
                            paper["paper_id"]: paper.get("relevance_judgement", {}).get("role")
                            for paper in evidence_papers
                        }

                        def normalize_novelty_items(
                            response: Any,
                            expected: set[str] = expected_candidate_ids,
                            evidence_roles: dict[str, Any] = roles,
                        ) -> tuple[Any, list[dict[str, Any]]]:
                            source = response if isinstance(response, dict) else {}
                            raw_items = source.get("verifications", [])
                            if not isinstance(raw_items, list):
                                raw_items = []
                            items: list[dict[str, Any]] = []
                            operations: list[dict[str, Any]] = []
                            seen: set[str] = set()
                            for item_index, raw_item in enumerate(raw_items):
                                candidate_id = (
                                    str(raw_item.get("candidate_id") or "")
                                    if isinstance(raw_item, dict)
                                    else ""
                                )
                                novelty_review_packet = agent_helpers.prepare_novelty_review_packet(
                                    context.get(
                                        "validated_revised_candidate",
                                        context["revised_candidate"],
                                    ),
                                    context["alternate_candidate"],
                                    context.get("novelty_papers", []),
                                )
                                candidate_packet = next(
                                    (
                                        packet
                                        for packet in novelty_review_packet.get(
                                            "candidate_packets", []
                                        )
                                        if packet.get("candidate", {}).get("candidate_id")
                                        == candidate_id
                                    ),
                                    {},
                                )
                                slot_map = candidate_packet.get("citation_slot_map", {})
                                if isinstance(raw_item, dict) and isinstance(slot_map, dict):
                                    raw_item = dict(raw_item)
                                    for citation_key in (
                                        "closest_prior_art_paper_ids",
                                        "materially_matching_paper_ids",
                                    ):
                                        raw_item[citation_key] = [
                                            slot_map.get(str(reference), str(reference))
                                            for reference in raw_item.get(citation_key, [])
                                        ]
                                allowed_candidate_ids = set(slot_map.values())
                                item, item_operations = normalize_novelty_item(
                                    raw_item,
                                    expected_candidate_ids=expected,
                                    allowed_paper_ids=allowed_candidate_ids,
                                )
                                operations.extend(
                                    {
                                        **event,
                                        "path": (
                                            f"verifications[{item_index}].{event.get('path', '')}"
                                        ).rstrip("."),
                                    }
                                    for event in item_operations
                                )
                                if item is None:
                                    operations.append(
                                        {
                                            "path": f"verifications[{item_index}]",
                                            "operation": "drop",
                                            "reason": (
                                                "invalid novelty decision for "
                                                f"{candidate_id or 'unknown'}"
                                            ),
                                        }
                                    )
                                    continue
                                if candidate_id in seen:
                                    operations.append(
                                        {
                                            "path": f"verifications[{item_index}]",
                                            "operation": "drop",
                                            "reason": (
                                                f"duplicate novelty decision for {candidate_id}"
                                            ),
                                        }
                                    )
                                    continue
                                seen.add(candidate_id)
                                items.append(item)
                            return {"verifications": items}, operations

                        response_normalizer = normalize_novelty_items

                        def validate_novelty(
                            response: Any,
                            expected: set[str] = expected_candidate_ids,
                            exact: bool = bool(step.map_input),
                            normalize: Callable[[Any], Any] | None = base_validator,
                            evidence_roles: dict[str, Any] = roles,
                            events: list[str] = normalization_events,
                        ) -> Any:
                            normalized = normalize(response) if normalize else response
                            normalized, duplicates = _normalize_keyed_response_items(
                                normalized, "verifications", "candidate_id"
                            )
                            events.extend(duplicates)
                            actual = {
                                item["candidate_id"] for item in normalized.get("verifications", [])
                            }
                            extra = actual - expected
                            missing = expected - actual if exact else set()
                            if extra or missing:
                                raise ValueError(
                                    "novelty candidate IDs do not match requested candidates; "
                                    f"missing={sorted(missing)}, extra={sorted(extra)}"
                                )
                            for item in normalized.get("verifications", []):
                                cited = item.get("closest_prior_art_paper_ids", []) + item.get(
                                    "materially_matching_paper_ids", []
                                )
                                non_core = [
                                    paper_id
                                    for paper_id in cited
                                    if evidence_roles.get(paper_id)
                                    not in {"direct_evidence", "transferable_method"}
                                ]
                                if non_core:
                                    raise ValueError(
                                        "novelty conclusions require core evidence: "
                                        + ", ".join(sorted(non_core))
                                    )
                            return normalized

                        reference_validator = validate_novelty

                        def recover_novelty(
                            _response: Any | None,
                            _error: Exception,
                            expected: set[str] = expected_candidate_ids,
                        ) -> Any:
                            return recover_novelty_verifications(expected)

                        final_recovery = recover_novelty
                    if step.response_schema == "research_proposal":
                        base_validator = reference_validator

                        def validate_proposal(
                            response: Any,
                            normalize: Callable[[Any], Any] | None = base_validator,
                        ) -> Any:
                            normalized = normalize(response) if normalize else response
                            validated, _ = agent_helpers.validate_research_proposal(
                                normalized,
                                context.get("evidence_papers", []),
                                context["novelty_verification"],
                            )
                            return validated

                        reference_validator = validate_proposal

                        def recover_proposal(
                            response: Any | None,
                            _error: Exception,
                            selected_model: str = resolved_model,
                            draft_holder: dict[str, str | None] = semantic_draft_holder,
                        ) -> Any:
                            return build_conservative_proposal(
                                topic=str(context.get("topic", "research topic")),
                                final_candidate=context["final_candidate"],
                                novelty_verification=context["novelty_verification"],
                                evidence_papers=context.get("evidence_papers", []),
                                model=selected_model,
                                semantic_draft=draft_holder["value"],
                                model_content=response if isinstance(response, dict) else None,
                            )

                        final_recovery = recover_proposal
                    if step.invalid_output != "recover":
                        final_recovery = None

                    try:
                        if not prompt_metrics.get("prompt_can_execute", True):
                            raise ModelOutputError(
                                "the bounded request cannot fit the model's allocated context",
                                diagnostic={
                                    "category": "prompt_budget_exhausted",
                                    "attempts": 0,
                                    "prompt_plan": prompt_metrics.get("prompt_plan", {}),
                                },
                            )
                        active_agent = (
                            self._get_agent(resolved_model, step.inactivity_timeout_seconds)
                            if output_route != "deterministic_fallback"
                            else None
                        )
                        call_prompt = prompt
                        draft_method = getattr(active_agent, "draft", None)
                        if (
                            output_route != "deterministic_fallback"
                            and step.generation_strategy == "draft_conditioned"
                            and callable(draft_method)
                        ):
                            self._emit(
                                "model_request_started",
                                step_name=step.name,
                                data={"request_kind": "semantic_draft", "attempt": 1},
                            )
                            semantic_draft, draft_metrics = draft_method(
                                prompt,
                                system_prompt=step.persona or step.system_prompt,
                                temperature=step.temperature,
                                max_output_tokens=min(step.max_output_tokens, 1536),
                                progress_callback=self._stream_progress_callback(step),
                                reasoning_mode=self._validated_model_metadata.get(
                                    resolved_model, {}
                                ).get("reasoning_mode"),
                                generation_token_limit=prompt_metrics.get("safe_generation_tokens"),
                            )
                            semantic_draft_holder["value"] = semantic_draft
                            self._emit(
                                "model_request_completed",
                                step_name=step.name,
                                data={"request_kind": "semantic_draft"},
                            )
                            call_prompt = (
                                prompt
                                + "\n\nSemantic draft to realize into the contract:\n"
                                + semantic_draft
                            )
                            prompt_metrics = self._prompt_metrics(
                                step, call_prompt, resolved_model, format_schema_override
                            )
                        if output_route == "deterministic_fallback":
                            if final_recovery is None:
                                raise ModelOutputError(
                                    f"{step.name} has no deterministic output fallback"
                                )
                            compiled = compile_model_output(
                                ModelOutputArtifact(
                                    content="",
                                    route="deterministic_fallback",
                                    model=resolved_model,
                                    request_count=0,
                                ),
                                schema=RESPONSE_SCHEMAS.get(step.response_schema),
                                validator=reference_validator,
                                fallback=final_recovery,
                                policy_id=output_policy.policy_id,
                                prompt_id=step.prompt_id or output_policy.policy_id,
                                prompt_version=step.prompt_version,
                                prompt_hash=prompt_resource_hash,
                                contract_version=output_policy.contract_version,
                            )
                            generated = compiled.value
                            metrics = {
                                "request_count": 0,
                                "response_characters": 0,
                                "generation_characters": 0,
                                "recovery_mode": "fallback",
                                "recovery_outcome": "fallback",
                                "compilation_outcome": "fallback",
                                "compilation_trace": compiled.trace.as_dict(),
                                "output_route": "deterministic_fallback",
                                "format_mode": "application_fallback",
                            }
                        else:
                            structured_mode = {
                                "content_schema": "schema",
                                "content_json": "json",
                                "tagged_text": "prompt",
                            }[output_route]
                            if active_agent is None:  # pragma: no cover - guarded by route
                                raise ModelOutputError("model agent was not initialized")
                            self._emit(
                                "model_request_started",
                                step_name=step.name,
                                data={
                                    "request_kind": (
                                        "focused_repair" if step.map_input else "contract"
                                    ),
                                    "attempt": 1,
                                    "unit_index": unit_index,
                                    "unit_total": total_units,
                                    "unit_id": (
                                        str(variant.get("unit_id", ""))
                                        if isinstance(variant, dict)
                                        else str(variant or "")
                                    ),
                                },
                            )
                            generated, metrics = active_agent.ask(
                                call_prompt,
                                system_prompt=step.persona or step.system_prompt,
                                temperature=step.temperature,
                                response_schema=step.response_schema,
                                response_validator=reference_validator,
                                response_normalizer=response_normalizer,
                                allow_schema_bound_normalization=step.invalid_output == "recover",
                                format_schema_override=format_schema_override,
                                prompt_schema_override=prompt_schema_override,
                                final_recovery=final_recovery,
                                max_output_tokens=step.max_output_tokens,
                                repair_output_tokens=step.repair_output_tokens,
                                allow_partial_recovery=step.allow_partial_recovery,
                                progress_callback=self._stream_progress_callback(step),
                                reasoning_mode=self._validated_model_metadata.get(
                                    resolved_model, {}
                                ).get("reasoning_mode"),
                                structured_output_mode=structured_mode,
                                generation_token_limit=prompt_metrics.get("safe_generation_tokens"),
                                expected_candidate_id=expected_candidate_id
                                if step.response_schema == "hypothesis_candidate"
                                else None,
                                validation_attempts=step.validation_attempts,
                                output_route=output_route,
                                policy_id=output_policy.policy_id,
                                prompt_id=step.prompt_id or output_policy.policy_id,
                                prompt_version=step.prompt_version,
                                prompt_hash=prompt_resource_hash,
                                contract_version=output_policy.contract_version,
                                defer_domain_validation=output_policy.model_owned_contract,
                            )
                            self._emit(
                                "model_request_completed",
                                step_name=step.name,
                                data={
                                    "request_kind": (
                                        "focused_repair" if step.map_input else "contract"
                                    )
                                },
                            )
                    except ModelTransportError as error:
                        if step.map_input and step.item_failure == "degrade":
                            generated = StageOutcome(
                                status=StageStatus.DEGRADED,
                                category="model_transport_failure",
                                reason=str(error),
                                retryable=True,
                                metadata={
                                    "model": resolved_model,
                                    "unit_id": (
                                        str(variant.get("unit_id", ""))
                                        if isinstance(variant, dict)
                                        else ""
                                    ),
                                    "diagnostic": error.diagnostic,
                                },
                            ).as_dict()
                            metrics = {
                                "model": resolved_model,
                                "request_count": int(error.diagnostic.get("attempts", 1)),
                                "outcome": "transport_degraded",
                                "compilation_outcome": "fallback",
                                "recovery_outcome": "degraded_unit",
                                **error.diagnostic,
                            }
                            context.setdefault("_stage_outcomes", []).append(
                                {"step": step.name, **generated}
                            )
                            message = (
                                f"{step.name}: one bounded unit was unavailable; continued "
                                "without promoting incomplete model content"
                            )
                            context.setdefault("_warnings", []).append(message)
                            self._emit("warning", message, step_name=step.name)
                        elif step.transport_failure == "conservative_fallback" and final_recovery:
                            compiled = compile_model_output(
                                ModelOutputArtifact(
                                    content="",
                                    route="deterministic_fallback",
                                    model=resolved_model,
                                    request_count=int(error.diagnostic.get("attempts", 1)),
                                ),
                                schema=RESPONSE_SCHEMAS.get(step.response_schema),
                                validator=reference_validator,
                                fallback=final_recovery,
                                policy_id=output_policy.policy_id,
                                prompt_id=step.prompt_id or output_policy.policy_id,
                                prompt_version=step.prompt_version,
                                prompt_hash=prompt_resource_hash,
                                contract_version=output_policy.contract_version,
                            )
                            generated = compiled.value
                            metrics = {
                                "request_count": int(error.diagnostic.get("attempts", 1)),
                                "response_characters": 0,
                                "generation_characters": 0,
                                "recovery_mode": "transport_fallback",
                                "recovery_outcome": "fallback",
                                "compilation_outcome": "fallback",
                                "compilation_trace": compiled.trace.as_dict(),
                                "output_route": "deterministic_fallback",
                                "format_mode": "application_fallback",
                                "transport_error": type(error).__name__,
                            }
                            self._emit(
                                "model_request_completed",
                                step_name=step.name,
                                data={"request_kind": "focused_repair", "fallback": True},
                            )
                        else:
                            failure_metric = {
                                "model": resolved_model,
                                "request_count": int(error.diagnostic.get("attempts", 1)),
                                "outcome": "interrupted",
                                **error.diagnostic,
                            }
                            context["_llm_metrics"][step.name] = [
                                *generated_metrics,
                                failure_metric,
                            ]
                            if step.variants or step.map_input:
                                pending_steps[step.name] = {
                                    "values": generated_values,
                                    "metrics": generated_metrics,
                                    "completed_count": len(generated_values),
                                    "failed_variant": variant,
                                }
                            context["_failure_diagnostic"] = {
                                "step": step.name,
                                **error.diagnostic,
                            }
                            raise type(error)(
                                f"{step.name}: {error}", diagnostic=error.diagnostic
                            ) from error
                    except ModelOutputError as error:
                        failure_metric = {
                            "model": resolved_model,
                            "request_count": int(error.diagnostic.get("attempts", 1)),
                            "outcome": "invalid_output",
                            "compilation_outcome": "fallback",
                            "recovery_outcome": "degraded_unit",
                            **error.diagnostic,
                        }
                        if step.map_input and step.item_failure == "degrade":
                            generated = StageOutcome(
                                status=StageStatus.DEGRADED,
                                category="model_output_invalid",
                                reason=str(error),
                                retryable=False,
                                metadata={
                                    "model": resolved_model,
                                    "unit_id": (
                                        str(variant.get("unit_id", ""))
                                        if isinstance(variant, dict)
                                        else ""
                                    ),
                                    "diagnostic": error.diagnostic,
                                },
                            ).as_dict()
                            metrics = failure_metric
                            context.setdefault("_stage_outcomes", []).append(
                                {"step": step.name, **generated}
                            )
                            message = (
                                f"{step.name}: one bounded unit could not satisfy its contract; "
                                "continued without promoting incomplete model content"
                            )
                            context.setdefault("_warnings", []).append(message)
                            self._emit("warning", message, step_name=step.name)
                        elif final_recovery is not None and step.invalid_output == "recover":
                            try:
                                generated = final_recovery(None, error)
                                if reference_validator is not None:
                                    generated = reference_validator(generated)
                            except (ValidationError, ValueError, TypeError) as recovery_error:
                                context["_llm_metrics"][step.name] = [
                                    *generated_metrics,
                                    failure_metric,
                                ]
                                context["_failure_diagnostic"] = {
                                    "step": step.name,
                                    "category": "deterministic_recovery_failed",
                                    "recovery_error": str(recovery_error).splitlines()[0],
                                    **error.diagnostic,
                                }
                                raise ModelOutputError(
                                    f"{step.name}: deterministic recovery failed: {recovery_error}",
                                    diagnostic=context["_failure_diagnostic"],
                                ) from recovery_error
                            metrics = {
                                **failure_metric,
                                "outcome": "fallback",
                                "compilation_outcome": "fallback",
                                "recovery_mode": "deterministic_fallback",
                                "recovery_outcome": "fallback",
                                "output_route": "deterministic_fallback",
                                "format_mode": "application_fallback",
                            }
                            message = (
                                f"{step.name}: the model contract was unavailable; used the "
                                "bounded application fallback"
                            )
                            context.setdefault("_warnings", []).append(message)
                            self._emit("warning", message, step_name=step.name)
                        else:
                            context["_llm_metrics"][step.name] = [
                                *generated_metrics,
                                failure_metric,
                            ]
                            if step.variants or step.map_input:
                                pending_steps[step.name] = {
                                    "values": generated_values,
                                    "metrics": generated_metrics,
                                    "completed_count": len(generated_values),
                                    "failed_variant": variant,
                                }
                            context["_failure_diagnostic"] = {
                                "step": step.name,
                                "category": "output_compilation_error",
                                **error.diagnostic,
                            }
                            raise ModelOutputError(
                                f"{step.name}: {error}", diagnostic=error.diagnostic
                            ) from error
                    metrics.update(prompt_metrics)
                    if hypothesis_axis_resolution:
                        metrics.update(
                            {
                                "configured_axis": hypothesis_axis_resolution.configured_axis,
                                "effective_axis": hypothesis_axis_resolution.effective_axis,
                                "axis_substitution_reason": (
                                    hypothesis_axis_resolution.substitution_reason
                                ),
                            }
                        )
                    if draft_metrics is not None:
                        metrics["draft_request_count"] = int(draft_metrics.get("request_count", 1))
                        metrics["draft_response_characters"] = int(
                            draft_metrics.get("response_characters", len(semantic_draft or ""))
                        )
                        metrics["request_count"] = int(metrics.get("request_count", 1)) + int(
                            draft_metrics.get("request_count", 1)
                        )
                    compilation_trace = metrics.get("compilation_trace")
                    recovery_source = (
                        compilation_trace.get("recovery_source", "none")
                        if isinstance(compilation_trace, dict)
                        else "none"
                    )
                    if str(metrics.get("compilation_outcome")) == "fallback":
                        recovery_source = "deterministic_fallback"
                    elif step.map_input:
                        recovery_source = "focused_model_repair"
                    elif str(metrics.get("compilation_outcome")) == "normalized":
                        recovery_source = "local_normalization"
                    metrics["recovery_source"] = recovery_source
                    if isinstance(compilation_trace, dict):
                        compilation_trace["recovery_source"] = recovery_source
                    if step.response_schema == "hypothesis_candidate":
                        pre_projection_outcome = str(metrics.get("compilation_outcome") or "exact")
                        canonical = isinstance(generated, dict) and {
                            "candidate_id",
                            "supporting_paper_ids",
                            "generation_provenance",
                        }.issubset(generated)
                        if not canonical:
                            generated_mode = {
                                "exact": "structured_exact",
                                "normalized": "structured_normalized",
                                "projected": "draft_projected",
                                "fallback": "conservative_fallback",
                            }.get(pre_projection_outcome, "draft_projected")
                            try:
                                generated = project_hypothesis_candidate(
                                    generated,
                                    topic=str(context.get("topic", "research topic")),
                                    candidate_id=expected_candidate_id or "candidate",
                                    research_scope=selected_research_scope,
                                    evidence_scope_options=available_scope_options,
                                    evidence_ledger=context.get("evidence_ledger", {}),
                                    evidence_papers=evidence_papers,
                                    model=resolved_model,
                                    semantic_draft=semantic_draft,
                                    generation_mode=generated_mode,
                                    validation_attempts=int(metrics.get("request_count", 1)),
                                    axis_resolution=hypothesis_axis_resolution,
                                )
                            except (ValidationError, ValueError, TypeError) as error:
                                _raise_post_generation_error(
                                    error,
                                    context=context,
                                    step=step,
                                    model=resolved_model,
                                    metrics=metrics,
                                    generated_values=generated_values,
                                    generated_metrics=generated_metrics,
                                    pending_steps=pending_steps,
                                    configured_variant=configured_variant,
                                    axis_resolution=hypothesis_axis_resolution,
                                )
                        if pre_projection_outcome == "normalized":
                            metrics["compilation_outcome"] = "normalized"
                        elif (
                            generated.get("generation_provenance", {}).get("mode")
                            == "structured_normalized"
                        ):
                            metrics["compilation_outcome"] = "normalized"
                            metrics["recovery_source"] = "local_normalization"
                        if step.name == "Revise Winning Candidate":
                            original = context.get("selected_candidate", {})
                            compared_fields = (
                                "hypothesis",
                                "mechanism",
                                "proposed_experiment",
                                "falsifier",
                            )
                            changed_fields = [
                                key
                                for key in compared_fields
                                if " ".join(str(generated.get(key, "")).split()).casefold()
                                != " ".join(str(original.get(key, "")).split()).casefold()
                            ]
                            selected_score = next(
                                (
                                    item
                                    for item in context.get("candidate_scorecard", [])
                                    if item.get("candidate_id") == original.get("candidate_id")
                                ),
                                {},
                            )
                            unresolved = selected_score.get("unresolved_major_revisions", [])
                            meaningful = len(changed_fields) >= 2
                            replacement_attempted = bool(selected_score.get("replacement_required"))
                            context["_revision_audit"] = {
                                "candidate_id": original.get("candidate_id"),
                                "changed_fields": changed_fields,
                                "meaningful_change": meaningful,
                                "unresolved_major_revisions": unresolved,
                                "replacement_attempted": replacement_attempted,
                                "acceptance": "pending_authoritative_gate",
                            }
                        if reference_validator:
                            try:
                                generated = reference_validator(generated)
                            except (ValidationError, ValueError, TypeError) as error:
                                _raise_post_generation_error(
                                    error,
                                    context=context,
                                    step=step,
                                    model=resolved_model,
                                    metrics=metrics,
                                    generated_values=generated_values,
                                    generated_metrics=generated_metrics,
                                    pending_steps=pending_steps,
                                    configured_variant=configured_variant,
                                    axis_resolution=hypothesis_axis_resolution,
                                )
                    if step.response_schema == "research_proposal":
                        canonical = isinstance(generated, dict) and {
                            "proposal_status",
                            "claim_evidence",
                            "references",
                            "generation_provenance",
                        }.issubset(generated)
                        if not canonical:
                            generation_mode = {
                                "exact": "structured_exact",
                                "normalized": "structured_normalized",
                                "projected": "draft_projected",
                                "fallback": "conservative_fallback",
                            }.get(
                                str(metrics.get("compilation_outcome") or "exact"),
                                "draft_projected",
                            )
                            generated = build_conservative_proposal(
                                topic=str(context.get("topic", "research topic")),
                                final_candidate=context["final_candidate"],
                                novelty_verification=context["novelty_verification"],
                                evidence_papers=context.get("evidence_papers", []),
                                model=resolved_model,
                                semantic_draft=semantic_draft,
                                model_content=generated if isinstance(generated, dict) else None,
                                generation_mode=generation_mode,
                                validation_attempts=int(metrics.get("request_count", 1)),
                            )
                        if reference_validator:
                            generated = reference_validator(generated)
                    if isinstance(generated, dict) and isinstance(
                        generated.get("generation_provenance"), dict
                    ):
                        provenance = generated["generation_provenance"]
                        if hypothesis_axis_resolution:
                            provenance.update(hypothesis_axis_resolution.as_dict())
                        provenance["validation_attempts"] = int(metrics.get("request_count", 1))
                        provenance["requested_route"] = output_route
                        provenance["effective_route"] = str(
                            metrics.get("output_route") or output_route
                        )
                        if provenance.get("mode") == "draft_projected":
                            recovery_source = "semantic_projection"
                        elif provenance.get("mode") == "conservative_fallback":
                            recovery_source = "deterministic_fallback"
                        metrics["recovery_source"] = recovery_source
                        if isinstance(compilation_trace, dict):
                            compilation_trace["recovery_source"] = recovery_source
                    compilation_outcome = str(metrics.get("compilation_outcome") or "exact")
                    context.setdefault("_compilation_events", []).append(
                        {
                            "step": step.name,
                            "model": resolved_model,
                            "policy_id": output_policy.policy_id,
                            "requested_route": output_route,
                            "effective_route": str(metrics.get("output_route") or output_route),
                            "route": str(metrics.get("output_route") or output_route),
                            "outcome": compilation_outcome,
                            "prompt_id": step.prompt_id or output_policy.policy_id,
                            "prompt_version": step.prompt_version,
                            "prompt_hash": prompt_resource_hash,
                            "contract_version": output_policy.contract_version,
                            "human_review_required": compilation_outcome
                            in {"projected", "fallback"},
                            "output_limit_reached": bool(
                                metrics.get("output_limit_reached", False)
                            ),
                            "recovery_source": recovery_source,
                            "repair_attempted": bool(metrics.get("repair_attempted", False)),
                        }
                    )
                    if model_digest:
                        structural_failure = is_structural_compilation_failure(metrics)
                        grammar_error = str(metrics.get("grammar_fallback_error") or "")
                        profile_observations.append(
                            {
                                "model_digest": model_digest,
                                "policy_id": output_policy.policy_id,
                                "prompt_hash": prompt_resource_hash,
                                "contract_version": output_policy.contract_version,
                                "route": output_route,
                                "outcome": compilation_outcome,
                                "structural_failure": structural_failure,
                                "confirmed_incompatibility": bool(
                                    grammar_error
                                    and grammar_error
                                    != "schema mode returned structurally nonconforming output"
                                ),
                            }
                        )
                    generated_items = generated if isinstance(generated, list) else [generated]
                    review_required = compilation_outcome in {"projected", "fallback"}
                    for generated_item in generated_items:
                        if not isinstance(generated_item, dict) or not isinstance(
                            generated_item.get("generation_provenance"), dict
                        ):
                            continue
                        provenance = generated_item["generation_provenance"]
                        if not provenance.get("model"):
                            provenance["model"] = resolved_model
                        mode = str(provenance.get("mode", "structured_exact"))
                        provenance.update(
                            {
                                "route": str(metrics.get("output_route") or output_route),
                                "compilation_outcome": compilation_outcome,
                                "prompt_id": step.prompt_id or output_policy.policy_id,
                                "prompt_version": step.prompt_version,
                                "prompt_hash": prompt_resource_hash,
                                "contract_version": output_policy.contract_version,
                                "compilation_issues": [
                                    str(issue.get("code", "output_recovery"))
                                    for issue in metrics.get("compilation_trace", {}).get(
                                        "issues", []
                                    )
                                ][:12],
                            }
                        )
                        context.setdefault("_generation_events", []).append(
                            {
                                "step": step.name,
                                "model": resolved_model,
                                "mode": mode,
                                "human_review_required": bool(
                                    provenance.get("human_review_required", False)
                                ),
                                "configured_axis": provenance.get("configured_axis"),
                                "effective_axis": provenance.get("effective_axis"),
                                "axis_substitution_reason": provenance.get(
                                    "axis_substitution_reason"
                                ),
                            }
                        )
                        review_required = review_required or bool(
                            provenance.get("human_review_required", False)
                        )
                    if (
                        hypothesis_axis_resolution
                        and hypothesis_axis_resolution.substitution_reason
                    ):
                        substitution_key = (
                            f"{step.name}:{hypothesis_axis_resolution.configured_axis}:"
                            f"{hypothesis_axis_resolution.effective_axis}"
                        )
                        warned_substitutions = context.setdefault(
                            "_hypothesis_axis_substitution_warnings", []
                        )
                        if substitution_key not in warned_substitutions:
                            warned_substitutions.append(substitution_key)
                            message = (
                                f"{step.name}: replaced "
                                f"{hypothesis_axis_resolution.configured_axis} with "
                                f"{hypothesis_axis_resolution.effective_axis} because "
                                f"{hypothesis_axis_resolution.substitution_reason.casefold()}"
                            )
                            context.setdefault("_warnings", []).append(message)
                            self._emit("warning", message, step_name=step.name)
                    warned_review_steps = context.setdefault("_human_review_warning_steps", [])
                    if review_required and step.name not in warned_review_steps:
                        warned_review_steps.append(step.name)
                        if metrics.get("output_limit_reached"):
                            message = (
                                f"{step.name}: the model reached its output token limit; "
                                "recovered a schema valid candidate with application owned "
                                "projection; human review is required"
                            )
                        elif compilation_outcome == "fallback":
                            message = (
                                f"{step.name}: used a deterministic fallback because no valid "
                                "model owned contract remained; human review is required"
                            )
                        else:
                            message = (
                                f"{step.name}: recovered usable scientific content from a "
                                "malformed model contract; human review is required"
                            )
                        context.setdefault("_warnings", []).append(message)
                        self._emit("warning", message, step_name=step.name)
                    if metrics.get("schema_normalization_count"):
                        paths = metrics.get("schema_normalized_paths", [])
                        context.setdefault("_normalization_events", []).append(
                            {
                                "step": step.name,
                                "kind": "schema_bounds",
                                "fields": paths,
                                "operations": metrics.get("schema_normalizations", []),
                            }
                        )
                        count = metrics["schema_normalization_count"]
                        message = (
                            f"{step.name}: shortened {count} overlong generated "
                            f"field{'s' if count != 1 else ''} to schema limits"
                        )
                        context.setdefault("_warnings", []).append(message)
                        self._emit("warning", message, step_name=step.name)
                    if metrics.get("policy_normalization_count"):
                        operations = metrics.get("policy_normalizations", [])
                        recovery_event = {
                            "step": step.name,
                            "kind": "response_policy",
                            "outcome": metrics.get("recovery_outcome", "normalized"),
                            "operations": operations,
                        }
                        context.setdefault("_recovery_events", []).append(recovery_event)
                        context.setdefault("_normalization_events", []).append(recovery_event)
                        counts: dict[str, int] = {}
                        for operation in operations:
                            name = str(operation.get("operation", "normalize"))
                            counts[name] = counts.get(name, 0) + 1
                        summary = ", ".join(
                            f"{count} {name}" for name, count in sorted(counts.items())
                        )
                        message = f"{step.name}: applied safe response recovery ({summary})"
                        context.setdefault("_warnings", []).append(message)
                        self._emit("warning", message, step_name=step.name)
                    if metrics.get("grammar_fallback_count"):
                        message = (
                            f"{step.name}: Ollama could not reliably enforce the requested "
                            "structured output grammar; "
                            "continued in JSON mode with application level validation"
                        )
                        context.setdefault("_warnings", []).append(message)
                        self._emit("warning", message, step_name=step.name)
                    if normalization_events:
                        normalized_ids = list(dict.fromkeys(normalization_events))
                        message = (
                            f"{step.name}: normalized semantically identical duplicate "
                            f"decision(s) for {', '.join(normalized_ids)}"
                        )
                        context.setdefault("_normalization_events", []).append(
                            {"step": step.name, "identities": normalized_ids}
                        )
                        context.setdefault("_warnings", []).append(message)
                        self._emit("warning", message, step_name=step.name)
                    if evidence_card_events:
                        for event in evidence_card_events:
                            context.setdefault("_normalization_events", []).append(
                                {"step": step.name, "kind": "evidence_card_duplicates", **event}
                            )
                        identical = sorted(
                            {
                                paper_id
                                for event in evidence_card_events
                                for paper_id in event["identical_paper_ids"]
                            }
                        )
                        recovery = sorted(
                            {
                                paper_id
                                for event in evidence_card_events
                                for paper_id in event["recovery_paper_ids"]
                            }
                        )
                        details = []
                        if identical:
                            details.append(f"deduplicated {', '.join(identical)}")
                        if recovery:
                            details.append(f"scheduled focused recovery for {', '.join(recovery)}")
                        message = f"{step.name}: " + "; ".join(details)
                        context.setdefault("_warnings", []).append(message)
                        self._emit("warning", message, step_name=step.name)
                    if relevance_recovery_events:
                        for recovery in relevance_recovery_events:
                            context.setdefault("_normalization_events", []).append(
                                {"step": step.name, "kind": "relevance_batch_recovery", **recovery}
                            )
                        unresolved = sum(
                            len(set(event["missing_slots"]) | set(event.get("invalid_slots", [])))
                            for event in relevance_recovery_events
                        )
                        normalized = sum(
                            len(event.get("normalized_slots", []))
                            for event in relevance_recovery_events
                        )
                        discarded = sum(
                            len(event["discarded_slots"]) for event in relevance_recovery_events
                        )
                        message = f"{step.name}: recovered relevance output per paper"
                        if normalized:
                            message += f"; normalized {normalized} valid decision(s)"
                        if unresolved:
                            message += f"; conservatively excluded {unresolved} invalid paper(s)"
                        if discarded:
                            message += f" and discarded {discarded} unexpected slot(s)"
                        context.setdefault("_warnings", []).append(message)
                        self._emit("warning", message, step_name=step.name)
                    if metrics.get("recovery_mode") == "validated_partial":
                        message = (
                            f"{step.name}: continued with complete schema valid items recovered "
                            "from a truncated model response"
                        )
                        context.setdefault("_warnings", []).append(message)
                        self._emit("warning", message, step_name=step.name)
                    context["_reference_corrections"].extend(
                        {"step": step.name, **correction} for correction in response_corrections
                    )
                    if (
                        step.map_input
                        and step.item_failure == "degrade"
                        and not (
                            isinstance(generated, dict)
                            and generated.get("status")
                            in {"succeeded", "degraded", "skipped", "terminal", "failed"}
                        )
                    ):
                        integrity = (
                            "normalized"
                            if str(metrics.get("compilation_outcome")) == "normalized"
                            else "exact"
                        )
                        generated = StageOutcome(
                            status=StageStatus.SUCCEEDED,
                            value=generated,
                            metadata={
                                "model": resolved_model,
                                "unit_id": (
                                    str(variant.get("unit_id", ""))
                                    if isinstance(variant, dict)
                                    else ""
                                ),
                                "integrity": integrity,
                                "repair_attempted": bool(metrics.get("repair_attempted", False)),
                            },
                        ).as_dict()
                        context.setdefault("_stage_outcomes", []).append(
                            {"step": step.name, **generated}
                        )
                    generated_values.append(generated)
                    generated_metrics.append(metrics)
                if profile_observations:
                    aggregate_outcome = (
                        "fallback"
                        if any(item["outcome"] == "fallback" for item in profile_observations)
                        else "projected"
                        if any(item["outcome"] == "projected" for item in profile_observations)
                        else "normalized"
                        if any(item["outcome"] == "normalized" for item in profile_observations)
                        else "exact"
                    )
                    profile = profile_observations[-1]
                    self.output_profile_store.record(
                        model_digest=profile["model_digest"],
                        policy_id=profile["policy_id"],
                        prompt_hash=profile["prompt_hash"],
                        contract_version=profile["contract_version"],
                        route=profile["route"],
                        outcome=aggregate_outcome,  # type: ignore[arg-type]
                        run_id=profile_run_id,
                        structural_failure=any(
                            item["structural_failure"] for item in profile_observations
                        ),
                        confirmed_incompatibility=any(
                            item["confirmed_incompatibility"] for item in profile_observations
                        ),
                    )
                value = generated_values if step.variants or step.map_input else generated_values[0]
                result_data = {step.outputs[0]: value}
                if step.response_schema:
                    evidence_pools = (
                        context.get("literature_candidates", []),
                        context.get("top_papers", []),
                        context.get("novelty_candidates", []),
                        context.get("novelty_papers", []),
                        context.get("evidence_papers", []),
                    )
                    known_ids = {
                        paper["paper_id"]
                        for pool in evidence_pools
                        for paper in pool
                        if isinstance(paper, dict) and paper.get("paper_id")
                    }
                    unknown_ids = _referenced_paper_ids(value) - known_ids
                    if unknown_ids:
                        raise ModelOutputError(
                            f"{step.name} referenced unknown paper IDs: "
                            + ", ".join(sorted(unknown_ids))
                        )
                    if step.response_schema == "hypothesis_candidate":
                        candidates = value if isinstance(value, list) else [value]
                        roles = {
                            paper["paper_id"]: paper.get("relevance_judgement", {}).get("role")
                            for pool in evidence_pools
                            for paper in pool
                            if isinstance(paper, dict) and paper.get("paper_id")
                        }
                        core_roles = {"direct_evidence", "transferable_method"}
                        for candidate in candidates:
                            invalid = [
                                paper_id
                                for paper_id in candidate.get("supporting_paper_ids", [])
                                if roles.get(paper_id) not in core_roles
                            ]
                            if invalid:
                                raise ModelOutputError(
                                    f"{step.name} used non core supporting evidence: "
                                    + ", ".join(sorted(invalid))
                                )
                    if step.response_schema == "novelty_verifications":
                        roles = {
                            paper["paper_id"]: paper.get("relevance_judgement", {}).get("role")
                            for pool in evidence_pools
                            for paper in pool
                            if isinstance(paper, dict) and paper.get("paper_id")
                        }
                        core_roles = {"direct_evidence", "transferable_method"}
                        verification_batches = value if isinstance(value, list) else [value]
                        for verification in (
                            item
                            for batch in verification_batches
                            for item in batch.get("verifications", [])
                        ):
                            invalid = [
                                paper_id
                                for paper_id in verification.get("closest_prior_art_paper_ids", [])
                                if roles.get(paper_id) not in core_roles
                            ]
                            if invalid:
                                raise ModelOutputError(
                                    f"{step.name} used non core prior art: "
                                    + ", ".join(sorted(invalid))
                                )
                context["_llm_metrics"][step.name] = (
                    generated_metrics if step.variants or step.map_input else generated_metrics[0]
                )
            committed_values: Any = (
                result_data[step.outputs[0]]
                if len(step.outputs) == 1
                else tuple(result_data[key] for key in step.outputs)
            )
            state.commit(step.outputs, committed_values)
            context.get("_pending_step_outputs", {}).pop(step.name, None)
            if "evidence_scope_options" in result_data:
                options = result_data["evidence_scope_options"].get("options", [])
                if self.scope_selector and self.requested_scope == "ask" and options:
                    self._emit("scope_selection_started", data={"options": options})
                    selected_scope = self.scope_selector(str(context["topic"]), options)
                    self._emit("scope_selection_completed")
                elif self.requested_scope.casefold() not in {"ask", "broad"}:
                    selected = next(
                        (
                            option
                            for option in options
                            if option["label"].casefold() == self.requested_scope.casefold()
                            or option["scope_id"].casefold() == self.requested_scope.casefold()
                        ),
                        None,
                    )
                    if selected is None:
                        available = ", ".join(option["label"] for option in options) or "none"
                        raise ValueError(
                            f"requested scope '{self.requested_scope}' was not found; "
                            f"available evidence scopes: {available}"
                        )
                    selected_scope = {
                        "mode": "domain",
                        "label": selected["label"],
                        "source": "cli",
                        "supporting_paper_ids": selected["paper_ids"],
                    }
                else:
                    selected_scope = {
                        "mode": "broad",
                        "label": "Broad topic",
                        "source": "default",
                        "supporting_paper_ids": [],
                    }
                context["research_scope"] = ResearchScope.model_validate(
                    selected_scope
                ).model_dump()
            for status_key in ("source_statuses", "novelty_source_statuses"):
                if status_key not in result_data:
                    continue
                failures = [
                    status
                    for status in result_data[status_key]
                    if status.get("status") == "error" or status.get("failed_query_count", 0)
                ]
                if failures:
                    summaries = []
                    for status in failures:
                        skipped = int(status.get("skipped_query_count", 0) or 0)
                        count_key = "planned_query_count" if skipped else "attempted_query_count"
                        total = status.get(count_key, status.get("query_count", 1))
                        label = "planned" if skipped else "attempted"
                        failed = status.get(
                            "failed_query_count", int(status.get("status") == "error")
                        )
                        item = (
                            f"{status.get('provider', 'unknown')}: "
                            f"{failed}/{total} {label} queries failed"
                        )
                        if skipped:
                            item += f"; {skipped} skipped"
                        summaries.append(item)
                    summary = ", ".join(summaries)
                    message = f"{step.name}: literature provider issues: {summary}"
                    context.setdefault("_warnings", []).append(message)
                    self._emit("warning", message, step_name=step.name)
            if "top_papers" in result_data:
                count = len(result_data["top_papers"])
                covered_queries = {
                    query
                    for paper in result_data["top_papers"]
                    for query in paper.get("matched_queries", [])
                }
                failures = []
                if count < self.minimum_papers:
                    failures.append(
                        f"only {count} unique relevant papers were found; "
                        f"at least {self.minimum_papers} are required"
                    )
                admission = context.get("evidence_admission", {})
                if admission and admission.get("core_count", 0) < admission.get("minimum_core", 0):
                    failures.append(
                        f"only {admission.get('core_count', 0)} direct or transferable papers "
                        f"were found; at least {admission.get('minimum_core', 0)} are required"
                    )
                expected_queries = context.get("literature_queries", [])
                if len(expected_queries) >= 2 and len(covered_queries) < 2:
                    failures.append("literature evidence covered fewer than two query facets")
                for message in failures:
                    if not self.allow_insufficient_literature:
                        context["_outcome"] = "insufficient_evidence"
                        context["research_outcome"] = {
                            "status": "insufficient_evidence",
                            "reason": message,
                        }
                    context.setdefault("_warnings", []).append(message)
                    self._emit("warning", message, step_name=step.name)
            event_data: dict[str, Any] = {}
            if "top_papers" in result_data:
                event_data["paper_count"] = len(result_data["top_papers"])
            if "evidence_extraction_metrics" in result_data:
                event_data["evidence_extraction"] = result_data["evidence_extraction_metrics"]
            if isinstance(step, LlmStep):
                metrics = context["_llm_metrics"][step.name]
                metric_items = metrics if isinstance(metrics, list) else [metrics]
                event_data["model_calls"] = sum(
                    int(item.get("request_count", 1)) for item in metric_items
                )
                event_data["generation_characters"] = sum(
                    int(item.get("generation_characters", 0)) for item in metric_items
                )
                event_data["output_tokens"] = sum(
                    int(item.get("eval_count", 0)) for item in metric_items
                )
            elapsed = round(time.monotonic() - step_started_at, 3)
            event_data["elapsed_seconds"] = elapsed
            context.setdefault("_step_metrics", []).append(
                {
                    "step": step.name,
                    "type": "model" if isinstance(step, LlmStep) else "tool",
                    "elapsed_seconds": elapsed,
                    "model_calls": event_data.get("model_calls", 0),
                    "generation_characters": event_data.get("generation_characters", 0),
                    "output_tokens": event_data.get("output_tokens", 0),
                }
            )
            self._emit(
                "step_completed",
                step_name=step.name,
                step_index=index,
                total_steps=len(steps),
                data=event_data,
            )
            context["_last_completed_step"] = step.name
            context.pop("_active_step", None)
            bounded_terminal = next(
                (
                    context.get(key)
                    for key in (
                        "research_outcome",
                        "candidate_pool_outcome",
                        "candidate_selection_outcome",
                    )
                    if isinstance(context.get(key), dict)
                    and context[key].get("status") in {"evidence_only", "insufficient_evidence"}
                ),
                None,
            )
            if bounded_terminal:
                outcome_status = str(bounded_terminal.get("status", "evidence_only"))
                context["_outcome"] = outcome_status
                context["research_outcome"] = {
                    "status": outcome_status,
                    "reason": bounded_terminal.get("reason"),
                }
                message = (
                    f"{bounded_terminal.get('reason')} Source backed evidence was preserved; "
                    "no incomplete model content was promoted to a proposal."
                )
                context.setdefault("_warnings", []).append(message)
                self._emit("warning", message, step_name=step.name)
                break
            if context.get("research_outcome", {}).get("status") == (
                "insufficient_valid_candidates"
            ):
                context["_outcome"] = "insufficient_valid_candidates"
                message = (
                    "No candidate passed the scientific validity and grounding gate; "
                    "preserved the source backed evidence without generating a proposal."
                )
                context.setdefault("_warnings", []).append(message)
                self._emit("warning", message, step_name=step.name)
                break
        context["_warnings"] = list(dict.fromkeys(context.get("_warnings", [])))
        context["_model_capabilities"] = build_capability_reports(
            self._validated_model_metadata,
            compilation_events=context.get("_compilation_events", []),
            stage_outcomes=context.get("_stage_outcomes", []),
        )
        context["_pipeline_duration_seconds"] = round(time.monotonic() - started, 3)
        if context.get("_outcome") not in {
            "evidence_only",
            "insufficient_evidence",
            "insufficient_valid_candidates",
        }:
            context["_outcome"] = "proposal_complete"
            if isinstance(context.get("research_outcome"), dict):
                context["research_outcome"]["status"] = context.get(
                    "audited_research_proposal", context.get("research_proposal", {})
                ).get("proposal_status", "unverified_research_direction")
        store = self._ensure_knowledge_store()
        if store is not None:
            try:
                run_id = store.ingest_run(str(context.get("topic", "")), context)
                context["_knowledge"] = {
                    **context["_knowledge"],
                    "run_id": run_id,
                    "status": store.status(),
                }
            except (OSError, ValueError, TypeError) as error:
                message = (
                    "Could not project the completed run into local knowledge; result output "
                    f"is unaffected ({type(error).__name__})."
                )
                context.setdefault("_warnings", []).append(message)
                self._emit("warning", message)
        self._emit(
            "pipeline_completed",
            f"{context['_pipeline_duration_seconds']:.1f}s",
            total_steps=len(steps),
            data={"duration_seconds": context["_pipeline_duration_seconds"]},
        )
        return context

    def close(self) -> None:
        for agent in self._agents.values():
            agent.close()
        self._agents.clear()
        if self._embedding_service is not None:
            self._embedding_service.close()
            self._embedding_service = None
        if self._knowledge_store is not None:
            self._knowledge_store.close()
            self._knowledge_store = None

    def persist_partial(self, context: dict[str, Any]) -> None:
        store = self._ensure_knowledge_store()
        if store is not None:
            store.ingest_run(
                str(context.get("topic", "")),
                context,
                status=str(context.get("_outcome", "interrupted")),
            )


# Compatibility exports preserve the former ``agent.agent`` API while all implementation lives
# in the modular pipeline package.
def slugify(text: str) -> str:
    return _pipeline_slugify(text)


def save_results(
    result_context: dict[str, Any],
    config: PipelineConfig,
    topic: str,
    output_dir: str | Path,
) -> tuple[Path, Path]:
    return _pipeline_save_results(result_context, config, topic, output_dir)


def save_failure_diagnostic(
    result_context: dict[str, Any],
    config: PipelineConfig,
    topic: str,
    output_dir: str | Path,
    error: str,
    *,
    status: str = "insufficient_evidence",
) -> Path:
    return _pipeline_save_failure_diagnostic(
        result_context, config, topic, output_dir, error, status=status
    )


def _is_resumable_context(context: dict[str, Any], config: PipelineConfig) -> bool:
    return _pipeline_is_resumable_context(context, config)


def load_resume_checkpoint(
    path: str | Path, config: PipelineConfig
) -> tuple[str, dict[str, Any], str]:
    return _pipeline_load_resume_checkpoint(path, config)


def _try_save_failure_diagnostic(
    result_context: dict[str, Any],
    config: PipelineConfig,
    topic: str,
    output_dir: str | Path,
    error: str,
    reporter: PipelineReporter,
    *,
    status: str,
) -> Path | None:
    try:
        return save_failure_diagnostic(
            result_context, config, topic, output_dir, error, status=status
        )
    except (OSError, TypeError, ValueError) as diagnostic_error:
        reporter.emit(
            PipelineEvent(
                "warning",
                f"could not save failure diagnostic: {type(diagnostic_error).__name__}",
            )
        )
        return None


def display_final_results(result_context: dict[str, Any], config: PipelineConfig) -> None:
    console.rule("[bold green]Final Results Summary[/bold green]")
    for key in config.final_outputs:
        value = result_context.get(key)
        title = key.replace("_", " ").title()
        if key == "top_papers" and isinstance(value, list):
            table = Table(title="Top Papers Used", show_lines=True, box=box.ROUNDED)
            for column in ("Title", "Score", "Year", "Sources", "URL"):
                table.add_column(column)
            for paper in value:
                table.add_row(
                    Text(str(paper.get("title", "N/A"))),
                    f"{paper.get('composite_score', 0):.2f}",
                    str(paper.get("year", "N/A")),
                    Text(", ".join(paper.get("sources", [paper.get("source", "N/A")]))),
                    Text(str(paper.get("url", "N/A"))),
                )
            console.print(table)
        elif value is not None:
            rendered = (
                json.dumps(value, indent=2, ensure_ascii=False)
                if isinstance(value, (dict, list))
                else str(value)
            )
            console.print(Panel(Text(rendered), title=title))


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not 0 < parsed <= 3600:
        raise argparse.ArgumentTypeError("must be between 1 and 3600")
    return parsed


def execute_run(
    args: argparse.Namespace,
    *,
    reporter: PipelineReporter | None = None,
    display_results: bool = True,
    scope_selector: Callable[[str, list[dict[str, Any]]], dict[str, Any]] | None = None,
) -> int:
    active_reporter = reporter or ConsoleReporter(console)
    try:
        config = load_pipeline_config()
    except ValueError as error:
        active_reporter.emit(PipelineEvent("error", str(error)))
        active_reporter.close()
        return EXIT_CONFIG
    resume_context: dict[str, Any] | None = None
    resume_step: str | None = None
    diagnostic = getattr(args, "diagnostic", None)
    if diagnostic:
        try:
            topic, resume_context, resume_step = load_resume_checkpoint(diagnostic, config)
            args.topic = topic
        except ValueError as error:
            active_reporter.emit(PipelineEvent("error", str(error)))
            active_reporter.close()
            return EXIT_CONFIG
    try:
        model_assignments = ModelAssignments.model_validate(
            getattr(args, "model_assignments", None)
            or (resume_context or {}).get("_pipeline_model_assignments")
            or {}
        )
        cli_overrides = {
            "all_model": getattr(args, "model", None),
            "planning_evidence": getattr(args, "planning_model", None),
            "creative": getattr(args, "creative_model", None),
            "critique": getattr(args, "critique_model", None),
            "writing": getattr(args, "writing_model", None),
        }
        model_assignments = model_assignments.model_copy(
            update={key: value for key, value in cli_overrides.items() if value}
        )
        manager = PipelineManager(
            config,
            ollama_host=args.ollama_host,
            ollama_timeout=args.ollama_timeout,
            minimum_papers=getattr(args, "minimum_papers", None)
            or (resume_context or {}).get("_minimum_papers"),
            allow_insufficient_literature=getattr(args, "allow_insufficient_literature", False)
            or bool((resume_context or {}).get("_insufficient_literature_override", False)),
            search_depth=getattr(args, "search_depth", None)
            or str((resume_context or {}).get("_search_depth", "standard")),
            reporter=active_reporter,
            model_assignments=model_assignments,
            requested_scope=getattr(args, "scope", None)
            or str((resume_context or {}).get("_requested_scope", "broad")),
            scope_selector=scope_selector,
            knowledge_enabled=getattr(args, "knowledge_enabled", True),
            knowledge_dir=getattr(args, "knowledge_dir", "knowledge"),
            embedding_model=getattr(args, "embedding_model", None),
            knowledge_refresh=getattr(args, "knowledge_refresh", "balanced"),
        )
    except ValueError as error:
        active_reporter.emit(PipelineEvent("error", f"invalid runtime configuration: {error}"))
        active_reporter.close()
        return EXIT_CONFIG
    cloud_models = sorted(model for model in manager.resolved_models if is_cloud_model(model))
    if cloud_models and not getattr(args, "allow_cloud_models", False):
        active_reporter.emit(
            PipelineEvent(
                "error",
                "cloud model(s) require explicit --allow-cloud-models consent: "
                + ", ".join(cloud_models),
            )
        )
        manager.close()
        active_reporter.close()
        return EXIT_CONFIG
    try:
        public_host, remote_host = parse_ollama_host(manager.ollama_host)
    except ValueError as error:
        active_reporter.emit(PipelineEvent("error", f"invalid Ollama host: {error}"))
        manager.close()
        active_reporter.close()
        return EXIT_CONFIG
    if remote_host:
        active_reporter.emit(
            PipelineEvent(
                "warning",
                f"Ollama host {public_host} is remote; prompts may leave this machine",
            )
        )
    try:
        active_reporter.emit(
            PipelineEvent(
                "preflight_started",
                "Validating selected Ollama models",
                data={"models": sorted(manager.resolved_models)},
            )
        )
        manager.preflight()
        active_reporter.emit(
            PipelineEvent(
                "preflight_completed", "Basic model transport and response mode checks completed"
            )
        )
        initial_context = resume_context or {
            "topic": args.topic,
            "_pipeline_start_time_iso": datetime.now().isoformat(),
        }
        context = manager.run_pipeline(initial_context, start_step=resume_step)
        if display_results:
            display_final_results(context, config)
        text_path, json_path = save_results(context, config, args.topic, args.output_dir)
        active_reporter.emit(
            PipelineEvent(
                "saved",
                data={"text_path": str(text_path), "json_path": str(json_path)},
            )
        )
        return EXIT_SUCCESS
    except InsufficientEvidenceError as error:
        diagnostic_path = _try_save_failure_diagnostic(
            manager.last_context,
            config,
            args.topic,
            args.output_dir,
            str(error),
            active_reporter,
            status="insufficient_evidence",
        )
        recoverable = bool(diagnostic_path) and _is_resumable_context(manager.last_context, config)
        active_reporter.emit(
            PipelineEvent(
                "error",
                str(error),
                data={
                    **({"diagnostic_path": str(diagnostic_path)} if diagnostic_path else {}),
                    "recoverable": recoverable,
                },
            )
        )
        return EXIT_EVIDENCE
    except ModelOutputError as error:
        manager.last_context["_outcome"] = "output_contract_failed"
        manager.last_context["_model_limitation"] = str(error)
        diagnostic_path = _try_save_failure_diagnostic(
            manager.last_context,
            config,
            args.topic,
            args.output_dir,
            str(error),
            active_reporter,
            status="model_output_error",
        )
        recoverable = bool(diagnostic_path) and _is_resumable_context(manager.last_context, config)
        active_reporter.emit(
            PipelineEvent(
                "error",
                "A model response could not be compiled into the stage contract. " + str(error),
                data={
                    **({"diagnostic_path": str(diagnostic_path)} if diagnostic_path else {}),
                    "recoverable": recoverable,
                    "category": "model_output_error",
                },
            )
        )
        return EXIT_OLLAMA
    except OllamaRuntimeError as error:
        if manager.last_context.get("evidence_ledger"):
            partial_context = manager.last_context
            partial_context["_outcome"] = "interrupted"
            partial_context["_model_limitation"] = str(error)
            warning = (
                "Model execution was interrupted. Source backed evidence is preserved and the "
                "active stage can be resumed after checking the model or connection."
            )
            partial_context.setdefault("_warnings", []).append(warning)
            diagnostic_path: Path | None = None
            try:
                persist_partial = getattr(manager, "persist_partial", None)
                if callable(persist_partial):
                    persist_partial(partial_context)
                text_path, json_path = save_results(
                    partial_context, config, args.topic, args.output_dir
                )
                diagnostic_path = _try_save_failure_diagnostic(
                    partial_context,
                    config,
                    args.topic,
                    args.output_dir,
                    str(error),
                    active_reporter,
                    status="model_runtime_interrupted",
                )
            except OSError as output_error:
                active_reporter.emit(
                    PipelineEvent("error", f"partial output failure: {output_error}")
                )
                return EXIT_OUTPUT
            active_reporter.emit(PipelineEvent("warning", warning))
            active_reporter.emit(
                PipelineEvent(
                    "saved",
                    "Saved an honest partial report",
                    data={
                        "text_path": str(text_path),
                        "json_path": str(json_path),
                        "outcome": "interrupted",
                    },
                )
            )
            recoverable = bool(diagnostic_path) and _is_resumable_context(partial_context, config)
            active_reporter.emit(
                PipelineEvent(
                    "error",
                    str(error),
                    data={
                        **({"diagnostic_path": str(diagnostic_path)} if diagnostic_path else {}),
                        "recoverable": recoverable,
                    },
                )
            )
            return EXIT_OLLAMA
        diagnostic_path = _try_save_failure_diagnostic(
            manager.last_context,
            config,
            args.topic,
            args.output_dir,
            str(error),
            active_reporter,
            status="model_transport_error",
        )
        recoverable = bool(diagnostic_path) and _is_resumable_context(manager.last_context, config)
        active_reporter.emit(
            PipelineEvent(
                "error",
                str(error),
                data={
                    **({"diagnostic_path": str(diagnostic_path)} if diagnostic_path else {}),
                    "recoverable": recoverable,
                },
            )
        )
        return EXIT_OLLAMA
    except OSError as error:
        active_reporter.emit(PipelineEvent("error", f"output failure: {error}"))
        return EXIT_OUTPUT
    except KeyboardInterrupt:
        active_reporter.emit(PipelineEvent("cancelled", "research run cancelled"))
        return EXIT_CANCELLED
    except Exception as error:
        diagnostic_data: dict[str, Any] = {}
        try:
            diagnostic_path = _try_save_failure_diagnostic(
                manager.last_context,
                config,
                args.topic,
                args.output_dir,
                f"{type(error).__name__}: {error}",
                active_reporter,
                status="pipeline_error",
            )
            if diagnostic_path:
                resumable = _is_resumable_context(manager.last_context, config)
                diagnostic_data.update(
                    {"diagnostic_path": str(diagnostic_path), "recoverable": resumable}
                )
        except (TypeError, ValueError):
            pass
        active_step = manager.last_context.get("_active_step")
        location = f" at {active_step}" if active_step else ""
        active_reporter.emit(
            PipelineEvent(
                "error",
                f"pipeline failure{location}: {type(error).__name__}: {error}",
                data=diagnostic_data,
            )
        )
        if os.getenv("ABSTRACT_AGENT_DEBUG"):
            console.print(Text(traceback.format_exc(), style="dim"))
        return EXIT_SOURCE
    finally:
        manager.close()
        active_reporter.close()


def interactive() -> int:
    try:
        from .ui import InteractiveApplication
    except ImportError:  # pragma: no cover - direct script execution
        from ui import InteractiveApplication

    return InteractiveApplication(console=console).run()


def _add_runtime_arguments(parser: argparse.ArgumentParser, *, resume: bool = False) -> None:
    if resume:
        parser.add_argument("--diagnostic", required=True)
    else:
        parser.add_argument("--topic", required=True)
    parser.add_argument("--output-dir", default="agent/output")
    parser.add_argument("--ollama-host")
    parser.add_argument(
        "--ollama-timeout",
        "--ollama-inactivity-timeout",
        dest="ollama_timeout",
        type=_positive_float,
        help="maximum seconds without Ollama response data",
    )
    parser.add_argument("--minimum-papers", type=_positive_int)
    parser.add_argument(
        "--search-depth", choices=("standard", "deep"), default=None if resume else "standard"
    )
    parser.add_argument(
        "--scope",
        default=None if resume else "broad",
        help="stay broad by default or select an exact evidence scope label",
    )
    parser.add_argument("--allow-insufficient-literature", action="store_true")
    parser.add_argument(
        "--allow-cloud-models",
        action="store_true",
        help="allow selected Ollama cloud models to send prompts off device",
    )
    parser.add_argument("--model", help="use one Ollama model for every LLM stage")
    parser.add_argument("--planning-model")
    parser.add_argument("--creative-model")
    parser.add_argument("--critique-model")
    parser.add_argument("--writing-model")
    parser.add_argument("--knowledge-dir", default="knowledge")
    parser.add_argument("--embedding-model")
    parser.add_argument(
        "--knowledge-refresh",
        choices=("balanced", "always", "offline"),
        default="balanced",
    )
    parser.add_argument(
        "--no-knowledge",
        dest="knowledge_enabled",
        action="store_false",
        default=True,
        help="disable local knowledge recall and persistence for this run",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="abstract-agent")
    commands = parser.add_subparsers(dest="command")

    run = commands.add_parser("run", help="run the research pipeline")
    _add_runtime_arguments(run)

    resume = commands.add_parser("resume", help="resume a failed run from its active stage")
    _add_runtime_arguments(resume, resume=True)

    sources = commands.add_parser("sources", help="scholarly source operations")
    source_commands = sources.add_subparsers(dest="source_command", required=True)
    check = source_commands.add_parser("check", help="perform live source health checks")
    check.add_argument("--topic", default="machine learning")

    knowledge_parser = commands.add_parser("knowledge", help="local knowledge operations")
    knowledge_parser.add_argument("--knowledge-dir", default="knowledge")
    knowledge_commands = knowledge_parser.add_subparsers(dest="knowledge_command", required=True)
    knowledge_commands.add_parser("status")
    search = knowledge_commands.add_parser("search")
    search.add_argument("query")
    search.add_argument("--limit", type=_positive_int, default=10)
    knowledge_commands.add_parser("reindex")
    knowledge_commands.add_parser("export")
    import_command = knowledge_commands.add_parser("import")
    import_command.add_argument("path")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        return interactive()
    if args.command in {"run", "resume"}:
        return execute_run(args)
    if args.command == "sources" and args.source_command == "check":
        statuses = multi_source.check_sources(args.topic)
        console.print_json(data=statuses)
        return (
            EXIT_SUCCESS
            if any(item["status"] in {"ok", "cached"} for item in statuses)
            else EXIT_SOURCE
        )
    if args.command == "knowledge":
        with knowledge.KnowledgeStore(args.knowledge_dir) as store:
            if args.knowledge_command == "status":
                result = store.status()
            elif args.knowledge_command == "search":
                result = store.search_papers(args.query, args.limit)
            elif args.knowledge_command == "reindex":
                result = {"reindexed": store.reindex()}
            elif args.knowledge_command == "export":
                result = store.export_projections()
            else:
                result = store.import_projections(args.path)
        console.print_json(data=result)
        return EXIT_SUCCESS
    parser.error("unknown command")
    return EXIT_CONFIG


if __name__ == "__main__":
    raise SystemExit(main())
