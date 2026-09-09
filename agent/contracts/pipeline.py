"""Pipeline topology and interactive preference contracts."""

from __future__ import annotations

from string import Formatter
from typing import Annotated, Any, Literal

from pydantic import Field, field_validator, model_validator

from .base import StrictModel, validate_ollama_host
from .models import ModelAssignments, ModelRole
from .research import RESPONSE_SCHEMAS


class BaseStep(StrictModel):
    name: str = Field(min_length=3)
    inputs: list[str] = Field(default_factory=list)
    outputs: list[str] = Field(default_factory=list)

    @field_validator("inputs", "outputs")
    @classmethod
    def unique_keys(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("context keys must be unique within a step")
        return value


class LlmStep(BaseStep):
    type: Literal["ollama_call"]
    model: str = Field(min_length=1)
    model_role: ModelRole = "general"
    persona: str | None = None
    system_prompt: str | None = None
    prompt_template: str = Field(min_length=10)
    prompt_id: str | None = Field(default=None, pattern=r"^[A-Za-z_][A-Za-z0-9_.-]*$")
    prompt_version: str = Field(default="1", min_length=1, max_length=32)
    prompt_view: str | None = Field(default=None, pattern=r"^[A-Za-z_][A-Za-z0-9_.-]*$")
    prompt_view_budget: int = Field(default=3_500, ge=1_024, le=16_000)
    temperature: float = Field(default=0.7, ge=0, le=2)
    max_output_tokens: int = Field(default=2048, ge=128, le=16384)
    repair_output_tokens: int | None = Field(default=None, ge=128, le=16384)
    validation_attempts: Literal[1, 2] = 1
    inactivity_timeout_seconds: float | None = Field(default=None, gt=0, le=3600)
    transport_failure: Literal["abort", "conservative_fallback"] = "abort"
    allow_partial_recovery: bool = False
    generation_strategy: Literal["direct_structured", "draft_conditioned"] = "direct_structured"
    invalid_output: Literal["abort", "recover"] = "abort"
    policy: str | None = Field(default=None, pattern=r"^[A-Za-z_][A-Za-z0-9_.-]*$")
    response_schema: str | None = None
    variants: list[str] = Field(default_factory=list)
    variant_key: str = Field(default="variant", pattern=r"^[A-Za-z_]\w*$")
    map_input: str | None = None
    map_item_key: str = Field(default="map_item", pattern=r"^[A-Za-z_]\w*$")
    item_failure: Literal["abort", "degrade"] = "abort"

    @model_validator(mode="after")
    def validate_llm_step(self) -> LlmStep:
        if len(self.outputs) != 1:
            raise ValueError("LLM steps must define exactly one output")
        allowed = set(self.inputs) | {"name", "persona"}
        if self.prompt_view:
            allowed.add("input_view")
        if self.variants:
            allowed.add(self.variant_key)
        if self.map_input:
            if self.map_input not in self.inputs:
                raise ValueError("map_input must be declared in step inputs")
            if self.variants:
                raise ValueError("LLM steps cannot combine variants and map_input")
            allowed.add(self.map_item_key)
        elif self.item_failure == "degrade":
            raise ValueError("item_failure='degrade' is only valid for mapped LLM steps")
        placeholders = {
            field_name
            for _, field_name, _, _ in Formatter().parse(self.prompt_template)
            if field_name
        }
        unknown = placeholders - allowed
        if unknown:
            raise ValueError(f"prompt placeholders are not declared inputs: {sorted(unknown)}")
        if self.response_schema and self.response_schema not in RESPONSE_SCHEMAS:
            raise ValueError(f"unknown response schema: {self.response_schema}")
        if (
            self.repair_output_tokens is not None
            and self.repair_output_tokens < self.max_output_tokens
        ):
            raise ValueError("repair_output_tokens cannot be below max_output_tokens")
        if self.transport_failure == "conservative_fallback" and self.invalid_output != "recover":
            raise ValueError("transport fallback requires invalid_output='recover'")
        if self.allow_partial_recovery and not self.response_schema:
            raise ValueError("partial recovery requires a response schema")
        if self.generation_strategy == "draft_conditioned" and not self.response_schema:
            raise ValueError("draft conditioned generation requires a response schema")
        if len(self.variants) != len(set(self.variants)):
            raise ValueError("LLM step variants must be unique")
        return self


class ToolStep(BaseStep):
    type: Literal["tool_call"]
    function: str = Field(pattern=r"^[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+$")
    params: dict[str, Any] = Field(default_factory=dict)
    input_bindings: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_input_bindings(self) -> ToolStep:
        invalid_names = [name for name in self.input_bindings if not name.isidentifier()]
        if invalid_names:
            raise ValueError(f"invalid function parameter names: {sorted(invalid_names)}")
        undeclared = set(self.input_bindings.values()) - set(self.inputs)
        if undeclared:
            raise ValueError(
                f"input bindings reference undeclared context keys: {sorted(undeclared)}"
            )
        collisions = set(self.input_bindings) & set(self.params)
        if collisions:
            raise ValueError(f"input bindings collide with params: {sorted(collisions)}")
        unbound_inputs = set(self.inputs) - set(self.input_bindings.values())
        duplicate_arguments = set(self.input_bindings) & unbound_inputs
        if duplicate_arguments:
            raise ValueError(
                f"input bindings duplicate implicit arguments: {sorted(duplicate_arguments)}"
            )
        return self

    def argument_sources(self) -> dict[str, str]:
        bound_context_keys = set(self.input_bindings.values())
        implicit = {key: key for key in self.inputs if key not in bound_context_keys}
        return implicit | self.input_bindings


PipelineStep = Annotated[LlmStep | ToolStep, Field(discriminator="type")]


class PipelineConfig(StrictModel):
    pipeline_name: str = Field(default="Research Hypothesis Generation", min_length=3)
    version: str | float = "5.0"
    ollama_host: str = "http://localhost:11434"
    ollama_timeout_seconds: float = Field(default=180, gt=0)
    minimum_unique_papers: int = Field(default=3, ge=1)
    context_init: list[str] = Field(default_factory=lambda: ["topic"])
    steps: list[PipelineStep] = Field(min_length=1)
    final_outputs: list[str] = Field(default_factory=list)

    _valid_ollama_host = field_validator("ollama_host")(validate_ollama_host)

    @model_validator(mode="after")
    def validate_data_flow(self) -> PipelineConfig:
        available = set(self.context_init)
        names: set[str] = set()
        for step in self.steps:
            if step.name in names:
                raise ValueError(f"duplicate step name: {step.name}")
            names.add(step.name)
            missing = set(step.inputs) - available
            if missing:
                raise ValueError(f"step '{step.name}' has unavailable inputs: {sorted(missing)}")
            duplicate_outputs = set(step.outputs) & available
            if duplicate_outputs:
                raise ValueError(
                    f"step '{step.name}' overwrites context keys: {sorted(duplicate_outputs)}"
                )
            available.update(step.outputs)
        missing_final = set(self.final_outputs) - available
        if missing_final:
            raise ValueError(f"final outputs are unavailable: {sorted(missing_final)}")
        return self


class InteractiveSettings(StrictModel):
    output_dir: str = Field(default="agent/output", min_length=1)
    minimum_papers: int = Field(default=6, ge=1, le=100)
    allow_insufficient_literature: bool = False
    history_page_size: int = Field(default=10, ge=5, le=50)
    confirm_before_run: bool = True
    search_depth: Literal["standard", "deep"] = "standard"
    ollama_host: str | None = None
    ollama_timeout: float | None = Field(default=None, gt=0, le=3600)
    knowledge_enabled: bool = True
    knowledge_dir: str = Field(default="knowledge", min_length=1)
    embedding_model: str | None = Field(default=None, min_length=1, max_length=200)
    knowledge_refresh: Literal["balanced", "always", "offline"] = "balanced"
    model_assignments: ModelAssignments = Field(default_factory=ModelAssignments)

    _valid_ollama_host = field_validator("ollama_host")(validate_ollama_host)
