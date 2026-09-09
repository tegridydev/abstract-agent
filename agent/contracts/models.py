"""Ollama model selection and capability contracts."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from .base import StrictModel

ModelRole = Literal["general", "planning_evidence", "creative", "critique", "writing"]


class ModelAssignments(StrictModel):
    all_model: str | None = Field(default=None, min_length=1)
    planning_evidence: str | None = Field(default=None, min_length=1)
    creative: str | None = Field(default=None, min_length=1)
    critique: str | None = Field(default=None, min_length=1)
    writing: str | None = Field(default=None, min_length=1)

    def resolve(self, role: ModelRole, configured_model: str) -> str:
        role_model = getattr(self, role, None) if role != "general" else None
        return role_model or self.all_model or configured_model


class InstalledModel(StrictModel):
    name: str = Field(min_length=1)
    digest: str = ""
    size: int = Field(default=0, ge=0)
    parameter_size: str = "Unknown"
    quantization: str = "Unknown"
    family: str = "Unknown"
    modified_at: str = ""
    capabilities: list[str] = Field(default_factory=list)
    context_length: int | None = Field(default=None, ge=1)
    allocated_context_length: int | None = Field(default=None, ge=1)
    running: bool = False
    reasoning_mode: Literal["not_applicable", "disabled", "low", "enabled"] | None = None
    structured_output_mode: Literal["schema", "json", "prompt"] | None = None
    contract_tier: Literal["basic_schema", "basic_json", "text_projection"] | None = None
    probe_warning: str | None = None


class ModelProbeResponse(StrictModel):
    status: Literal["ok"]
