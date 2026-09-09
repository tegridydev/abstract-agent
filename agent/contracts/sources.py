"""Literature provider status contracts."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from .base import StrictModel


class SourceStatus(StrictModel):
    provider: str
    status: Literal["ok", "partial", "empty", "error", "cached", "skipped"]
    result_count: int = Field(ge=0)
    duration_seconds: float = Field(ge=0)
    error_category: str | None = None
    message: str | None = None
    rate_limit: dict[str, str] = Field(default_factory=dict)
    query_count: int = Field(default=1, ge=0)
    planned_query_count: int = Field(default=1, ge=0)
    attempted_query_count: int = Field(default=1, ge=0)
    succeeded_query_count: int = Field(default=0, ge=0)
    skipped_query_count: int = Field(default=0, ge=0)
    circuit_state: Literal["closed", "open"] = "closed"
    circuit_reason: str | None = None
    failed_query_count: int = Field(default=0, ge=0)
    cached_query_count: int = Field(default=0, ge=0)
    warning_categories: list[str] = Field(default_factory=list)
    warning_query_count: int = Field(default=0, ge=0)
    retry_count: int = Field(default=0, ge=0)
    retry_categories: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_query_counts(self) -> SourceStatus:
        if self.failed_query_count > self.query_count:
            raise ValueError("failed_query_count cannot exceed query_count")
        if self.cached_query_count > self.query_count:
            raise ValueError("cached_query_count cannot exceed query_count")
        if self.warning_query_count > self.query_count:
            raise ValueError("warning_query_count cannot exceed query_count")
        if self.attempted_query_count > self.planned_query_count:
            raise ValueError("attempted_query_count cannot exceed planned_query_count")
        if self.succeeded_query_count > self.attempted_query_count:
            raise ValueError("succeeded_query_count cannot exceed attempted_query_count")
        if self.skipped_query_count > self.planned_query_count:
            raise ValueError("skipped_query_count cannot exceed planned_query_count")
        if self.attempted_query_count + self.skipped_query_count != self.planned_query_count:
            raise ValueError(
                "planned_query_count must equal attempted_query_count plus skipped_query_count"
            )
        # A partially successful request can legitimately count as both a
        # success and a failure (for example, usable cached/results data plus a
        # provider-side 429).  Each counter is bounded independently above.
        return self
