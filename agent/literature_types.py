"""Strict boundary types for scholarly search and provider records."""

from typing import Any, Literal

from pydantic import Field, model_validator

from .schemas import StrictModel


class SearchQuery(StrictModel):
    query_id: str = Field(min_length=1, max_length=160)
    text: str = Field(min_length=1, max_length=500)
    phase: Literal["literature", "novelty"] = "literature"
    candidate_id: str | None = Field(default=None, max_length=160)

    @model_validator(mode="after")
    def validate_candidate(self) -> "SearchQuery":
        if self.phase == "novelty" and not self.candidate_id:
            raise ValueError("novelty queries require candidate_id")
        if self.phase == "literature" and self.candidate_id is not None:
            raise ValueError("literature queries cannot have candidate_id")
        return self


class PaperRecord(StrictModel):
    provider: str = Field(min_length=1, max_length=80)
    provider_id: str | None = None
    doi: str | None = None
    pmid: str | None = None
    pmcid: str | None = None
    arxiv_id: str | None = None
    title: str = Field(min_length=1)
    authors: str = ""
    year: int | str | None = None
    summary: str = Field(min_length=1)
    url: str | None = None
    citations: int = Field(default=0, ge=0)
    source: str = Field(min_length=1)
    paper_id: str | None = None
    sources: list[str] = Field(default_factory=list)
    matched_queries: list[str] = Field(default_factory=list)
    matched_query_ids: list[str] = Field(default_factory=list)
    matched_candidate_ids: list[str] = Field(default_factory=list)
    relevance_score: float | None = None
    recency_score: float | None = None
    citation_score: float | None = None
    composite_score: float | None = None
    selection_score: float | None = Field(default=None, ge=0, le=1)
    best_matching_query_id: str | None = None
    relevance_judgement: dict[str, Any] | None = None
