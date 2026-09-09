"""Strict research domain output contracts."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from .base import StrictModel


class ResearchPlan(StrictModel):
    topic_interpretation: str = Field(min_length=1)
    assumptions: list[str] = Field(default_factory=list)
    required_concepts: list[str] = Field(min_length=1, max_length=4)
    mechanism_query: str = Field(min_length=3, max_length=160)
    approaches_query: str = Field(min_length=3, max_length=160)
    prior_art_query: str = Field(min_length=3, max_length=160)
    exclusions: list[str] = Field(default_factory=list)


class EvidenceClaim(StrictModel):
    claim: str = Field(min_length=1, max_length=360)
    source_passage: str = Field(min_length=1, max_length=360)
    source_start: int | None = Field(default=None, ge=0)
    source_end: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def validate_source_offsets(self) -> EvidenceClaim:
        if (self.source_start is None) != (self.source_end is None):
            raise ValueError("source_start and source_end must be supplied together")
        if self.source_start is not None and self.source_end <= self.source_start:
            raise ValueError("source_end must be greater than source_start")
        return self


EvidenceLimitation = Annotated[str, Field(min_length=1, max_length=300)]


class EvidenceProvenance(StrictModel):
    extraction_mode: Literal["legacy_model", "embedding_extractive", "lexical_extractive"] = (
        "legacy_model"
    )
    source_content_hash: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    source_field: Literal["summary", "title"] = "summary"
    embedding_model: str | None = Field(default=None, min_length=1, max_length=200)
    evidence_contract_version: str = Field(default="legacy", min_length=1, max_length=32)
    reused: bool = False


class EvidenceCard(StrictModel):
    paper_id: str = Field(min_length=1)
    findings: list[EvidenceClaim] = Field(min_length=1, max_length=2)
    method: str = Field(default="Not reported", max_length=360)
    limitations: list[EvidenceLimitation] = Field(default_factory=list, max_length=2)
    relevance: str = Field(min_length=1, max_length=240)
    provenance: EvidenceProvenance = Field(default_factory=EvidenceProvenance)


class EvidenceLedger(StrictModel):
    cards: list[EvidenceCard] = Field(min_length=1, max_length=20)


class GapItem(StrictModel):
    statement: str = Field(min_length=1, max_length=240)
    paper_ids: list[str] = Field(default_factory=list, max_length=4)
    status: Literal["established", "inferred", "unknown", "contradictory"]


class GapMap(StrictModel):
    established: list[GapItem] = Field(default_factory=list, max_length=5)
    contradictions: list[GapItem] = Field(default_factory=list, max_length=3)
    unresolved_questions: list[GapItem] = Field(default_factory=list, max_length=5)


GenerationMode = Literal[
    "structured_exact",
    "structured_normalized",
    "draft_projected",
    "conservative_fallback",
]


class SecondaryPrediction(StrictModel):
    outcome: str = Field(min_length=1, max_length=240)
    expected_relation: Literal[
        "increases",
        "decreases",
        "differs",
        "noninferior",
        "positive_association",
        "negative_association",
    ]


ExpectedRelation = Literal[
    "increases",
    "decreases",
    "differs",
    "noninferior",
    "positive_association",
    "negative_association",
]


class OperationalDefinition(StrictModel):
    """A named exposure/intervention plus the operation that makes it reproducible."""

    name: str = Field(min_length=1, max_length=160)
    operational_definition: str = Field(min_length=8, max_length=360)


class OutcomeSpecification(StrictModel):
    """Atomic outcome fields rendered into prose only by the application."""

    name: str = Field(min_length=1, max_length=160)
    metric: str = Field(min_length=1, max_length=160)
    expected_relation: ExpectedRelation
    decision_rule: str = Field(min_length=8, max_length=300)


class MechanismSpecification(StrictModel):
    cause: str = Field(min_length=1, max_length=240)
    mediator: str = Field(min_length=1, max_length=240)
    measurable_consequence: str = Field(min_length=1, max_length=240)


class ExperimentSpecification(StrictModel):
    units: str = Field(min_length=1, max_length=240)
    procedure: str = Field(min_length=8, max_length=600)
    controls: list[str] = Field(min_length=1, max_length=4)


class ScientificHypothesisSpec(StrictModel):
    """v5 scientific IR kept separate from the stable rendered candidate contract."""

    title: str = Field(min_length=1, max_length=160)
    study_design: Literal["experimental", "observational", "measurement"]
    system: str = Field(min_length=1, max_length=240)
    exposure_or_intervention: OperationalDefinition
    comparator: OperationalDefinition
    primary_outcome: OutcomeSpecification
    secondary_outcomes: list[OutcomeSpecification] = Field(default_factory=list, max_length=2)
    mechanism: MechanismSpecification
    novelty_baseline: str = Field(min_length=1, max_length=240)
    novelty_difference: str = Field(min_length=1, max_length=300)
    experiment: ExperimentSpecification
    evidence_slots: list[str] = Field(min_length=1, max_length=3)
    confounders: list[str] = Field(min_length=1, max_length=4)
    risks: list[str] = Field(min_length=1, max_length=4)
    potential_insights: list[str] = Field(min_length=1, max_length=4)
    addressed_finding_ids: list[str] = Field(default_factory=list, max_length=12)


class CandidateKernel(StrictModel):
    """Small causal comparison contract with application owned identity and citations."""

    title: str = Field(min_length=1, max_length=160)
    study_design: Literal["experimental", "observational", "measurement"]
    system: str = Field(min_length=1, max_length=240)
    exposure_or_intervention: OperationalDefinition
    comparator: OperationalDefinition
    primary_outcome: OutcomeSpecification
    evidence_slots: list[str] = Field(min_length=1, max_length=2)


class CandidateProtocol(StrictModel):
    """Small model owned mechanism and experiment contract for one validated kernel."""

    secondary_outcomes: list[OutcomeSpecification] = Field(default_factory=list, max_length=2)
    mechanism: MechanismSpecification
    novelty_baseline: str = Field(min_length=1, max_length=240)
    novelty_difference: str = Field(min_length=1, max_length=300)
    experiment: ExperimentSpecification
    confounders: list[str] = Field(min_length=1, max_length=4)
    risks: list[str] = Field(min_length=1, max_length=4)
    potential_insights: list[str] = Field(min_length=1, max_length=3)


class CritiqueFindingDraft(StrictModel):
    category: Literal[
        "evidence",
        "internal_consistency",
        "intervention",
        "outcome",
        "falsifiability",
        "mechanism",
        "confounding",
        "feasibility",
        "clarity",
    ]
    severity: Literal["minor", "major", "fatal"]
    issue: str = Field(min_length=1, max_length=360)
    required_revision: str = Field(min_length=1, max_length=360)


class CandidateCritiqueDraft(StrictModel):
    """Identity free critic response for exactly one application owned candidate packet."""

    grounding: int = Field(ge=1, le=5)
    falsifiability: int = Field(ge=1, le=5)
    feasibility: int = Field(ge=1, le=5)
    clarity: int = Field(ge=1, le=5)
    verdict: Literal["pass", "revise", "exclude"]
    findings: list[CritiqueFindingDraft] = Field(default_factory=list, max_length=4)

    @model_validator(mode="after")
    def findings_match_verdict(self) -> CandidateCritiqueDraft:
        if self.verdict == "pass" and self.findings:
            raise ValueError("passing critiques must not contain findings")
        if self.verdict != "pass" and not self.findings:
            raise ValueError("revise/exclude critiques require at least one finding")
        return self


class HypothesisSpec(StrictModel):
    """Legacy v4 scientific IR retained for saved result and checkpoint compatibility."""

    title: str = Field(min_length=1, max_length=160)
    study_design: Literal["experimental", "observational", "measurement"]
    system: str = Field(min_length=1, max_length=240)
    exposure_or_intervention: str = Field(min_length=1, max_length=300)
    comparator: str = Field(min_length=1, max_length=240)
    primary_outcome: str = Field(min_length=1, max_length=240)
    expected_relation: Literal[
        "increases",
        "decreases",
        "differs",
        "noninferior",
        "positive_association",
        "negative_association",
    ]
    mechanism: str = Field(min_length=1, max_length=500)
    novelty_delta: str = Field(min_length=1, max_length=300)
    experiment: str = Field(min_length=1, max_length=600)
    evidence_slots: list[str] = Field(min_length=1, max_length=3)
    confounders: list[str] = Field(min_length=1, max_length=4)
    risks: list[str] = Field(min_length=1, max_length=4)
    secondary_predictions: list[SecondaryPrediction] = Field(default_factory=list, max_length=3)
    potential_insights: list[str] = Field(min_length=1, max_length=4)


class GenerationProvenance(StrictModel):
    """Application owned record of how model authored content reached its contract."""

    mode: GenerationMode = "structured_exact"
    model: str | None = Field(default=None, min_length=1, max_length=200)
    validation_attempts: int = Field(default=1, ge=0, le=20)
    human_review_required: bool = False
    application_enriched: bool = False
    limitations: list[str] = Field(default_factory=list, max_length=8)
    route: (
        Literal["content_schema", "content_json", "tagged_text", "deterministic_fallback"] | None
    ) = None
    compilation_outcome: Literal["exact", "normalized", "projected", "fallback"] | None = None
    prompt_id: str | None = Field(default=None, min_length=1, max_length=120)
    prompt_version: str | None = Field(default=None, min_length=1, max_length=32)
    prompt_hash: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    contract_version: str | None = Field(default=None, min_length=1, max_length=32)
    compilation_issues: list[str] = Field(default_factory=list, max_length=12)
    requested_route: str | None = Field(default=None, min_length=1, max_length=40)
    effective_route: str | None = Field(default=None, min_length=1, max_length=40)
    contract_integrity: Literal["exact", "normalized", "projected", "fallback"] | None = None
    scientific_integrity: Literal["pass", "revisable", "fail", "not_audited"] = "not_audited"
    configured_axis: str | None = Field(default=None, min_length=1, max_length=120)
    effective_axis: str | None = Field(default=None, min_length=1, max_length=120)
    axis_substitution_reason: str | None = Field(default=None, min_length=1, max_length=300)


class GroundingLink(StrictModel):
    """Deterministic link between a generated direction and source backed evidence."""

    paper_id: str = Field(min_length=1)
    evidence_role: Literal["direct_evidence", "transferable_method"]
    alignment_method: Literal["embedding", "lexical", "application_fallback"]
    alignment_score: float = Field(ge=0, le=1)
    source_passage: str = Field(min_length=1, max_length=360)


class HypothesisCandidate(StrictModel):
    candidate_id: str = Field(min_length=1)
    title: str = Field(min_length=1, max_length=240)
    hypothesis: str = Field(min_length=1, max_length=1000)
    mechanism: str = Field(min_length=1, max_length=1000)
    novelty_delta: str = Field(min_length=1, max_length=600)
    supporting_paper_ids: list[str] = Field(min_length=1, max_length=8)
    conflicting_paper_ids: list[str] = Field(default_factory=list, max_length=8)
    predictions: list[str] = Field(min_length=1, max_length=6)
    falsifier: str = Field(min_length=1, max_length=800)
    proposed_experiment: str = Field(min_length=1, max_length=1200)
    risks: list[str] = Field(default_factory=list, max_length=8)
    novel_insights: list[str] = Field(min_length=1, max_length=6)
    applicability: Literal["broad", "cross_domain", "domain_specific"]
    target_domain: str | None = Field(default=None, max_length=100)
    grounding_links: list[GroundingLink] = Field(default_factory=list, max_length=8)
    scientific_spec: ScientificHypothesisSpec | None = None
    generation_provenance: GenerationProvenance = Field(default_factory=GenerationProvenance)

    @model_validator(mode="after")
    def validate_applicability(self) -> HypothesisCandidate:
        if self.applicability == "domain_specific" and not self.target_domain:
            raise ValueError("domain specific candidates require target_domain")
        if self.applicability != "domain_specific" and self.target_domain:
            raise ValueError("only domain specific candidates may set target_domain")
        return self


class CandidateScore(StrictModel):
    candidate_id: str = Field(min_length=1)
    grounding: int = Field(ge=1, le=5)
    novelty: int = Field(ge=1, le=5)
    falsifiability: int = Field(ge=1, le=5)
    feasibility: int = Field(ge=1, le=5)
    clarity: int = Field(ge=1, le=5)
    issues: list[str] = Field(default_factory=list, max_length=6)
    required_revisions: list[str] = Field(default_factory=list, max_length=6)
    score_source: Literal["model", "recovered", "fallback"] = "model"
    verdict: Literal["pass", "revise", "exclude"] = "revise"
    findings: list[CritiqueFinding] = Field(default_factory=list, max_length=8)
    fatal_flags: list[
        Literal[
            "internal_inconsistency",
            "undefined_intervention",
            "undefined_outcome",
            "non_falsifiable",
            "unsupported_mechanism",
        ]
    ] = Field(default_factory=list, max_length=5)


class CritiqueFinding(StrictModel):
    finding_id: str | None = Field(default=None, min_length=1, max_length=80)
    source: Literal["deterministic", "evidence_critic", "methodology_critic", "legacy"] = "legacy"
    category: Literal[
        "evidence",
        "novelty",
        "internal_consistency",
        "intervention",
        "outcome",
        "falsifiability",
        "mechanism",
        "confounding",
        "feasibility",
        "clarity",
    ]
    severity: Literal["minor", "major", "fatal"]
    issue: str = Field(min_length=1, max_length=360)
    required_revision: str = Field(min_length=1, max_length=360)


class ScientificFinding(StrictModel):
    finding_id: str = Field(min_length=1, max_length=80)
    source: Literal["deterministic", "evidence_critic", "methodology_critic", "legacy"]
    category: Literal[
        "evidence",
        "novelty",
        "internal_consistency",
        "intervention",
        "outcome",
        "falsifiability",
        "mechanism",
        "confounding",
        "feasibility",
        "clarity",
    ]
    severity: Literal["minor", "major", "fatal"]
    field_paths: list[str] = Field(default_factory=list, max_length=6)
    issue: str = Field(min_length=1, max_length=360)
    required_revision: str = Field(min_length=1, max_length=360)


class ScientificValidityAudit(StrictModel):
    candidate_id: str = Field(min_length=1)
    status: Literal["pass", "revisable", "fail"]
    findings: list[ScientificFinding] = Field(default_factory=list, max_length=20)
    grounding_tier: Literal["mechanism_aligned", "contextual", "weak", "insufficient"] = (
        "insufficient"
    )
    grounding_method: Literal["embedding", "lexical", "application_fallback"] = (
        "application_fallback"
    )
    contextual_link_count: int = Field(default=0, ge=0, le=8)
    mechanism_link_count: int = Field(default=0, ge=0, le=8)
    checked_contract_version: str = "5"


class CandidateCritiques(StrictModel):
    critiques: list[CandidateScore] = Field(min_length=1, max_length=6)


class PaperRelevanceAssessment(StrictModel):
    paper_id: str = Field(min_length=1)
    score: int = Field(ge=1, le=5)
    role: Literal["direct_evidence", "transferable_method", "background", "irrelevant"]
    rationale: str = Field(min_length=1, max_length=240)


class PaperRelevanceAssessments(StrictModel):
    assessments: list[PaperRelevanceAssessment] = Field(max_length=30)


class PaperRelevanceJudgement(StrictModel):
    score: int = Field(ge=1, le=5)
    role: Literal["direct_evidence", "transferable_method", "background", "irrelevant"]
    rationale: str = Field(min_length=1, max_length=240)


class SlottedPaperRelevanceAssessments(StrictModel):
    assessments: dict[str, PaperRelevanceJudgement] = Field(max_length=15)


class EvidenceScopeOption(StrictModel):
    scope_id: str = Field(pattern=r"^scope_[1-4]$")
    label: str = Field(min_length=2, max_length=80)
    description: str = Field(min_length=3, max_length=240)
    paper_ids: list[str] = Field(min_length=2, max_length=12)


class EvidenceScopeOptions(StrictModel):
    options: list[EvidenceScopeOption] = Field(default_factory=list, max_length=4)


class ResearchScope(StrictModel):
    mode: Literal["broad", "domain"] = "broad"
    label: str = Field(default="Broad topic", min_length=2, max_length=100)
    source: Literal["default", "interactive", "cli"] = "default"
    supporting_paper_ids: list[str] = Field(default_factory=list, max_length=12)


class NoveltyVerification(StrictModel):
    candidate_id: str = Field(min_length=1)
    status: Literal["novelty_supported", "prior_art_overlap", "uncertain"]
    closest_prior_art_paper_ids: list[str] = Field(default_factory=list, max_length=8)
    materially_matching_paper_ids: list[str] = Field(default_factory=list, max_length=8)
    reasoning: str = Field(min_length=1, max_length=1000)
    limitations: list[str] = Field(default_factory=list, max_length=6)

    @model_validator(mode="after")
    def validate_novelty_evidence(self) -> NoveltyVerification:
        if self.status != "uncertain" and not self.closest_prior_art_paper_ids:
            raise ValueError(
                f"novelty status '{self.status}' requires cited closest prior art; "
                "use 'uncertain' when evidence is inconclusive"
            )
        matching = set(self.materially_matching_paper_ids)
        closest = set(self.closest_prior_art_paper_ids)
        if not matching <= closest:
            raise ValueError("materially matching papers must be included in closest prior art")
        if self.status == "prior_art_overlap" and not matching:
            raise ValueError("prior_art_overlap requires materially matching prior art")
        if self.status == "novelty_supported" and matching:
            raise ValueError("novelty_supported cannot include materially matching prior art")
        return self


class NoveltyDecisionDraft(StrictModel):
    """Identity free novelty assessment for one application owned candidate packet."""

    status: Literal["novelty_supported", "prior_art_overlap", "uncertain"]
    closest_prior_art_paper_ids: list[str] = Field(default_factory=list, max_length=6)
    materially_matching_paper_ids: list[str] = Field(default_factory=list, max_length=6)
    reasoning: str = Field(min_length=1, max_length=800)
    limitations: list[str] = Field(default_factory=list, max_length=4)

    @model_validator(mode="after")
    def validate_novelty_evidence(self) -> NoveltyDecisionDraft:
        closest = set(self.closest_prior_art_paper_ids)
        matching = set(self.materially_matching_paper_ids)
        if self.status != "uncertain" and not closest:
            raise ValueError("non uncertain novelty decisions require cited prior art")
        if not matching <= closest:
            raise ValueError("material matches must also be listed as closest prior art")
        if self.status == "prior_art_overlap" and not matching:
            raise ValueError("prior_art_overlap requires a material match")
        if self.status == "novelty_supported" and matching:
            raise ValueError("novelty_supported cannot include a material match")
        return self


class NoveltyVerifications(StrictModel):
    verifications: list[NoveltyVerification] = Field(min_length=1, max_length=4)


class ClaimEvidenceLink(StrictModel):
    claim_id: str | None = Field(default=None, min_length=1, max_length=80)
    claim_kind: Literal["hypothesis", "prediction", "context"] | None = None
    claim: str = Field(min_length=1, max_length=500)
    paper_ids: list[str] = Field(default_factory=list, max_length=8)
    relationship: Literal["supports", "conflicts", "context", "proposed_inference"]
    confidence: Literal["low", "medium", "high"]

    @field_validator("paper_ids")
    @classmethod
    def unique_paper_ids(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("paper_ids must be unique")
        return value


class ResearchProposal(StrictModel):
    proposal_status: Literal["novelty_supported", "unverified_research_direction"]
    title: str = Field(min_length=1, max_length=240)
    hypothesis: str = Field(min_length=1, max_length=1200)
    rationale: str = Field(min_length=1, max_length=1600)
    proposed_method: str = Field(min_length=1, max_length=2000)
    predictions: list[str] = Field(min_length=1, max_length=8)
    falsifier: str = Field(min_length=1, max_length=1000)
    novel_insights: list[str] = Field(min_length=1, max_length=8)
    limitations_and_risks: list[str] = Field(default_factory=list, max_length=10)
    claim_evidence: list[ClaimEvidenceLink] = Field(min_length=1, max_length=16)
    abstract: str = Field(min_length=1, max_length=2400)
    references: list[str] = Field(min_length=1, max_length=30)
    generation_provenance: GenerationProvenance = Field(default_factory=GenerationProvenance)

    @field_validator("references")
    @classmethod
    def unique_references(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("references must be unique")
        return value


RESPONSE_SCHEMAS: dict[str, type[BaseModel]] = {
    "research_plan": ResearchPlan,
    "evidence_ledger": EvidenceLedger,
    "gap_map": GapMap,
    "hypothesis_candidate": HypothesisCandidate,
    "candidate_kernel": CandidateKernel,
    "candidate_protocol": CandidateProtocol,
    "candidate_critique_draft": CandidateCritiqueDraft,
    "novelty_decision_draft": NoveltyDecisionDraft,
    "hypothesis_spec": HypothesisSpec,
    "candidate_critiques": CandidateCritiques,
    "paper_relevance_assessments": PaperRelevanceAssessments,
    "slotted_paper_relevance_assessments": SlottedPaperRelevanceAssessments,
    "evidence_scope_options": EvidenceScopeOptions,
    "novelty_verifications": NoveltyVerifications,
    "research_proposal": ResearchProposal,
}

__all__ = [name for name in globals() if not name.startswith("_")]
