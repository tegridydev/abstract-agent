# ruff: noqa: F401
"""Backward compatible facade for typed application contracts."""

try:
    from .contracts.base import StrictModel
    from .contracts.base import validate_ollama_host as _validate_ollama_host
    from .contracts.models import InstalledModel, ModelAssignments, ModelProbeResponse, ModelRole
    from .contracts.pipeline import (
        BaseStep,
        InteractiveSettings,
        LlmStep,
        PipelineConfig,
        PipelineStep,
        ToolStep,
    )
    from .contracts.research import (
        RESPONSE_SCHEMAS,
        CandidateCritiques,
        CandidateScore,
        ClaimEvidenceLink,
        EvidenceCard,
        EvidenceClaim,
        EvidenceLedger,
        EvidenceLimitation,
        EvidenceScopeOption,
        EvidenceScopeOptions,
        GapItem,
        GapMap,
        GenerationProvenance,
        GroundingLink,
        HypothesisCandidate,
        NoveltyVerification,
        NoveltyVerifications,
        PaperRelevanceAssessment,
        PaperRelevanceAssessments,
        PaperRelevanceJudgement,
        ResearchPlan,
        ResearchProposal,
        ResearchScope,
        SlottedPaperRelevanceAssessments,
    )
    from .contracts.sources import SourceStatus
except ImportError:  # pragma: no cover - supports direct script execution
    from contracts.base import StrictModel
    from contracts.base import validate_ollama_host as _validate_ollama_host
    from contracts.models import InstalledModel, ModelAssignments, ModelProbeResponse, ModelRole
    from contracts.pipeline import (
        BaseStep,
        InteractiveSettings,
        LlmStep,
        PipelineConfig,
        PipelineStep,
        ToolStep,
    )
    from contracts.research import (
        RESPONSE_SCHEMAS,
        CandidateCritiques,
        CandidateScore,
        ClaimEvidenceLink,
        EvidenceCard,
        EvidenceClaim,
        EvidenceLedger,
        EvidenceLimitation,
        EvidenceScopeOption,
        EvidenceScopeOptions,
        GapItem,
        GapMap,
        GenerationProvenance,
        GroundingLink,
        HypothesisCandidate,
        NoveltyVerification,
        NoveltyVerifications,
        PaperRelevanceAssessment,
        PaperRelevanceAssessments,
        PaperRelevanceJudgement,
        ResearchPlan,
        ResearchProposal,
        ResearchScope,
        SlottedPaperRelevanceAssessments,
    )
    from contracts.sources import SourceStatus

__all__ = [name for name in globals() if not name.startswith("__")]
