"""Stage specific scientific policies used by the generic pipeline executor."""

from .candidate_units import (
    audit_candidate_pool,
    compile_candidate_critiques,
    compile_candidate_units,
    prepare_all_candidate_critique_packets,
    prepare_candidate_kernel_packets,
    prepare_candidate_protocol_packets,
    select_candidate_v6,
)
from .critiques import normalize_critique_item, recover_critiques
from .hypotheses import (
    HypothesisAxisResolution,
    project_hypothesis_candidate,
    resolve_hypothesis_axis,
)
from .novelty import compile_novelty_decisions, normalize_novelty_item, recover_novelty
from .planning import normalize_research_plan, recover_research_plan
from .proposal import build_conservative_proposal
from .scopes_and_gaps import recover_gap_map, recover_scopes
from .validity import (
    audit_hypothesis_candidates,
    finalize_revision_candidates,
    gate_revised_candidate,
    initialize_revision_candidate,
    prepare_revision_repair_packets,
)

__all__ = [
    "build_conservative_proposal",
    "audit_candidate_pool",
    "compile_candidate_critiques",
    "compile_candidate_units",
    "compile_novelty_decisions",
    "prepare_all_candidate_critique_packets",
    "prepare_candidate_kernel_packets",
    "prepare_candidate_protocol_packets",
    "select_candidate_v6",
    "HypothesisAxisResolution",
    "audit_hypothesis_candidates",
    "finalize_revision_candidates",
    "gate_revised_candidate",
    "initialize_revision_candidate",
    "prepare_revision_repair_packets",
    "project_hypothesis_candidate",
    "resolve_hypothesis_axis",
    "recover_critiques",
    "normalize_critique_item",
    "recover_gap_map",
    "recover_novelty",
    "normalize_novelty_item",
    "normalize_research_plan",
    "recover_research_plan",
    "recover_scopes",
]
