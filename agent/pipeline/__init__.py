"""Pipeline configuration and execution support."""

from .config import load_config
from .config_loader import hydrate_prompt_files
from .context import PipelineState, prepare_pipeline_context
from .errors import (
    InsufficientEvidenceError,
    ModelOutputError,
    ModelRuntimeError,
    ModelTransportError,
    PipelineInvariantError,
)
from .executor import execute_tool_step, map_step_result
from .model_io import (
    build_capability_reports,
    consume_chat_stream,
    identity_scoped_format_schema,
    is_structural_compilation_failure,
    ollama_format_schema,
    slotted_relevance_format_schema,
)
from .outcomes import RunOutcome, StageOutcome, StageStatus, terminal_result
from .output import (
    CompiledOutput,
    ModelContractProfileStore,
    ModelOutputArtifact,
    build_prompt_packet,
    build_prompt_view,
    compile_model_output,
    escape_prompt_data,
    get_output_policy,
    parse_output,
    validate_compact_instance,
)
from .persistence import (
    is_resumable_context,
    load_resume_checkpoint,
    save_failure_diagnostic,
    save_results,
    slugify,
)
from .prompting import BudgetPolicy, PromptFit, PromptPlan, TokenEstimator, plan_prompt
from .recovery import DEFAULT_BOUNDED_RECOVERY, BoundedRecoveryPolicy
from .registry import resolve_tool, validate_tool_contracts

__all__ = [
    "InsufficientEvidenceError",
    "ModelOutputError",
    "ModelRuntimeError",
    "ModelTransportError",
    "PipelineInvariantError",
    "PipelineState",
    "hydrate_prompt_files",
    "load_config",
    "execute_tool_step",
    "map_step_result",
    "consume_chat_stream",
    "identity_scoped_format_schema",
    "is_structural_compilation_failure",
    "ollama_format_schema",
    "slotted_relevance_format_schema",
    "is_resumable_context",
    "load_resume_checkpoint",
    "save_failure_diagnostic",
    "save_results",
    "slugify",
    "prepare_pipeline_context",
    "resolve_tool",
    "validate_tool_contracts",
    "CompiledOutput",
    "ModelContractProfileStore",
    "ModelOutputArtifact",
    "build_prompt_packet",
    "build_capability_reports",
    "build_prompt_view",
    "compile_model_output",
    "escape_prompt_data",
    "get_output_policy",
    "parse_output",
    "validate_compact_instance",
    "BoundedRecoveryPolicy",
    "DEFAULT_BOUNDED_RECOVERY",
    "RunOutcome",
    "StageOutcome",
    "StageStatus",
    "terminal_result",
    "BudgetPolicy",
    "PromptFit",
    "PromptPlan",
    "TokenEstimator",
    "plan_prompt",
]
