"""Public model output compiler API."""

from .compiler import compile_model_output, validate_compact_instance
from .models import (
    CompilationIssue,
    CompilationTrace,
    CompiledOutput,
    ModelOutputArtifact,
)
from .packets import PromptPacket, build_prompt_packet, escape_prompt_data
from .parser import DuplicateKeyError, is_schema_echo, parse_output
from .policies import POLICIES, StageOutputPolicy, get_output_policy
from .profiles import ContractProfile, ModelContractProfileStore, profile_key
from .views import PromptView, build_prompt_view

__all__ = [
    "CompilationIssue",
    "CompilationTrace",
    "CompiledOutput",
    "ContractProfile",
    "DuplicateKeyError",
    "ModelContractProfileStore",
    "ModelOutputArtifact",
    "POLICIES",
    "PromptPacket",
    "PromptView",
    "StageOutputPolicy",
    "build_prompt_packet",
    "build_prompt_view",
    "compile_model_output",
    "escape_prompt_data",
    "get_output_policy",
    "is_schema_echo",
    "parse_output",
    "profile_key",
    "validate_compact_instance",
]
