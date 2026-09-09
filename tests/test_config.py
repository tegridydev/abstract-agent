import pytest

from agent.agent import _validate_tool_contracts, load_pipeline_config
from agent.schemas import PipelineConfig, ToolStep


def test_built_in_pipeline_is_valid_and_fixed() -> None:
    config = load_pipeline_config()

    assert config.pipeline_name == "Evidence-Centred Research Hypothesis Generation"
    assert str(config.version) == "6.0"
    assert config.minimum_unique_papers == 6
    assert config.ollama_timeout_seconds == 600
    assert len(config.steps) == 40
    assert config.steps[0].name == "Plan Research"
    assert config.steps[-1].name == "Audit Research Proposal"
    evidence_step = next(step for step in config.steps if step.name == "Build Evidence Ledger")
    assert isinstance(evidence_step, ToolStep)
    assert evidence_step.function == "knowledge.build_evidence_ledger"
    assert not any(
        step.response_schema == "evidence_ledger"
        for step in config.steps
        if hasattr(step, "response_schema")
    )
    assert {step.name for step in config.steps if isinstance(step, ToolStep)} >= {
        "Assess Literature Relevance",
        "Assess Prior Art Relevance",
        "Audit Candidate Pool",
        "Compile Candidate Units",
        "Audit Initial Revision",
        "Prepare Revision Repair",
        "Gate Revised Candidate",
    }

    scope_step = next(step for step in config.steps if step.name == "Validate Evidence Scopes")
    assert isinstance(scope_step, ToolStep)
    assert scope_step.argument_sources()["evidence_scope_options"] == (
        "initial_evidence_scope_options"
    )


def test_pipeline_rejects_ollama_host_credentials() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="embedded credentials"):
        PipelineConfig.model_validate(
            {
                "ollama_host": f"https://{SYNTHETIC_CREDENTIALS}@example.test",
                "steps": [
                    {
                        "name": "Run",
                        "type": "tool_call",
                        "function": "agent_helpers.slugify",
                        "inputs": ["topic"],
                        "outputs": ["result"],
                    }
                ],
            }
        )


def test_tool_contract_validation_reports_argument_drift() -> None:
    config = PipelineConfig.model_validate(
        {
            "context_init": ["initial_evidence_scope_options", "top_papers"],
            "steps": [
                {
                    "name": "Broken evidence scopes",
                    "type": "tool_call",
                    "function": "agent_helpers.validate_scope_options",
                    "inputs": ["initial_evidence_scope_options", "top_papers"],
                    "outputs": ["options", "scope"],
                }
            ],
        }
    )

    with pytest.raises(ValueError) as error:
        _validate_tool_contracts(config)

    message = str(error.value)
    assert "missing required arguments ['evidence_scope_options']" in message
    assert "unexpected arguments ['initial_evidence_scope_options']" in message


SYNTHETIC_CREDENTIALS = "user:secret"
