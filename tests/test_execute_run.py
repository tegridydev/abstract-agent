from __future__ import annotations

import argparse
import io
from pathlib import Path
from typing import Any

import pytest
from rich.console import Console

import agent.agent as agent_module
from agent.agent import (
    EXIT_CANCELLED,
    EXIT_CONFIG,
    EXIT_EVIDENCE,
    EXIT_OLLAMA,
    EXIT_OUTPUT,
    EXIT_SOURCE,
    EXIT_SUCCESS,
    InsufficientEvidenceError,
    OllamaRuntimeError,
    execute_run,
)
from agent.reporting import PipelineEvent
from agent.schemas import PipelineConfig


class RecordingReporter:
    def __init__(self) -> None:
        self.events: list[PipelineEvent] = []
        self.close_count = 0

    def emit(self, event: PipelineEvent) -> None:
        self.events.append(event)

    def close(self) -> None:
        self.close_count += 1


class Manager:
    def __init__(
        self,
        *args: Any,
        outcome: dict[str, Any] | BaseException | None = None,
        models: set[str] | None = None,
        host: str = "http://localhost:11434",
        last_context: dict[str, Any] | None = None,
        preflight_error: BaseException | None = None,
        **kwargs: Any,
    ) -> None:
        del args
        self.kwargs = kwargs
        self.outcome = outcome if outcome is not None else {"topic": "topic", "result": "done"}
        self.resolved_models = models or {"model:1"}
        self.ollama_host = host
        self.last_context = last_context or {"topic": "topic", "_active_step": "Generate"}
        self.preflight_error = preflight_error
        self.closed = False
        self.initial_context: dict[str, Any] | None = None
        self.start_step: str | None = None

    def preflight(self) -> None:
        if self.preflight_error:
            raise self.preflight_error

    def run_pipeline(
        self, initial_context: dict[str, Any], *, start_step: str | None = None
    ) -> dict[str, Any]:
        self.initial_context = initial_context
        self.start_step = start_step
        if isinstance(self.outcome, BaseException):
            raise self.outcome
        return self.outcome

    def close(self) -> None:
        self.closed = True


def config() -> PipelineConfig:
    return PipelineConfig.model_validate(
        {
            "context_init": ["topic"],
            "steps": [
                {
                    "name": "Generate",
                    "type": "ollama_call",
                    "model": "model:1",
                    "inputs": ["topic"],
                    "outputs": ["result"],
                    "prompt_template": "Write {topic}",
                }
            ],
            "final_outputs": ["result"],
        }
    )


def args(**overrides: Any) -> argparse.Namespace:
    values = {
        "topic": "topic",
        "diagnostic": None,
        "output_dir": "output",
        "ollama_host": None,
        "ollama_timeout": None,
        "minimum_papers": None,
        "allow_insufficient_literature": False,
        "allow_cloud_models": False,
        "search_depth": "standard",
        "model_assignments": None,
        "model": None,
        "planning_model": None,
        "creative_model": None,
        "critique_model": None,
        "writing_model": None,
        "scope": "broad",
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def install_manager(monkeypatch: pytest.MonkeyPatch, manager: Manager) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []

    def factory(*factory_args: Any, **kwargs: Any) -> Manager:
        del factory_args
        calls.append(kwargs)
        manager.kwargs = kwargs
        return manager

    monkeypatch.setattr(agent_module, "load_pipeline_config", config)
    monkeypatch.setattr(agent_module, "PipelineManager", factory)
    return calls


def test_execute_run_success_emits_lifecycle_displays_saves_and_closes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    manager = Manager(host="http://remote.example.test:11434")
    install_manager(monkeypatch, manager)
    displayed: list[dict[str, Any]] = []
    saved: list[tuple[dict[str, Any], str, str]] = []
    monkeypatch.setattr(
        agent_module, "display_final_results", lambda context, cfg: displayed.append(context)
    )

    def save(context: dict[str, Any], cfg: PipelineConfig, topic: str, output: str):
        del cfg
        saved.append((context, topic, output))
        return tmp_path / "result.txt", tmp_path / "result.json"

    monkeypatch.setattr(agent_module, "save_results", save)
    reporter = RecordingReporter()

    result = execute_run(args(), reporter=reporter)

    assert result == EXIT_SUCCESS
    assert displayed == [manager.outcome]
    assert saved == [(manager.outcome, "topic", "output")]
    assert [event.kind for event in reporter.events] == [
        "warning",
        "preflight_started",
        "preflight_completed",
        "saved",
    ]
    assert manager.initial_context["topic"] == "topic"
    assert manager.closed and reporter.close_count == 1


def test_execute_run_success_can_skip_display_and_apply_cli_model_overrides(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    manager = Manager()
    calls = install_manager(monkeypatch, manager)
    monkeypatch.setattr(
        agent_module,
        "display_final_results",
        lambda *unused: (_ for _ in ()).throw(AssertionError("displayed")),
    )
    monkeypatch.setattr(
        agent_module,
        "save_results",
        lambda *unused: (tmp_path / "result.txt", tmp_path / "result.json"),
    )

    result = execute_run(
        args(model="all:1", writing_model="writer:1"),
        reporter=RecordingReporter(),
        display_results=False,
    )

    assert result == EXIT_SUCCESS
    assignments = calls[0]["model_assignments"]
    assert assignments.all_model == "all:1"
    assert assignments.writing == "writer:1"


@pytest.mark.parametrize("stage", ["config", "checkpoint", "manager"])
def test_execute_run_configuration_failures_close_reporter(
    monkeypatch: pytest.MonkeyPatch, stage: str
) -> None:
    reporter = RecordingReporter()
    monkeypatch.setattr(agent_module, "load_pipeline_config", config)
    run_args = args()
    if stage == "config":
        monkeypatch.setattr(
            agent_module,
            "load_pipeline_config",
            lambda: (_ for _ in ()).throw(ValueError("bad config")),
        )
    elif stage == "checkpoint":
        run_args.diagnostic = "failed.json"
        monkeypatch.setattr(
            agent_module,
            "load_resume_checkpoint",
            lambda *unused: (_ for _ in ()).throw(ValueError("bad checkpoint")),
        )
    else:
        monkeypatch.setattr(
            agent_module,
            "PipelineManager",
            lambda *unused, **kwargs: (_ for _ in ()).throw(ValueError("bad runtime")),
        )

    assert execute_run(run_args, reporter=reporter) == EXIT_CONFIG
    assert reporter.events[-1].kind == "error"
    assert reporter.close_count == 1


def test_execute_run_resume_uses_checkpoint_context_and_fallback_settings(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    manager = Manager()
    calls = install_manager(monkeypatch, manager)
    resume_context = {
        "topic": "resumed topic",
        "result": "partial",
        "_minimum_papers": 9,
        "_search_depth": "deep",
        "_requested_scope": "Focused",
        "_insufficient_literature_override": True,
        "_pipeline_model_assignments": {"creative": "creative:2"},
    }
    monkeypatch.setattr(
        agent_module,
        "load_resume_checkpoint",
        lambda path, cfg: ("resumed topic", resume_context, "Generate"),
    )
    monkeypatch.setattr(
        agent_module,
        "save_results",
        lambda *unused: (tmp_path / "result.txt", tmp_path / "result.json"),
    )

    run_args = args(diagnostic="failed.json", search_depth=None, scope=None)
    assert execute_run(run_args, reporter=RecordingReporter()) == EXIT_SUCCESS

    assert run_args.topic == "resumed topic"
    assert manager.initial_context is resume_context
    assert manager.start_step == "Generate"
    assert calls[0]["minimum_papers"] == 9
    assert calls[0]["search_depth"] == "deep"
    assert calls[0]["requested_scope"] == "Focused"
    assert calls[0]["allow_insufficient_literature"] is True
    assert calls[0]["model_assignments"].creative == "creative:2"


def test_execute_run_blocks_cloud_models_without_consent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = Manager(models={"model:cloud"})
    install_manager(monkeypatch, manager)
    reporter = RecordingReporter()

    assert execute_run(args(), reporter=reporter) == EXIT_CONFIG
    assert "explicit --allow-cloud-models" in reporter.events[-1].message
    assert manager.closed and reporter.close_count == 1


def test_execute_run_rejects_invalid_manager_host(monkeypatch: pytest.MonkeyPatch) -> None:
    manager = Manager(host="ftp://invalid")
    install_manager(monkeypatch, manager)
    reporter = RecordingReporter()

    assert execute_run(args(), reporter=reporter) == EXIT_CONFIG
    assert "invalid Ollama host" in reporter.events[-1].message
    assert manager.closed and reporter.close_count == 1


@pytest.mark.parametrize(
    ("outcome", "expected"),
    [
        (InsufficientEvidenceError("too few"), EXIT_EVIDENCE),
        (OllamaRuntimeError("bad model"), EXIT_OLLAMA),
        (OSError("disk full"), EXIT_OUTPUT),
        (KeyboardInterrupt(), EXIT_CANCELLED),
        (RuntimeError("unexpected"), EXIT_SOURCE),
    ],
)
def test_execute_run_maps_runtime_failures_and_always_closes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    outcome: BaseException,
    expected: int,
) -> None:
    manager = Manager(outcome=outcome)
    install_manager(monkeypatch, manager)
    reporter = RecordingReporter()
    monkeypatch.setattr(
        agent_module,
        "_try_save_failure_diagnostic",
        lambda *args, **kwargs: tmp_path / "failed.json",
    )
    monkeypatch.setattr(agent_module, "_is_resumable_context", lambda *args: True)

    assert execute_run(args(), reporter=reporter) == expected
    assert manager.closed and reporter.close_count == 1
    assert reporter.events[-1].kind in {"error", "cancelled"}
    if expected in {EXIT_EVIDENCE, EXIT_OLLAMA, EXIT_SOURCE}:
        assert reporter.events[-1].data["diagnostic_path"].endswith("failed.json")
        assert reporter.events[-1].data["recoverable"] is True


def test_execute_run_generic_failure_survives_diagnostic_validation_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = Manager(outcome=RuntimeError("unexpected"), last_context={"topic": "topic"})
    install_manager(monkeypatch, manager)
    reporter = RecordingReporter()
    monkeypatch.setattr(
        agent_module,
        "_try_save_failure_diagnostic",
        lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("cannot diagnose")),
    )

    assert execute_run(args(), reporter=reporter) == EXIT_SOURCE
    assert reporter.events[-1].data == {}
    assert "pipeline failure" in reporter.events[-1].message


def test_execute_run_saves_interrupted_snapshot_and_resumable_diagnostic(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    manager = Manager(
        outcome=OllamaRuntimeError("candidate schema failed"),
        last_context={
            "topic": "compression",
            "evidence_ledger": {"cards": [{"paper_id": "paper:1"}]},
            "_active_step": "Generate Independent Hypotheses",
        },
    )
    install_manager(monkeypatch, manager)
    saved: list[dict[str, Any]] = []

    def fake_save(context, *unused):
        saved.append(context)
        return tmp_path / "partial.txt", tmp_path / "partial.json"

    monkeypatch.setattr(agent_module, "save_results", fake_save)
    diagnostic = tmp_path / "failed.json"
    monkeypatch.setattr(agent_module, "_try_save_failure_diagnostic", lambda *a, **k: diagnostic)
    reporter = RecordingReporter()

    assert execute_run(args(), reporter=reporter) == EXIT_OLLAMA
    assert saved[0]["_outcome"] == "interrupted"
    assert saved[0]["_active_step"] == "Generate Independent Hypotheses"
    assert reporter.events[-1].kind == "error"
    assert reporter.events[-1].data["diagnostic_path"] == str(diagnostic)


def test_execute_run_preflight_failure_is_classified_as_source_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = Manager(preflight_error=RuntimeError("offline"))
    install_manager(monkeypatch, manager)
    reporter = RecordingReporter()
    monkeypatch.setattr(agent_module, "_try_save_failure_diagnostic", lambda *args, **kwargs: None)

    assert execute_run(args(), reporter=reporter) == EXIT_SOURCE
    assert "offline" in reporter.events[-1].message


def test_try_save_failure_diagnostic_warns_when_writer_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reporter = RecordingReporter()
    monkeypatch.setattr(
        agent_module,
        "save_failure_diagnostic",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("read-only")),
    )

    result = agent_module._try_save_failure_diagnostic(
        {}, config(), "topic", "output", "error", reporter, status="failed"
    )

    assert result is None
    assert reporter.events[-1].kind == "warning"
    assert "OSError" in reporter.events[-1].message


def test_argument_validators_and_final_display(monkeypatch: pytest.MonkeyPatch) -> None:
    assert agent_module._positive_int("2") == 2
    with pytest.raises(argparse.ArgumentTypeError):
        agent_module._positive_int("0")
    assert agent_module._positive_float("2.5") == 2.5
    with pytest.raises(argparse.ArgumentTypeError):
        agent_module._positive_float("0")

    output = io.StringIO()
    monkeypatch.setattr(agent_module, "console", Console(file=output, force_terminal=False))
    display_config = PipelineConfig.model_validate(
        {
            "context_init": ["topic", "top_papers", "summary", "nothing"],
            "steps": [
                {
                    "name": "Noop",
                    "type": "tool_call",
                    "function": "example.noop",
                    "inputs": ["topic"],
                    "outputs": ["done"],
                }
            ],
            "final_outputs": ["top_papers", "summary", "nothing"],
        }
    )
    agent_module.display_final_results(
        {
            "top_papers": [
                {
                    "title": "Paper",
                    "composite_score": 0.75,
                    "year": 2025,
                    "source": "Provider",
                    "url": "https://example.test",
                }
            ],
            "summary": {"answer": 42},
            "nothing": None,
        },
        display_config,
    )
    assert "Paper" in output.getvalue()
    assert '"answer": 42' in output.getvalue()
