from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import pytest
from rich.console import Console

from agent.agent import EXIT_CANCELLED, EXIT_CONFIG, EXIT_EVIDENCE, EXIT_SUCCESS
from agent.schemas import InstalledModel, InteractiveSettings, ModelAssignments
from agent.ui import InteractiveApplication, RunRecord, discover_runs


class FakeReporter:
    def __init__(self, console: Console) -> None:
        del console
        self.saved_paths: tuple[Path, Path] | None = None
        self.error_data: dict[str, Any] = {}


class FakeService:
    def __init__(
        self,
        *args: Any,
        models: list[InstalledModel] | None = None,
        error: Exception | None = None,
        **kwargs: Any,
    ) -> None:
        del args, kwargs
        self.models = models or []
        self.error = error
        self.closed = False
        self.pulled: list[str] = []

    def list_models(self) -> list[InstalledModel]:
        if self.error:
            raise self.error
        return self.models

    def show_model(self, name: str, fallback: InstalledModel) -> InstalledModel:
        del name
        if self.error:
            raise self.error
        return fallback

    def probe_model(self, model: InstalledModel) -> InstalledModel:
        return model

    def pull_model(self, name: str, callback: Any) -> None:
        self.pulled.append(name)
        callback({"status": "downloading", "total": 10, "completed": 5})

    def close(self) -> None:
        self.closed = True


def make_app(tmp_path: Path, **kwargs: Any) -> InteractiveApplication:
    app = InteractiveApplication(
        console=Console(file=io.StringIO(), force_terminal=True, width=120),
        settings_path=tmp_path / "settings.toml",
        **kwargs,
    )
    assert app._load_config()
    return app


def prompts(monkeypatch: pytest.MonkeyPatch, answers: list[str]) -> None:
    iterator = iter(answers)
    monkeypatch.setattr("agent.ui.Prompt.ask", lambda *args, **kwargs: next(iterator))


def result_payload(topic: str = "A topic") -> dict[str, Any]:
    return {
        "run_info": {"topic": topic, "timestamp_iso": "2026-08-25T10:11:12"},
        "warnings": ["one"],
        "llm_metrics": {"draft": {}},
        "source_statuses": [
            {
                "provider": "example",
                "status": "ok",
                "result_count": 1,
                "failed_query_count": 0,
                "warning_query_count": 0,
            }
        ],
        "final_context": {
            "topic": topic,
            "_pipeline_duration_seconds": 1.25,
            "novelty_verification": {"status": "novelty_supported"},
            "evidence_audit": {"evidence_coverage": 0.5, "claim_count": 1},
            "audited_research_proposal": {
                "title": "Proposal",
                "hypothesis": "A useful hypothesis",
                "abstract": "A useful abstract",
                "novel_insights": ["Insight"],
                "references": ["p1"],
                "claim_evidence": [{"paper_ids": ["p2"]}],
            },
            "evidence_papers": [
                {
                    "paper_id": "p1",
                    "title": "Paper one",
                    "year": 2025,
                    "sources": ["Example"],
                    "url": "https://example.test/one",
                },
                {
                    "paper_id": "p3",
                    "title": "Unreferenced",
                    "source": "Other",
                },
            ],
        },
    }


def test_run_record_defaults_and_malformed_nested_values(tmp_path: Path) -> None:
    result = RunRecord(tmp_path / "result.json", {"final_context": "bad", "run_info": "bad"})
    failure = RunRecord(tmp_path / "failed.json", {"diagnostic_context": [], "resumable": 1})

    assert result.context == {}
    assert result.kind == "Result"
    assert result.status == "completed"
    assert result.topic == "Unknown topic"
    assert result.timestamp == "Unknown"
    assert result.novelty_status == "N/A"
    assert result.evidence_coverage == "N/A"
    assert failure.kind == "Failure"
    assert failure.status == "failed"
    assert failure.resumable is True
    assert failure.error == "No error detail recorded"


def test_discover_runs_handles_absent_directory(tmp_path: Path) -> None:
    records, warnings = discover_runs(tmp_path / "missing")
    assert records == []
    assert warnings == []


def test_main_menu_dispatches_all_screens_and_handles_interrupt(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    app = make_app(tmp_path)
    calls: list[str] = []
    for name in ("new_research", "recent_runs", "models_menu", "system_status", "preferences"):
        monkeypatch.setattr(app, name, lambda name=name: calls.append(name))
    prompts(monkeypatch, ["1", "2", "3", "4", "5", "6"])

    assert app.run() == EXIT_SUCCESS
    assert calls == ["new_research", "recent_runs", "models_menu", "system_status", "preferences"]

    monkeypatch.setattr(
        "agent.ui.Prompt.ask", lambda *args, **kwargs: (_ for _ in ()).throw(EOFError())
    )
    assert app.run() == EXIT_SUCCESS


def test_load_config_error_and_notice_render(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    app = make_app(tmp_path)
    app._set_notice("Saved")
    app._screen("Screen", "Subtitle")
    assert app._notice is None
    assert "Saved" in app.console.file.getvalue()

    monkeypatch.setattr(
        "agent.ui.load_pipeline_config",
        lambda: (_ for _ in ()).throw(ValueError("bad config")),
    )
    assert app._load_config() is False
    assert "bad config" in app.console.file.getvalue()


def test_scope_selector_returns_broad_or_selected(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    app = make_app(tmp_path)
    options = [{"label": "Domain", "description": "Focused", "paper_ids": ["p1"]}]
    monkeypatch.setattr("agent.ui.IntPrompt.ask", lambda *args, **kwargs: 0)
    assert app._select_research_scope("topic", options)["mode"] == "broad"
    monkeypatch.setattr("agent.ui.IntPrompt.ask", lambda *args, **kwargs: 1)
    selected = app._select_research_scope("topic", options)
    assert selected == {
        "mode": "domain",
        "label": "Domain",
        "source": "interactive",
        "supporting_paper_ids": ["p1"],
    }


def test_new_research_reprompts_blank_topic_then_cancels(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    app = make_app(tmp_path)
    prompts(monkeypatch, ["", "valid topic", "b"])

    assert app.new_research() == EXIT_CANCELLED
    assert app._notice == ("Research run cancelled.", "yellow")
    assert "Please enter a research topic" in app.console.file.getvalue()


def test_new_research_rejects_invalid_host(tmp_path: Path) -> None:
    app = make_app(tmp_path)
    app.settings = app.settings.model_copy(update={"ollama_host": "ftp://invalid"})

    assert app.new_research("topic") == EXIT_CONFIG
    assert "Invalid Ollama host" in app.console.file.getvalue()


def test_new_research_cloud_confirmation_can_cancel(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    app = make_app(tmp_path)
    app.settings = app.settings.model_copy(
        update={
            "confirm_before_run": False,
            "model_assignments": ModelAssignments(all_model="cloud-model:cloud"),
        }
    )
    monkeypatch.setattr("agent.ui.Confirm.ask", lambda *args, **kwargs: False)

    assert app.new_research("topic") == EXIT_CANCELLED
    assert "off device" in app._notice[0]


def test_new_research_evidence_recovery_deep_then_success(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[str] = []

    def execute(args: Any, **kwargs: Any) -> int:
        del kwargs
        calls.append(args.search_depth)
        return EXIT_EVIDENCE if len(calls) == 1 else EXIT_SUCCESS

    app = make_app(tmp_path, run_executor=execute, reporter_factory=FakeReporter)
    monkeypatch.setattr(app, "_pause", lambda: None)
    prompts(monkeypatch, ["s", "d"])

    assert app.new_research("topic") == EXIT_SUCCESS
    assert calls == ["standard", "deep"]


def test_new_research_customizes_every_run_setting(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured: list[Any] = []

    def execute(run_args: Any, **kwargs: Any) -> int:
        del kwargs
        captured.append(run_args)
        return EXIT_SUCCESS

    app = make_app(tmp_path, run_executor=execute, reporter_factory=FakeReporter)
    app.settings = app.settings.model_copy(update={"ollama_host": "https://remote.example.test"})
    monkeypatch.setattr(app, "_pause", lambda: None)
    monkeypatch.setattr(
        app,
        "_assignment_editor",
        lambda current: ModelAssignments(all_model="custom:1"),
    )
    monkeypatch.setattr("agent.ui.IntPrompt.ask", lambda *args, **kwargs: 8)
    monkeypatch.setattr("agent.ui.Confirm.ask", lambda *args, **kwargs: True)
    prompts(
        monkeypatch,
        ["c", "1", "2", "3", "deep", "4", "custom-output", "5", "b", "s"],
    )

    assert app.new_research("topic") == EXIT_SUCCESS
    assert captured[0].minimum_papers == 8
    assert captured[0].allow_insufficient_literature is True
    assert captured[0].search_depth == "deep"
    assert captured[0].output_dir == "custom-output"
    assert captured[0].model_assignments.all_model == "custom:1"
    assert "remote" in app.console.file.getvalue()


def test_new_research_adjusts_minimum_then_cancels_recovery(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    minimums: list[int] = []

    def execute(run_args: Any, **kwargs: Any) -> int:
        del kwargs
        minimums.append(run_args.minimum_papers)
        return EXIT_EVIDENCE

    app = make_app(tmp_path, run_executor=execute, reporter_factory=FakeReporter)
    monkeypatch.setattr(app, "_pause", lambda: None)
    monkeypatch.setattr("agent.ui.IntPrompt.ask", lambda *args, **kwargs: 2)
    prompts(monkeypatch, ["s", "m", "c"])

    assert app.new_research("topic") == EXIT_EVIDENCE
    assert minimums == [6, 2]


def test_new_research_resumes_recoverable_diagnostic(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[str | None] = []

    def execute(args: Any, reporter: FakeReporter, **kwargs: Any) -> int:
        del kwargs
        calls.append(getattr(args, "diagnostic", None))
        if len(calls) == 1:
            reporter.error_data = {"recoverable": True, "diagnostic_path": "failed.json"}
            return EXIT_CONFIG
        return EXIT_SUCCESS

    app = make_app(tmp_path, run_executor=execute, reporter_factory=FakeReporter)
    monkeypatch.setattr(app, "_pause", lambda: None)
    prompts(monkeypatch, ["s"])
    monkeypatch.setattr("agent.ui.Confirm.ask", lambda *args, **kwargs: True)

    assert app.new_research("topic") == EXIT_SUCCESS
    assert calls == [None, "failed.json"]


def test_new_research_renders_saved_result_and_handles_invalid_saved_json(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    text_path = tmp_path / "result.txt"
    json_path = tmp_path / "result.json"
    json_path.write_text(json.dumps(result_payload()))

    class SavedReporter(FakeReporter):
        def __init__(self, console: Console) -> None:
            super().__init__(console)
            self.saved_paths = (text_path, json_path)

    app = make_app(
        tmp_path,
        run_executor=lambda *args, **kwargs: EXIT_SUCCESS,
        reporter_factory=SavedReporter,
    )
    app.settings = app.settings.model_copy(update={"confirm_before_run": False})
    monkeypatch.setattr(app, "_pause", lambda: None)
    assert app.new_research("topic") == EXIT_SUCCESS
    assert "Proposal" in app.console.file.getvalue()

    json_path.write_text("invalid")
    assert app.new_research("topic") == EXIT_SUCCESS
    assert "could not be displayed" in app.console.file.getvalue()


def test_render_failure_unstructured_and_unverified_records(tmp_path: Path) -> None:
    app = make_app(tmp_path)
    failure = RunRecord(
        tmp_path / "failed.json",
        {
            "error": "boom",
            "resumable": True,
            "resume_step": "Verify",
            "diagnostic_context": {"topic": "Failed topic"},
        },
    )
    app._render_record(failure)
    app._render_record(RunRecord(tmp_path / "result.json", {"final_context": {"topic": "Raw"}}))
    payload = result_payload("Unverified")
    proposal = payload["final_context"]["audited_research_proposal"]
    proposal["proposal_status"] = "unverified_research_direction"
    proposal["novel_insights"] = []
    app._render_record(RunRecord(tmp_path / "unverified.json", payload))

    rendered = app.console.file.getvalue()
    assert "Failed run" in rendered
    assert "Evidence only result" in rendered
    assert "Unverified research direction" in rendered
    assert "None recorded" in rendered


def test_render_record_details_filters_evidence_and_handles_wrong_types(tmp_path: Path) -> None:
    app = make_app(tmp_path)
    app._render_record_details(RunRecord(tmp_path / "result.json", result_payload()))
    app._render_record_details(
        RunRecord(
            tmp_path / "odd.json",
            {"source_statuses": "bad", "final_context": {"evidence_papers": "bad"}},
        )
    )
    rendered = app.console.file.getvalue()
    assert "Paper one" in rendered
    assert "Unreferenced" not in rendered
    assert "Initial Literature Source Health" in rendered


def test_recent_runs_empty_paginates_and_reruns(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    app = make_app(tmp_path)
    monkeypatch.setattr(app, "_pause", lambda: None)
    app.settings = app.settings.model_copy(update={"output_dir": str(tmp_path)})
    app.recent_runs()
    assert "No saved runs" in app.console.file.getvalue()

    records = [
        RunRecord(tmp_path / f"result_{index}.json", result_payload(f"Topic {index}"))
        for index in range(6)
    ]
    app.settings = app.settings.model_copy(update={"history_page_size": 5})
    monkeypatch.setattr("agent.ui.discover_runs", lambda path: (records, ["one warning"]))
    rerun: list[str] = []
    monkeypatch.setattr(app, "new_research", lambda topic=None: rerun.append(topic) or EXIT_SUCCESS)
    prompts(monkeypatch, ["n", "p", "1", "r"])
    app.recent_runs()
    assert rerun == ["Topic 0"]


def test_recent_runs_details_and_paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    app = make_app(tmp_path)
    monkeypatch.setattr(app, "_pause", lambda: None)
    record = RunRecord(tmp_path / "result.json", result_payload())
    monkeypatch.setattr("agent.ui.discover_runs", lambda path: ([record], []))
    prompts(monkeypatch, ["1", "d", "1", "p", "b"])

    app.recent_runs()

    rendered = app.console.file.getvalue()
    assert "Referenced Evidence" in rendered
    assert "JSON:" in rendered
    assert "Text:" in rendered


def test_resume_record_cloud_decline_and_success(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[str] = []
    app = make_app(
        tmp_path,
        run_executor=lambda args, **kwargs: calls.append(args.diagnostic) or EXIT_SUCCESS,
        reporter_factory=FakeReporter,
    )
    monkeypatch.setattr(app, "_pause", lambda: None)
    record = RunRecord(
        tmp_path / "failed.json",
        {"resumable": True, "diagnostic_context": {"topic": "Topic"}},
    )
    app.settings = app.settings.model_copy(
        update={"model_assignments": ModelAssignments(all_model="remote:cloud")}
    )
    monkeypatch.setattr("agent.ui.Confirm.ask", lambda *args, **kwargs: False)
    app._resume_record(record)
    assert calls == []

    monkeypatch.setattr("agent.ui.Confirm.ask", lambda *args, **kwargs: True)
    app._resume_record(record)
    assert calls == [str(record.path)]


@pytest.mark.parametrize(
    ("size", "expected"),
    [(1, "1.0 B"), (1024, "1.0 KiB"), (1024**2, "1.0 MiB"), (1024**4, "1.0 TiB")],
)
def test_format_size_units(size: int, expected: str) -> None:
    assert InteractiveApplication._format_size(size) == expected


def test_select_model_empty_back_incompatible_error_and_success(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    app = make_app(tmp_path)
    complete = InstalledModel(name="complete", capabilities=["completion"])
    embedding = InstalledModel(name="embedding", capabilities=["embedding"])
    service = FakeService(models=[complete, embedding])
    assert app._select_model(service, [], "test") is None  # type: ignore[arg-type]

    prompts(monkeypatch, ["b"])
    assert app._select_model(service, [complete], "test") is None  # type: ignore[arg-type]
    prompts(monkeypatch, ["1"])
    assert app._select_model(service, [embedding], "test") is None  # type: ignore[arg-type]

    broken = FakeService(error=RuntimeError("offline"))
    prompts(monkeypatch, ["1"])
    assert app._select_model(broken, [complete], "test") is None  # type: ignore[arg-type]
    prompts(monkeypatch, ["1"])
    assert app._select_model(service, [complete], "test") == "complete"  # type: ignore[arg-type]


def test_assignment_editor_defaults_cancel_save_and_service_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    service = FakeService(models=[InstalledModel(name="model", capabilities=["completion"])])
    app = make_app(tmp_path, model_service_factory=lambda *args, **kwargs: service)
    prompts(monkeypatch, ["d", "s"])
    assert app._assignment_editor(ModelAssignments(all_model="old")) == ModelAssignments()
    assert service.closed

    service.closed = False
    prompts(monkeypatch, ["b"])
    assert app._assignment_editor(ModelAssignments()) is None
    assert service.closed

    broken = FakeService(error=RuntimeError("offline"))
    app._model_service_factory = lambda *args, **kwargs: broken
    assert app._assignment_editor(ModelAssignments()) is None
    assert broken.closed


def test_assignment_editor_sets_all_and_one_role(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    service = FakeService(models=[InstalledModel(name="model", capabilities=["completion"])])
    app = make_app(tmp_path, model_service_factory=lambda *args, **kwargs: service)
    monkeypatch.setattr(app, "_select_model", lambda *args: "model")
    prompts(monkeypatch, ["a", "1", "s"])
    assignments = app._assignment_editor(ModelAssignments())
    assert assignments.all_model == "model"
    assert assignments.planning_evidence == "model"


def test_pull_model_declines_cloud_and_handles_success_and_failures(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    app = make_app(tmp_path)
    service = FakeService()
    prompts(monkeypatch, ["model:cloud"])
    monkeypatch.setattr("agent.ui.Confirm.ask", lambda *args, **kwargs: False)
    assert app._pull_model(service) is False  # type: ignore[arg-type]

    prompts(monkeypatch, ["local-model"])
    monkeypatch.setattr("agent.ui.Confirm.ask", lambda *args, **kwargs: True)
    assert app._pull_model(service) is True  # type: ignore[arg-type]
    assert service.pulled == ["local-model"]

    class BrokenService(FakeService):
        def pull_model(self, name: str, callback: Any) -> None:
            del name, callback
            raise RuntimeError("pull failed")

    prompts(monkeypatch, ["broken"])
    assert app._pull_model(BrokenService()) is False  # type: ignore[arg-type]

    class CancelledService(FakeService):
        def pull_model(self, name: str, callback: Any) -> None:
            del name, callback
            raise KeyboardInterrupt

    prompts(monkeypatch, ["cancelled"])
    assert app._pull_model(CancelledService()) is False  # type: ignore[arg-type]


def test_models_menu_refresh_pull_defaults_assign_and_role_detail(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    model = InstalledModel(name="model", capabilities=["completion"], context_length=4096)
    service = FakeService(models=[model])
    app = make_app(tmp_path, model_service_factory=lambda *args, **kwargs: service)
    app.settings = app.settings.model_copy(
        update={"model_assignments": ModelAssignments(all_model="old")}
    )
    monkeypatch.setattr(app, "_pull_model", lambda svc: True)
    monkeypatch.setattr(
        app, "_assignment_editor", lambda current: ModelAssignments(all_model="assigned")
    )
    monkeypatch.setattr("agent.ui.Confirm.ask", lambda *args, **kwargs: True)
    prompts(monkeypatch, ["r", "p", "d", "a", "1", "t", "1", "b", "b"])

    app.models_menu()

    assert app.settings.model_assignments.planning_evidence == "model"
    assert service.closed


def test_models_menu_noncompletion_and_operation_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    embedding = InstalledModel(name="embed", capabilities=["embedding"])
    service = FakeService(models=[embedding])
    app = make_app(tmp_path, model_service_factory=lambda *args, **kwargs: service)
    prompts(monkeypatch, ["1", "b", "b"])
    app.models_menu()
    assert "cannot be assigned" in app.console.file.getvalue()

    broken = FakeService(error=RuntimeError("offline"))
    app._model_service_factory = lambda *args, **kwargs: broken
    monkeypatch.setattr(app, "_pause", lambda: None)
    app.models_menu()
    assert "Ollama operation failed" in app.console.file.getvalue()


def test_models_menu_can_reset_output_compatibility_profiles(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class Profiles:
        resets = 0

        def reset(self) -> None:
            self.resets += 1

    profiles_store = Profiles()
    model = InstalledModel(name="model", capabilities=["completion"])
    service = FakeService(models=[model])
    app = make_app(tmp_path, model_service_factory=lambda *args, **kwargs: service)
    app._output_profile_store = profiles_store  # type: ignore[assignment]
    monkeypatch.setattr("agent.ui.Confirm.ask", lambda *args, **kwargs: True)
    prompts(monkeypatch, ["x", "b"])

    app.models_menu()

    assert profiles_store.resets == 1
    assert "compatibility routes reset" in app.console.file.getvalue()


def test_system_status_checks_manager_and_providers(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    events: list[str] = []

    class Manager:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            del args, kwargs

        def preflight(self) -> None:
            events.append("preflight")

        def close(self) -> None:
            events.append("close")

    statuses = lambda: [  # noqa: E731 - compact injected health check
        {"provider": "fake", "status": "ok", "result_count": 1, "message": None}
    ]
    app = make_app(
        tmp_path,
        pipeline_manager_factory=Manager,
        source_checker=statuses,
    )
    monkeypatch.setattr(app, "_pause", lambda: None)
    prompts(monkeypatch, ["1"])
    app.system_status()
    prompts(monkeypatch, ["2"])
    app.system_status()

    assert events == ["preflight", "close"]
    assert "fake" in app.console.file.getvalue()


def test_system_status_reports_checker_failures(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    app = make_app(
        tmp_path,
        source_checker=lambda: (_ for _ in ()).throw(RuntimeError("providers offline")),
    )
    monkeypatch.setattr(app, "_pause", lambda: None)
    prompts(monkeypatch, ["2"])
    app.system_status()
    assert "Provider check failed" in app.console.file.getvalue()


@pytest.mark.parametrize(
    ("choice", "prompt_value", "expected_key", "expected_value"),
    [
        ("1", "other-output", "output_dir", "other-output"),
        ("6", "deep", "search_depth", "deep"),
        ("7", "auto", "ollama_host", None),
        ("8", "45", "ollama_timeout", 45.0),
    ],
)
def test_preferences_prompt_value_branches(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    choice: str,
    prompt_value: str,
    expected_key: str,
    expected_value: Any,
) -> None:
    app = make_app(tmp_path)
    prompts(monkeypatch, [choice, prompt_value, "b"])
    app.preferences()
    assert getattr(app.settings, expected_key) == expected_value


def test_preferences_boolean_models_invalid_and_reset(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    app = make_app(tmp_path)
    monkeypatch.setattr("agent.ui.Confirm.ask", lambda *args, **kwargs: True)
    prompts(monkeypatch, ["3", "b"])
    app.preferences()
    assert app.settings.allow_insufficient_literature is True

    monkeypatch.setattr(app, "_assignment_editor", lambda current: ModelAssignments(all_model="m"))
    prompts(monkeypatch, ["m", "b"])
    app.preferences()
    assert app.settings.model_assignments.all_model == "m"

    prompts(monkeypatch, ["8", "not-a-number", "b"])
    app.preferences()
    assert "Preference was not saved" in app.console.file.getvalue()

    app.settings_path.write_text("minimum_papers = 7\n")
    prompts(monkeypatch, ["r", "b"])
    app.preferences()
    assert not app.settings_path.exists()
    assert app.settings == InteractiveSettings()
