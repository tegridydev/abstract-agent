import io
import json

from rich.console import Console

from agent.agent import EXIT_CONFIG, EXIT_SUCCESS
from agent.schemas import InstalledModel, InteractiveSettings, ModelAssignments
from agent.ui import (
    InteractiveApplication,
    discover_runs,
    load_interactive_settings,
    save_interactive_settings,
)


def test_settings_round_trip_without_secrets(tmp_path) -> None:
    path = tmp_path / ".abstract-agent.toml"
    settings = InteractiveSettings(
        output_dir="results",
        minimum_papers=8,
        ollama_host="http://localhost:11434",
        model_assignments=ModelAssignments(
            planning_evidence="planner:latest", writing="writer:latest"
        ),
    )

    save_interactive_settings(path, settings)
    loaded, warning = load_interactive_settings(path)

    assert loaded == settings
    assert warning is None
    assert "API_KEY" not in path.read_text()
    assert loaded.model_assignments.writing == "writer:latest"


def test_invalid_settings_fall_back_safely(tmp_path) -> None:
    path = tmp_path / ".abstract-agent.toml"
    path.write_text("minimum_papers = -1\n")

    settings, warning = load_interactive_settings(path)

    assert settings.minimum_papers == 6
    assert warning and "Ignoring invalid settings" in warning


def test_run_discovery_sorts_and_skips_corrupt_results(tmp_path) -> None:
    valid = tmp_path / "result_valid.json"
    valid.write_text(
        json.dumps(
            {
                "schema_version": "3.0",
                "run_info": {"topic": "A topic", "timestamp_iso": "2026-08-21T10:00:00"},
                "warnings": [],
                "final_context": {
                    "novelty_verification": {"status": "uncertain"},
                    "evidence_audit": {"evidence_coverage": 0.75},
                },
            }
        )
    )
    (tmp_path / "result_broken.json").write_text("not-json")

    records, warnings = discover_runs(tmp_path)

    assert len(records) == 1
    assert records[0].topic == "A topic"
    assert records[0].novelty_status == "uncertain"
    assert records[0].evidence_coverage == "75%"
    assert len(warnings) == 1


def test_run_discovery_includes_failure_diagnostics(tmp_path) -> None:
    diagnostic = tmp_path / "failed_topic.json"
    diagnostic.write_text(
        json.dumps(
            {
                "schema_version": "4.3",
                "status": "pipeline_error",
                "error": "failed safely",
                "resumable": True,
                "resume_step": "Verify",
                "run_info": {"topic": "A topic", "timestamp_iso": "2026-08-21T10:00:00"},
                "diagnostic_context": {"topic": "A topic", "_active_step": "Verify"},
            }
        )
    )

    records, warnings = discover_runs(tmp_path)

    assert warnings == []
    assert records[0].kind == "Failure"
    assert records[0].resumable is True
    assert records[0].error == "failed safely"


def test_non_tty_launcher_never_waits_for_input(tmp_path) -> None:
    output = io.StringIO()
    app = InteractiveApplication(
        console=Console(file=output, force_terminal=False),
        settings_path=tmp_path / "settings.toml",
    )

    result = app.run()

    assert result == EXIT_CONFIG
    assert "requires a terminal" in output.getvalue()


def test_menu_launch_is_local_and_can_exit(monkeypatch, tmp_path) -> None:
    output = io.StringIO()
    app = InteractiveApplication(
        console=Console(file=output, force_terminal=True, width=100),
        settings_path=tmp_path / "settings.toml",
    )
    monkeypatch.setattr("agent.ui.Prompt.ask", lambda *args, **kwargs: "6")
    monkeypatch.setattr(
        "agent.ui.PipelineManager.preflight",
        lambda self: (_ for _ in ()).throw(AssertionError("unexpected network call")),
    )
    monkeypatch.setattr(
        "agent.ui.multi_source.check_sources",
        lambda: (_ for _ in ()).throw(AssertionError("unexpected network call")),
    )

    result = app.run()

    assert result == EXIT_SUCCESS
    assert "New Research" in output.getvalue()


def _app(tmp_path, width=120):
    app = InteractiveApplication(
        console=Console(file=io.StringIO(), force_terminal=True, width=width),
        settings_path=tmp_path / "settings.toml",
    )
    assert app._load_config()
    return app


def test_model_table_shows_current_role_usage(tmp_path) -> None:
    app = _app(tmp_path)
    app.settings = app.settings.model_copy(
        update={"model_assignments": ModelAssignments(all_model="research:9b")}
    )

    app._display_models(
        [InstalledModel(name="research:9b", parameter_size="9B", size=6_000_000_000)]
    )

    rendered = app.console.file.getvalue()
    assert "Used by" in rendered
    assert "research:9b" in rendered
    assert "All" in rendered


def test_models_detail_assigns_selected_model_to_all_roles(monkeypatch, tmp_path) -> None:
    app = _app(tmp_path)
    model = InstalledModel(
        name="research:9b",
        parameter_size="9B",
        capabilities=["completion"],
    )

    class FakeService:
        def __init__(self, *args, **kwargs):
            pass

        def list_models(self):
            return [model]

        def show_model(self, *args, **kwargs):
            return model

        def close(self):
            pass

    answers = iter(["1", "a", "b", "b"])
    monkeypatch.setattr("agent.ui.OllamaModelService", FakeService)
    monkeypatch.setattr("agent.ui.Prompt.ask", lambda *args, **kwargs: next(answers))

    app.models_menu()

    assert app.settings.model_assignments.all_model == "research:9b"
    loaded, warning = load_interactive_settings(app.settings_path)
    assert warning is None
    assert loaded.model_assignments.all_model == "research:9b"


def test_preferences_edits_only_selected_setting(monkeypatch, tmp_path) -> None:
    app = _app(tmp_path)
    original_output = app.settings.output_dir
    answers = iter(["2", "b"])
    monkeypatch.setattr("agent.ui.Prompt.ask", lambda *args, **kwargs: next(answers))
    monkeypatch.setattr("agent.ui.IntPrompt.ask", lambda *args, **kwargs: 8)

    app.preferences()

    assert app.settings.minimum_papers == 8
    assert app.settings.output_dir == original_output


def test_local_knowledge_menu_reports_status_without_network(monkeypatch, tmp_path) -> None:
    app = _app(tmp_path)
    knowledge_dir = tmp_path / "knowledge"
    app.settings = app.settings.model_copy(update={"knowledge_dir": str(knowledge_dir)})
    monkeypatch.setattr("agent.ui.Prompt.ask", lambda *args, **kwargs: "b")

    app.knowledge_menu()

    rendered = app.console.file.getvalue()
    assert "Local Knowledge" in rendered
    assert "Papers" in rendered
    assert (knowledge_dir / "index.sqlite3").exists()


def test_new_research_fast_path_starts_from_summary(monkeypatch, tmp_path) -> None:
    app = _app(tmp_path)
    captured = {}
    monkeypatch.setattr("agent.ui.Prompt.ask", lambda *args, **kwargs: "s")
    monkeypatch.setattr(app, "_pause", lambda: None)

    def execute(args, **kwargs):
        captured["topic"] = args.topic
        captured["search_depth"] = args.search_depth
        return EXIT_SUCCESS

    monkeypatch.setattr("agent.ui.execute_run", execute)

    result = app.new_research("compression")

    assert result == EXIT_SUCCESS
    assert captured == {"topic": "compression", "search_depth": "standard"}
    assert "Run summary" in app.console.file.getvalue()
