import io
from pathlib import Path

from rich.console import Console

from agent.reporting import ConsoleReporter, NullReporter, PipelineEvent, RichProgressReporter


class SpyProgress:
    instances: list["SpyProgress"] = []

    def __init__(self, *columns: object, console: Console, **kwargs: object) -> None:
        self.columns = columns
        self.console = console
        self.kwargs = kwargs
        self.started = False
        self.stopped = False
        self.tasks: list[tuple[str, dict[str, object]]] = []
        self.updates: list[tuple[object, dict[str, object]]] = []
        self.__class__.instances.append(self)

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True

    def add_task(self, description: str, **kwargs: object) -> int:
        self.tasks.append((description, kwargs))
        return len(self.tasks)

    def update(self, task_id: object, **kwargs: object) -> None:
        self.updates.append((task_id, kwargs))


def test_null_reporter_accepts_events() -> None:
    reporter = NullReporter()
    assert reporter.emit(PipelineEvent("anything")) is None
    assert reporter.close() is None


def test_console_reporter_renders_every_supported_event() -> None:
    output = io.StringIO()
    reporter = ConsoleReporter(Console(file=output, force_terminal=False, width=140))
    events = [
        PipelineEvent("preflight_started", data={"models": ["a", "b"]}),
        PipelineEvent("preflight_completed"),
        PipelineEvent("pipeline_started", "Starting"),
        PipelineEvent("step_started", step_name="Search", step_index=1, total_steps=2),
        PipelineEvent("step_completed", step_name="Search"),
        PipelineEvent("step_skipped", "Already complete"),
        PipelineEvent("warning", "Careful"),
        PipelineEvent("error", "Broken"),
        PipelineEvent("saved", data={"text_path": "a.txt", "json_path": "a.json"}),
        PipelineEvent("unknown"),
    ]

    for event in events:
        reporter.emit(event)

    assert reporter.close() is None
    rendered = output.getvalue()
    for expected in (
        "Validating Ollama models: a, b",
        "Basic model transport and response checks completed",
        "Starting",
        "[1/2] Search...",
        "Already complete",
        "Warning: Careful",
        "Broken",
        "Saved: a.txt and a.json",
    ):
        assert expected in rendered


def test_rich_reporter_full_lifecycle_uses_injected_progress() -> None:
    SpyProgress.instances.clear()
    output = io.StringIO()
    console = Console(file=output, force_terminal=False, width=140)
    reporter = RichProgressReporter(console, progress_factory=SpyProgress)  # type: ignore[arg-type]

    reporter.emit(PipelineEvent("warning", "before progress"))
    reporter.emit(PipelineEvent("preflight_started", data={"models": ["tiny"]}))
    preflight = SpyProgress.instances[-1]
    reporter.emit(PipelineEvent("preflight_completed"))
    reporter.emit(PipelineEvent("pipeline_started", total_steps=0))
    pipeline = SpyProgress.instances[-1]
    reporter.emit(PipelineEvent("step_started", step_name=None, data={"model_step": True}))
    reporter.emit(
        PipelineEvent(
            "model_request_started", data={"request_kind": "focused_repair", "attempt": 1}
        )
    )
    reporter.emit(PipelineEvent("model_progress", data={"streamed_characters": "12"}))
    reporter.emit(PipelineEvent("model_request_completed"))
    reporter.emit(PipelineEvent("step_completed", data={"paper_count": 3, "model_calls": 2}))
    reporter.emit(PipelineEvent("step_skipped", data={"paper_count": 1, "model_calls": 0}))
    reporter.emit(PipelineEvent("warning", "during progress"))
    reporter.emit(PipelineEvent("scope_selection_started"))
    reporter.emit(PipelineEvent("scope_selection_completed"))
    restarted = SpyProgress.instances[-1]
    reporter.emit(
        PipelineEvent("saved", data={"text_path": "report.txt", "json_path": "report.json"})
    )
    reporter.emit(PipelineEvent("pipeline_completed", "all done"))

    assert preflight.started and preflight.stopped
    assert pipeline.started and pipeline.stopped
    assert restarted.started and restarted.stopped
    assert (
        pipeline.tasks[0][1]["pipeline_started_at"] == restarted.tasks[0][1]["pipeline_started_at"]
    )
    assert reporter.saved_paths == (Path("report.txt"), Path("report.json"))
    assert reporter.paper_count == 3
    assert reporter.model_calls == 2
    assert reporter.completed_steps == 2
    assert reporter.warning_count == 2
    assert any(update[1].get("description") == "Working" for update in pipeline.updates)
    assert any(update[1].get("response_chars") == 12 for update in pipeline.updates)
    assert any(
        update[1].get("request_phase") == "repair 1/1 · waiting for first token"
        for update in pipeline.updates
    )
    assert "Selected models passed" in output.getvalue()
    assert "Research completed" in output.getvalue()


def test_rich_reporter_cancel_error_and_close_paths() -> None:
    SpyProgress.instances.clear()
    output = io.StringIO()
    console = Console(file=output, force_terminal=False)
    reporter = RichProgressReporter(console, progress_factory=SpyProgress)  # type: ignore[arg-type]

    reporter.emit(PipelineEvent("pipeline_started", total_steps=2))
    reporter.emit(PipelineEvent("cancelled"))
    reporter.emit(PipelineEvent("pipeline_started", total_steps=2))
    reporter.emit(
        PipelineEvent("error", "failed", data={"diagnostic_path": "failed.json", "code": 2})
    )
    reporter.close()

    assert reporter.error_data == {"diagnostic_path": "failed.json", "code": 2}
    assert reporter.progress is None
    assert "cancelled" in output.getvalue()
    assert "Diagnostic saved: failed.json" in output.getvalue()


def test_rich_stream_progress_reports_counts_without_model_content() -> None:
    output = io.StringIO()
    reporter = RichProgressReporter(Console(file=output, force_terminal=True, width=140))

    reporter.emit(PipelineEvent("pipeline_started", total_steps=1))
    reporter.emit(PipelineEvent("step_started", step_name="Extract Evidence"))
    reporter.emit(
        PipelineEvent(
            "model_progress",
            message="SECRET MODEL CONTENT",
            data={"attempt": 2, "streamed_characters": 1234},
        )
    )
    reporter.emit(PipelineEvent("pipeline_completed", "done"))

    rendered = output.getvalue()
    assert "1234 chars" in rendered
    assert "attempt 2" in rendered
    assert "SECRET MODEL CONTENT" not in rendered


def _render_responsive_progress(width: int) -> str:
    output = io.StringIO()
    reporter = RichProgressReporter(Console(file=output, force_terminal=False, width=width))
    reporter.emit(PipelineEvent("pipeline_started", total_steps=36))
    reporter.emit(
        PipelineEvent(
            "step_started",
            step_name="Recover Missing Novelty Verifications",
            data={"model_step": True},
        )
    )
    reporter.emit(PipelineEvent("model_request_started", data={"attempt": 2}))
    reporter.emit(
        PipelineEvent(
            "model_progress",
            data={"attempt": 2, "streamed_characters": 123456},
        )
    )
    reporter.close()
    return output.getvalue()


def test_rich_progress_preserves_all_information_at_supported_widths() -> None:
    required = (
        "Recover Missing Novelty Verifications",
        "0/36",
        "0:00:00",
        "receiving model output",
        "papers 0",
        "model calls 1",
        "output 123456 chars",
        "cumulative 123456",
        "tokens 0",
        "attempt 2",
        "warnings 0",
    )

    for width in (60, 80, 100, 140, 170, 200, 240):
        rendered = _render_responsive_progress(width)
        normalized = " ".join(rendered.split())
        assert "…" not in rendered
        assert all(value in normalized for value in required), (width, rendered)


def test_rich_progress_uses_responsive_row_counts() -> None:
    narrow_lines = [line for line in _render_responsive_progress(60).splitlines() if line.strip()]
    screenshot_lines = [
        line for line in _render_responsive_progress(170).splitlines() if line.strip()
    ]
    wide_lines = [line for line in _render_responsive_progress(240).splitlines() if line.strip()]

    assert len(narrow_lines) >= 3
    assert len(screenshot_lines) == 2
    assert len(wide_lines) == 1


def test_pipeline_elapsed_time_does_not_truncate_after_one_day(monkeypatch) -> None:
    clock = [10.0]
    monkeypatch.setattr("agent.reporting.time.monotonic", lambda: clock[0])
    output = io.StringIO()
    reporter = RichProgressReporter(Console(file=output, force_terminal=False, width=240))
    reporter.emit(PipelineEvent("pipeline_started", total_steps=36))
    reporter.emit(PipelineEvent("step_started", step_name="Plan Research"))
    clock[0] += 25 * 3600 + 61
    reporter.close()

    assert "25:01:01" in output.getvalue()
    assert "…" not in output.getvalue()
