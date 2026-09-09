"""Pipeline progress events and terminal reporters."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from rich.cells import cell_len
from rich.console import Console, ConsoleOptions, RenderableType, RenderResult
from rich.progress import (
    BarColumn,
    Progress,
    ProgressColumn,
    SpinnerColumn,
    Task,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Column, Table
from rich.text import Text


@dataclass(frozen=True)
class PipelineEvent:
    """A presentation neutral pipeline lifecycle event."""

    kind: str
    message: str = ""
    step_name: str | None = None
    step_index: int | None = None
    total_steps: int | None = None
    data: dict[str, Any] = field(default_factory=dict)


class PipelineReporter(Protocol):
    def emit(self, event: PipelineEvent) -> None: ...

    def close(self) -> None: ...


class NullReporter:
    def emit(self, event: PipelineEvent) -> None:
        del event

    def close(self) -> None:
        return None


class ConsoleReporter:
    """Simple readable output for subcommands and non interactive terminals."""

    def __init__(self, console: Console) -> None:
        self.console = console

    def emit(self, event: PipelineEvent) -> None:
        if event.kind == "preflight_started":
            models = ", ".join(event.data.get("models", []))
            self.console.print(f"[cyan]Validating Ollama models:[/cyan] {models}")
        elif event.kind == "preflight_completed":
            self.console.print(
                "[green]✓[/green] Basic model transport and response checks completed"
            )
        elif event.kind == "pipeline_started":
            self.console.print(f"[bold cyan]{event.message}[/bold cyan]")
        elif event.kind == "step_started":
            self.console.print(
                f"[cyan][{event.step_index}/{event.total_steps}][/cyan] {event.step_name}..."
            )
        elif event.kind == "step_completed":
            self.console.print(f"[green]✓[/green] {event.step_name}")
        elif event.kind == "step_skipped":
            self.console.print(f"[yellow]↷[/yellow] {event.message}")
        elif event.kind == "warning":
            self.console.print(Text(f"Warning: {event.message}", style="yellow"))
        elif event.kind == "error":
            self.console.print(Text(event.message, style="bold red"))
        elif event.kind == "saved":
            self.console.print(
                f"[green]Saved:[/green] {event.data['text_path']} and {event.data['json_path']}"
            )

    def close(self) -> None:
        return None


class PipelineElapsedColumn(ProgressColumn):
    """Render elapsed time from the original pipeline start across UI interruptions."""

    def render(self, task: Task) -> Text:
        started = task.fields.get("pipeline_started_at")
        elapsed = max(0.0, time.monotonic() - float(started)) if started is not None else 0.0
        seconds = round(elapsed)
        hours, remainder = divmod(seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return Text(f"{hours}:{minutes:02}:{seconds:02}", style="progress.elapsed")


class _ResponsivePipelineRenderable:
    """Render one pipeline task without allowing Rich to discard trailing fields."""

    _BAR_MINIMUM = 24
    _NARROW_WIDTH = 80

    def __init__(
        self,
        *,
        spinner: RenderableType,
        description: Text,
        bar: RenderableType,
        progress_count: Text,
        elapsed: Text,
        details: Text,
    ) -> None:
        self.spinner = spinner
        self.description = description
        self.bar = bar
        self.progress_count = progress_count
        self.elapsed = elapsed
        self.details = details

    def _wide_required_width(self) -> int:
        # Six table cells create five padded boundaries. Keep a small safety
        # margin so Rich never has to ellipsize a supposedly fitting row.
        return (
            1
            + cell_len(self.description.plain)
            + self._BAR_MINIMUM
            + cell_len(self.progress_count.plain)
            + cell_len(self.elapsed.plain)
            + cell_len(self.details.plain)
            + 12
        )

    def _wide(self) -> Table:
        table = Table.grid(
            Column(width=1, no_wrap=True),
            Column(width=cell_len(self.description.plain), no_wrap=True),
            Column(ratio=1, min_width=self._BAR_MINIMUM),
            Column(width=cell_len(self.progress_count.plain), no_wrap=True),
            Column(width=cell_len(self.elapsed.plain), no_wrap=True),
            Column(width=cell_len(self.details.plain), no_wrap=True),
            padding=(0, 1),
            expand=True,
        )
        table.add_row(
            self.spinner,
            self.description,
            self.bar,
            self.progress_count,
            self.elapsed,
            self.details,
        )
        return table

    def _header(self, *, include_bar: bool) -> Table:
        columns = [
            Column(width=1, no_wrap=True),
            Column(ratio=2, min_width=12, overflow="fold"),
        ]
        if include_bar:
            columns.append(Column(ratio=1, min_width=8))
        columns.extend(
            (
                Column(width=cell_len(self.progress_count.plain), no_wrap=True),
                Column(width=cell_len(self.elapsed.plain), no_wrap=True),
            )
        )
        table = Table.grid(*columns, padding=(0, 1), expand=True)
        cells: list[RenderableType] = [self.spinner, self.description]
        if include_bar:
            cells.append(self.bar)
        cells.extend((self.progress_count, self.elapsed))
        table.add_row(*cells)
        return table

    def _full_width_bar(self) -> Table:
        table = Table.grid(Column(ratio=1), expand=True)
        table.add_row(self.bar)
        return table

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        del console
        width = options.max_width
        if self._wide_required_width() <= width:
            yield self._wide()
            return
        if width >= self._NARROW_WIDTH:
            yield self._header(include_bar=True)
            yield self.details
            return
        yield self._header(include_bar=False)
        yield self._full_width_bar()
        yield self.details


class ResponsivePipelineColumn(ProgressColumn):
    """Build a width aware multi line progress display for a single pipeline task."""

    def __init__(self) -> None:
        super().__init__(table_column=Column(ratio=1, overflow="fold"))
        self._spinner = SpinnerColumn()
        self._bar = BarColumn(bar_width=None)
        self._elapsed = PipelineElapsedColumn()

    def render(self, task: Task) -> RenderableType:
        total = task.total if task.total is not None else 0
        progress_count = Text(f"{task.completed:.0f}/{total:.0f}")
        fields = task.fields
        details = Text(
            f"{fields.get('unit_progress', '')}"
            f"{fields.get('request_phase', 'preparing')} • "
            f"papers {fields.get('papers', 0)} • "
            f"model calls {fields.get('model_calls', 0)} • "
            f"output {fields.get('current_chars', 0)} chars • "
            f"cumulative {fields.get('total_chars', 0)} • "
            f"tokens {fields.get('output_tokens', 0)} • "
            f"attempt {fields.get('attempt', 1)} • "
            f"warnings {fields.get('warnings', 0)}",
            style="dim",
            overflow="fold",
            no_wrap=False,
        )
        return _ResponsivePipelineRenderable(
            spinner=self._spinner.render(task),
            description=Text(task.description, style="bold cyan", overflow="fold"),
            bar=self._bar.render(task),
            progress_count=progress_count,
            elapsed=self._elapsed.render(task),
            details=details,
        )


class RichProgressReporter:
    """Live progress display used by the focused interactive workspace."""

    def __init__(
        self,
        console: Console,
        *,
        progress_factory: Callable[..., Progress] = Progress,
    ) -> None:
        self.console = console
        self._progress_factory = progress_factory
        self.progress: Progress | None = None
        self.task_id: int | None = None
        self.saved_paths: tuple[Path, Path] | None = None
        self.warning_count = 0
        self.paper_count = 0
        self.model_calls = 0
        self.generated_characters = 0
        self.output_tokens = 0
        self.completed_steps = 0
        self.total_steps = 1
        self.error_data: dict[str, Any] = {}
        self._pipeline_started_at: float | None = None
        self._step_started_calls = 0
        self._displayed_warning_keys: set[tuple[str | None, str]] = set()

    def _stop(self) -> None:
        if self.progress is not None:
            self.progress.stop()
            self.progress = None
            self.task_id = None

    def _start_pipeline_progress(self) -> None:
        self.progress = self._progress_factory(
            ResponsivePipelineColumn(),
            console=self.console,
            transient=False,
            expand=True,
        )
        self.progress.start()
        self.task_id = self.progress.add_task(
            "Preparing research",
            total=self.total_steps,
            completed=self.completed_steps,
            papers=self.paper_count,
            model_calls=self.model_calls,
            current_chars=0,
            response_chars=0,
            total_chars=self.generated_characters,
            output_tokens=self.output_tokens,
            attempt=1,
            warnings=self.warning_count,
            request_phase="preparing",
            unit_progress="",
            pipeline_started_at=self._pipeline_started_at or time.monotonic(),
        )

    def emit(self, event: PipelineEvent) -> None:
        if event.kind == "preflight_started":
            models = ", ".join(event.data.get("models", []))
            self.progress = self._progress_factory(
                SpinnerColumn(),
                TextColumn("[bold cyan]{task.description}"),
                TimeElapsedColumn(),
                console=self.console,
            )
            self.progress.start()
            self.task_id = self.progress.add_task(f"Validating models: {models}", total=None)
        elif event.kind == "preflight_completed":
            self._stop()
            self.console.print(
                "[green]✓ Selected models passed basic transport and response mode checks.[/green]"
            )
        elif event.kind == "pipeline_started":
            self._stop()
            if self._pipeline_started_at is None:
                self._pipeline_started_at = time.monotonic()
            self.total_steps = event.total_steps or 1
            self.completed_steps = 0
            self._start_pipeline_progress()
        elif event.kind == "step_started" and self.progress is not None:
            self._step_started_calls = 0
            self.progress.update(
                self.task_id,
                description=event.step_name or "Working",
                current_chars=0,
                response_chars=0,
                attempt=1,
                request_phase=(
                    "preparing model request" if event.data.get("model_step") else "local"
                ),
                unit_progress="",
            )
        elif event.kind == "model_request_started" and self.progress is not None:
            self.model_calls += 1
            self._step_started_calls += 1
            request_kind = str(event.data.get("request_kind", "contract"))
            phase = (
                "repair 1/1 · waiting for first token"
                if request_kind == "focused_repair"
                else "waiting for first token"
            )
            unit_index = int(event.data.get("unit_index", 0) or 0)
            unit_total = int(event.data.get("unit_total", 0) or 0)
            unit_progress = (
                f"unit {unit_index}/{unit_total} • " if unit_index > 0 and unit_total > 0 else ""
            )
            self.progress.update(
                self.task_id,
                model_calls=self.model_calls,
                request_phase=phase,
                current_chars=0,
                response_chars=0,
                attempt=int(event.data.get("attempt", 1)),
                unit_progress=unit_progress,
            )
        elif event.kind == "model_progress" and self.progress is not None:
            self.progress.update(
                self.task_id,
                current_chars=int(event.data.get("streamed_characters", 0)),
                response_chars=int(event.data.get("streamed_characters", 0)),
                total_chars=self.generated_characters
                + int(event.data.get("streamed_characters", 0)),
                attempt=int(event.data.get("attempt", 1)),
                request_phase="receiving model output",
            )
        elif event.kind == "model_request_completed" and self.progress is not None:
            self.progress.update(
                self.task_id,
                request_phase=(
                    "conservative fallback" if event.data.get("fallback") else "validating output"
                ),
            )
        elif event.kind in {"step_completed", "step_skipped"} and self.progress is not None:
            self.completed_steps += 1
            self.paper_count = max(self.paper_count, int(event.data.get("paper_count", 0)))
            actual_calls = int(event.data.get("model_calls", 0))
            self.model_calls += max(0, actual_calls - self._step_started_calls)
            current_chars = int(event.data.get("generation_characters", 0))
            self.generated_characters += current_chars
            self.output_tokens += int(event.data.get("output_tokens", 0))
            self._step_started_calls = 0
            self.progress.update(
                self.task_id,
                advance=1,
                papers=self.paper_count,
                model_calls=self.model_calls,
                warnings=self.warning_count,
                current_chars=current_chars,
                response_chars=current_chars,
                total_chars=self.generated_characters,
                output_tokens=self.output_tokens,
                request_phase="complete",
                unit_progress="",
            )
        elif event.kind == "scope_selection_started":
            self._stop()
        elif event.kind == "scope_selection_completed":
            self._start_pipeline_progress()
        elif event.kind == "warning":
            category = event.message.split(":", 1)[-1].strip().split(";", 1)[0]
            key = (event.step_name, category)
            if key in self._displayed_warning_keys:
                return
            self._displayed_warning_keys.add(key)
            self.warning_count += 1
            if self.progress is not None:
                self.progress.update(self.task_id, warnings=self.warning_count)
                self.progress.console.print(Text(f"Warning: {event.message}", style="yellow"))
        elif event.kind in {"pipeline_completed", "cancelled", "error"}:
            self._stop()
            if event.kind == "pipeline_completed":
                self.console.print(
                    f"[bold green]✓ Research completed[/bold green]: {event.message}"
                )
            elif event.kind == "cancelled":
                self.console.print(
                    "[yellow]Research run cancelled; no incomplete result was saved.[/yellow]"
                )
            else:
                self.error_data = dict(event.data)
                self.console.print(Text(event.message, style="bold red"))
                if event.data.get("diagnostic_path"):
                    self.console.print(
                        Text(
                            f"Diagnostic saved: {event.data['diagnostic_path']}",
                            style="yellow",
                        )
                    )
            self._pipeline_started_at = None
        elif event.kind == "saved":
            self.saved_paths = (Path(event.data["text_path"]), Path(event.data["json_path"]))

    def close(self) -> None:
        self._stop()
