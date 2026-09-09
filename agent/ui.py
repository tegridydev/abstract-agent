"""Focused Rich terminal workspace for Abstract Agent."""

from __future__ import annotations

import argparse
import json
import os
import platform
from collections.abc import Callable
from pathlib import Path
from typing import Any

from pydantic import ValidationError
from rich import box
from rich.console import Console, Group
from rich.panel import Panel
from rich.progress import BarColumn, DownloadColumn, Progress, SpinnerColumn, TextColumn
from rich.prompt import Confirm, IntPrompt, Prompt
from rich.table import Table
from rich.text import Text

from . import __version__, multi_source
from .agent import (
    EXIT_CANCELLED,
    EXIT_CONFIG,
    EXIT_EVIDENCE,
    EXIT_SUCCESS,
    PipelineManager,
    execute_run,
    load_pipeline_config,
)
from .knowledge import KnowledgeStore
from .ollama_models import (
    OllamaModelService,
    describe_ollama_error,
    is_cloud_model,
    parse_ollama_host,
)
from .pipeline.output import ModelContractProfileStore
from .presentation import (
    RunRecord,
    discover_runs,
    load_interactive_settings,
    save_interactive_settings,
)
from .reporting import RichProgressReporter
from .schemas import InstalledModel, InteractiveSettings, ModelAssignments, PipelineConfig

DEFAULT_SETTINGS_PATH = Path(".abstract-agent.toml")
MODEL_ROLES = (
    ("planning_evidence", "Planning & evidence"),
    ("creative", "Creative generation"),
    ("critique", "Critique & verification"),
    ("writing", "Final writing"),
)


class InteractiveApplication:
    """Small, testable state machine for the terminal workspace."""

    def __init__(
        self,
        *,
        console: Console,
        settings_path: Path = DEFAULT_SETTINGS_PATH,
        model_service_factory: Callable[..., OllamaModelService] | None = None,
        run_executor: Callable[..., int] | None = None,
        source_checker: Callable[[], list[Any]] | None = None,
        pipeline_manager_factory: Callable[..., PipelineManager] | None = None,
        reporter_factory: Callable[[Console], RichProgressReporter] | None = None,
        output_profile_store: ModelContractProfileStore | None = None,
    ) -> None:
        self.console = console
        self.settings_path = settings_path
        self._model_service_factory = model_service_factory
        self._run_executor = run_executor
        self._source_checker = source_checker
        self._pipeline_manager_factory = pipeline_manager_factory
        self._reporter_factory = reporter_factory
        self._output_profile_store = output_profile_store or ModelContractProfileStore()
        self.settings, self.settings_warning = load_interactive_settings(settings_path)
        self.config: PipelineConfig | None = None
        self._notice: tuple[str, str] | None = None

    def _load_config(self) -> bool:
        try:
            self.config = load_pipeline_config()
            return True
        except ValueError as error:
            self.console.print(Panel(Text(str(error), style="red"), title="Configuration error"))
            return False

    def _header(self) -> None:
        assert self.config is not None
        if self.console.is_terminal:
            self.console.clear()
        models = sorted(set(self._resolved_role_models(self.settings.model_assignments).values()))
        subtitle = (
            f"Models: {', '.join(models) or 'N/A'}  •  Evidence minimum: "
            f"{self.settings.minimum_papers}  •  Output: {self.settings.output_dir}"
        )
        self.console.print(
            Panel(
                Text("ABSTRACT AGENT", style="bold cyan", justify="center"),
                subtitle=subtitle,
                border_style="cyan",
            )
        )

    def _screen(self, title: str, subtitle: str | None = None) -> None:
        """Draw one focused terminal view and consume any pending notice."""
        self._header()
        heading = Text(title, style="bold cyan")
        if subtitle:
            heading.append(f"  {subtitle}", style="dim")
        self.console.print(heading)
        if self._notice:
            message, style = self._notice
            self.console.print(Panel(Text(message, style=style), border_style=style))
            self._notice = None

    def _set_notice(self, message: str, style: str = "green") -> None:
        self._notice = (message, style)

    def run(self) -> int:
        if not self.console.is_terminal:
            self.console.print(
                "Interactive mode requires a terminal. Use "
                '[bold]uv run abs.py run --topic "..."[/bold] or --help.'
            )
            return EXIT_CONFIG
        if not self._load_config():
            return EXIT_CONFIG
        while True:
            self._screen("Main menu")
            if self.settings_warning:
                self.console.print(Text(self.settings_warning, style="yellow"))
                self.settings_warning = None
            self.console.print(
                Group(
                    Text("1  New Research", style="bold"),
                    Text("2  Recent Runs & Failures"),
                    Text("3  Models & Ollama"),
                    Text("4  System Status"),
                    Text("5  Preferences"),
                    Text("k  Local Knowledge"),
                    Text("6  Exit"),
                )
            )
            try:
                choice = Prompt.ask(
                    "\nSelect",
                    choices=["1", "2", "3", "4", "5", "6", "k"],
                    console=self.console,
                )
                if choice == "1":
                    self.new_research()
                elif choice == "2":
                    self.recent_runs()
                elif choice == "3":
                    self.models_menu()
                elif choice == "4":
                    self.system_status()
                elif choice == "5":
                    self.preferences()
                elif choice == "k":
                    self.knowledge_menu()
                elif choice == "6":
                    self.console.print("[cyan]Goodbye.[/cyan]")
                    return EXIT_SUCCESS
            except (KeyboardInterrupt, EOFError):
                self.console.print("\n[cyan]Goodbye.[/cyan]")
                return EXIT_SUCCESS

    def _pause(self) -> None:
        Prompt.ask(
            "\nPress Enter to continue", default="", show_default=False, console=self.console
        )

    def _configured_role_models(self) -> dict[str, str]:
        assert self.config is not None
        configured: dict[str, str] = {}
        for step in self.config.steps:
            role = getattr(step, "model_role", "general")
            model = getattr(step, "model", None)
            if role != "general" and model and role not in configured:
                configured[role] = model
        fallback = next(
            (step.model for step in self.config.steps if getattr(step, "model", None)),
            "N/A",
        )
        return {key: configured.get(key, fallback) for key, _ in MODEL_ROLES}

    def _resolved_role_models(self, assignments: ModelAssignments) -> dict[str, str]:
        configured = self._configured_role_models()
        return {
            role: assignments.resolve(role, configured[role])  # type: ignore[arg-type]
            for role, _ in MODEL_ROLES
        }

    def new_research(self, topic: str | None = None) -> int:
        assert self.config is not None
        self._screen("New research", "Enter a topic to begin")
        while not topic or not topic.strip():
            topic = Prompt.ask("Research topic", console=self.console).strip()
            if not topic:
                self.console.print("[yellow]Please enter a research topic.[/yellow]")

        minimum_papers = self.settings.minimum_papers
        allow_insufficient = self.settings.allow_insufficient_literature
        output_dir = self.settings.output_dir
        model_assignments = self.settings.model_assignments
        search_depth = self.settings.search_depth
        while True:
            self._screen("New research", "Review and start")
            summary = Table(box=box.SIMPLE, show_header=False)
            summary.add_column(style="cyan", no_wrap=True)
            summary.add_column()
            summary.add_row("Topic", topic)
            summary.add_row("Minimum papers", str(minimum_papers))
            summary.add_row("Evidence override", "enabled" if allow_insufficient else "disabled")
            summary.add_row("Search depth", search_depth)
            summary.add_row("Research scope", "Ask after evidence discovery")
            summary.add_row("Output", output_dir)
            summary.add_row("Pipeline", f"{self.config.pipeline_name} v{self.config.version}")
            summary.add_row(
                "Ollama inactivity timeout",
                f"{self.settings.ollama_timeout or self.config.ollama_timeout_seconds:g} seconds",
            )
            resolved_models = self._resolved_role_models(model_assignments)
            for role, label in MODEL_ROLES:
                summary.add_row(label, resolved_models[role])
            self.console.print(Panel(summary, title="Run summary"))
            if len(set(resolved_models.values())) > 1:
                self.console.print(
                    "[yellow]Multiple models may increase loading time and memory pressure as "
                    "Ollama switches between roles.[/yellow]"
                )
            host = self.settings.ollama_host or self.config.ollama_host
            try:
                public_host, remote = parse_ollama_host(host)
            except ValueError as error:
                self.console.print(f"[red]Invalid Ollama host: {error}[/red]")
                return EXIT_CONFIG
            if remote:
                self.console.print(
                    f"[yellow]Ollama host {public_host} is remote; prompts may leave this machine."
                    "[/yellow]"
                )
            if not self.settings.confirm_before_run:
                break
            self.console.print(
                "[bold]s[/bold] Start research · [bold]c[/bold] Customize · [bold]b[/bold] Cancel"
            )
            action = Prompt.ask(
                "Action", choices=["s", "c", "b"], default="s", console=self.console
            )
            if action == "s":
                break
            if action == "b":
                self._set_notice("Research run cancelled.", "yellow")
                return EXIT_CANCELLED
            while True:
                self._screen("Customize run", topic)
                custom = Table(box=box.ROUNDED)
                custom.add_column("#")
                custom.add_column("Setting")
                custom.add_column("Value")
                custom.add_row("1", "Minimum papers", str(minimum_papers))
                custom.add_row("2", "Evidence override", str(allow_insufficient))
                custom.add_row("3", "Search depth", search_depth)
                custom.add_row("4", "Output directory", output_dir)
                custom.add_row("5", "Model assignments", "Edit roles")
                self.console.print(custom)
                setting = Prompt.ask(
                    "Select setting or b=summary",
                    choices=["1", "2", "3", "4", "5", "b"],
                    default="b",
                    console=self.console,
                )
                if setting == "b":
                    break
                if setting == "1":
                    minimum_papers = IntPrompt.ask(
                        "Minimum unique papers", default=minimum_papers, console=self.console
                    )
                elif setting == "2":
                    allow_insufficient = Confirm.ask(
                        "Continue when evidence is insufficient?",
                        default=allow_insufficient,
                        console=self.console,
                    )
                elif setting == "3":
                    search_depth = Prompt.ask(
                        "Search depth",
                        choices=["standard", "deep"],
                        default=search_depth,
                        console=self.console,
                    )
                elif setting == "4":
                    output_dir = Prompt.ask(
                        "Output directory", default=output_dir, console=self.console
                    )
                else:
                    updated = self._assignment_editor(model_assignments)
                    if updated is not None:
                        model_assignments = updated
                self._set_notice("Run settings updated.", "cyan")

        cloud_models = sorted(model for model in resolved_models.values() if is_cloud_model(model))
        allow_cloud_models = not cloud_models or Confirm.ask(
            "Selected cloud models send prompts to Ollama's cloud service. Continue?",
            default=False,
            console=self.console,
        )
        if not allow_cloud_models:
            self._set_notice("Research run cancelled before off device inference.", "yellow")
            return EXIT_CANCELLED
        args = argparse.Namespace(
            topic=topic,
            output_dir=output_dir,
            ollama_host=self.settings.ollama_host,
            ollama_timeout=self.settings.ollama_timeout,
            minimum_papers=minimum_papers,
            allow_insufficient_literature=allow_insufficient,
            allow_cloud_models=bool(cloud_models),
            search_depth=search_depth,
            model_assignments=model_assignments,
            model=None,
            planning_model=None,
            creative_model=None,
            critique_model=None,
            writing_model=None,
            scope="ask",
            knowledge_enabled=self.settings.knowledge_enabled,
            knowledge_dir=self.settings.knowledge_dir,
            embedding_model=self.settings.embedding_model,
            knowledge_refresh=self.settings.knowledge_refresh,
        )
        while True:
            reporter = (self._reporter_factory or RichProgressReporter)(self.console)
            exit_code = (self._run_executor or execute_run)(
                args,
                reporter=reporter,
                display_results=False,
                scope_selector=self._select_research_scope,
            )
            if exit_code == EXIT_SUCCESS and reporter.saved_paths:
                try:
                    payload = json.loads(reporter.saved_paths[1].read_text(encoding="utf-8"))
                    self._render_record(RunRecord(reporter.saved_paths[1], payload))
                except (OSError, json.JSONDecodeError) as error:
                    self.console.print(
                        Text(f"Result saved but could not be displayed: {error}", style="yellow")
                    )
                break
            if (
                reporter.error_data.get("recoverable")
                and reporter.error_data.get("diagnostic_path")
                and Confirm.ask("Resume from the failed stage?", default=True, console=self.console)
            ):
                args.diagnostic = reporter.error_data["diagnostic_path"]
                continue
            if exit_code != EXIT_EVIDENCE:
                break
            choices = ["e", "m", "c"]
            labels = "[e] Edit topic  [m] Adjust minimum  [c] Cancel"
            if args.search_depth != "deep":
                choices.insert(0, "d")
                labels = "[d] Deep search and retry  " + labels
            self.console.print(Panel(labels, title="Evidence recovery", border_style="yellow"))
            action = Prompt.ask(
                "Recovery action",
                choices=choices,
                default="d" if "d" in choices else "e",
                console=self.console,
            )
            if action == "d":
                args.search_depth = "deep"
                continue
            if action == "m":
                args.minimum_papers = IntPrompt.ask(
                    "Minimum papers for this retry",
                    default=max(1, min(args.minimum_papers, 5)),
                    console=self.console,
                )
                continue
            if action == "e":
                return self.new_research()
            break
        self._pause()
        return exit_code

    def _select_research_scope(self, topic: str, options: list[dict[str, Any]]) -> dict[str, Any]:
        self._screen("Choose research scope", topic)
        table = Table(box=box.ROUNDED)
        table.add_column("#", justify="right", style="cyan")
        table.add_column("Scope")
        table.add_column("Evidence")
        table.add_row("0", "Stay broad", "Use evidence across domains")
        for index, option in enumerate(options, start=1):
            table.add_row(
                str(index),
                str(option["label"]),
                f"{len(option.get('paper_ids', []))} papers · {option['description']}",
            )
        self.console.print(table)
        choice = IntPrompt.ask(
            "Scope",
            default=0,
            choices=[str(index) for index in range(len(options) + 1)],
            console=self.console,
        )
        if choice == 0:
            return {
                "mode": "broad",
                "label": "Broad topic",
                "source": "interactive",
                "supporting_paper_ids": [],
            }
        selected = options[choice - 1]
        return {
            "mode": "domain",
            "label": selected["label"],
            "source": "interactive",
            "supporting_paper_ids": selected["paper_ids"],
        }

    def recent_runs(self) -> None:
        self._screen("Recent runs & failures")
        records, warnings = discover_runs(Path(self.settings.output_dir))
        if not records:
            self.console.print("[yellow]No saved runs found.[/yellow]")
            self._pause()
            return
        page_size = self.settings.history_page_size
        page = 0
        while True:
            page_count = max(1, (len(records) + page_size - 1) // page_size)
            page = min(page, page_count - 1)
            visible = records[page * page_size : (page + 1) * page_size]
            self._screen("Recent runs & failures", f"Page {page + 1} of {page_count}")
            for warning in warnings[:3]:
                self.console.print(Text(warning, style="yellow"))
            table = Table(
                title="Saved research",
                box=box.ROUNDED,
                show_lines=False,
            )
            for heading in ("#", "Type", "When", "Topic", "Status", "Grounding", "Warnings"):
                table.add_column(heading)
            for index, record in enumerate(visible, start=1):
                table.add_row(
                    str(index),
                    record.kind,
                    record.timestamp,
                    record.topic[:60],
                    record.status if record.kind == "Failure" else record.novelty_status,
                    record.evidence_coverage,
                    str(len(record.payload.get("warnings", []))),
                )
            self.console.print(table)
            choices = [str(index) for index in range(1, len(visible) + 1)] + ["b"]
            if page > 0:
                choices.append("p")
            if page + 1 < page_count:
                choices.append("n")
            choice = Prompt.ask(
                "Select run, n=next, p=previous, or b=back",
                choices=choices,
                default="b",
                console=self.console,
            )
            if choice == "b":
                return
            if choice == "n":
                page += 1
                continue
            if choice == "p":
                page -= 1
                continue
            record = visible[int(choice) - 1]
            self._screen("Run result", record.topic)
            self._render_record(record)
            action_choices = ["d", "p", "r", "b"]
            if record.resumable:
                action_choices.insert(0, "s")
            action = Prompt.ask(
                "s=resume, d=details, p=file paths, r=rerun topic, or b=back"
                if record.resumable
                else "d=details, p=file paths, r=rerun topic, or b=back",
                choices=action_choices,
                default="b",
                console=self.console,
            )
            if action == "d":
                self._screen("Run details", record.topic)
                self._render_record_details(record)
                self._pause()
            elif action == "p":
                self.console.print(f"JSON: {record.path}")
                if record.kind == "Result":
                    self.console.print(f"Text: {record.path.with_suffix('.txt')}")
                self._pause()
            elif action == "s":
                self._resume_record(record)
                return
            elif action == "r":
                self.new_research(record.topic)
                return

    def _resume_record(self, record: RunRecord) -> None:
        resolved = self._resolved_role_models(self.settings.model_assignments)
        cloud_models = sorted(model for model in resolved.values() if is_cloud_model(model))
        if cloud_models and not Confirm.ask(
            "Selected cloud models send prompts to Ollama's cloud service. Resume?",
            default=False,
            console=self.console,
        ):
            return
        args = argparse.Namespace(
            diagnostic=str(record.path),
            topic=record.topic,
            output_dir=self.settings.output_dir,
            ollama_host=self.settings.ollama_host,
            ollama_timeout=self.settings.ollama_timeout,
            minimum_papers=None,
            allow_insufficient_literature=False,
            allow_cloud_models=bool(cloud_models),
            search_depth=None,
            model_assignments=self.settings.model_assignments,
            model=None,
            planning_model=None,
            creative_model=None,
            critique_model=None,
            writing_model=None,
            scope=None,
            knowledge_enabled=self.settings.knowledge_enabled,
            knowledge_dir=self.settings.knowledge_dir,
            embedding_model=self.settings.embedding_model,
            knowledge_refresh=self.settings.knowledge_refresh,
        )
        reporter = (self._reporter_factory or RichProgressReporter)(self.console)
        exit_code = (self._run_executor or execute_run)(
            args,
            reporter=reporter,
            display_results=False,
            scope_selector=self._select_research_scope,
        )
        if exit_code == EXIT_SUCCESS and reporter.saved_paths:
            try:
                payload = json.loads(reporter.saved_paths[1].read_text(encoding="utf-8"))
                self._render_record(RunRecord(reporter.saved_paths[1], payload))
            except (OSError, json.JSONDecodeError) as error:
                self.console.print(
                    f"[yellow]Result saved but could not be displayed: {error}[/yellow]"
                )
        self._pause()

    def _render_record(self, record: RunRecord) -> None:
        context = record.context
        if record.kind == "Failure":
            active_step = record.payload.get("resume_step") or context.get("_active_step") or "N/A"
            self.console.print(
                Panel(
                    Text(
                        f"Topic: {record.topic}\nStatus: {record.status}\n"
                        f"Active step: {active_step}\nResumable: {record.resumable}\n\n"
                        f"{record.error}"
                    ),
                    title="Failed run",
                    border_style="red",
                )
            )
            return
        proposal = context.get("audited_research_proposal") or context.get("research_proposal")
        if not isinstance(proposal, dict):
            outcome = context.get("research_outcome", {})
            findings = outcome.get("findings", []) if isinstance(outcome, dict) else []
            reason = outcome.get("reason") if isinstance(outcome, dict) else None
            details = "\n".join(
                f"• {item.get('issue')}" for item in findings[:5] if isinstance(item, dict)
            )
            self.console.print(
                Panel(
                    Text(
                        f"Topic: {record.topic}\n"
                        f"Outcome: {record.outcome}\n"
                        "No proposal was generated. "
                        + (f"{reason} " if reason else "")
                        + "Any source backed evidence is saved."
                        + (f"\n\nBlocking findings\n{details}" if details else "")
                    ),
                    title="Evidence only result",
                    border_style="yellow",
                )
            )
            return
        insights = proposal.get("novel_insights", [])
        novelty = context.get("novelty_verification", {})
        proposal_status = proposal.get("proposal_status") or (
            "novelty_supported"
            if novelty.get("status") == "novelty_supported"
            else "unverified_research_direction"
        )
        unverified = proposal_status == "unverified_research_direction"
        generation = proposal.get("generation_provenance", {})
        lineage_modes = [
            str(item.get("generation_provenance", {}).get("mode", "structured_exact"))
            for item in (
                context.get("selected_candidate", {}),
                context.get("revised_candidate", {}),
                context.get("final_candidate", {}),
                proposal,
            )
            if isinstance(item, dict)
        ]
        integrity_order = {
            "structured_exact": 3,
            "structured_normalized": 2,
            "draft_projected": 1,
            "conservative_fallback": 0,
        }
        generation_mode = min(
            lineage_modes or [str(generation.get("mode", "structured_exact"))],
            key=lambda mode: integrity_order.get(mode, 0),
        )
        projected = generation_mode not in {"structured_exact", "structured_normalized"}
        scientific_integrity = str(
            context.get("revision_validity_audit", {}).get(
                "status", generation.get("scientific_integrity", "not_audited")
            )
        )
        run_integrity = (
            "degraded"
            if any(
                event.get("outcome") in {"projected", "fallback"}
                for event in record.payload.get("compilation_events", [])
                if isinstance(event, dict)
            )
            else "clean"
        )
        body = Group(
            *(
                [
                    Text(
                        "This proposal required application owned model output projection and "
                        "must be reviewed by a human.",
                        style="bold yellow",
                    ),
                    Text(""),
                ]
                if projected
                else []
            ),
            *(
                [
                    Text(
                        "Novelty has not been established; treat this as a research direction.",
                        style="bold yellow",
                    ),
                    Text(""),
                ]
                if unverified
                else []
            ),
            Text(str(proposal.get("hypothesis", "Not available")), style="bold"),
            Text(""),
            Text(str(proposal.get("abstract", "Not available"))),
            Text(""),
            Text(
                "Potential insights to test" if unverified else "Novel insights",
                style="bold cyan",
            ),
            Text("\n".join(f"• {item}" for item in insights) or "None recorded"),
        )
        panel_title = str(proposal.get("title", record.topic))
        if unverified:
            panel_title = f"Unverified research direction: {panel_title}"
        self.console.print(
            Panel(body, title=panel_title, border_style="yellow" if unverified else "none")
        )
        audit = context.get("evidence_audit", {})
        final_candidate_id = context.get("final_candidate", {}).get("candidate_id")
        score = next(
            (
                item
                for item in context.get("candidate_scorecard", [])
                if item.get("candidate_id") == final_candidate_id
            ),
            {},
        )
        open_circuits = sorted(
            {
                status.get("provider")
                for key in ("source_statuses", "novelty_source_statuses")
                for status in context.get(key, [])
                if status.get("circuit_state") == "open" and status.get("provider")
            }
        )
        slow_stages = sorted(
            context.get("_step_metrics", []),
            key=lambda item: float(item.get("elapsed_seconds", 0)),
            reverse=True,
        )[:2]
        outcome = "Unverified research direction" if unverified else "Novelty supported"
        self.console.print(
            f"Outcome: [bold]{outcome}[/bold]  •  "
            f"Contract: [bold]{generation_mode}[/bold]  •  "
            f"Scientific validity: [bold]{scientific_integrity}[/bold]  •  "
            f"Run integrity: [bold]{run_integrity}[/bold]  •  "
            f"Selection: [bold]{score.get('selection_confidence', 'N/A')}[/bold]  •  "
            f"Grounding tier: [bold]{score.get('grounding_tier', 'N/A')}[/bold]  •  "
            f"Novelty: [bold]{novelty.get('status', 'N/A')}[/bold]  •  "
            f"Citation coverage: [bold]{record.evidence_coverage}[/bold]  •  "
            f"Direct support: [bold]{record.support_coverage}[/bold]  •  "
            f"Claims: {audit.get('claim_count', 'N/A')}  •  References: "
            f"{len(proposal.get('references', []))}"
        )
        if score.get("unresolved_major_revisions"):
            self.console.print(
                f"[yellow]Unresolved major revisions: "
                f"{len(score['unresolved_major_revisions'])}[/yellow]"
            )
        if open_circuits:
            self.console.print(f"Provider circuits open: {', '.join(open_circuits)}")
        if slow_stages:
            self.console.print(
                "Slowest stages: "
                + ", ".join(
                    f"{item.get('step')} ({float(item.get('elapsed_seconds', 0)):.1f}s)"
                    for item in slow_stages
                )
            )

    def _render_record_details(self, record: RunRecord) -> None:
        context = record.context
        proposal = context.get("audited_research_proposal") or context.get("research_proposal", {})
        references = set(proposal.get("references", [])) if isinstance(proposal, dict) else set()
        if isinstance(proposal, dict):
            references.update(
                paper_id
                for link in proposal.get("claim_evidence", [])
                for paper_id in link.get("paper_ids", [])
            )
        papers = context.get("evidence_papers", [])
        table = Table(title="Referenced Evidence", box=box.ROUNDED)
        for heading in ("ID", "Title", "Year", "Sources", "URL"):
            table.add_column(heading)
        for paper in papers if isinstance(papers, list) else []:
            if references and paper.get("paper_id") not in references:
                continue
            table.add_row(
                str(paper.get("paper_id", "N/A")),
                str(paper.get("title", "N/A"))[:60],
                str(paper.get("year", "N/A")),
                ", ".join(paper.get("sources", [paper.get("source", "N/A")])),
                str(paper.get("url", "N/A")),
            )
        self.console.print(table)

        for title, key in (
            ("Initial Literature Source Health", "source_statuses"),
            ("Novelty Prior Art Source Health", "novelty_source_statuses"),
        ):
            statuses = record.payload.get(key, context.get(key, []))
            status_table = Table(title=title)
            for heading in ("Provider", "Status", "Results", "Failed", "Warnings"):
                status_table.add_column(heading)
            for status in statuses if isinstance(statuses, list) else []:
                if isinstance(status, dict):
                    status_table.add_row(
                        str(status.get("provider", "N/A")),
                        str(status.get("status", "N/A")),
                        str(status.get("result_count", "N/A")),
                        str(status.get("failed_query_count", 0)),
                        str(status.get("warning_query_count", 0)),
                    )
            self.console.print(status_table)
        self.console.print(
            f"Duration: {context.get('_pipeline_duration_seconds', 'N/A')}s  •  "
            f"Model stages: {len(record.payload.get('llm_metrics', {}))}  •  "
            f"Warnings: {len(record.payload.get('warnings', []))}"
        )

    @staticmethod
    def _format_size(size: int) -> str:
        value = float(size)
        for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
            if value < 1024 or unit == "TiB":
                return f"{value:.1f} {unit}"
            value /= 1024
        return f"{value:.1f} TiB"

    def _model_usage(self, assignments: ModelAssignments) -> dict[str, str]:
        usage: dict[str, list[str]] = {}
        labels = {
            "planning_evidence": "Planning",
            "creative": "Creative",
            "critique": "Critique",
            "writing": "Writing",
        }
        for role, model in self._resolved_role_models(assignments).items():
            usage.setdefault(model, []).append(labels[role])
        return {
            model: "All" if len(roles) == len(MODEL_ROLES) else ", ".join(roles)
            for model, roles in usage.items()
        }

    def _display_models(
        self, models: list[InstalledModel], assignments: ModelAssignments | None = None
    ) -> None:
        table = Table(title="Installed models", box=box.ROUNDED)
        headings = ["#", "Model", "Parameters"]
        if self.console.width >= 90:
            headings.extend(["Quantization", "Size"])
        headings.extend(["State", "Used by"])
        for heading in headings:
            table.add_column(heading)
        usage = self._model_usage(assignments or self.settings.model_assignments)
        for index, model in enumerate(models, start=1):
            values = [
                str(index),
                model.name,
                model.parameter_size,
            ]
            if self.console.width >= 90:
                values.extend([model.quantization, self._format_size(model.size)])
            values.extend(
                ["loaded" if model.running else "installed", usage.get(model.name, "not assigned")]
            )
            style = "green" if model.name in usage else None
            table.add_row(*values, style=style)
        self.console.print(table)

    def _display_assignments(self, assignments: ModelAssignments) -> None:
        table = Table(title="Role assignments", box=box.SIMPLE, show_header=False)
        table.add_column(style="cyan", no_wrap=True)
        table.add_column()
        resolved = self._resolved_role_models(assignments)
        for role, label in MODEL_ROLES:
            table.add_row(label, resolved[role])
        self.console.print(table)

    def _select_model(
        self,
        service: OllamaModelService,
        models: list[InstalledModel],
        label: str,
    ) -> str | None:
        if not models:
            self.console.print("[yellow]No installed models are available.[/yellow]")
            return None
        self._display_models(models)
        choice = Prompt.ask(
            f"Select {label} model or b=back",
            choices=[str(index) for index in range(1, len(models) + 1)] + ["b"],
            default="b",
            console=self.console,
        )
        if choice == "b":
            return None
        selected = models[int(choice) - 1]
        try:
            with self.console.status(f"Checking {selected.name} capabilities..."):
                details = service.show_model(selected.name, selected)
            if "completion" not in details.capabilities:
                self.console.print(
                    Text(
                        f"{selected.name} cannot be selected because it does not advertise "
                        "completion capability.",
                        style="red",
                    )
                )
                return None
        except Exception as error:
            self.console.print(
                Text(f"Could not inspect model: {describe_ollama_error(error)}", style="red")
            )
            return None
        return selected.name

    def _assignment_editor(self, current: ModelAssignments) -> ModelAssignments | None:
        assert self.config is not None
        host = self.settings.ollama_host or self.config.ollama_host
        service = (self._model_service_factory or OllamaModelService)(
            host, self.settings.ollama_timeout or self.config.ollama_timeout_seconds
        )
        try:
            with self.console.status(f"Loading models from {host}..."):
                models = service.list_models()
            working = current.model_copy(deep=True)
            while True:
                self._screen(
                    "Model assignments",
                    "Choose a role, use one model everywhere, then save",
                )
                self._display_assignments(working)
                self.console.print(
                    "[bold]a[/bold]  Set all roles   "
                    "[bold]1 to 4[/bold]  Change one role   "
                    "[bold]d[/bold]  YAML defaults   "
                    "[bold]s[/bold]  Save   [bold]b[/bold]  Cancel"
                )
                action = Prompt.ask(
                    "Action",
                    choices=["a", "1", "2", "3", "4", "d", "s", "b"],
                    default="s",
                    console=self.console,
                )
                if action == "b":
                    return None
                if action == "s":
                    return working
                if action == "d":
                    working = ModelAssignments()
                    self._set_notice("YAML model defaults selected.")
                    continue
                if action == "a":
                    selected = self._select_model(service, models, "all roles")
                    if selected:
                        working = ModelAssignments(all_model=selected)
                        self._set_notice(f"{selected} selected for all roles.")
                    continue
                role, label = MODEL_ROLES[int(action) - 1]
                selected = self._select_model(service, models, label)
                if selected:
                    working = working.model_copy(update={role: selected})
                    self._set_notice(f"{label} will use {selected}.")
        except Exception as error:
            self.console.print(
                Text(f"Model selection failed: {describe_ollama_error(error)}", style="red")
            )
            return None
        finally:
            service.close()

    def _pull_model(self, service: OllamaModelService) -> bool:
        name = Prompt.ask("Exact Ollama model name", console=self.console).strip()
        if is_cloud_model(name) and not Confirm.ask(
            "This appears to be a cloud model and may send prompts off device. Continue?",
            default=False,
            console=self.console,
        ):
            return False
        host = self.settings.ollama_host or self.config.ollama_host
        if not Confirm.ask(
            f"Pull '{name}' to Ollama host {host}? This may download several gigabytes.",
            default=False,
            console=self.console,
        ):
            return False
        progress = Progress(
            SpinnerColumn(),
            TextColumn("{task.description}"),
            BarColumn(),
            DownloadColumn(),
            console=self.console,
        )
        task = progress.add_task(f"Pulling {name}", total=None)

        def update(item: dict[str, Any]) -> None:
            total = item.get("total") or None
            completed = item.get("completed") or 0
            progress.update(
                task,
                description=str(item.get("status") or f"Pulling {name}"),
                total=total,
                completed=completed,
            )

        try:
            with progress:
                service.pull_model(name, update)
            self.console.print(f"[green]✓ Model '{name}' is available.[/green]")
            return True
        except KeyboardInterrupt:
            self.console.print(
                "[yellow]Model pull cancelled; Ollama may retain reusable layers.[/yellow]"
            )
            return False
        except Exception as error:
            self.console.print(
                Text(f"Model pull failed: {describe_ollama_error(error)}", style="red")
            )
            return False

    def models_menu(self) -> None:
        assert self.config is not None
        host = self.settings.ollama_host or self.config.ollama_host
        service = (self._model_service_factory or OllamaModelService)(
            host, self.settings.ollama_timeout or self.config.ollama_timeout_seconds
        )
        try:
            while True:
                try:
                    with self.console.status(f"Loading models from {host}..."):
                        models = service.list_models()
                    self._screen(
                        "Models & Ollama",
                        f"{len(models)} installed · {host}",
                    )
                    self._display_models(models)
                    self._display_assignments(self.settings.model_assignments)
                    choices = [str(index) for index in range(1, len(models) + 1)] + [
                        "a",
                        "d",
                        "p",
                        "r",
                        "x",
                        "b",
                    ]
                    self.console.print(
                        "Select a model for details · [bold]a[/bold] Assign roles · "
                        "[bold]p[/bold] Pull · [bold]r[/bold] Refresh · "
                        "[bold]d[/bold] Defaults · [bold]x[/bold] Reset output adaptation · "
                        "[bold]b[/bold] Back"
                    )
                    choice = Prompt.ask(
                        "Action",
                        choices=choices,
                        default="b",
                        console=self.console,
                    )
                    if choice == "b":
                        return
                    if choice == "r":
                        self._set_notice("Model list refreshed.", "cyan")
                        continue
                    if choice == "p":
                        if self._pull_model(service):
                            self._set_notice("Model pull completed.")
                        continue
                    if choice == "x":
                        if Confirm.ask(
                            "Reset learned model output compatibility routes?",
                            default=False,
                            console=self.console,
                        ):
                            self._output_profile_store.reset()
                            self._set_notice("Model output compatibility routes reset.")
                        continue
                    if choice == "d":
                        if Confirm.ask(
                            "Restore YAML model defaults?", default=False, console=self.console
                        ):
                            self.settings = self.settings.model_copy(
                                update={"model_assignments": ModelAssignments()}
                            )
                            save_interactive_settings(self.settings_path, self.settings)
                            self._set_notice("YAML model defaults restored.")
                        continue
                    if choice == "a":
                        assignments = self._assignment_editor(self.settings.model_assignments)
                        if assignments is not None:
                            self.settings = self.settings.model_copy(
                                update={"model_assignments": assignments}
                            )
                            save_interactive_settings(self.settings_path, self.settings)
                            self._set_notice("Default model assignments saved.")
                        continue

                    selected = models[int(choice) - 1]
                    with self.console.status(f"Inspecting {selected.name}..."):
                        details = service.show_model(selected.name, selected)
                    while True:
                        self._screen("Model details", selected.name)
                        detail_table = Table(box=box.ROUNDED, show_header=False)
                        detail_table.add_column("Property", style="cyan")
                        detail_table.add_column("Value")
                        detail_table.add_row("Family", details.family)
                        detail_table.add_row("Parameters", details.parameter_size)
                        detail_table.add_row("Quantization", details.quantization)
                        detail_table.add_row("Context", str(details.context_length or "Unknown"))
                        detail_table.add_row(
                            "Capabilities", ", ".join(details.capabilities) or "Unknown"
                        )
                        detail_table.add_row("Digest", details.digest or "Unknown")
                        self.console.print(detail_table)
                        self._display_assignments(self.settings.model_assignments)
                        assignable = "completion" in details.capabilities
                        if assignable:
                            self.console.print(
                                "[bold]a[/bold] Use for all · "
                                "[bold]1 to 4[/bold] Assign one role · "
                                "[bold]t[/bold] Test · [bold]b[/bold] Back"
                            )
                            detail_choices = ["a", "1", "2", "3", "4", "t", "b"]
                        else:
                            self.console.print(
                                "[yellow]This model cannot be assigned because it does not "
                                "advertise completion capability.[/yellow]\n"
                                "[bold]b[/bold] Back"
                            )
                            detail_choices = ["b"]
                        action = Prompt.ask(
                            "Action", choices=detail_choices, default="b", console=self.console
                        )
                        if action == "b":
                            break
                        if action == "t":
                            with self.console.status(f"Testing {selected.name}..."):
                                details = service.probe_model(details)
                            if details.probe_warning:
                                self._set_notice(details.probe_warning, "yellow")
                            else:
                                self._set_notice(
                                    "Basic transport and structured response mode check completed."
                                )
                            continue
                        if action == "a":
                            assignments = ModelAssignments(all_model=selected.name)
                            message = f"{selected.name} assigned to all roles."
                        else:
                            role, label = MODEL_ROLES[int(action) - 1]
                            assignments = self.settings.model_assignments.model_copy(
                                update={role: selected.name}
                            )
                            message = f"{selected.name} assigned to {label}."
                        self.settings = self.settings.model_copy(
                            update={"model_assignments": assignments}
                        )
                        save_interactive_settings(self.settings_path, self.settings)
                        self._set_notice(message)
                except (KeyboardInterrupt, EOFError):
                    return
                except Exception as error:
                    self.console.print(
                        Text(
                            f"Ollama operation failed: {describe_ollama_error(error)}", style="red"
                        )
                    )
                    self._pause()
                    return
        finally:
            service.close()

    def system_status(self) -> None:
        assert self.config is not None
        self._screen("System status", "Local runtime and integrations")
        output = Path(self.settings.output_dir)
        writable_target = output if output.exists() else output.parent
        table = Table(title="Local Status", box=box.ROUNDED)
        table.add_column("Check")
        table.add_column("Status")
        table.add_row("Configuration", "valid")
        table.add_row("Python", platform.python_version())
        table.add_row("Abstract Agent", __version__)
        table.add_row("uv.lock", "present" if Path("uv.lock").exists() else "missing")
        table.add_row(
            "Output directory",
            "writable" if os.access(writable_target, os.W_OK) else "not writable",
        )
        table.add_row("Ollama", self.settings.ollama_host or self.config.ollama_host)
        for variable in (
            "SEMANTIC_SCHOLAR_API_KEY",
            "OPENALEX_API_KEY",
            "NCBI_API_KEY",
            "ABSTRACT_AGENT_CONTACT_EMAIL",
        ):
            table.add_row(variable, "configured" if os.getenv(variable) else "not configured")
        self.console.print(table)
        choice = Prompt.ask(
            "1=check Ollama, 2=check literature providers, or b=back",
            choices=["1", "2", "b"],
            default="b",
            console=self.console,
        )
        if choice == "1":
            manager = (self._pipeline_manager_factory or PipelineManager)(
                self.config,
                ollama_host=self.settings.ollama_host,
                ollama_timeout=self.settings.ollama_timeout,
                minimum_papers=self.settings.minimum_papers,
                model_assignments=self.settings.model_assignments,
                knowledge_enabled=self.settings.knowledge_enabled,
                knowledge_dir=self.settings.knowledge_dir,
                embedding_model=self.settings.embedding_model,
                knowledge_refresh=self.settings.knowledge_refresh,
            )
            try:
                with self.console.status("Checking Ollama and configured models..."):
                    manager.preflight()
                self.console.print("[green]✓ Ollama and configured models are ready.[/green]")
            except Exception as error:
                self.console.print(Text(f"Ollama check failed: {error}", style="red"))
            finally:
                manager.close()
            self._pause()
        elif choice == "2":
            try:
                with self.console.status("Checking scholarly providers..."):
                    statuses = (self._source_checker or multi_source.check_sources)()
                provider_table = Table(title="Provider Status")
                provider_table.add_column("Provider")
                provider_table.add_column("Status")
                provider_table.add_column("Results")
                provider_table.add_column("Detail")
                for status in statuses:
                    provider_table.add_row(
                        status["provider"],
                        status["status"],
                        str(status["result_count"]),
                        status.get("message") or "",
                    )
                self.console.print(provider_table)
            except Exception as error:
                self.console.print(Text(f"Provider check failed: {error}", style="red"))
            self._pause()

    def knowledge_menu(self) -> None:
        while True:
            self._screen("Local Knowledge", self.settings.knowledge_dir)
            try:
                with KnowledgeStore(self.settings.knowledge_dir) as store:
                    status = store.status()
                    table = Table(box=box.ROUNDED)
                    table.add_column("Records")
                    table.add_column("Count", justify="right")
                    for name, count in status["counts"].items():
                        table.add_row(name.replace("_", " ").title(), str(count))
                    table.add_row("FTS5", "ready" if status["fts5"] else "unavailable")
                    self.console.print(table)
                    self.console.print(
                        "[bold]s[/bold] Search · [bold]r[/bold] Reindex · "
                        "[bold]e[/bold] Export · [bold]i[/bold] Import · [bold]b[/bold] Back"
                    )
                    choice = Prompt.ask(
                        "Action",
                        choices=["s", "r", "e", "i", "b"],
                        default="b",
                        console=self.console,
                    )
                    if choice == "b":
                        return
                    if choice == "s":
                        query = Prompt.ask("Search", console=self.console)
                        results = store.search_papers(query)
                        results_table = Table(title="Knowledge results", box=box.ROUNDED)
                        results_table.add_column("Paper ID")
                        results_table.add_column("Title")
                        for paper in results:
                            results_table.add_row(
                                str(paper.get("paper_id", "")), str(paper.get("title", ""))
                            )
                        self.console.print(results_table)
                        self._pause()
                    elif choice == "r":
                        self._set_notice(f"Reindexed {store.reindex()} papers.")
                    elif choice == "e":
                        counts = store.export_projections()
                        self._set_notice(
                            "Exported readable projections: "
                            + ", ".join(f"{key}={value}" for key, value in counts.items())
                        )
                    else:
                        source = Prompt.ask(
                            "Directory containing exported records", console=self.console
                        )
                        counts = store.import_projections(source)
                        self._set_notice(
                            f"Imported {counts['papers']} papers and {counts['evidence']} cards."
                        )
            except (OSError, ValueError, json.JSONDecodeError) as error:
                self._set_notice(f"Knowledge operation failed: {error}", "red")

    def preferences(self) -> None:
        assert self.config is not None
        labels = (
            ("output_dir", "Output directory"),
            ("minimum_papers", "Minimum papers"),
            ("allow_insufficient_literature", "Allow insufficient evidence"),
            ("history_page_size", "History page size"),
            ("confirm_before_run", "Review before each run"),
            ("search_depth", "Default search depth"),
            ("ollama_host", "Ollama host"),
            ("ollama_timeout", "Ollama inactivity timeout"),
            ("knowledge_enabled", "Local knowledge enabled"),
            ("knowledge_dir", "Knowledge directory"),
            ("embedding_model", "Embedding model"),
            ("knowledge_refresh", "Knowledge refresh policy"),
        )
        while True:
            self._screen("Preferences", "Select one setting to edit")
            table = Table(box=box.ROUNDED)
            table.add_column("#")
            table.add_column("Setting")
            table.add_column("Value")
            for index, (key, label) in enumerate(labels, start=1):
                value = getattr(self.settings, key)
                if key == "ollama_host" and value is None:
                    value = "automatic"
                if key == "ollama_timeout":
                    value = value or self.config.ollama_timeout_seconds
                    value = f"{value:g} seconds"
                table.add_row(str(index), label, str(value))
            table.add_row("m", "Model assignments", "Manage roles")
            self.console.print(table)
            self.console.print(
                "Select a setting · [bold]m[/bold] Models · [bold]r[/bold] Reset · "
                "[bold]b[/bold] Back"
            )
            choice = Prompt.ask(
                "Action",
                choices=[str(index) for index in range(1, len(labels) + 1)] + ["m", "r", "b"],
                default="b",
                console=self.console,
            )
            if choice == "b":
                return
            if choice == "r":
                if Confirm.ask("Reset local preferences?", default=False, console=self.console):
                    self.settings = InteractiveSettings()
                    if self.settings_path.exists():
                        self.settings_path.unlink()
                    self._load_config()
                    self._set_notice("Preferences reset to defaults.")
                continue
            if choice == "m":
                updated = self._assignment_editor(self.settings.model_assignments)
                if updated is not None:
                    self.settings = self.settings.model_copy(update={"model_assignments": updated})
                    save_interactive_settings(self.settings_path, self.settings)
                    self._set_notice("Model assignments saved.")
                continue
            key, label = labels[int(choice) - 1]
            try:
                if key == "output_dir":
                    value: Any = Prompt.ask(
                        label, default=self.settings.output_dir, console=self.console
                    )
                elif key in {"minimum_papers", "history_page_size"}:
                    value = IntPrompt.ask(
                        label, default=getattr(self.settings, key), console=self.console
                    )
                elif key in {
                    "allow_insufficient_literature",
                    "confirm_before_run",
                    "knowledge_enabled",
                }:
                    value = Confirm.ask(
                        label, default=getattr(self.settings, key), console=self.console
                    )
                elif key == "search_depth":
                    value = Prompt.ask(
                        label,
                        choices=["standard", "deep"],
                        default=self.settings.search_depth,
                        console=self.console,
                    )
                elif key == "knowledge_refresh":
                    value = Prompt.ask(
                        label,
                        choices=["balanced", "always", "offline"],
                        default=self.settings.knowledge_refresh,
                        console=self.console,
                    )
                elif key == "knowledge_dir":
                    value = Prompt.ask(
                        label, default=self.settings.knowledge_dir, console=self.console
                    )
                elif key == "embedding_model":
                    value = Prompt.ask(
                        "Embedding model (or 'auto')",
                        default=self.settings.embedding_model or "auto",
                        console=self.console,
                    )
                    value = None if value == "auto" else value
                elif key == "ollama_host":
                    value = Prompt.ask(
                        "Ollama host (or 'auto')",
                        default=self.settings.ollama_host or "auto",
                        console=self.console,
                    )
                    value = None if value == "auto" else value
                else:
                    value = float(
                        Prompt.ask(
                            label,
                            default=str(
                                self.settings.ollama_timeout or self.config.ollama_timeout_seconds
                            ),
                            console=self.console,
                        )
                    )
                candidate = self.settings.model_copy(update={key: value})
                candidate = InteractiveSettings.model_validate(candidate.model_dump())
                save_interactive_settings(self.settings_path, candidate)
                self.settings = candidate
                self._set_notice(f"{label} saved.")
            except (ValidationError, ValueError, OSError) as error:
                self._set_notice(f"Preference was not saved: {error}", "red")
