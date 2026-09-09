from __future__ import annotations

import runpy
from typing import Any

import pytest

import agent.agent as agent_module


class RecordingConsole:
    def __init__(self) -> None:
        self.payloads: list[Any] = []

    def print_json(self, *, data: Any) -> None:
        self.payloads.append(data)


def test_main_dispatches_interactive_run_and_resume(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(agent_module, "interactive", lambda: 11)
    assert agent_module.main([]) == 11

    commands: list[str] = []
    monkeypatch.setattr(
        agent_module,
        "execute_run",
        lambda args: commands.append(args.command) or 12,
    )
    assert agent_module.main(["run", "--topic", "topic"]) == 12
    assert agent_module.main(["resume", "--diagnostic", "failed.json"]) == 12
    assert commands == ["run", "resume"]


def test_main_source_check_returns_success_or_source_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    console = RecordingConsole()
    monkeypatch.setattr(agent_module, "console", console)
    monkeypatch.setattr(
        agent_module.multi_source,
        "check_sources",
        lambda topic: [{"provider": "p", "status": "cached", "topic": topic}],
    )
    assert agent_module.main(["sources", "check", "--topic", "custom"]) == (
        agent_module.EXIT_SUCCESS
    )
    assert console.payloads[0][0]["topic"] == "custom"

    monkeypatch.setattr(
        agent_module.multi_source,
        "check_sources",
        lambda topic: [{"provider": "p", "status": "error"}],
    )
    assert agent_module.main(["sources", "check"]) == agent_module.EXIT_SOURCE


def test_interactive_constructs_application(monkeypatch: pytest.MonkeyPatch) -> None:
    import agent.ui as ui_module

    class Application:
        def __init__(self, *, console: Any) -> None:
            self.console = console

        def run(self) -> int:
            return 13

    monkeypatch.setattr(ui_module, "InteractiveApplication", Application)
    assert agent_module.interactive() == 13


def test_package_main_module_exits_with_main_result(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(agent_module, "main", lambda: 14)
    with pytest.raises(SystemExit) as caught:
        runpy.run_module("agent.__main__", run_name="__main__")
    assert caught.value.code == 14
