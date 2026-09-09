from __future__ import annotations

import json
import runpy
import sys
from pathlib import Path

import pytest

from agent.agent import PipelineManager, load_pipeline_config
from agent.application.io import atomic_write
from agent.checkpoints import context_checksum
from agent.pipeline.persistence import load_resume_checkpoint, save_failure_diagnostic


@pytest.mark.parametrize("launcher", ["abs.py", "agent/agent.py"])
def test_file_launchers_delegate_to_package_main(monkeypatch, launcher):
    import agent.agent as module

    monkeypatch.setattr(module, "main", lambda: 17)
    monkeypatch.setattr(sys, "path", list(sys.path))
    with pytest.raises(SystemExit) as caught:
        runpy.run_path(str(Path(__file__).resolve().parents[1] / launcher), run_name="__main__")
    assert caught.value.code == 17


@pytest.mark.parametrize("operation", ["fsync", "replace"])
def test_atomic_write_failure_keeps_original_and_removes_temporary(
    monkeypatch, tmp_path, operation
):
    import agent.application.io as module

    target = tmp_path / "settings.toml"
    target.write_text("original")

    def fail(*args):
        raise OSError("disk unavailable")

    monkeypatch.setattr(module.os, operation, fail)
    with pytest.raises(OSError, match="disk unavailable"):
        atomic_write(target, "replacement")
    assert target.read_text() == "original"
    assert list(tmp_path.iterdir()) == [target]


def test_offline_refresh_cannot_silently_enable_provider_requests():
    with pytest.raises(ValueError, match="offline literature refresh requires"):
        PipelineManager(
            load_pipeline_config(), knowledge_refresh="offline", knowledge_enabled=False
        )


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"schema_version": "1.0"}, "only v4.7"),
        ({"pipeline": None}, "different pipeline"),
        ({"diagnostic_context": None}, "no valid diagnostic context"),
        ({"context_checksum": "wrong"}, "checksum"),
        ({"resumable": False}, "not resumable"),
        ({"resume_step": "unknown"}, "step metadata"),
    ],
)
def test_checkpoint_rejects_invalid_envelope(tmp_path, changes, message):
    config = load_pipeline_config()
    context = {"topic": "compression", "_active_step": config.steps[0].name}
    path = save_failure_diagnostic(context, config, "compression", tmp_path, "interrupted")
    payload = json.loads(path.read_text())
    payload.update(changes)
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match=message):
        load_resume_checkpoint(path, config)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"_active_step": "unknown"}, "current pipeline step"),
        ({"topic": None}, "no research topic"),
        ({"_resume_count": "bad"}, "invalid resume count"),
        ({"_resume_count": 3}, "resume limit"),
        ({"invented": []}, "unknown context keys"),
        ({"top_papers": [{}]}, "invalid 'top_papers'"),
        ({"source_statuses": [{}]}, "invalid 'source_statuses'"),
    ],
)
def test_checkpoint_revalidates_checksummed_content(tmp_path, changes, message):
    config = load_pipeline_config()
    context = {"topic": "compression", "_active_step": config.steps[0].name}
    path = save_failure_diagnostic(context, config, "compression", tmp_path, "interrupted")
    payload = json.loads(path.read_text())
    context.update(changes)
    payload.update(
        diagnostic_context=context,
        context_checksum=context_checksum(context),
        resume_step=context["_active_step"],
        run_info={},
    )
    path.write_text(json.dumps(payload))
    original = path.read_bytes()
    with pytest.raises(ValueError, match=message):
        load_resume_checkpoint(path, config)
    assert path.read_bytes() == original


def test_knowledge_cli_round_trip_stays_local(monkeypatch, tmp_path):
    import agent.agent as module
    from agent.knowledge import KnowledgeStore

    source = tmp_path / "source"
    destination = tmp_path / "destination"
    with KnowledgeStore(source) as store:
        store.ingest_papers(
            [
                {
                    "paper_id": "paper:one",
                    "title": "Model compression",
                    "summary": "Quantization reduces model memory.",
                }
            ]
        )
        store.export_projections()
    outputs = []
    monkeypatch.setattr(module.console, "print_json", lambda *, data: outputs.append(data))
    prefix = ["knowledge", "--knowledge-dir", str(destination)]
    for command in [
        ["import", str(source)],
        ["status"],
        ["search", "compression"],
        ["reindex"],
        ["export"],
    ]:
        assert module.main(prefix + command) == 0
    assert outputs[0]["papers"] == 1
    assert outputs[1]["counts"]["papers"] == 1
    assert outputs[2][0]["paper_id"] == "paper:one"
    assert outputs[3]["reindexed"] == 1
    assert outputs[4]["papers"] == 1


@pytest.mark.parametrize("refresh", ["balanced", "always", "offline"])
def test_prior_art_runtime_preserves_provider_circuits_and_refresh_policy(refresh):
    from agent.pipeline.stages.runtime import tool_overrides

    config = load_pipeline_config()
    step = next(step for step in config.steps if step.name == "Retrieve Closest Prior Art")
    store = object()
    result = tool_overrides(
        step,
        {
            "source_statuses": [
                {"provider": "limited", "error_category": "http_429"},
                {"provider": "failed", "status": "error"},
                {"provider": "open", "circuit_state": "open"},
                {"provider": "healthy", "status": "ok"},
            ]
        },
        search_depth="deep",
        knowledge_enabled=True,
        knowledge_refresh=refresh,
        minimum_papers=2,
        knowledge_store_factory=lambda: store,
        embedding_service=None,
    )
    assert result["disabled_providers"] == ["limited", "failed", "open"]
    assert result["knowledge_store"] is store
    assert result["max_results"] == 60
    if refresh == "offline":
        assert result["offline"] is True
    elif refresh == "always":
        assert result["use_cache"] is False
    else:
        assert result["cache_ttl_seconds"] == 86400
