from __future__ import annotations

from typing import Any

import pytest
from ollama import ResponseError

from agent import ollama_models
from agent.ollama_models import OllamaModelService
from agent.schemas import InstalledModel


def service(client: Any) -> OllamaModelService:
    result = OllamaModelService.__new__(OllamaModelService)
    result.host = "http://localhost:11434"
    result.client = client
    return result


def test_value_timeout_name_and_error_helpers() -> None:
    class Item:
        value = 1

    assert ollama_models._value(Item(), "value") == 1
    assert ollama_models._value({}, "missing", 2) == 2
    assert ollama_models._value(None, "missing", 3) == 3
    with pytest.raises(ValueError, match="positive"):
        ollama_models.ollama_http_timeout(0)
    assert ollama_models.validate_model_name("  model:7b ") == "model:7b"
    with pytest.raises(ValueError, match="invalid"):
        ollama_models.validate_model_name("a" * 201)
    assert "HTTP 400" in ollama_models.describe_ollama_error(ResponseError("bad", 400))
    assert ollama_models.describe_ollama_error(RuntimeError("bad")) == "RuntimeError: bad"


def test_host_parser_rejects_invalid_forms_and_supports_ipv6() -> None:
    for host in ("relative", "ftp://host", "http://host:bad"):
        with pytest.raises(ValueError):
            ollama_models.parse_ollama_host(host)
    assert ollama_models.parse_ollama_host("http://[::1]:11434/") == (
        "http://[::1]:11434",
        False,
    )


def test_list_models_tolerates_ps_failure_skips_unnamed_and_sorts() -> None:
    class Client:
        def list(self) -> dict[str, Any]:
            return {
                "models": [
                    {"name": "z-model", "details": None},
                    {"model": "A-model", "size": "2"},
                    {"digest": "unnamed"},
                ]
            }

        def ps(self) -> None:
            raise RuntimeError("unavailable")

    models = service(Client()).list_models()
    assert [model.name for model in models] == ["A-model", "z-model"]
    assert all(not model.running for model in models)
    assert models[0].size == 2


def test_show_model_defaults_without_base_and_ignores_non_mapping_info() -> None:
    class Client:
        def show(self, name: str) -> dict[str, Any]:
            return {"details": {}, "model_info": ["bad"], "capabilities": None}

    model = service(Client()).show_model("model:1")
    assert model.parameter_size == "Unknown"
    assert model.context_length is None
    assert model.capabilities == []


def test_pull_without_progress_and_close_variants() -> None:
    class Client:
        close = "not callable"

        def pull(self, **kwargs: Any):
            del kwargs
            return [{"status": None, "digest": None, "total": None, "completed": None}]

    service(Client()).pull_model("model:1")
    service(Client()).close()

    closed: list[bool] = []

    class Closable(Client):
        def close(self) -> None:
            closed.append(True)

    service(Closable()).close()
    assert closed == [True]


def test_probe_retries_thinking_exhaustion_with_larger_budget() -> None:
    ollama_models._PROBE_CACHE.clear()

    class Client:
        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []

        def chat(self, **kwargs: Any) -> dict[str, Any]:
            self.calls.append(kwargs)
            if len(self.calls) == 1:
                return {
                    "message": {"content": "", "thinking": "still reasoning"},
                    "done_reason": "length",
                }
            return {"message": {"content": '{"status":"ok"}'}}

    client = Client()
    model = InstalledModel(
        name="thinking:1", digest="retry", capabilities=["completion", "thinking"]
    )
    profiled = service(client).probe_model(model)
    assert profiled.structured_output_mode == "schema"
    assert client.calls[1]["options"]["num_predict"] == 16384


def test_probe_reports_no_usable_text_after_all_modes() -> None:
    ollama_models._PROBE_CACHE.clear()

    class Client:
        def chat(self, **kwargs: Any) -> dict[str, Any]:
            del kwargs
            return {"message": {"content": ""}}

    with pytest.raises(ValueError, match="produced no usable text"):
        service(Client()).probe_model(
            InstalledModel(name="model:1", digest="empty", capabilities=["completion"])
        )


def test_validate_models_missing_capability_and_probe_toggle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selected = InstalledModel(name="selected", capabilities=["completion"])
    svc = service(object())
    monkeypatch.setattr(svc, "list_models", lambda: [selected])
    with pytest.raises(ValueError, match="missing Ollama"):
        svc.validate_models({"missing"})

    monkeypatch.setattr(
        svc,
        "show_model",
        lambda name, base: InstalledModel(name=name, capabilities=["embedding"]),
    )
    with pytest.raises(ValueError, match="completion capability"):
        svc.validate_models({"selected"})

    monkeypatch.setattr(svc, "show_model", lambda name, base: selected)
    monkeypatch.setattr(
        svc,
        "probe_model",
        lambda model: (_ for _ in ()).throw(AssertionError("probe called")),
    )
    assert svc.validate_models({"selected"}, probe=False) == {"selected": selected}
