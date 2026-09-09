import pytest
from ollama import ResponseError

from agent import ollama_models
from agent.ollama_models import (
    OllamaModelService,
    is_cloud_model,
    ollama_http_timeout,
    validate_model_name,
)
from agent.schemas import InstalledModel


class FakeClient:
    def __init__(self):
        self.chat_calls = 0
        self.chat_kwargs = []

    def list(self):
        return {
            "models": [
                {
                    "model": "writer:7b",
                    "digest": "digest-1",
                    "size": 1_000,
                    "modified_at": "2026-01-01T00:00:00Z",
                    "details": {
                        "parameter_size": "7B",
                        "quantization_level": "Q4_K_M",
                        "family": "test",
                    },
                }
            ]
        }

    def ps(self):
        return {"models": [{"model": "writer:7b", "context_length": 4096}]}

    def show(self, name):
        assert name == "writer:7b"
        return {
            "capabilities": ["completion"],
            "details": {
                "parameter_size": "7B",
                "quantization_level": "Q4_K_M",
                "family": "test",
            },
            "model_info": {"test.context_length": 8192},
        }

    def chat(self, **kwargs):
        self.chat_calls += 1
        self.chat_kwargs.append(kwargs)
        return {"message": {"content": '{"status":"ok"}'}}

    def pull(self, **kwargs):
        return iter(
            [
                {"status": "downloading", "total": 100, "completed": 50},
                {"status": "success", "total": 100, "completed": 100},
            ]
        )

    def close(self):
        return None


def service_with_fake() -> tuple[OllamaModelService, FakeClient]:
    service = OllamaModelService.__new__(OllamaModelService)
    service.host = "http://localhost:11434"
    service.client = FakeClient()
    return service, service.client


def test_inventory_and_model_details_are_normalized() -> None:
    service, _ = service_with_fake()

    inventory = service.list_models()
    details = service.show_model(inventory[0].name, inventory[0])

    assert inventory[0].running is True
    assert inventory[0].allocated_context_length == 4096
    assert details.capabilities == ["completion"]
    assert details.context_length == 8192
    assert details.digest == "digest-1"


def test_model_details_accept_current_ollama_modelinfo_field() -> None:
    service, client = service_with_fake()
    original_show = client.show

    def show(name):
        response = original_show(name)
        response["modelinfo"] = response.pop("model_info")
        return response

    client.show = show

    assert service.show_model("writer:7b").context_length == 8192


def test_structured_probe_is_cached_by_digest() -> None:
    ollama_models._PROBE_CACHE.clear()
    service, client = service_with_fake()
    model = InstalledModel(name="writer:7b", digest="digest-1", capabilities=["completion"])

    service.probe_model(model)
    service.probe_model(model)

    assert client.chat_calls == 1


def test_thinking_model_probe_disables_reasoning_for_structured_output() -> None:
    ollama_models._PROBE_CACHE.clear()
    service, client = service_with_fake()
    model = InstalledModel(
        name="writer:7b",
        digest="thinking-digest",
        capabilities=["completion", "thinking"],
    )

    profiled = service.probe_model(model)

    assert profiled.reasoning_mode == "disabled"
    assert profiled.structured_output_mode == "schema"
    assert client.chat_kwargs[0]["think"] is False
    assert client.chat_kwargs[0]["options"]["num_predict"] == 256


def test_level_only_thinking_model_negotiates_low_instead_of_being_rejected() -> None:
    ollama_models._PROBE_CACHE.clear()
    service, client = service_with_fake()

    def chat(**kwargs):
        client.chat_calls += 1
        client.chat_kwargs.append(kwargs)
        if kwargs.get("think") is False:
            raise ResponseError("think must be low, medium, or high", 400)
        return {"message": {"content": '{"status":"ok"}'}}

    client.chat = chat
    model = InstalledModel(
        name="writer:7b",
        digest="level-digest",
        capabilities=["completion", "thinking"],
    )

    profiled = service.probe_model(model)

    assert profiled.reasoning_mode == "low"
    assert client.chat_kwargs[-1]["think"] == "low"
    assert client.chat_kwargs[-1]["options"]["num_predict"] == 4096


def test_nonconforming_completion_model_uses_prompt_mode_with_warning() -> None:
    ollama_models._PROBE_CACHE.clear()
    service, client = service_with_fake()
    client.chat = lambda **kwargs: {"message": {"content": "I can still answer text prompts"}}
    model = InstalledModel(name="writer:7b", digest="prompt-digest", capabilities=["completion"])

    profiled = service.probe_model(model)

    assert profiled.structured_output_mode == "prompt"
    assert profiled.probe_warning


def test_probe_rejects_embedding_only_model() -> None:
    service, _ = service_with_fake()

    with pytest.raises(ValueError, match="completion capability"):
        service.probe_model(InstalledModel(name="embed:latest", capabilities=["embedding"]))


def test_embedding_selection_prefers_embeddinggemma_without_pulling(monkeypatch) -> None:
    service, _ = service_with_fake()
    models = [
        InstalledModel(name="zeta:embed", digest="z"),
        InstalledModel(name="embeddinggemma:latest", digest="g"),
    ]
    monkeypatch.setattr(service, "list_models", lambda: models)
    monkeypatch.setattr(
        service,
        "show_model",
        lambda name, base: base.model_copy(update={"capabilities": ["embedding"]}),
    )

    selected = service.select_embedding_model()

    assert selected.name == "embeddinggemma:latest"


def test_preferred_embedding_model_must_be_installed_and_capable(monkeypatch) -> None:
    service, _ = service_with_fake()
    monkeypatch.setattr(service, "list_models", lambda: [InstalledModel(name="writer:7b")])

    with pytest.raises(ValueError, match="not installed"):
        service.select_embedding_model("missing:embed")
    with pytest.raises(ValueError, match="embedding capability"):
        service.select_embedding_model("writer:7b")


def test_pull_streams_progress_without_shelling_out() -> None:
    service, _ = service_with_fake()
    updates = []

    service.pull_model("writer:7b", updates.append)

    assert updates[-1]["completed"] == 100
    assert updates[-1]["status"] == "success"


@pytest.mark.parametrize("name", ["../bad model", "bad;command", "-invalid"])
def test_model_name_validation_rejects_unsafe_values(name) -> None:
    with pytest.raises(ValueError, match="invalid Ollama model name"):
        validate_model_name(name)


def test_cloud_model_detection_requires_explicit_warning() -> None:
    assert is_cloud_model("gpt-oss:120b-cloud")
    assert not is_cloud_model("ornith:9b")


def test_ollama_timeout_separates_connection_and_generation_limits() -> None:
    timeout = ollama_http_timeout(600)

    assert timeout.connect == 10
    assert timeout.read == 600
    assert timeout.write == 30
    assert timeout.pool == 10


def test_ollama_host_parsing_redacts_and_classifies() -> None:
    from agent.ollama_models import parse_ollama_host

    assert parse_ollama_host("http://localhost:11434") == (
        "http://localhost:11434",
        False,
    )
    assert parse_ollama_host("https://ollama.example/api?token=secret") == (
        "https://ollama.example/api",
        True,
    )
    with pytest.raises(ValueError, match="embedded credentials"):
        parse_ollama_host(f"https://{SYNTHETIC_CREDENTIALS}@ollama.example")


SYNTHETIC_CREDENTIALS = "user:secret"
