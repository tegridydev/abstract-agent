"""Ollama model discovery, pulling, and basic response mode checks."""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import httpx
from ollama import Client, ResponseError
from pydantic import ValidationError

from .schemas import InstalledModel, ModelProbeResponse

MODEL_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]*(?::[A-Za-z0-9._-]+)?$")
_PROBE_CACHE: dict[tuple[str, str, str], tuple[str, str, str, str | None]] = {}


def ollama_http_timeout(inactivity_seconds: float) -> httpx.Timeout:
    """Use a long generation read window without weakening connection safety."""
    if inactivity_seconds <= 0:
        raise ValueError("Ollama inactivity timeout must be positive")
    return httpx.Timeout(connect=10, read=inactivity_seconds, write=30, pool=10)


def _value(value: Any, key: str, default: Any = None) -> Any:
    if hasattr(value, key):
        return getattr(value, key)
    if isinstance(value, dict):
        return value.get(key, default)
    return default


def validate_model_name(name: str) -> str:
    normalized = name.strip()
    if len(normalized) > 200 or not MODEL_NAME_PATTERN.fullmatch(normalized):
        raise ValueError("invalid Ollama model name")
    return normalized


def is_cloud_model(name: str) -> bool:
    lowered = name.casefold()
    return lowered.endswith("-cloud") or ":cloud" in lowered


def parse_ollama_host(host: str) -> tuple[str, bool]:
    """Validate a host and return a credential free display URL and remote flag."""
    parsed = urlsplit(host)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("Ollama host must be an absolute http(s) URL")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("Ollama host must not contain embedded credentials")
    try:
        port = parsed.port
    except ValueError as error:
        raise ValueError("Ollama host contains an invalid port") from error
    hostname = parsed.hostname
    bracketed = f"[{hostname}]" if ":" in hostname else hostname
    netloc = f"{bracketed}:{port}" if port is not None else bracketed
    display = urlunsplit((parsed.scheme, netloc, parsed.path.rstrip("/"), "", ""))
    return display, hostname not in {"localhost", "127.0.0.1", "::1"}


class OllamaModelService:
    def __init__(self, host: str, timeout: float = 30) -> None:
        self.host = host.rstrip("/")
        self.client = Client(host=self.host, timeout=ollama_http_timeout(timeout))

    def list_models(self) -> list[InstalledModel]:
        response = self.client.list()
        entries = _value(response, "models", []) or []
        try:
            running_response = self.client.ps()
            running_entries = _value(running_response, "models", []) or []
            running_contexts = {
                _value(item, "model") or _value(item, "name"): (
                    _value(item, "context_length") or _value(item, "context")
                )
                for item in running_entries
            }
            running = set(running_contexts)
        except Exception:
            running = set()
            running_contexts = {}
        models = []
        for entry in entries:
            name = _value(entry, "model") or _value(entry, "name")
            if not name:
                continue
            details = _value(entry, "details", {}) or {}
            models.append(
                InstalledModel(
                    name=name,
                    digest=_value(entry, "digest", "") or "",
                    size=int(_value(entry, "size", 0) or 0),
                    parameter_size=_value(details, "parameter_size", "Unknown") or "Unknown",
                    quantization=_value(details, "quantization_level", "Unknown") or "Unknown",
                    family=_value(details, "family", "Unknown") or "Unknown",
                    modified_at=str(_value(entry, "modified_at", "") or ""),
                    running=name in running,
                    allocated_context_length=(
                        int(running_contexts[name])
                        if name in running_contexts and running_contexts[name]
                        else None
                    ),
                )
            )
        return sorted(models, key=lambda model: model.name.casefold())

    def show_model(self, name: str, base: InstalledModel | None = None) -> InstalledModel:
        name = validate_model_name(name)
        response = self.client.show(name)
        details = _value(response, "details", {}) or {}
        info = _value(response, "model_info", None) or _value(response, "modelinfo", {}) or {}
        context_length = next(
            (
                int(value)
                for key, value in (info.items() if isinstance(info, dict) else [])
                if str(key).endswith(".context_length") and isinstance(value, int | float)
            ),
            None,
        )
        return InstalledModel(
            name=name,
            digest=base.digest if base else "",
            size=base.size if base else 0,
            parameter_size=_value(details, "parameter_size", None)
            or (base.parameter_size if base else "Unknown"),
            quantization=_value(details, "quantization_level", None)
            or (base.quantization if base else "Unknown"),
            family=_value(details, "family", None) or (base.family if base else "Unknown"),
            modified_at=base.modified_at if base else "",
            capabilities=list(_value(response, "capabilities", []) or []),
            context_length=context_length,
            allocated_context_length=base.allocated_context_length if base else None,
            running=base.running if base else False,
        )

    def pull_model(
        self, name: str, progress: Callable[[dict[str, Any]], None] | None = None
    ) -> None:
        name = validate_model_name(name)
        response = self.client.pull(model=name, stream=True)
        for chunk in response:
            update = {
                "status": str(_value(chunk, "status", "Downloading")),
                "digest": str(_value(chunk, "digest", "") or ""),
                "total": int(_value(chunk, "total", 0) or 0),
                "completed": int(_value(chunk, "completed", 0) or 0),
            }
            if progress:
                progress(update)

    def probe_model(self, model: InstalledModel) -> InstalledModel:
        if "completion" not in model.capabilities:
            raise ValueError(f"model '{model.name}' does not advertise completion capability")
        cache_key = (self.host, model.name, model.digest)
        if cache_key in _PROBE_CACHE:
            reasoning_mode, output_mode, contract_tier, warning = _PROBE_CACHE[cache_key]
            return model.model_copy(
                update={
                    "reasoning_mode": reasoning_mode,
                    "structured_output_mode": output_mode,
                    "contract_tier": contract_tier,
                    "probe_warning": warning,
                }
            )

        reasoning_candidates: list[tuple[str, bool | str | None]]
        if "thinking" in model.capabilities:
            reasoning_candidates = [
                ("disabled", False),
                ("low", "low"),
                ("enabled", True),
                ("not_applicable", None),
            ]
        else:
            reasoning_candidates = [("not_applicable", None)]
        output_candidates: list[tuple[str, dict[str, Any] | str | None]] = [
            ("schema", ModelProbeResponse.model_json_schema()),
            ("json", "json"),
            ("prompt", None),
        ]
        successful_response: tuple[str, str, str] | None = None
        last_error: Exception | None = None
        for reasoning_mode, think in reasoning_candidates:
            for output_mode, response_format in output_candidates:
                kwargs: dict[str, Any] = {
                    "model": model.name,
                    "messages": [
                        {
                            "role": "user",
                            "content": 'Return exactly {"status":"ok"} as structured JSON.',
                        }
                    ],
                    "options": {
                        "temperature": 0,
                        "num_predict": 4096 if think in ("low", True) else 256,
                    },
                }
                if think is not None:
                    kwargs["think"] = think
                if response_format is not None:
                    kwargs["format"] = response_format
                try:
                    response = self.client.chat(**kwargs)
                    message = _value(response, "message", {}) or {}
                    content = str(_value(message, "content", "") or "").strip()
                    if (
                        not content
                        and _value(message, "thinking", "")
                        and _value(response, "done_reason") == "length"
                        and kwargs["options"]["num_predict"] < 16384
                    ):
                        kwargs["options"] = {**kwargs["options"], "num_predict": 16384}
                        response = self.client.chat(**kwargs)
                        message = _value(response, "message", {}) or {}
                        content = str(_value(message, "content", "") or "").strip()
                    if content and successful_response is None:
                        successful_response = (reasoning_mode, output_mode, content)
                    ModelProbeResponse.model_validate_json(content)
                except (ValidationError, ValueError) as error:
                    last_error = error
                    continue
                except Exception as error:
                    last_error = error
                    continue
                contract_tier = "basic_schema" if output_mode == "schema" else "basic_json"
                profile = (reasoning_mode, output_mode, contract_tier, None)
                _PROBE_CACHE[cache_key] = profile
                return model.model_copy(
                    update={
                        "reasoning_mode": reasoning_mode,
                        "structured_output_mode": output_mode,
                        "contract_tier": contract_tier,
                    }
                )

        if successful_response is not None:
            reasoning_mode, _, _ = successful_response
            warning = (
                "The model returned text but did not pass the small structured output probe; "
                "runtime JSON validation and repair will be used."
            )
            profile = (reasoning_mode, "prompt", "text_projection", warning)
            _PROBE_CACHE[cache_key] = profile
            return model.model_copy(
                update={
                    "reasoning_mode": reasoning_mode,
                    "structured_output_mode": "prompt",
                    "contract_tier": "text_projection",
                    "probe_warning": warning,
                }
            )

        raise ValueError(
            f"model '{model.name}' produced no usable text during the completion probe: "
            f"{type(last_error).__name__}: {last_error}"
        )

    def validate_models(self, names: set[str], *, probe: bool = True) -> dict[str, InstalledModel]:
        installed = {model.name: model for model in self.list_models()}
        missing = sorted(names - set(installed))
        if missing:
            raise ValueError("missing Ollama model(s): " + ", ".join(missing))
        selected = {}
        for name in sorted(names):
            model = self.show_model(name, installed[name])
            if "completion" not in model.capabilities:
                raise ValueError(f"model '{name}' does not advertise completion capability")
            if probe:
                model = self.probe_model(model)
            selected[name] = model
        if probe:
            try:
                running_entries = _value(self.client.ps(), "models", []) or []
                allocated = {
                    _value(item, "model") or _value(item, "name"): (
                        _value(item, "context_length") or _value(item, "context")
                    )
                    for item in running_entries
                }
                selected = {
                    name: model.model_copy(
                        update={
                            "running": name in allocated,
                            "allocated_context_length": (
                                int(allocated[name]) if allocated.get(name) else None
                            ),
                        }
                    )
                    for name, model in selected.items()
                }
            except Exception:
                pass
        return selected

    def select_embedding_model(self, preferred: str | None = None) -> InstalledModel | None:
        """Select an installed embedding model without downloading or starting one."""
        installed = self.list_models()
        if preferred:
            preferred = validate_model_name(preferred)
            candidate = next((model for model in installed if model.name == preferred), None)
            if candidate is None:
                raise ValueError(f"preferred embedding model '{preferred}' is not installed")
            detailed = self.show_model(candidate.name, candidate)
            if "embedding" not in detailed.capabilities:
                raise ValueError(
                    f"preferred model '{preferred}' does not advertise embedding capability"
                )
            return detailed
        detailed_models = [self.show_model(model.name, model) for model in installed]
        capable = [model for model in detailed_models if "embedding" in model.capabilities]
        if not capable:
            return None
        return min(
            capable,
            key=lambda model: (
                0 if model.name.casefold().startswith("embeddinggemma") else 1,
                model.name.casefold(),
            ),
        )

    def close(self) -> None:
        close = getattr(self.client, "close", None)
        if callable(close):
            close()


def describe_ollama_error(error: Exception) -> str:
    if isinstance(error, ResponseError):
        return f"Ollama returned HTTP {error.status_code}: {error.error}"
    return f"{type(error).__name__}: {error}"
