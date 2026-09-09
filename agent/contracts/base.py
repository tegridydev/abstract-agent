"""Base validation contracts and shared host validation."""

from __future__ import annotations

from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


def validate_ollama_host(value: str | None) -> str | None:
    if value is None:
        return None
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("Ollama host must be an absolute http(s) URL")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("Ollama host must not contain embedded credentials")
    try:
        _ = parsed.port
    except ValueError as error:
        raise ValueError("Ollama host contains an invalid port") from error
    return value.rstrip("/")
