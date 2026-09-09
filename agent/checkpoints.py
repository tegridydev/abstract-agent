"""Bounded, deterministic serialization primitives for result checkpoints."""

import hashlib
import json
from pathlib import Path
from typing import Any

MAX_CHECKPOINT_BYTES = 25 * 1024 * 1024


def normalize_context(value: dict[str, Any]) -> dict[str, Any]:
    """Normalize values through the exact JSON representation written to disk."""
    return json.loads(json.dumps(value, ensure_ascii=False, default=str))


def context_checksum(context: dict[str, Any]) -> str:
    canonical = json.dumps(
        context, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(canonical).hexdigest()


def read_checkpoint_object(path: str | Path) -> dict[str, Any]:
    checkpoint_path = Path(path)
    try:
        if checkpoint_path.stat().st_size > MAX_CHECKPOINT_BYTES:
            raise ValueError("resume diagnostic exceeds the 25 MiB safety limit")
        payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"could not load resume diagnostic: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError("resume diagnostic root must be an object")
    return payload
