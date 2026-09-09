"""Loading and atomic persistence of non secret interactive preferences."""

from __future__ import annotations

import json
import tomllib
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from ..application.io import atomic_write
from ..schemas import InteractiveSettings


def load_interactive_settings(path: Path) -> tuple[InteractiveSettings, str | None]:
    defaults: dict[str, Any] = {}
    if not path.exists():
        return InteractiveSettings.model_validate(defaults), None
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
        raw.pop("config_path", None)
        return InteractiveSettings.model_validate({**defaults, **raw}), None
    except (OSError, tomllib.TOMLDecodeError, ValidationError) as error:
        return (
            InteractiveSettings.model_validate(defaults),
            f"Ignoring invalid settings in {path}: {error}",
        )


def save_interactive_settings(path: Path, settings: InteractiveSettings) -> None:
    lines = []
    values = settings.model_dump()
    model_assignments = values.pop("model_assignments")
    for key, value in values.items():
        if value is None:
            continue
        if isinstance(value, bool):
            rendered = "true" if value else "false"
        elif isinstance(value, str):
            rendered = json.dumps(value, ensure_ascii=False)
        else:
            rendered = str(value)
        lines.append(f"{key} = {rendered}")
    lines.append("")
    lines.append("[model_assignments]")
    for key, value in model_assignments.items():
        if value:
            lines.append(f"{key} = {json.dumps(value, ensure_ascii=False)}")
    atomic_write(path, "\n".join(lines) + "\n")
