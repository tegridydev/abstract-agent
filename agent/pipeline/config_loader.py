"""Safe loading of prompt resources referenced by pipeline YAML."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def hydrate_prompt_files(raw: Any, config_path: Path) -> Any:
    """Replace trusted prompt resource keys with inline text before validation."""
    if not isinstance(raw, dict) or not isinstance(raw.get("steps"), list):
        return raw
    base = config_path.parent.resolve()
    for index, step in enumerate(raw["steps"]):
        if not isinstance(step, dict):
            continue
        for file_key, value_key in (
            ("system_prompt_file", "system_prompt"),
            ("prompt_file", "prompt_template"),
        ):
            if file_key not in step:
                continue
            if value_key in step:
                raise ValueError(f"step {index} cannot define both {file_key} and {value_key}")
            relative = step.pop(file_key)
            if not isinstance(relative, str) or not relative.strip():
                raise ValueError(f"step {index} has an invalid {file_key}")
            prompt_path = (base / relative).resolve()
            if not prompt_path.is_relative_to(base):
                raise ValueError(f"step {index} {file_key} escapes the pipeline directory")
            try:
                prompt = prompt_path.read_text(encoding="utf-8").strip()
            except OSError as error:
                raise ValueError(
                    f"step {index} prompt resource is unavailable: {relative}"
                ) from error
            if len(prompt) < 10:
                raise ValueError(f"step {index} prompt resource is empty: {relative}")
            step[value_key] = prompt
    return raw
