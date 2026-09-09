"""Validated loading for versioned pipeline definitions."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import ValidationError

from ..contracts.pipeline import PipelineConfig
from .config_loader import hydrate_prompt_files
from .registry import validate_tool_contracts


def load_config(path: str | Path) -> PipelineConfig:
    """Load, hydrate, and validate a trusted local pipeline resource."""
    config_path = Path(path)
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        raw = hydrate_prompt_files(raw, config_path)
        config = PipelineConfig.model_validate(raw)
        validate_tool_contracts(config)
        return config
    except FileNotFoundError as error:
        raise ValueError("built in pipeline definition is missing") from error
    except yaml.YAMLError as error:
        raise ValueError(f"invalid built in pipeline definition: {error}") from error
    except ValidationError as error:
        raise ValueError(f"invalid built in pipeline definition:\n{error}") from error
    except ValueError as error:
        raise ValueError(f"invalid built in pipeline definition: {error}") from error


__all__ = ["load_config"]
