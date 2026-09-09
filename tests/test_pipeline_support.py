from pathlib import Path

import pytest

from agent.pipeline.config_loader import hydrate_prompt_files
from agent.pipeline.state import PipelineState


def test_prompt_resource_hydration_is_relative_and_non_destructive(tmp_path: Path) -> None:
    prompts = tmp_path / "prompts"
    prompts.mkdir()
    (prompts / "stage.txt").write_text("Return a small validated response.", encoding="utf-8")
    raw = {"steps": [{"name": "Stage", "prompt_file": "prompts/stage.txt"}]}

    hydrated = hydrate_prompt_files(raw, tmp_path / "pipeline.yaml")

    assert hydrated["steps"][0] == {
        "name": "Stage",
        "prompt_template": "Return a small validated response.",
    }


def test_prompt_resource_hydration_rejects_ambiguous_or_escaping_paths(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="both prompt_file and prompt_template"):
        hydrate_prompt_files(
            {"steps": [{"prompt_file": "one.txt", "prompt_template": "inline"}]},
            tmp_path / "pipeline.yaml",
        )
    with pytest.raises(ValueError, match="escapes the pipeline directory"):
        hydrate_prompt_files(
            {"steps": [{"prompt_file": "../outside.txt"}]},
            tmp_path / "pipeline.yaml",
        )
    with pytest.raises(ValueError, match="invalid prompt_file"):
        hydrate_prompt_files(
            {"steps": [{"prompt_file": ""}]},
            tmp_path / "pipeline.yaml",
        )
    with pytest.raises(ValueError, match="unavailable"):
        hydrate_prompt_files(
            {"steps": [{"prompt_file": "missing.txt"}]},
            tmp_path / "pipeline.yaml",
        )
    (tmp_path / "short.txt").write_text("short", encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        hydrate_prompt_files(
            {"steps": [{"prompt_file": "short.txt"}]},
            tmp_path / "pipeline.yaml",
        )

    unchanged = ["not", "a", "pipeline"]
    assert hydrate_prompt_files(unchanged, tmp_path / "pipeline.yaml") is unchanged


def test_pipeline_state_commits_multi_output_atomically() -> None:
    state = PipelineState({"input": 1})

    with pytest.raises(ValueError, match="returned 1 values for 2 outputs"):
        state.commit(["first", "second"], [2])
    assert state.context == {"input": 1}

    state.commit(["first", "second"], (2, 3))
    assert state.values_for(["first", "second"]) == {"first": 2, "second": 3}

    with pytest.raises(ValueError, match="missing context inputs"):
        state.values_for(["missing"])

    with pytest.raises(ValueError, match="overwrite context keys"):
        state.commit(["first"], 4)
    assert state.context["first"] == 2
