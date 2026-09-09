from __future__ import annotations

import hashlib
import importlib.util
import json
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("public_export", ROOT / "scripts/export_public.py")
assert spec and spec.loader
exporter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exporter)


def fixture_source(tmp_path):
    source = tmp_path / "source"
    for name in exporter.REQUIRED:
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# source\n")
    (source / exporter.MANIFEST).write_text("\n".join(sorted(exporter.REQUIRED)) + "\n")
    return source


def test_export_is_reproducible_and_omits_private_working_files(tmp_path):
    source = fixture_source(tmp_path)
    for name in (
        ".env",
        ".git/config",
        "agent/output/private.py",
        ".venv/private.py",
        "knowledge/records.jsonl",
        "agent/local_notes.py",
        "backup.tar.gz",
    ):
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("private local content")
    first = tmp_path / "first"
    second = tmp_path / "second"
    report = exporter.export(source, first)
    exporter.export(source, second)
    assert (first / "abstract-agent.zip").read_bytes() == (
        second / "abstract-agent.zip"
    ).read_bytes()
    assert report["file_count"] == len(exporter.REQUIRED)
    with zipfile.ZipFile(first / "abstract-agent.zip") as archive:
        assert set(archive.namelist()) == {f"abstract-agent/{p}" for p in exporter.REQUIRED}
        for name, digest in report["files"].items():
            assert hashlib.sha256(archive.read(f"abstract-agent/{name}")).hexdigest() == digest
    assert (source / ".env").read_text() == "private local content"
    with pytest.raises(ValueError, match="already exists"):
        exporter.export(source, first)


@pytest.mark.parametrize(
    "name",
    [
        "../outside.py",
        "/outside.py",
        "agent/output/result.py",
        "agent/.env",
        "agent/cache.sqlite3",
        ".git/config",
    ],
)
def test_manifest_rejects_unsafe_or_local_paths(tmp_path, name):
    source = fixture_source(tmp_path)
    with (source / exporter.MANIFEST).open("a") as manifest:
        manifest.write(name + "\n")
    with pytest.raises(ValueError):
        exporter.export(source, tmp_path / "public")
    assert not (tmp_path / "public").exists()


def test_manifest_rejects_links_and_missing_files(tmp_path):
    source = fixture_source(tmp_path)
    entry = source / "abs.py"
    entry.unlink()
    with pytest.raises(ValueError, match="missing"):
        exporter.export(source, tmp_path / "public")
    outside = tmp_path / "outside.py"
    outside.write_text("# private\n")
    entry.symlink_to(outside)
    with pytest.raises(ValueError, match="symbolic link"):
        exporter.export(source, tmp_path / "public")


@pytest.mark.parametrize(
    "content",
    [
        "token = '" + "gh" + "p_" + "a" * 36 + "'",
        "path = '" + "/Users" + "/someone/private/'",
        "# text\u2014with punctuation",
        "# text\u2013with punctuation",
        "# text\x00",
    ],
)
def test_scan_stops_export_without_exposing_values(tmp_path, content):
    source = fixture_source(tmp_path)
    (source / "abs.py").write_text(content)
    with pytest.raises(ValueError) as caught:
        exporter.export(source, tmp_path / "public")
    assert content not in str(caught.value)
    assert not (tmp_path / "public").exists()


def test_docs_check_preserves_commands_but_rejects_broken_links_and_compounds(tmp_path):
    source = fixture_source(tmp_path)
    readme = source / "README.md"
    readme.write_text("Use `abstract-agent --help`.\n")
    exporter.scan(source, exporter.public_paths(source))
    readme.write_text("[missing](missing.md)\n")
    with pytest.raises(ValueError, match="link missing"):
        exporter.scan(source, exporter.public_paths(source))
    readme.write_text("A well" + "-" + "formed sentence.\n")
    with pytest.raises(ValueError, match="hyphenated prose"):
        exporter.scan(source, exporter.public_paths(source))


def test_export_rejects_output_in_source(tmp_path):
    source = fixture_source(tmp_path)
    with pytest.raises(ValueError, match="outside"):
        exporter.export(source, source / "public")


def test_real_public_inventory_is_complete_and_passes_scan():
    paths = exporter.public_paths(ROOT)
    hashes = exporter.scan(ROOT, paths)
    assert "tests/test_public_export.py" in hashes
    assert "docs/export.md" in hashes
    assert len(hashes) == len(paths)
    json.dumps(hashes)
