"""Tests for the reproducible environment baseline collector."""

from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "generate_environment_baseline", ROOT / "tools" / "generate_environment_baseline.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_collect_has_traceability_and_explicit_statuses(monkeypatch, tmp_path):
    def git_value(_root, *args):
        return "abc123" if args == ("rev-parse", "HEAD") else ""
    monkeypatch.setattr(MODULE, "_git_value", git_value)
    monkeypatch.setattr(MODULE.shutil, "which", lambda name: None)
    payload = MODULE.collect(tmp_path)
    assert payload["source"]["git_commit"] == "abc123"
    assert payload["source"]["git_dirty"] is False
    assert payload["r"]["status"] == "unavailable"
    assert payload["wsl"]["status"] == "unavailable"
    assert "x64_enabled" in payload["jax"]


def test_collect_records_r_and_wsl_probe(monkeypatch, tmp_path):
    monkeypatch.setattr(MODULE, "_git_value", lambda *_args: "")
    monkeypatch.setattr(MODULE.shutil, "which", lambda name: "Rscript.exe" if name == "Rscript" else "wsl.exe")
    calls = []

    def fake_run(command, timeout=30):
        calls.append(command)
        if command[0] == "Rscript.exe":
            return "R version 4.6.0\ngamlss = 5.5.0\ngamlss.dist = 6.1.1\njsonlite = 1.9.0"
        if command[0] == "wsl.exe":
            return "3.11.15\n0.10.0\nTrue\n[CudaDevice(id=0)]"
        return ""

    monkeypatch.setattr(MODULE, "_run", fake_run)
    payload = MODULE.collect(tmp_path)
    assert payload["r"]["status"] == "available"
    assert payload["wsl"]["available"] is True
    assert "Rscript.exe" in calls[0]
    assert calls[1][:4] == ["wsl.exe", "-d", "debianjaxgpu", "--"]
