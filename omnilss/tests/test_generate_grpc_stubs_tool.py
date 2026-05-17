"""Tests for gRPC stub generation helper tool."""

from __future__ import annotations

import importlib.util
from pathlib import Path


_TOOL_PATH = Path(__file__).resolve().parents[1] / "tools" / "generate_grpc_stubs.py"
_SPEC = importlib.util.spec_from_file_location("generate_grpc_stubs_tool", _TOOL_PATH)
generate_grpc_stubs = importlib.util.module_from_spec(_SPEC)
assert _SPEC is not None and _SPEC.loader is not None
_SPEC.loader.exec_module(generate_grpc_stubs)


def test_generate_grpc_stubs_reports_missing_grpc_tools(monkeypatch, capsys) -> None:
    """Tool should return 2 with actionable message when grpc_tools is unavailable."""

    original_import = __import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("grpc_tools"):
            raise ImportError("grpc_tools missing")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)
    code = generate_grpc_stubs.main()
    captured = capsys.readouterr()

    assert code == 2
    assert "pip install 'omnilss[grpc]'" in captured.err
