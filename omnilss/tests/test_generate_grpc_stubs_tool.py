"""Tests for gRPC stub generation helper tool."""

from __future__ import annotations

import importlib.util
from pathlib import Path

_TOOL_PATH = Path(__file__).resolve().parents[1] / "tools" / "generate_grpc_stubs.py"
_SPEC = importlib.util.spec_from_file_location("generate_grpc_stubs_tool", _TOOL_PATH)
generate_grpc_stubs = importlib.util.module_from_spec(_SPEC)
assert _SPEC is not None and _SPEC.loader is not None
_SPEC.loader.exec_module(generate_grpc_stubs)


def test_generate_grpc_stubs_reports_missing_compilers(monkeypatch, capsys) -> None:
    """Tool should return 2 when neither grpc_tools nor system protoc is available."""

    original_import = __import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("grpc_tools"):
            raise ImportError("grpc_tools missing")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)
    monkeypatch.setattr(generate_grpc_stubs.shutil, "which", lambda name: None)
    code = generate_grpc_stubs.main()
    captured = capsys.readouterr()

    assert code == 2
    assert "pip install 'omnilss[grpc]'" in captured.err


def test_fallback_generator_supports_multiple_unary_methods() -> None:
    """Capability service fallback metadata should include both unary RPCs."""

    service, methods = generate_grpc_stubs.SERVICE_SPECS["capability"]

    assert service == "CapabilityService"
    assert ("CapabilityMatrix", "CapabilityMatrixRequest", "CapabilityMatrixResponse") in methods
    assert ("RouteCapability", "RouteCapabilityRequest", "RouteCapabilityResponse") in methods
