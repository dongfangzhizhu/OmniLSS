from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

_MODULE_PATH = Path(__file__).resolve().parents[1] / "tools" / "audit_gamlss_migration.py"
_SPEC = importlib.util.spec_from_file_location("audit_gamlss_migration", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
mod = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = mod
_SPEC.loader.exec_module(mod)


def test_audit_payload_shape() -> None:
    payload = mod.build_audit()
    assert "rows" in payload
    assert payload["requirements_total"] >= 20
    assert payload["requirements_ok"] + payload["requirements_missing"] == payload["requirements_total"]
