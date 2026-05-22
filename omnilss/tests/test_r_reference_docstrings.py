"""Ensure key R-ported modules keep explicit R reference docstrings."""

from __future__ import annotations

import importlib


MODULES = [
    "omnilss.extra",
    "omnilss.lpred",
    "omnilss.pdfplot",
    "omnilss.smoothers.pb",
    "omnilss.smoothers.ps",
    "omnilss.smoothers.cs",
]


def test_modules_include_r_reference_marker() -> None:
    for name in MODULES:
        mod = importlib.import_module(name)
        doc = mod.__doc__ or ""
        assert "R reference:" in doc, f"missing explicit R reference marker in {name}"
