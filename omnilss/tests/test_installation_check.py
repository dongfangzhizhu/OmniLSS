"""Tests for installation diagnostics exposed in omnilss.__init__."""

from __future__ import annotations


def test_check_installation_returns_core_keys() -> None:
    """check_installation should return status keys for major subsystems."""
    import omnilss

    status = omnilss.check_installation()
    for key in {
        "core",
        "distributions",
        "smoothers",
        "diagnostics",
        "prediction",
        "serialization",
        "deep",
        "sklearn",
        "grpc",
    }:
        assert key in status
        assert isinstance(status[key], str)


def test_check_installation_core_is_ok() -> None:
    """Core fitting import should be available in test environment."""
    import omnilss

    status = omnilss.check_installation()
    assert status["core"] == "ok"
