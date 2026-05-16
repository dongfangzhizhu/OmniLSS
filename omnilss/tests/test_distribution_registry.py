import pytest

from omnilss.distribution_registry import create_default_registry


def test_default_registry_contains_phase1_core_set():
    reg = create_default_registry()
    names = reg.names()
    for fam in ("NO", "GA", "TF", "BE", "ZIP", "ZINBI"):
        assert fam in names


def test_registry_get_unknown_raises():
    reg = create_default_registry()
    with pytest.raises(KeyError):
        reg.get("UNKNOWN")
