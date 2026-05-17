"""Ensure batch14 families are present in default distribution registry."""

from omnilss.distribution_registry import get_default_registry


def test_batch14_names_in_registry() -> None:
    reg = get_default_registry()
    names = set(reg.names())
    for name in {"SHASHO2", "JSUO", "ST5", "BCPEO", "BCTO"}:
        assert name in names
