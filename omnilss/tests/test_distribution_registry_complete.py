"""Registry coverage and resolve-family integration tests."""

import pytest

from omnilss.distribution_registry import get_default_registry


def test_registry_coverage() -> None:
    reg = get_default_registry()
    assert len(reg.names()) >= 50, f"Expected 50+ distributions, got {{len(reg.names())}}"


@pytest.mark.parametrize("name", get_default_registry().names())
def test_each_distribution_instantiates(name: str) -> None:
    reg = get_default_registry()
    fam = reg.get(name)
    assert fam is not None
    assert fam.name is not None
    assert len(fam.parameters) >= 1


@pytest.mark.parametrize("name", get_default_registry().names())
def test_resolve_family_via_string(name: str) -> None:
    from omnilss.distributions import resolve_family

    fam = resolve_family(name)
    assert fam is not None
