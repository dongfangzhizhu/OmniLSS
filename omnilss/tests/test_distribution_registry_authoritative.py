# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for the authoritative distribution registry API."""

import pytest

from omnilss.distribution_registry import list_families, register, resolve
from omnilss.distributions import NO, resolve_family
from omnilss.families import FamilyDefinition


def test_list_families_exposes_registered_family_names():
    names = list_families()
    assert names == sorted(names)
    assert len(names) >= 47
    assert {"NO", "GA", "WEI", "TF"}.issubset(names)


def test_resolve_is_case_insensitive_and_accepts_family_instances():
    upper = resolve("NO")
    lower = resolve("no")
    instance = NO()

    assert upper.name == "NO"
    assert lower.name == "NO"
    assert resolve(instance) is instance
    assert resolve_family("no").name == "NO"


def test_register_adds_new_family_factory():
    register("UNITTEST_NO_ALIAS", NO)
    try:
        family = resolve("unittest_no_alias")
        assert isinstance(family, FamilyDefinition)
        assert family.name == "NO"
        assert "UNITTEST_NO_ALIAS" in list_families()
    finally:
        # Keep this test isolated from any later tests in the same process.
        from omnilss import distribution_registry

        distribution_registry._REGISTRY.pop("UNITTEST_NO_ALIAS", None)


def test_resolve_unknown_family_reports_available_names():
    with pytest.raises(ValueError, match="Unknown family"):
        resolve("UNKNOWN_TEST_FAMILY")
