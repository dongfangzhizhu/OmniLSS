"""No initial-value branch may be written with a name no family answers to.

`_fitting_init.py` branches on family.name -- the name on the FAMILY OBJECT --
which is not always the registry key.  Every "original parameterisation" alias
differs, e.g. JSUO/JSUo, SHASHO/SHASHo, BCCGO/BCCGo, GEOMO/GEOMo.

Scanning for missing branches with registry keys instead of .name reports a
screenful of false positives (D2.110), and "fixing" one by rewriting the
literal in the branch -- "SHASHo" -> "SHASHO" -- silently DISABLES a branch
that was working: SHASHO/SHASHO2 then fall through to the generic default,
sigma lands on the eps floor and the fit collapses to ~1e13 (D2.113/D2.118).

So this test guards the direction that actually bites: a literal that looks
like a family (it is a registry key) but is not any family object name is a
branch that can never fire.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT.parent / "tools"))

from omnilss.distributions import resolve_family  # noqa: E402

import rs_family_parity_scan as scan  # noqa: E402


def _registry_keys() -> list[str]:
    return scan.discover_families()


def _family_names() -> set[str]:
    return {resolve_family(key).name for key in _registry_keys()}


def _branch_literals() -> set[str]:
    path = ROOT / "src" / "omnilss" / "_fitting_init.py"
    source = path.read_text(encoding="utf-8")
    return set(re.findall(chr(34) + "([A-Za-z][A-Za-z0-9_]*)" + chr(34), source))


def test_no_branch_is_written_with_a_name_no_family_answers_to():
    keys = set(_registry_keys())
    names = _family_names()
    dead = sorted(n for n in _branch_literals() if n in keys and n not in names)
    assert dead == [], dead


def test_the_alias_registry_keys_are_documented():
    """The key/name pairs that caused the D2.110 false positive."""
    expected = {
        "BCCGO": "BCCGo",
        "BCPEO": "BCPEo",
        "BCTO": "BCTo",
        "BEO": "BEo",
        "GEOMO": "GEOMo",
        "JSUO": "JSUo",
        "LOGSHASHO": "LOGSHASHo",
        "PARETO2O": "PARETO2o",
        "SHASHO": "SHASHo",
        "SHASHO2": "SHASHo2",
    }
    for key, name in expected.items():
        got = resolve_family(key).name
        assert got == name, (key, got)