"""Capability admission must match the evidence that exists (D2.73).

_R_CONSISTENCY_EVIDENCE in omnilss/family_capabilities.py is hand-maintained,
and it had drifted: three families with live-R test modules in this repository
were still recorded as unsupported, so the generated coverage artifacts
claimed their d/p/q was blocked when the parity gate measures it.

The set is deliberately NOT "every family the parity scanner can compare" --
that is 116 families and the scanner is a tool, not a test module.  Admission
here means: a test in omnilss/tests holds values taken from live R.  The gap
between the two sets is recorded, not closed by assertion.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[2] / "omnilss" / "src"
sys.path.insert(0, str(SRC))

from omnilss.family_capabilities import (  # noqa: E402
    _R_CONSISTENCY_EVIDENCE,
    get_family_capability,
    list_family_capabilities,
)

COVERAGE = (
    Path(__file__).resolve().parents[2] / "docs" / "development"
    / "r-export-coverage.json"
)

#: (family, the test module holding its live-R evidence).
_ADMITTED = {
    "ZABI": "omnilss/tests/test_binomial_zero_variants_bd.py",
    "ZIBI": "omnilss/tests/test_binomial_zero_variants_bd.py",
    "NET": "omnilss/tests/test_net_probe_reachability.py",
}


@pytest.mark.parametrize("family,test_module", sorted(_ADMITTED.items()))
def test_the_admitted_families_are_in_the_evidence_set(family, test_module):
    assert family in _R_CONSISTENCY_EVIDENCE, family
    module = Path(__file__).resolve().parents[2] / test_module
    assert module.exists(), test_module
    capability = get_family_capability(family)
    assert capability.status("r_consistency").value == "validated"


def test_the_generated_artifact_agrees_with_the_runtime_capability():
    """No drift between the committed artifact and the capability matrix."""
    if not COVERAGE.exists():  # pragma: no cover - artifact is committed
        pytest.skip("coverage artifact not generated")
    coverage = json.loads(COVERAGE.read_text(encoding="utf-8"))
    families = coverage["families"]
    for capability in list_family_capabilities():
        entry = families.get(capability.name)
        assert entry is not None, capability.name
        expected = capability.status("r_consistency").value
        actual = entry.get("evidence_tier_dpqr")
        if expected == "validated":
            assert actual == "VALIDATED", (capability.name, actual)


def test_admission_never_claims_evidence_the_scanner_does_not_measure():
    """Every admitted family must still pass the deriv gate.

    The reverse is not asserted on purpose: 79 families pass the dpqr parity
    gate with no test module holding R values, and closing that gap is a
    decision about whether a tool run counts as evidence (D2.73 log).
    """
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
    import rs_family_parity_scan as scan  # noqa: E402

    for family in _ADMITTED:
        row = scan.gate_derivatives(family)
        assert row["status"] == "pass", (family, row)