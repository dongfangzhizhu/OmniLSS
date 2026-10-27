"""Sichel zero-variants: the hessian slots and the CDF budget (D2.75).

Two defects, both found while measuring quantile-residual coverage (D2.74):

1. ``_neg_info`` wrapped a score slot into R's ``-score^2`` with
   ``def hess(*args)``.  The RS engine calls a hessian slot as
   ``hess_fn(**param_dict)``, so ZASICHEL and ZISICHEL raised
   "hess() got an unexpected keyword argument 'y'" on **every** fit -- which
   is why their rsdev gate has always been a skip.
2. Both CDFs summed the full ``SICHEL_PROBABILITY_MAX_COUNT`` = 100001-term
   support for every observation, whatever the quantile.  A lower tail only
   needs the terms up to the largest q, so the budget is now derived from the
   call itself.  The truncation is exact (logsumexp over ``support <= q``
   never sees a term past q); the tests below pin the values against live R in
   both tails.

The fits are still slow -- ``tofySICHEL`` costs ~0.15 s per score evaluation
and the RS inner loop needs hundreds of them, so a one-cycle fit on twelve
observations takes about four minutes.  That is a documented limitation, not
a claim: the tests here do not run a fit.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from omnilss.distribution_registry import resolve  # noqa: E402
from omnilss.dpqr_functions import _sichel_lower_tail_cap  # noqa: E402

_Q = np.array([0.0, 1.0, 2.0, 5.0, 9.0])
_ARGS = {"mu": 3.0, "sigma": 0.5, "nu": -0.5}
_ZA_ARGS = {**_ARGS, "tau": 0.3}

#: live R 4.6.0 + gamlss.dist 6.1.1, pSICHEL / pZASICHEL at the grid above.
_R_CDF = {
    "SICHEL": [
        0.13533528323661276, 0.33833820809153192, 0.52865345014301857,
        0.85259452493881971, 0.96900380445936196,
    ],
    "ZASICHEL": [
        0.29999999999999999, 0.46434352488714903, 0.61841557946885128,
        0.88066607721769441, 0.97490664709939345,
    ],
}
_R_UPPER = {
    "SICHEL": [
        0.86466471676338719, 0.66166179190846808, 0.47134654985698143,
        0.14740547506118029, 0.030996195540638039,
    ],
    "ZASICHEL": [
        0.69999999999999996, 0.53565647511285097, 0.38158442053114872,
        0.11933392278230559, 0.025093352900606547,
    ],
}


class TestHessianSlotsAcceptKeywords:
    """The regression: the engine calls hessian slots by keyword."""

    @pytest.mark.parametrize("family", ["ZASICHEL", "ZISICHEL"])
    @pytest.mark.parametrize("parameter", ["mu", "sigma", "nu", "tau"])
    def test_the_slot_answers_a_keyword_call(self, family, parameter):
        fam = resolve(family)
        n = 4
        kwargs = {
            "y": np.array([0.0, 1.0, 2.0, 3.0]),
            "mu": np.full(n, 3.0),
            "sigma": np.full(n, 0.5),
            "nu": np.full(n, -0.5),
            "tau": np.full(n, 0.3),
        }
        values = np.asarray(fam.hessian_functions[parameter](**kwargs), dtype=float)
        assert values.shape == (n,)
        assert np.isfinite(values).all()

    @pytest.mark.parametrize("family", ["ZASICHEL", "ZISICHEL"])
    def test_the_derivative_gate_passes(self, family):
        """The scores are still correct once they can be called at all.

        gate_derivatives compares every score slot against central finite
        differences of the density, so this covers all four parameters without
        hard-coding R's formulas.
        """
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
        import rs_family_parity_scan as scan  # noqa: E402

        row = scan.gate_derivatives(family)
        assert row["status"] == "pass", row


class TestCdfBudget:
    """The truncated CDF must be exactly R's full-support answer."""

    @pytest.mark.parametrize("family", ["SICHEL", "ZASICHEL"])
    def test_lower_tail_matches_live_r(self, family):
        fam = resolve(family)
        args = _ARGS if family == "SICHEL" else _ZA_ARGS
        ours = np.asarray(fam.p(_Q, **args), dtype=float)
        np.testing.assert_allclose(ours, _R_CDF[family], rtol=1e-13)

    @pytest.mark.parametrize("family", ["SICHEL", "ZASICHEL"])
    def test_upper_tail_matches_live_r(self, family):
        """The upper tail still scans the full budget -- and stays correct."""
        fam = resolve(family)
        args = _ARGS if family == "SICHEL" else _ZA_ARGS
        ours = np.asarray(fam.p(_Q, **args, lower_tail=False), dtype=float)
        np.testing.assert_allclose(ours, _R_UPPER[family], rtol=1e-13)

    def test_the_cap_is_the_largest_quantile(self):
        assert _sichel_lower_tail_cap(np.array([0.0, 1.0, 4.2]), True, 100000) == 5
        assert _sichel_lower_tail_cap(np.array([0.0, 7.0]), True, 100000) == 7

    def test_the_cap_never_exceeds_the_budget(self):
        assert _sichel_lower_tail_cap(np.array([1e9]), True, 1000) == 1000

    def test_the_upper_tail_keeps_the_full_budget(self):
        assert _sichel_lower_tail_cap(np.array([0.0, 1.0]), False, 1000) == 1000

    def test_an_empty_or_infinite_grid_does_not_raise(self):
        assert _sichel_lower_tail_cap(np.array([np.inf]), True, 1000) == 0
        assert _sichel_lower_tail_cap(np.array([]), True, 1000) == 0