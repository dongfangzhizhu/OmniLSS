"""NET: the family whose gates never ran, and why (D2.72).

NET was the only family in the scan with a deriv *and* a dpqr skip at the
same time.  Neither skip said "R has no such function" -- R exports
dNET/pNET/qNET/rNET.  Both came from one probe defect:

* R's NET.R declares parameters = list(mu=TRUE, sigma=TRUE, nu=FALSE,
  tau=FALSE) -- nu/tau are non-estimable shape constants at 1.5 / 2 -- and
  dNET stops on nu <= 0.  The generic parameter grid for an identity-linked
  parameter runs -2.0 .. 3.0, so every probe answered "nu must be positive"
  and the gate recorded "no d/p/q counterpart in R".
* With the constants registered, the calibration could still not call the
  density: build_ad_family generates def _d_auto(*args, log=False), which
  takes no keyword arguments, so nu= raised TypeError and the score read -1
  for every candidate grid -- "no support grid yields a finite density".

The fixes are in tools/rs_family_parity_scan.py: NET joins
_PROBE_FIXED_CONSTANTS, and every call into one of OUR d/p/q functions goes
through _our_call_args, which assembles the positional vector in the
family's own parameters order (grids skip the constants and bd, and fixed
parameters kept out of the tuple are appended where the signatures expect
them -- DBI is that case).

With those in place NET measures all three gates, and its d/p/q agree with
live R exactly (d) and to 1e-16 (p, q).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

TOOLS = Path(__file__).resolve().parents[2] / "tools"
sys.path.insert(0, str(TOOLS))

import rs_family_parity_scan as scan  # noqa: E402

from omnilss.distributions import resolve_family  # noqa: E402

#: The standard probe grid: n = 8 points, mu/sigma 0.5..3.0, nu/tau held at
#: R's non-estimable 1.5 / 2.0.
_X = np.array([0.3, 0.8285714285714285, 1.3571428571428572,
              1.885714285714286, 2.414285714285714, 2.9428571428571426,
              3.4714285714285715, 4.0])
_MU = np.array([0.5, 0.8571428571428572, 1.2142857142857144,
               1.5714285714285714, 1.9285714285714286, 2.2857142857142856,
               2.642857142857143, 3.0])
_SIGMA = _MU.copy()

#: live R 4.6.0 + gamlss.dist 6.1.1.
_R_LOG_DENSITY = [
    -0.3826248348487778, -0.84217689113702032, -1.196848445074594, -1.4677571391517803,
    -1.6842662294383972, -1.8637787135931909, -2.0167779600154829, -2.1499398596323882,
]
_R_CDF = [
    0.35607262193868994, 0.48768767601892138, 0.54336325496115057, 0.57339798182208512,
    0.5920693468655297, 0.60476822348452475, 0.61395409256014666, 0.62090306926266647,
]
_R_QUANTILE = [
    -0.24521691311729354, 0.10928601570271557, 0.62892405218126435, 1.3273904114421309,
    2.2280728067366056, 3.3875715320285442, 4.9487490706309121, 7.4713014787037642,
]


class TestProbeConstants:
    def test_net_holds_r_s_non_estimable_shapes(self):
        """nu = 1.5, tau = 2.0 -- R's nu.initial / tau.initial."""
        assert scan._PROBE_FIXED_CONSTANTS["NET"] == {"nu": 1.5, "tau": 2.0}

    def test_the_probe_grid_holds_them_constant(self):
        jobs, _ = scan._build_jobs(["NET"], 8)
        d_jobs = [j for j in jobs if j.get("type") == "d"]
        assert len(d_jobs) == 1
        job = d_jobs[0]
        assert job["argnames"] == ["x", "mu", "sigma", "nu", "tau"]
        grid = dict(zip(job["argnames"], job["args"]))
        assert set(grid["nu"]) == {1.5}
        assert set(grid["tau"]) == {2.0}

    def test_calibration_finds_a_support_grid(self):
        """This is what used to fail: the density could not be called."""
        cal = scan.calibrate("NET", 8)
        assert cal.ok
        assert len(cal.params) == 2  # mu and sigma vary; nu/tau do not


class TestCallAssembly:
    def test_the_positional_vector_is_in_our_parameter_order(self):
        fam = resolve_family("NET")
        n = 8
        values = {
            "mu": np.linspace(0.5, 3.0, n),
            "sigma": np.linspace(0.5, 3.0, n),
        }
        args = scan._our_call_args(fam, _X, values, None, {"nu": 1.5, "tau": 2.0})
        assert len(args) == 5  # x, mu, sigma, nu, tau
        np.testing.assert_allclose(args[3], 1.5)
        np.testing.assert_allclose(args[4], 2.0)

    def test_a_fixed_parameter_outside_the_tuple_is_appended(self):
        """DBI declares parameters = (mu, sigma) and takes bd after them."""
        fam = resolve_family("DBI")
        n = 8
        values = {
            "mu": np.linspace(0.2, 0.8, n),
            "sigma": np.linspace(0.2, 0.8, n),
        }
        args = scan._our_call_args(fam, _X, values, 10.0, {})
        assert len(args) == 4  # x, mu, sigma, bd
        np.testing.assert_allclose(args[3], 10.0)


class TestGates:
    def test_the_derivative_gate_passes(self):
        row = scan.gate_derivatives("NET")
        assert row["status"] == "pass", row

    @pytest.mark.parametrize("slot", ["d", "p", "q"])
    def test_dpqr_matches_live_r_literals(self, slot):
        fam = resolve_family("NET")
        nu = np.full(8, 1.5)
        tau = np.full(8, 2.0)
        if slot == "d":
            ours = np.asarray(
                fam.d(_X, _MU, _SIGMA, nu, tau, log=True), dtype=float)
            expected = _R_LOG_DENSITY
            tol = 1e-15
        elif slot == "p":
            ours = np.asarray(fam.p(_X, _MU, _SIGMA, nu, tau), dtype=float)
            expected = _R_CDF
            tol = 1e-15
        else:
            # The quantile probe sweeps p over 0.1..0.9 while mu/sigma keep
            # their grids (and nu/tau their constants), exactly as the R job
            # was built.
            ours = np.asarray(
                fam.q(np.linspace(0.1, 0.9, 8), _MU, _SIGMA, nu, tau),
                dtype=float,
            )
            expected = _R_QUANTILE
            tol = 1e-14
        np.testing.assert_allclose(ours, expected, rtol=tol, atol=1e-15)


class TestNoFamilyLostItsGate:
    """The call-assembly refactor touched every family's probe path."""

    @pytest.mark.parametrize(
        "family",
        ["NO", "GA", "PO", "BI", "DBI", "BB", "ZABI", "ZIBI", "ZABB", "NET"],
    )
    def test_the_derivative_gate_still_runs(self, family):
        row = scan.gate_derivatives(family)
        assert row["status"] in {"pass", "skip"}, row
        assert "KeyError" not in row["detail"]