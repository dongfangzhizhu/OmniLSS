"""Support-bounded probe grids and the bd contract (D2.54).

The scanner read six families as dpqr failures / rsdev skips because the
probe grid spilled outside their response support:

- MN3/MN4/MN5 are categorical over 1..k.  R dMN3 says "x must be 1, 2 or 3"
  and stops, so the generic grids (0..7, 1..8, continuous) could not be
  evaluated on the R side at all; the reported "failure" was a probe
  defect, and the fit sample (drawn over 0..7) made R fit stop with
  "response variable out of range".
- ZABB/ZIBB are beta-binomial over 0..bd, but their family objects did not
  declare bd as a fixed parameter, so the scanner passed no bd and R fell
  back to bd = 1 ("x must be <= the binomial denominator 1").  The kernels
  also rejected the float/vector bd the probe carries (dZABB requires one
  static integer: it sizes an internal grid).

The fixes: _SUPPORT_BOUNDS in the scanner turns the support itself into the
probe grid for MN3-5; ZABB/ZIBB declare fixed_parameters=("bd",) and their
d/p/q collapse a constant vector bd to the static int (the shape BI already
had).  This file pins both contracts against live R.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

from omnilss.distributions import resolve_family

RSCRIPT = shutil.which("Rscript")
TOOLS = Path(__file__).resolve().parents[2] / "tools"

MN_FAMILIES = ("MN3", "MN4", "MN5")
MN_SUPPORT = {"MN3": (1, 3), "MN4": (1, 4), "MN5": (1, 5)}
BB_FAMILIES = ("ZABB", "ZIBB")
BB_BD = 10


def _r_vector(values) -> str:
    return "c(" + ", ".join(repr(float(v)) for v in values) + ")"


def _run_r(lines, timeout=600) -> str:
    if RSCRIPT is None:  # pragma: no cover - CI without R
        pytest.skip("Rscript not on PATH")
    with tempfile.NamedTemporaryFile("w", suffix=".R", delete=False,
                                     encoding="utf-8") as fh:
        fh.write(chr(10).join(lines))
        path = fh.name
    try:
        proc = subprocess.run([RSCRIPT, path], capture_output=True,
                              text=True, timeout=timeout)
    finally:
        Path(path).unlink(missing_ok=True)
    assert proc.returncode == 0, f"R failed: {proc.stderr[-800:]}"
    return proc.stdout

def _r_probe_script():
    """R program: MN d/p/q on the support and ZABB/ZIBB d/p/q with bd."""
    nl = chr(10)
    lines = [
        "suppressPackageStartupMessages({library(gamlss.dist); library(jsonlite)})",
        "out <- list()",
    ]
    for fam in MN_FAMILIES:
        lo, hi = MN_SUPPORT[fam]
        y = _r_vector(range(lo, hi + 1))
        probs = _r_vector([0.2, 0.5, 0.8])
        lines.append("out$" + fam + " <- list(")
        lines.append("  y = " + y + ",")
        lines.append("  d = d" + fam + "(" + y + ", mu = 1.2, sigma = 0.8),")
        lines.append("  p = p" + fam + "(" + y + ", mu = 1.2, sigma = 0.8),")
        lines.append("  q = q" + fam + "(" + probs + ", mu = 1.2, sigma = 0.8),")
        lines.append("  pgrid = " + probs)
        lines.append(")")
    grid = _r_vector(range(0, 8))
    for fam in BB_FAMILIES:
        lines.append("out$" + fam + " <- list(")
        lines.append("  y = " + grid + ",")
        lines.append("  d = d" + fam + "(" + grid + ", mu = 0.4, sigma = 1.1, nu = 0.3, bd = " + str(BB_BD) + "),")
        lines.append("  p = p" + fam + "(" + grid + ", mu = 0.4, sigma = 1.1, nu = 0.3, bd = " + str(BB_BD) + ")")
        lines.append(")")
    lines.append("cat(toJSON(out, auto_unbox = TRUE, digits = 17, na = \"null\"))")
    return nl.join(lines)


@pytest.fixture(scope="module")
def r_probe():
    if RSCRIPT is None:  # pragma: no cover
        pytest.skip("Rscript not on PATH")
    return json.loads(_run_r(_r_probe_script().splitlines()))


# ---------------------------------------------------------------------------
# MN3/MN4/MN5: the support is the probe grid
# ---------------------------------------------------------------------------


class TestCategoricalSupport:
    def test_calibration_uses_the_support(self):
        """The scanner probe grid must stay inside 1..k."""
        sys.path.insert(0, str(TOOLS))
        import rs_family_parity_scan as scan

        for fam in MN_FAMILIES:
            cal = scan.calibrate(fam, 8)
            lo, hi = MN_SUPPORT[fam]
            assert cal.ok, fam + " calibration failed on its own support"
            assert set(np.unique(cal.y)) == set(range(lo, hi + 1)), (
                fam + " probe grid " + str(cal.y) + " leaves the support"
            )

    def test_fit_sample_stays_inside_the_support(self):
        """The fit sample is drawn from the calibrated range (1..k)."""
        sys.path.insert(0, str(TOOLS))
        import rs_family_parity_scan as scan

        rng = np.random.default_rng(20261006)
        for fam in MN_FAMILIES:
            lo, hi = MN_SUPPORT[fam]
            cal = scan.calibrate(fam, 8)
            data = scan._fit_data(fam, rng, cal.y)
            y = np.asarray(data["y"])
            assert np.all((y >= lo) & (y <= hi)), fam + " fit sample leaves the support"
            assert np.all(y == np.round(y)), fam + " fit sample is not integral"

    @pytest.mark.skipif(RSCRIPT is None, reason="Rscript not on PATH")
    def test_dpqr_matches_live_r_on_the_support(self, r_probe):
        for fam in MN_FAMILIES:
            f = resolve_family(fam)
            r = r_probe[fam]
            y = np.asarray(r["y"], dtype=float)
            np.testing.assert_allclose(
                np.asarray(f.d(y, 1.2, 0.8)), np.asarray(r["d"]),
                rtol=1e-12, atol=0, err_msg="d" + fam + " vs live R",
            )
            np.testing.assert_allclose(
                np.asarray(f.p(y, 1.2, 0.8)), np.asarray(r["p"]),
                rtol=1e-12, atol=0, err_msg="p" + fam + " vs live R",
            )
            np.testing.assert_allclose(
                np.asarray(f.q(r["pgrid"], 1.2, 0.8)), np.asarray(r["q"]),
                rtol=1e-12, atol=0, err_msg="q" + fam + " vs live R",
            )

    def test_fractional_values_inside_the_range_fall_to_the_reference(self):
        """R ifelse fallthrough: values in [1, k] that are not 1..k-1 carry
        the reference-category mass, and our kernel reproduces that."""
        f = resolve_family("MN3")
        expected = 1.0 / (1.0 + 1.2 + 0.8)
        np.testing.assert_allclose(float(np.asarray(f.d(1.5, 1.2, 0.8))), expected,
                                   rtol=1e-12)


# ---------------------------------------------------------------------------
# ZABB/ZIBB: bd is a fixed parameter and the probe vector bd is accepted
# ---------------------------------------------------------------------------


class TestBdContract:
    def test_bd_is_a_fixed_parameter(self):
        for fam in BB_FAMILIES:
            f = resolve_family(fam)
            assert "bd" in (f.fixed_parameters or ()), fam + " must declare bd"

    def test_dpqr_accept_constant_vector_bd(self):
        """The scanner passes np.full(n, bd); the kernels need one int."""
        for fam in BB_FAMILIES:
            f = resolve_family(fam)
            grid = np.arange(0, 8, dtype=float)
            bd_vec = np.full(grid.shape, float(BB_BD))
            scalar = np.asarray(f.d(grid, 0.4, 1.1, 0.3))
            vector = np.asarray(f.d(grid, 0.4, 1.1, 0.3, bd=bd_vec))
            np.testing.assert_allclose(vector, scalar, rtol=1e-12)
            np.testing.assert_allclose(
                np.asarray(f.p(grid, 0.4, 1.1, 0.3, bd=bd_vec)),
                np.asarray(f.p(grid, 0.4, 1.1, 0.3)),
                rtol=1e-12,
            )

    def test_non_constant_bd_is_rejected(self):
        f = resolve_family("ZABB")
        bad = np.array([9.0, 10.0])
        with pytest.raises(Exception):
            np.asarray(f.d(np.array([0.0, 1.0]), 0.4, 1.1, 0.3, bd=bad))

    @pytest.mark.skipif(RSCRIPT is None, reason="Rscript not on PATH")
    def test_dpqr_matches_live_r_with_bd(self, r_probe):
        for fam in BB_FAMILIES:
            f = resolve_family(fam)
            r = r_probe[fam]
            grid = np.asarray(r["y"], dtype=float)
            np.testing.assert_allclose(
                np.asarray(f.d(grid, 0.4, 1.1, 0.3)),
                np.asarray(r["d"]),
                rtol=1e-10, atol=1e-300, err_msg="d" + fam + " vs live R",
            )
            np.testing.assert_allclose(
                np.asarray(f.p(grid, 0.4, 1.1, 0.3)),
                np.asarray(r["p"]),
                rtol=1e-10, atol=1e-300, err_msg="p" + fam + " vs live R",
            )
