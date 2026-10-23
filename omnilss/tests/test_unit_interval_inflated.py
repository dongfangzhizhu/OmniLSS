"""Unit-interval inflated-beta boundary contract (D2.55).

The four failing families were all the same shape of defect: the density
routed by y <= 0 / y >= 1 instead of by exact equality, so x > 1 fell
into the boundary atom and x = 1 leaked through the beta density.
Measured before the fix: dBEINF(2) returned tau/(1+nu+tau) where R answers
0; dBEZI(2.41) returned 3121.8; dBEINF0(1) returned 1.7e+11.

R rules (each family its own):
  BEINF  : x==0 -> nu, x==1 -> tau, 0<x<1 -> beta, outside [0,1] -> 0
  BEINF0 : x==0 -> nu, 0<x<1 -> beta, x>=1 or x<0 -> 0
  BEINF1 : x==1 -> nu, 0<x<1 -> beta, x<=0 or x>1 -> 0
  BEZI   : x==0 -> nu, 0<x<1 -> (1-nu)beta, x>=1 or x<0 -> 0
  BEOI   : x==1 -> nu, 0<x<1 -> (1-nu)beta, x<=0 or x>1 -> 0

The scanner probe grid is now linspace(0, 1, n) for these families (it
used to be 0..7, which scored every out-of-support point as "good" while
our permissive density answered with finite garbage), and _fit_data no
longer injects a zero atom into the two families that have none.
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

FAMILIES = ("BEINF", "BEINF0", "BEINF1", "BEZI", "BEOI")
# Which boundary carries an atom, and the parameter names in R order.
ATOMS = {"BEINF": (True, True), "BEINF0": (True, False), "BEINF1": (False, True),
         "BEZI": (True, False), "BEOI": (False, True)}
PARAM_ORDER = {"BEINF": ("mu", "sigma", "nu", "tau")}

MU = 0.4
SIGMA = 0.7
NU = 0.25
TAU = 0.15


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

def _r_probe_script() -> str:
    """R program: d/p on the unit grid for the five inflated-beta families."""
    nl = chr(10)
    grid = _r_vector(np.linspace(0.0, 1.0, 9))
    out = _r_vector([0.05, 0.3, 0.7, 0.95])
    lines = [
        "suppressPackageStartupMessages({library(gamlss.dist); library(jsonlite)})",
        "out <- list()",
    ]
    args = {
        "BEINF": f"mu = {MU}, sigma = {SIGMA}, nu = {NU}, tau = {TAU}",
        "BEINF0": f"mu = {MU}, sigma = {SIGMA}, nu = {NU}",
        "BEINF1": f"mu = {MU}, sigma = {SIGMA}, nu = {NU}",
        "BEZI": f"mu = {MU}, sigma = {SIGMA}, nu = {NU}",
        "BEOI": f"mu = {MU}, sigma = {SIGMA}, nu = {NU}",
    }
    for fam in FAMILIES:
        lines.append("out$" + fam + " <- list(")
        lines.append("  grid = " + grid + ",")
        lines.append("  d = d" + fam + "(" + grid + ", " + args[fam] + "),")
        lines.append("  p = p" + fam + "(" + grid + ", " + args[fam] + "),")
        lines.append("  out = " + out + ",")
        lines.append("  pout = p" + fam + "(" + out + ", " + args[fam] + ")")
        lines.append(")")
    lines.append("cat(toJSON(out, auto_unbox = TRUE, digits = 17))")
    return nl.join(lines)


@pytest.fixture(scope="module")
def r_probe():
    if RSCRIPT is None:  # pragma: no cover
        pytest.skip("Rscript not on PATH")
    return json.loads(_run_r(_r_probe_script().splitlines()))


def _call(f, fam, first):
    if fam == "BEINF":
        return f(first, MU, SIGMA, NU, TAU)
    return f(first, MU, SIGMA, NU)


# ---------------------------------------------------------------------------
# Density on [0, 1], including both boundary atoms
# ---------------------------------------------------------------------------


@pytest.mark.skipif(RSCRIPT is None, reason="Rscript not on PATH")
class TestUnitIntervalDensity:
    def test_density_matches_live_r_on_the_closed_interval(self, r_probe):
        for fam in FAMILIES:
            f = resolve_family(fam)
            r = r_probe[fam]
            grid = np.asarray(r["grid"], dtype=float)
            ours = np.asarray(_call(f.d, fam, grid))
            ref = np.asarray(r["d"])
            finite = ref > 0
            np.testing.assert_allclose(
                ours[finite], ref[finite], rtol=1e-10,
                err_msg=fam + " d on [0,1]",
            )
            # The atoms and the interior must all be positive somewhere;
            # a family whose every value was zero would be a routing bug.
            assert finite.sum() >= 8, fam + " lost the interior"

    def test_outside_the_unit_interval_is_zero(self):
        for fam in FAMILIES:
            f = resolve_family(fam)
            for x in (-0.5, 1.5, 2.0, 7.0):
                v = float(np.ravel(np.asarray(_call(f.d, fam, np.array([x]))))[0])
                assert v == 0.0, (fam, x, v)

    def test_each_family_has_its_own_atoms(self):
        """The boundary rules differ between families (D2.19)."""
        beinf = resolve_family("BEINF")
        beinf0 = resolve_family("BEINF0")
        beinf1 = resolve_family("BEINF1")
        bezi = resolve_family("BEZI")
        beoi = resolve_family("BEOI")
        one = lambda f, fam, x: float(np.ravel(np.asarray(_call(f.d, fam, np.array([x]))))[0])
        # BEINF: both ends are atoms
        np.testing.assert_allclose(one(beinf, "BEINF", 0.0), NU / (1 + NU + TAU),
                                   rtol=1e-12)
        np.testing.assert_allclose(one(beinf, "BEINF", 1.0), TAU / (1 + NU + TAU),
                                   rtol=1e-12)
        # BEINF0: only 0; x = 1 is outside its (0,1) support
        np.testing.assert_allclose(one(beinf0, "BEINF0", 0.0), NU / (1 + NU),
                                   rtol=1e-12)
        assert one(beinf0, "BEINF0", 1.0) == 0.0
        # BEINF1: only 1; x = 0 carries no mass
        assert one(beinf1, "BEINF1", 0.0) == 0.0
        np.testing.assert_allclose(one(beinf1, "BEINF1", 1.0), NU / (1 + NU),
                                   rtol=1e-12)
        # BEZI: only 0, atom value nu (no denominator)
        np.testing.assert_allclose(one(bezi, "BEZI", 0.0), NU, rtol=1e-12)
        assert one(bezi, "BEZI", 1.0) == 0.0
        # BEOI: only 1, atom value nu
        assert one(beoi, "BEOI", 0.0) == 0.0
        np.testing.assert_allclose(one(beoi, "BEOI", 1.0), NU, rtol=1e-12)


# ---------------------------------------------------------------------------
# Scanner calibration and fit sample
# ---------------------------------------------------------------------------


class TestProbeAndSample:
    def test_calibration_covers_the_atoms(self):
        sys.path.insert(0, str(TOOLS))
        import rs_family_parity_scan as scan

        for fam in FAMILIES:
            cal = scan.calibrate(fam, 8)
            assert cal.ok, fam + " calibration failed on [0,1]"
            assert np.all((cal.y >= 0.0) & (cal.y <= 1.0)), fam + " grid leaves [0,1]"
            assert 0.0 in set(np.round(cal.y, 12)) and 1.0 in set(np.round(cal.y, 12)), (
                fam + " grid misses a boundary atom"
            )

    def test_fit_sample_respects_the_atoms(self):
        sys.path.insert(0, str(TOOLS))
        import rs_family_parity_scan as scan

        rng = np.random.default_rng(20261023)
        for fam in FAMILIES:
            cal = scan.calibrate(fam, 8)
            y = np.asarray(scan._fit_data(fam, rng, cal.y)["y"])
            zero_atom, one_atom = ATOMS[fam]
            assert np.all((y >= 0.0) & (y <= 1.0)), fam + " sample leaves [0,1]"
            if not zero_atom:
                assert np.all(y > 0.0), fam + " sample carries a zero atom it does not have"
            if one_atom and not zero_atom:
                assert np.any(y == 1.0), fam + " sample misses its one atom"
