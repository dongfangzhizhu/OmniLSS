"""SHASH: a trajectory that splits from R's on a flat ridge (D2.67).

The rsdev gap is 136.647357 vs R 136.386600 (rel 1.9e-03) -- ours higher, the
"defect" direction, but the dossier says otherwise:

1. **Cycles 1-16 are bit-identical to live R.**  Every coefficient and the
   global deviance match to < 1e-12, individually, at k = 1 and k = 8 and
   k = 16.  A wrong density, score or Hessian cannot survive sixteen cycles
   of RS on the same data.
2. **The split is a threshold knife-edge.**  In cycle 17 mu is still
   identical; only sigma differs (ours 0.175949 against R's 0.219194 on the
   log link).  Instrumenting rs_step shows the sigma inner fit stopping after
   4 iterations (7 halvings) with a deviance change just under cc = 0.001,
   while R takes one more iteration.
3. **It is not the cc value.**  Tightening the inner tolerance does not
   reproduce R's path: cc = 5e-4 / 1e-4 / 1e-5 give 136.565 / 136.663 /
   136.691 at the same 20-cycle budget.
4. **Both sides converge to the same region.**  At n.cyc = 400 ours is
   131.676186 against R's 131.579879 -- rel 7.3e-04, inside the gate's own
   1e-3, with both sides converged = TRUE.

So the registration records a parity *limit* rather than a defect: closing it
would mean reproducing R's floating-point arithmetic step for step.  The gate
reports warn, not pass, and re-checks the pinned numbers so a real regression
cannot ride in on the note.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest

from omnilss import gamlss

RSCRIPT = shutil.which("Rscript")

# The scanner's full-family-stream SHASH fit sample (n = 48).
SHASH_Y = [
    3.3294101484688783, 2.305054378919365, 1.5458730743142448, 1.1995119234331297, 0.7094217184025652, 3.5304043084177756,
    0.8523100411064295, 3.910925273164421, 2.4247954437972075, 0.656077280174889, 1.7460062878591274, 0.7493547269946769,
    2.789327816491484, 3.0608214291142715, 3.7380114431047526, 1.5867634398004706, 2.2830101935322387, 2.0853446243384877,
    2.5301040204692287, 1.971285467957254, 2.272060148914163, 1.6650607577284862, 0.34170534988103174, 2.0516404478895924,
    1.9029733624469096, 3.5190595784438594, 0.7161712725432886, 3.0257360003229192, 1.20074484776365, 3.1538350109738396,
    2.319086637736592, 0.41508183117046016, 0.31136230149577465, 1.7379958302199903, 0.39497522932721046, 2.1468574290032443,
    1.0031338466981037, 2.931663046931425, 0.5864841383459435, 1.4045070398931483, 2.6915555525803607, 2.3332687933768788,
    0.4385397403347322, 3.070103376084037, 3.5304328209432265, 0.5667148468832977, 2.1548084870470237, 1.8397293965647636,
]
SHASH_X = [
    0.7216208287820969, -1.6790521584969615, 0.7761250196966891, 1.3421505085593513, -1.2889429639958772, 0.8372767920240787,
    -0.4022538949481902, -0.28945158929221904, 0.1788026245686188, -0.07390729375197465, -1.1091136420221708, -0.5640005699752483,
    -0.7586180486110984, -0.8137179284425666, -0.2659917217241748, 1.1415003160248693, -2.417749131009538, -0.7336862285578594,
    1.2014103669659832, -1.5348294726858867, 0.16155073634826453, -0.018218133252049195, -0.8724887762023459, 0.3406988038589152,
    -0.34541787355305686, -0.13532469438826866, -1.004205008939244, -1.1666993883521917, 1.4254281347815392, -0.6131312704425117,
    -0.5882775271881996, -0.7429701144623456, -1.7471371259201516, 0.09170413952016672, 0.28133625697552195, -0.25808503363394175,
    -0.08827064329854915, 0.8758467454543647, -0.2937630098213492, 0.7102659802847253, 0.3393988959480952, -0.11592260946497028,
    0.3867009132890376, 0.8866015408164352, -1.0858325992931688, -0.1282326967187828, -1.324449190232814, 2.082917175612372,
]


RECORDED = dict(dev_ours=136.647357, dev_r=136.386600,
                dev_ours_long=131.676186, dev_r_long=131.579879)


def _run_r(lines, timeout=1800) -> str:
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


def _r_fit_at(k: int) -> dict:
    nl = chr(10)
    lines = [
        "suppressPackageStartupMessages({library(gamlss); library(gamlss.dist); library(jsonlite)})",
        "y <- c(" + ", ".join(repr(v) for v in SHASH_Y) + ")",
        "x <- c(" + ", ".join(repr(v) for v in SHASH_X) + ")",
        "m <- suppressWarnings(gamlss(y ~ x, family = SHASH(), data = data.frame(y = y, x = x),",
        "      sigma.formula = ~1, control = gamlss.control(n.cyc = " + str(k) + ", trace = FALSE)))",
        "cat(toJSON(list(deviance = m$G.deviance, converged = m$converged,",
        '   mu = as.numeric(coef(m, parameter = "mu")),',
        '   sigma = as.numeric(coef(m, parameter = "sigma")),',
        '   nu = as.numeric(coef(m, parameter = "nu")),',
        '   tau = as.numeric(coef(m, parameter = "tau"))),',
        "   auto_unbox = TRUE, digits = 17))",
    ]
    return json.loads(_run_r(lines))


def _ours_at(k: int) -> dict:
    m = gamlss("y ~ x", family="SHASH",
               data={"y": np.asarray(SHASH_Y), "x": np.asarray(SHASH_X)},
               sigma_formula="~1", method="RS", max_iter=k,
               allow_compat_alias=True)
    out = {"deviance": float(m.deviance), "converged": bool(m.converged)}
    for p in m.parameters:
        out[p] = np.ravel(np.asarray(m.coefficients[p], dtype=float)).tolist()
    return out


@pytest.mark.skipif(RSCRIPT is None, reason="Rscript not on PATH")
class TestTrajectoryIdentity:
    @pytest.mark.parametrize("k", [1, 8, 16])
    def test_cycles_match_r_bit_for_bit(self, k):
        """Sixteen cycles of identical RS is the core of the dossier."""
        r = _r_fit_at(k)
        ours = _ours_at(k)
        np.testing.assert_allclose(ours["deviance"], float(r["deviance"]),
                                   rtol=1e-12,
                                   err_msg=f"deviance at k={k}")
        for p in ("mu", "sigma", "nu", "tau"):
            np.testing.assert_allclose(
                np.asarray(ours[p], dtype=float),
                np.ravel(np.asarray(r[p], dtype=float)),
                rtol=1e-12, atol=1e-12,
                err_msg=f"{p} coefficients at k={k}",
            )

    def test_cycle_17_splits_in_sigma_only(self):
        """First divergence: mu still identical, sigma differs."""
        r17 = _r_fit_at(17)
        o17 = _ours_at(17)
        np.testing.assert_allclose(
            np.asarray(o17["mu"], dtype=float),
            np.ravel(np.asarray(r17["mu"], dtype=float)),
            rtol=1e-12, atol=1e-12, err_msg="mu must still match at k=17",
        )
        assert abs(o17["sigma"][0] - float(np.ravel(np.asarray(r17["sigma"]))[0])) > 1e-3, (
            "sigma is expected to carry the split at k=17"
        )


@pytest.mark.skipif(RSCRIPT is None, reason="Rscript not on PATH")
class TestLongBudgetAgreement:
    def test_four_hundred_cycles_agree_inside_the_gate_tolerance(self):
        """The dossier's closing argument: same region once both converge."""
        r400 = _r_fit_at(400)
        o400 = _ours_at(400)
        np.testing.assert_allclose(o400["deviance"], float(r400["deviance"]),
                                   rtol=1e-3,
                                   err_msg="400-cycle deviances must agree")
        np.testing.assert_allclose(o400["deviance"], RECORDED["dev_ours_long"],
                                   rtol=1e-4, err_msg="our long-budget value drifted")
        np.testing.assert_allclose(float(r400["deviance"]), RECORDED["dev_r_long"],
                                   rtol=1e-4, err_msg="R's long-budget value drifted")


class TestRegistration:
    def test_registration_matches_the_measurement(self):
        import sys

        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
        import rs_family_parity_scan as scan

        rec = scan._RSDEV_PATH_DIVERGENT.get("SHASH")
        assert rec is not None
        ours = _ours_at(20)
        np.testing.assert_allclose(ours["deviance"], rec["dev_ours"], rtol=1e-5,
                                   err_msg="recorded 20-cycle deviance drifted")
        assert rec["max_rel"] <= 5e-3
