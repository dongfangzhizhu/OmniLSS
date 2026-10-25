"""GT: R's own fit diverges; ours converges once the initial values are R's (D2.68).

What the scanner found
----------------------
rsdev for GT failed at 1424.36 against R's 153864.96 -- both numbers are
absurd, which is the tell.  GT had no entries in _fitting_init.py at all, so
mu fell through to the generic target=y and sigma to the residual scale
against that same target (RMSE(y, y) = 0 -> the eps floor, measured 2.2e-16).
With sigma = 2.2e-16 the first mu inner fit ran away to 1.2e6 and the cycle-1
deviance was 1521.

Live R's GT() starts at mu = (y+mean(y))/2, sigma = sd(y)/4, nu = 5, tau = 2
(links identity / log / log / log, the same as ours).  With those starts our
fit descends monotonically:

    cycle   1        2        3        4        5        6        ...  20
    ours  139.983  132.934  129.350  126.9   124.699  120.597        116.799
    R     139.796  132.873  129.233  126.913 124.680  120.686  174199 at 7

R is well behaved through cycle 6 and then **its own iteration diverges**:
174198.8 at cycle 7, creeping down to 153865 by cycle 20 with converged =
FALSE, and its nu coefficient frozen at 7.5125 throughout.  There is no R
reference at the gate budget; the best value R ever reached is 120.6856
(cycle 6), and our 116.799 is below it.

The gate therefore reports warn with that comparison (see
_RSDEV_R_DIVERGED in tools/rs_family_parity_scan.py) rather than failing us
against R's blown-up endpoint.
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
from omnilss._fitting_init import (
    _initial_mu_target,
    _initial_parameter_value,
    _initial_sigma,
)
from omnilss.distributions import resolve_family

RSCRIPT = shutil.which("Rscript")

# The scanner's full-family-stream GT fit sample (n = 48).
GT_Y = [
    0.6962475455429877, 3.8067556163502956, 0.4905230069239036, 2.462173599922743, 1.5285030178400343, 1.8856228159612283,
    1.620649626691907, 0.5182902257511874, 3.297350521843577, 2.7146214204165235, 3.981888181805485, 1.461624383394118,
    2.1199189374914704, 3.2226124931371776, 1.3858668712838649, 1.021898920176694, 1.5600956495225862, 1.7524793730670865,
    1.8440377728086927, 1.3248167897173138, 3.113182521948154, 1.9954858475653119, 3.964354025043557, 3.655775276275538,
    1.5840314555191626, 1.684343482672965, 2.9781109280381926, 2.03022997034022, 2.5169735093175545, 3.163621717468683,
    0.8989833375162497, 0.8374300900691964, 3.1667067445411163, 2.826287725092228, 3.597788805735504, 2.4747290214335522,
    0.816527739225011, 2.8255567617236497, 0.6487887336325231, 1.4169532516243522, 2.937664208119153, 2.1252071612979195,
    2.488147032926908, 1.0278585038195436, 1.041769553946108, 0.7663918705989663, 1.7466524876437781, 0.5595628464804439,
]
GT_X = [
    -0.5851289077008408, -0.9871491180867609, -0.39455859507278623, 0.05682991022231527, 0.3552097657044223, -1.541348861305916,
    -0.33461696187122014, -0.1659641932531582, -0.46585911746869846, 0.854305540512301, 0.9224019976709613, 1.8547022092062262,
    -1.6226890941804628, 1.0041395620606715, 0.6298226115380388, -1.2994416388876548, 0.35598943148562817, -1.7013658085255317,
    0.9605180652977198, 1.1234449560365898, -0.06741945959445513, 0.6015991134068748, 1.1139757406008934, -0.2803302809766093,
    0.8821945151279228, -0.9130351436086653, -0.022554376388958176, -0.7198190776526209, -0.42737124737981025, -0.42635378155227455,
    1.2221504941949748, -1.6468197340470794, 0.8966514107090234, 0.45245821566157046, 0.8051795434485424, -1.0814485330567045,
    0.35697404119998793, 1.5040303675779052, -1.775203662283344, -0.8106362036174108, -1.2334922879831682, -0.5307145187636857,
    -0.037360490167269465, 0.027702529152410466, -0.07290220869237732, 0.0018502016231936992, 0.7599884547417661, 0.40856636996158696,
]


RECORDED = dict(dev_ours=116.798974, dev_r=153864.963870,
                best_trace=120.685644, best_trace_cycle=6)


def _run_r(lines, timeout=1200) -> str:
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


def _ours(max_iter: int = 20):
    return gamlss("y ~ x", family="GT",
                  data={"y": np.asarray(GT_Y), "x": np.asarray(GT_X)},
                  sigma_formula="~1", method="RS", max_iter=max_iter,
                  allow_compat_alias=True)


class TestInitials:
    def test_sigma_start_is_not_on_the_floor(self):
        fam = resolve_family("GT")
        y = np.asarray(GT_Y, dtype=float)
        w = np.ones_like(y)
        mu0 = _initial_mu_target(fam, y, w)
        sig0 = float(np.ravel(_initial_sigma(fam, y, mu0, w))[0])
        np.testing.assert_allclose(sig0, float(np.std(y, ddof=1) / 4.0), rtol=1e-12)
        assert sig0 > 1e-3

    def test_mu_target_is_the_half_mean_shift(self):
        fam = resolve_family("GT")
        y = np.asarray(GT_Y, dtype=float)
        w = np.ones_like(y)
        np.testing.assert_allclose(np.asarray(_initial_mu_target(fam, y, w)),
                                   (y + np.mean(y)) / 2.0, rtol=1e-12)

    def test_nu_and_tau_initials(self):
        fam = resolve_family("GT")
        y = np.asarray(GT_Y, dtype=float)
        w = np.ones_like(y)
        mu0 = _initial_mu_target(fam, y, w)
        assert float(np.ravel(_initial_parameter_value(fam, "nu", y, mu0, w))[0]) == 5.0
        assert float(np.ravel(_initial_parameter_value(fam, "tau", y, mu0, w))[0]) == 2.0


class TestOurFit:
    def test_deviance_matches_the_recorded_value(self):
        np.testing.assert_allclose(float(_ours().deviance), RECORDED["dev_ours"],
                                   rtol=1e-5)

    def test_our_fit_is_below_r_best_trace_value(self):
        assert float(_ours().deviance) < RECORDED["best_trace"]

    def test_first_cycle_no_longer_explodes(self):
        """Pre-fix cycle 1 was 1521 with mu = 1.2e6."""
        m = _ours(max_iter=1)
        assert float(m.deviance) < 200.0
        mu0 = float(np.ravel(np.asarray(m.coefficients["mu"], dtype=float))[0])
        assert abs(mu0) < 10.0, f"mu intercept {mu0} exploded on cycle 1"


@pytest.mark.skipif(RSCRIPT is None, reason="Rscript not on PATH")
class TestRDiverges:
    def test_r_trace_collapses_after_cycle_six(self):
        """R's own iteration must show the jump the registration rests on."""
        nl = chr(10)
        lines = [
            "suppressPackageStartupMessages({library(gamlss); library(gamlss.dist); library(jsonlite)})",
            "y <- c(" + ", ".join(repr(v) for v in GT_Y) + ")",
            "x <- c(" + ", ".join(repr(v) for v in GT_X) + ")",
            "tc <- capture.output(m <- suppressWarnings(gamlss(y ~ x, family = GT(),",
            "      data = data.frame(y = y, x = x), sigma.formula = ~1,",
            "      control = gamlss.control(n.cyc = 20, trace = TRUE))))",
            'it <- grep("iteration", tc, value = TRUE)',
            'devs <- as.numeric(sub(".*Global Deviance = ", "", it))',
            "cat(toJSON(list(devs = devs, reported = m$G.deviance, converged = m$converged),",
            "           auto_unbox = TRUE, digits = 17))",
        ]
        probe = json.loads(_run_r(lines))
        devs = [float(v) for v in probe["devs"]]
        assert len(devs) >= 7
        # Well behaved up to cycle 6 ...
        assert max(devs[:6]) < 145.0, f"early cycles should be sane: {devs[:6]}"
        # ... then it explodes.
        assert devs[6] > 1e3, f"R must diverge at cycle 7, got {devs[6]}"
        np.testing.assert_allclose(min(devs), RECORDED["best_trace"], rtol=1e-4)
        assert devs.index(min(devs)) == RECORDED["best_trace_cycle"] - 1
        np.testing.assert_allclose(float(probe["reported"]), RECORDED["dev_r"],
                                   rtol=1e-5)
        assert probe["converged"] is False


class TestRegistration:
    def test_registration_matches_the_measurement(self):
        import sys

        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
        import rs_family_parity_scan as scan

        rec = scan._RSDEV_R_DIVERGED.get("GT")
        assert rec is not None
        assert rec["dev_ours"] == pytest.approx(RECORDED["dev_ours"], rel=1e-5)
        assert rec["best_trace"] == pytest.approx(RECORDED["best_trace"], rel=1e-5)
        assert rec["reported"] == pytest.approx(RECORDED["dev_r"], rel=1e-5)
