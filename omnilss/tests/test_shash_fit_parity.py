"""SHASH / SHASHo: R's closed-form scores and -score^2 Hessians (D2.66).

What the scanner found
----------------------
Both families sat just above R on rsdev -- SHASH 136.687 vs 136.387 (2.2e-03)
and SHASHo 128.778 vs 128.618 (1.2e-03) -- the "ours is higher, so it is a
defect" direction, but far too small for a wrong density (the dpqr gates
pass).  R's SHASH()/SHASHo() carry the full derivative slots, and every
d2ldX2 there is -dldX^2 floored at -1e-15.  Our AD path feeds the *true*
second derivative, so the IRLS weights differ from R's from the first inner
step (the JSUO/ST5 signature).

Swapping only the Hessian moved SHASH 136.687 -> 136.647 and SHASHo
128.778 -> 128.683; transcribing the scores as well (so the family never
touches the AD path) reproduces exactly the same two numbers -- the scores
were already the true derivatives and the gate's deriv check still passes,
which is the expected outcome and pins the transcription.

Fit parity after the fix (full-family-stream sample, n_cyc = 20):
  SHASHo  128.682584 vs R 128.618200   rel 5.0e-04  -> gate pass
  SHASH   136.647357 vs R 136.386600   rel 1.9e-03  -> still fail

SHASH's residual is NOT the Hessian form: R is still descending at the gate
budget (per-cycle 151.31 -> 149.41 -> 147.77 -> ... -> 136.39, converged =
FALSE) and at n.cyc = 200 goes to 131.6735, i.e. the likelihood has a long
downhill path that our iteration walks more slowly.  The registration-free
record of that is below; the family stays a genuine rsdev fail.
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
from omnilss.distributions import resolve_family

RSCRIPT = shutil.which("Rscript")
R_NAMES = {"SHASH": "SHASH", "SHASHO": "SHASHo"}

# The scanner's full-family-stream fit samples (n = 48).
SAMPLES = {
    "SHASH": dict(
        y=[
            3.3294101484688783, 2.305054378919365, 1.5458730743142448, 1.1995119234331297, 0.7094217184025652, 3.5304043084177756,
            0.8523100411064295, 3.910925273164421, 2.4247954437972075, 0.656077280174889, 1.7460062878591274, 0.7493547269946769,
            2.789327816491484, 3.0608214291142715, 3.7380114431047526, 1.5867634398004706, 2.2830101935322387, 2.0853446243384877,
            2.5301040204692287, 1.971285467957254, 2.272060148914163, 1.6650607577284862, 0.34170534988103174, 2.0516404478895924,
            1.9029733624469096, 3.5190595784438594, 0.7161712725432886, 3.0257360003229192, 1.20074484776365, 3.1538350109738396,
            2.319086637736592, 0.41508183117046016, 0.31136230149577465, 1.7379958302199903, 0.39497522932721046, 2.1468574290032443,
            1.0031338466981037, 2.931663046931425, 0.5864841383459435, 1.4045070398931483, 2.6915555525803607, 2.3332687933768788,
            0.4385397403347322, 3.070103376084037, 3.5304328209432265, 0.5667148468832977, 2.1548084870470237, 1.8397293965647636,
        ],
        x=[
            0.7216208287820969, -1.6790521584969615, 0.7761250196966891, 1.3421505085593513, -1.2889429639958772, 0.8372767920240787,
            -0.4022538949481902, -0.28945158929221904, 0.1788026245686188, -0.07390729375197465, -1.1091136420221708, -0.5640005699752483,
            -0.7586180486110984, -0.8137179284425666, -0.2659917217241748, 1.1415003160248693, -2.417749131009538, -0.7336862285578594,
            1.2014103669659832, -1.5348294726858867, 0.16155073634826453, -0.018218133252049195, -0.8724887762023459, 0.3406988038589152,
            -0.34541787355305686, -0.13532469438826866, -1.004205008939244, -1.1666993883521917, 1.4254281347815392, -0.6131312704425117,
            -0.5882775271881996, -0.7429701144623456, -1.7471371259201516, 0.09170413952016672, 0.28133625697552195, -0.25808503363394175,
            -0.08827064329854915, 0.8758467454543647, -0.2937630098213492, 0.7102659802847253, 0.3393988959480952, -0.11592260946497028,
            0.3867009132890376, 0.8866015408164352, -1.0858325992931688, -0.1282326967187828, -1.324449190232814, 2.082917175612372,
        ],
    ),
    "SHASHO": dict(
        y=[
            1.1879442493205026, 1.2439748964951083, 1.022691237574978, 1.0581433756049878, 3.3107012134978038, 1.2381548701170386,
            3.0564851372022535, 2.847108582237461, 3.226748218133636, 3.4360920792786467, 3.737195608908902, 0.989021930130811,
            2.6992498757824266, 1.547395284210197, 1.6669415881188545, 1.6280372079798027, 2.201274398619694, 1.7423564444415043,
            1.5903277528484594, 2.633404300525317, 3.5820252971978044, 2.5553803320274566, 1.3104352456814776, 0.8308396867441332,
            1.6397426680720646, 2.8660898486749122, 0.919027404591021, 2.345645216585062, 2.7696284434011487, 3.619288232401345,
            1.7267343524096497, 0.48169367463770535, 3.9236470785217348, 1.6857447052974037, 3.3787819762129514, 3.2013848883402614,
            2.604414250037063, 3.525205204418984, 3.315921517463741, 1.8471029618145711, 1.7236964715308334, 1.9487118529920846,
            2.574398277282837, 2.4200199811924175, 3.463922986016864, 1.990695156357013, 0.4453849917383514, 3.42804722005897,
        ],
        x=[
            1.4359717298659187, 1.0025844585949528, -0.27944474685323617, 0.27884926415060324, 0.7810935814483907, -0.012581406962648471,
            -0.285746184652002, -1.1222428305504377, -0.823974563635071, -0.9525053206292072, -1.1775936917827217, 1.0534757694899024,
            0.1248238683160164, 1.9780031112148047, 0.8893301125047225, -1.059943933590276, -0.6646603825682086, -0.5946294355514793,
            -0.5547126021610046, -1.1339059306215442, 0.5613363355653757, 0.21039593712929938, 0.7742436786534503, -0.5906502668805143,
            -0.7532461962524047, -0.7873218921002777, -0.046598306716911425, 1.4971148266350445, 0.045182569254499476, 0.2598608255550236,
            0.01609717041907952, 1.0785324277698818, -0.25818409004180737, 0.9241784659760168, 0.5861801670998185, 1.802170071321472,
            0.10038617805587528, 2.5178112518045355, -1.6843349697768946, 0.1355583334611963, -0.36568341772999985, -0.20804974112865934,
            -0.3425468796119151, -0.1301960233456073, -0.5980662428874226, -0.4445438521744074, 0.2455141527266873, -0.24665330194288465,
        ],
    ),
}


PROBE_GRID = [
    (0.5, 2.0, 1.0, 0.5, 0.5),
    (2.5, 1.0, 0.8, -0.5, 1.0),
    (-1.0, 0.0, 1.0, 1.0, 2.0),
    (3.0, 2.0, 0.5, 0.2, 0.8),
]

_FAMILY_CACHE: dict[str, object] = {}


def _family(name: str):
    if name not in _FAMILY_CACHE:
        _FAMILY_CACHE[name] = resolve_family(name)
    return _FAMILY_CACHE[name]


def _run_r(lines, timeout=900) -> str:
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


def _r_scores(name: str) -> list[dict]:
    nl = chr(10)
    rname = R_NAMES[name]
    rows = []
    for y, mu, sigma, nu, tau in PROBE_GRID:
        args = f"y={y!r}, mu={mu!r}, sigma={sigma!r}, nu={nu!r}, tau={tau!r}"
        rows.append(
            f"  list(y={y!r}, mu={mu!r}, sigma={sigma!r}, nu={nu!r}, tau={tau!r},"
            f" dldm={rname}()$dldm({args}), dldd={rname}()$dldd({args}),"
            f" dldv={rname}()$dldv({args}), dldt={rname}()$dldt({args}))"
        )
    lines = [
        "suppressPackageStartupMessages({library(gamlss.dist); library(jsonlite)})",
        "cat(toJSON(list(" + nl + ("," + nl).join(rows) + nl + "), auto_unbox = TRUE, digits = 17))",
    ]
    return json.loads(_run_r(lines))


def _r_fit(name: str) -> dict:
    s = SAMPLES[name]
    nl = chr(10)
    lines = [
        "suppressPackageStartupMessages({library(gamlss); library(gamlss.dist); library(jsonlite)})",
        "y <- c(" + ", ".join(repr(v) for v in s["y"]) + ")",
        "x <- c(" + ", ".join(repr(v) for v in s["x"]) + ")",
        "m <- suppressWarnings(gamlss(y ~ x, family = " + R_NAMES[name] + "(), data = data.frame(y = y, x = x),",
        "      sigma.formula = ~1, control = gamlss.control(n.cyc = 20, trace = FALSE)))",
        "cat(toJSON(list(deviance = m$G.deviance, converged = m$converged), auto_unbox = TRUE, digits = 17))",
    ]
    return json.loads(_run_r(lines))


@pytest.mark.skipif(RSCRIPT is None, reason="Rscript not on PATH")
@pytest.mark.parametrize("name", ["SHASH", "SHASHO"])
class TestDerivativesAgainstLiveR:
    def test_scores_match_r_slots(self, name):
        fam = _family(name)
        for row in _r_scores(name):
            args = (row["y"], row["mu"], row["sigma"], row["nu"], row["tau"])
            for pname, slot in (("mu", "dldm"), ("sigma", "dldd"),
                                ("nu", "dldv"), ("tau", "dldt")):
                ours = float(np.ravel(np.asarray(
                    fam.score_functions[pname](*args), dtype=float))[0])
                np.testing.assert_allclose(
                    ours, float(row[slot]), rtol=1e-9, atol=1e-11,
                    err_msg=f"{name} {pname} score vs R's {slot}",
                )

    def test_hessians_are_minus_score_squared(self, name):
        fam = _family(name)
        for y, mu, sigma, nu, tau in PROBE_GRID:
            args = (y, mu, sigma, nu, tau)
            for pname in fam.parameters:
                d = float(np.ravel(np.asarray(
                    fam.score_functions[pname](*args), dtype=float))[0])
                h = float(np.ravel(np.asarray(
                    fam.hessian_functions[pname](*args), dtype=float))[0])
                # R: ifelse(d2 < -1e-15, d2, -1e-15) -- keep the value when it is
                # *more* negative than the floor, otherwise floor it.
                expected = -d * d if (-d * d) < -1e-15 else -1e-15
                np.testing.assert_allclose(h, expected, rtol=1e-12, atol=1e-15,
                                           err_msg=f"{name} {pname} hessian form")

    def test_scores_are_the_true_derivatives(self, name):
        """Self-check independent of R: central difference of the density."""
        fam = _family(name)
        h = 1e-6
        for y, mu, sigma, nu, tau in PROBE_GRID:
            for k, pname in enumerate(fam.parameters, start=1):
                base = [y, mu, sigma, nu, tau]
                plus, minus = list(base), list(base)
                plus[k] += h
                minus[k] -= h
                lp = float(np.ravel(np.asarray(fam.d(*plus, log=True), dtype=float))[0])
                lm = float(np.ravel(np.asarray(fam.d(*minus, log=True), dtype=float))[0])
                analytic = float(np.ravel(np.asarray(
                    fam.score_functions[pname](*base), dtype=float))[0])
                np.testing.assert_allclose(
                    analytic, (lp - lm) / (2 * h), rtol=1e-6, atol=1e-8,
                    err_msg=f"{name} {pname} score vs central difference",
                )


@pytest.mark.skipif(RSCRIPT is None, reason="Rscript not on PATH")
class TestFitParity:
    def test_shasho_matches_r_at_scanner_budget(self):
        """Ours 128.682584 vs R 128.618200 -- rel 5.0e-04, inside 1e-3."""
        r = _r_fit("SHASHO")
        s = SAMPLES["SHASHO"]
        py = gamlss("y ~ x", family="SHASHO",
                    data={"y": np.asarray(s["y"]), "x": np.asarray(s["x"])},
                    method="RS", max_iter=20, allow_compat_alias=True)
        np.testing.assert_allclose(float(py.deviance), float(r["deviance"]),
                                   rtol=1e-3, atol=1e-3,
                                   err_msg="SHASHo deviance vs live R")

    def test_shash_residual_is_a_descent_rate_gap(self):
        """SHASH still fails, and the reason is documented rather than hidden.

        R is still descending at the gate budget (136.3866 at cycle 20,
        131.6735 at cycle 200, converged = FALSE both times); our iteration
        walks the same descent more slowly and stalls at 136.647357.  R's own
        density at our coefficients gives 136.6854, i.e. the densities agree
        to 2.8e-04 and the gap is the iteration, not the likelihood.
        """
        s = SAMPLES["SHASH"]
        py = gamlss("y ~ x", family="SHASH",
                    data={"y": np.asarray(s["y"]), "x": np.asarray(s["x"])},
                    method="RS", max_iter=20, allow_compat_alias=True)
        dev = float(py.deviance)
        np.testing.assert_allclose(dev, 136.647357, rtol=1e-5,
                                   err_msg="SHASH deviance drifted")
        assert dev > 136.3866, (
            "SHASH is expected to remain above R's 20-cycle value; if this "
            "ever passes, delete the note in the D2.66 log"
        )
