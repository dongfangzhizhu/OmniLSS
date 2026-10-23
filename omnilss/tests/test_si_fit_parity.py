"""SI fit parity and the Bessel-kernel fix behind it (D2.53).

What the scanner found
----------------------
rsdev for SI failed at deviance 221.609 against live R's 220.619
(rel 4.49e-03, ours *above* R -- the defect direction).  Two independent
defects produced it:

1. **Missing initial values** (the EXGAUS/NBII signature again): SI had no
   entries in _fitting_init.py at all, so mu fell through to the generic
   target=y (which the log link consumes as exp(y)), sigma evaluated the
   residual scale against that very target (RMSE(y, y) = 0 -> the eps
   floor, measured 2.2e-16), and nu took the generic +1.0 while R starts
   at -0.5.

2. **A broken Bessel kernel** (found while validating the fix): the Sichel
   log-density normalises by log K_nu(alpha) - log K_nu(1/sigma) with
   alpha ~ 1/sigma, and _log_bessel_kv_scipy clamped the raw kv value to
   eps "for safety".  But K_nu(x) drops below double eps for x >~ 35
   (K_2(50) = 3.4e-23), so every large-argument log(K) silently became
   log(eps) = -36.04 and the normaliser difference collapsed to 0.
   Measured consequence: SI's density came out 1.04e+04 times too large
   at mu=10, sigma=0.02, nu=2 (and 1.65x at mu=0.5).  The same kernel
   feeds SICHEL / ZISICHEL / ZASICHEL / PIG / GIG / ZIPIG / ZAPIG /
   ZANBI, so the clamp was a latent defect for all of them -- invisible
   to the dpqr gates only because their calibrated probe grids stay in
   the small-argument region.

The fix: kve(v, x) = K_v(x) * exp(x) does not underflow, so
log(K) = log(kve) - x is exact across the whole range (verified here
against live R's log(besselK) point by point, including x = 745 where
raw kv is exactly 0).

Fit parity after the fix: at the scanner's 20-cycle budget ours sits at
220.6047 against R's 220.6189 (rel 6.4e-05).  Ours is *lower*; the
two-step adjudication for that direction is executed below -- R's own
dSI evaluated at our fitted per-observation parameters reproduces our
deviance to 2e-11 and shows our point is the better one (220.4414 vs
220.6189 at a 200-cycle budget): R's dldv slot is a *forward* difference
(numeric.deriv(..., delta = 1e-4)) that cannot follow the flat nu-ridge,
so its RS stops at an inferior stationary point -- the PE2 disposition.
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
from omnilss.bessel import log_bessel_kv
from omnilss.distributions import resolve_family

FAMILY = "SI"
RSCRIPT = shutil.which("Rscript")

# The scanner's full-family-stream SI fit sample (n = 48, counts 0..7).
FIT_Y = np.array([
    0.0, 6.0, 1.0, 7.0, 3.0, 1.0, 7.0, 7.0, 6.0, 3.0, 7.0, 1.0, 2.0, 6.0,
    7.0, 6.0, 0.0, 4.0, 4.0, 7.0, 2.0, 3.0, 1.0, 6.0, 5.0, 5.0, 7.0, 7.0,
    5.0, 5.0, 5.0, 1.0, 3.0, 2.0, 6.0, 1.0, 4.0, 5.0, 0.0, 6.0, 4.0, 0.0,
    3.0, 5.0, 4.0, 3.0, 6.0, 6.0,
])
FIT_X = np.array([
    0.6711337826963099, -0.5537664058013549, -0.445278495899371, 0.05947737594238893,
    1.1327175642149419, 0.84955700747031, -1.7578964723212798, -0.44658952818060016,
    -0.4273474232330522, 0.5595151474263224, 0.16714980112504113, 1.7362678851287001,
    -2.5342647152378124, 1.2306743199175398, -0.02361827373262674, 0.6887392823616041,
    0.7628754082442548, 0.8836752160468575, -1.0277104845392668, -0.5887107112257796,
    1.1372456872004322, -1.917047085373188, 0.511618850530522, 0.22654037292317877,
    -0.3912297361800437, -0.8698294861976061, -0.12893826316050347, 0.15851252658257878,
    0.530674615866927, 2.276584645851399, -0.9856244481958137, -0.06228229194106559,
    0.2988971498516023, -0.949361474679258, -0.8417533900708964, 0.44933112281544235,
    1.407509709572772, -0.5728833454811738, -1.094068758347576, 0.608238975073419,
    1.4922618211561776, -0.4432454528719345, 0.48874416688367406, 0.06331736995369232,
    -1.1760082786443282, -1.504684125442814, 0.3083153206397264, 0.5459703155124568,
])

# Bessel probe grid: orders and arguments.  The x >= 34 block is where the
# old eps-clamp destroyed the value (K_2(35) = 6.4e-17 < eps); x = 745 is
# where raw kv underflows to exactly 0.
NU_GRID = [-2.0, -0.5, 0.0, 0.5, 1.0, 2.0, 5.0]
X_GRID = [1e-3, 0.1, 1.0, 5.0, 20.0, 34.0, 35.0, 40.0, 50.0, 100.0, 300.0, 700.0, 745.0]


def _r_vector(values) -> str:
    return "c(" + ", ".join(repr(float(v)) for v in values) + ")"


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


def _r_probe_script() -> str:
    """R program: initial expressions, R's log(besselK) grid, R's dSI on the
    wide density grid, and the reference fit."""
    x_vec = _r_vector(X_GRID)
    # NOTE the argument order: R's besselK is (x, nu) while scipy's kv is
    # (nu, x) -- getting this backwards silently compares against the
    # transposed family.  R's log(besselK) underflows to -Inf at x = 745,
    # which jsonlite cannot serialise, so non-finite entries become NA
    # (rendered as null) and the test asserts our finiteness there.
    bessel_rows = [
        f"  ifelse(is.finite(v <- sapply({x_vec}, function(xx) log(besselK(xx, {nu!r})))),"
        f" v, NA_real_)"
        for nu in NU_GRID
    ]
    bessel_block = "list(" + chr(10) + ("," + chr(10)).join(bessel_rows) + chr(10) + ")"

    # Wide density grid (the profiles that exposed the clamp).
    profiles = [(mu, sigma, nu) for mu in (0.5, 2.0, 10.0)
                for sigma in (0.02, 0.2, 1.0) for nu in (-0.5, 0.5, 2.0)]
    dens_rows = []
    for mu, sigma, nu in profiles:
        yv = _r_vector(range(0, 31))
        dens_rows.append(
            f"  list(mu={mu}, sigma={sigma}, nu={nu}, y={yv}, "
            f"d=dSI({yv}, mu={mu}, sigma={sigma}, nu={nu}))"
        )
    dens_block = "list(" + chr(10) + ("," + chr(10)).join(dens_rows) + chr(10) + ")"

    nl = chr(10)
    return (
        "suppressPackageStartupMessages({library(gamlss); library(gamlss.dist); library(jsonlite)})" + nl +
        "out <- list()" + nl +
        "fam <- SI()" + nl +
        "y <- " + _r_vector(FIT_Y) + nl +
        "x <- " + _r_vector(FIT_X) + nl +
        "out$initial <- list(" + nl +
        "  mu = eval(fam$mu.initial)," + nl +
        "  sigma = eval(fam$sigma.initial)," + nl +
        "  nu = eval(fam$nu.initial)" + nl +
        ")" + nl +
        "out$bessel <- " + bessel_block + nl +
        "out$density <- " + dens_block + nl +
        "m <- gamlss(y ~ x, family = SI(), sigma.formula = ~1, nu.formula = ~1," + nl +
        "            control = gamlss.control(n.cyc = 20, trace = FALSE))" + nl +
        "out$fit <- list(" + nl +
        "  deviance = m$G.deviance," + nl +
        '  mu_coef = as.numeric(coef(m, parameter = "mu")),' + nl +
        '  sigma_coef = as.numeric(coef(m, parameter = "sigma")),' + nl +
        '  nu_coef = as.numeric(coef(m, parameter = "nu")),' + nl +
        "  converged = m$converged" + nl +
        ")" + nl +
        "cat(toJSON(out, auto_unbox = TRUE, digits = 17, na = \"null\"))"
    )


@pytest.fixture(scope="module")
def r_probe():
    if RSCRIPT is None:  # pragma: no cover
        pytest.skip("Rscript not on PATH")
    stdout = _run_r(_r_probe_script().splitlines())
    return json.loads(stdout)


# ---------------------------------------------------------------------------
# The Bessel kernel (the D2.53 core fix)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(RSCRIPT is None, reason="Rscript not on PATH")
class TestBesselKernelAgainstLiveR:
    def test_log_k_matches_r_besselK_everywhere(self, r_probe):
        """log(K_nu(x)) == log(besselK(nu, x)) on live R, point by point.

        The x >= 34 block is the regression guard: the old clamp turned
        every one of those into log(eps) = -36.04.
        """
        import jax.numpy as jnp

        for nu, r_row in zip(NU_GRID, r_probe["bessel"]):
            for x, r_val in zip(X_GRID, r_row):
                ours = float(jnp.asarray(log_bessel_kv(jnp.asarray(nu), jnp.asarray(x))))
                if r_val is None:
                    # R's own log(besselK) underflowed to -Inf here; the
                    # pre-fix clamp answered log(eps) = -36.04, so the guard
                    # is that ours stays finite and in the far tail.
                    assert np.isfinite(ours) and ours < -700.0, (
                        f"log K_{nu}({x}) = {ours} in R's underflow region"
                    )
                    continue
                np.testing.assert_allclose(
                    ours, float(r_val), rtol=1e-12, atol=1e-9,
                    err_msg=f"log K_{nu}({x}) vs live R",
                )


# ---------------------------------------------------------------------------
# Density parity on the wide grid (the case that exposed the clamp)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(RSCRIPT is None, reason="Rscript not on PATH")
class TestDensityAgainstLiveR:
    def test_wide_grid_matches_r_dsi(self, r_probe):
        """27 profiles x 31 counts; the pre-fix worst was rel 1.04e+04."""
        fam = resolve_family(FAMILY)
        for row in r_probe["density"]:
            ours = np.asarray(fam.d(row["y"], row["mu"], row["sigma"], row["nu"]))
            np.testing.assert_allclose(
                ours, np.asarray(row["d"]), rtol=1e-11, atol=1e-300,
                err_msg=f"dSI at mu={row['mu']}, sigma={row['sigma']}, nu={row['nu']}",
            )

    def test_density_sums_to_one(self):
        """A normalisation guard independent of R: the mass must sum to 1.

        Summed over 0..511 because the kernel's masked scan runs exactly
        512 steps (a documented performance trade-off): beyond that the
        density repeats the y = 512 value, which would accumulate a
        ~1e-9 artifact over a 0..600 sum.  The pre-fix clamp produced sums
        of 1.65 .. 1.04e+04, so 1e-8 sensitivity catches that class of
        defect with four orders of margin.
        """
        fam = resolve_family(FAMILY)
        for mu in (0.5, 2.0, 10.0):
            for sigma in (0.02, 0.2, 1.0):
                for nu in (-0.5, 0.5, 2.0):
                    y = np.arange(0, 512, dtype=float)
                    mass = np.asarray(fam.d(y, mu, sigma, nu))
                    np.testing.assert_allclose(
                        mass.sum(), 1.0, rtol=1e-8, atol=1e-8,
                        err_msg=f"mass sum at mu={mu}, sigma={sigma}, nu={nu}",
                    )


# ---------------------------------------------------------------------------
# Initial values (live-R expressions) and the sigma floor guard
# ---------------------------------------------------------------------------


@pytest.mark.skipif(RSCRIPT is None, reason="Rscript not on PATH")
class TestInitials:
    def test_initial_values_match_live_r(self, r_probe):
        fam = resolve_family(FAMILY)
        w = np.ones_like(FIT_Y)
        mu0 = _initial_mu_target(fam, FIT_Y, w)
        sig0 = _initial_sigma(fam, FIT_Y, mu0, w)
        nu0 = _initial_parameter_value(fam, "nu", FIT_Y, mu0, w)
        # R: mu <- y + 0.5 (log link -> link scale is log(y+0.5)),
        #     sigma <- max((var(y)-mean(y))/mean(y)^2, 0.1), nu <- -0.5.
        np.testing.assert_allclose(
            mu0, np.log(FIT_Y + 0.5), rtol=1e-12,
            err_msg="mu.initial must be log(y + 0.5)",
        )
        var_y = np.var(FIT_Y, ddof=1)
        np.testing.assert_allclose(
            float(np.ravel(sig0)[0]),
            max((var_y - np.mean(FIT_Y)) / np.mean(FIT_Y) ** 2, 0.1),
            rtol=1e-12, err_msg="sigma.initial",
        )
        assert nu0 == -0.5

    def test_initial_expressions_match_r_evaluation(self, r_probe):
        """R evaluated its own expressions on this sample; compare directly."""
        fam = resolve_family(FAMILY)
        w = np.ones_like(FIT_Y)
        mu0 = _initial_mu_target(fam, FIT_Y, w)
        sig0 = _initial_sigma(fam, FIT_Y, mu0, w)
        nu0 = _initial_parameter_value(fam, "nu", FIT_Y, mu0, w)
        r_init = r_probe["initial"]
        # R's eval gives the *response*-scale expression (y + 0.5); our
        # target is the link scale, so it is R's value through the log link.
        np.testing.assert_allclose(mu0, np.log(np.asarray(r_init["mu"])),
                                   rtol=1e-12, err_msg="mu.initial vs live R")
        np.testing.assert_allclose(sig0, r_init["sigma"][0], rtol=1e-12,
                                   err_msg="sigma.initial vs live R")
        np.testing.assert_allclose(nu0, r_init["nu"][0], rtol=1e-12,
                                   err_msg="nu.initial vs live R")

    def test_sigma_start_is_not_on_the_floor(self):
        """The pre-fix failure signature: sigma started at 2.2e-16."""
        fam = resolve_family(FAMILY)
        w = np.ones_like(FIT_Y)
        mu0 = _initial_mu_target(fam, FIT_Y, w)
        sig0 = float(np.ravel(_initial_sigma(fam, FIT_Y, mu0, w))[0])
        assert sig0 > 1e-3, f"sigma start {sig0} sits at the generic floor"


# ---------------------------------------------------------------------------
# End-to-end fit parity
# ---------------------------------------------------------------------------


@pytest.mark.skipif(RSCRIPT is None, reason="Rscript not on PATH")
class TestFitParity:
    def test_rs_fit_matches_r_at_scanner_budget(self, r_probe):
        """n_cyc = 20 on both sides: ours 220.6047 vs R 220.6189.

        rel 6.4e-05, inside the 1e-3 gate.  The pre-fix fit reported
        221.609 (ours *above* R -- the defect direction).
        """
        r = r_probe["fit"]
        py = gamlss("y ~ x", family=FAMILY, data={"y": FIT_Y, "x": FIT_X},
                    method="RS", max_iter=20)
        np.testing.assert_allclose(
            float(py.deviance), float(r["deviance"]), rtol=1e-3, atol=1e-3,
            err_msg=f"deviance {py.deviance} vs R {r['deviance']}",
        )

    def test_our_point_is_genuine_on_r_density(self):
        """The 'ours is lower' adjudication, step 2.

        Run a longer budget, then evaluate R's own dSI at our fitted
        per-observation parameters: it must reproduce our deviance.  At a
        200-cycle budget ours sits at 220.4414 against R's converged
        220.6189 -- R's own density confirms our point is the better one,
        so the residual is R's forward-difference dldv limiting its
        iteration (the PE2 disposition).
        """
        py = gamlss("y ~ x", family=FAMILY, data={"y": FIT_Y, "x": FIT_X},
                    method="RS", max_iter=200)
        mu_coef = np.ravel(np.asarray(py.coefficients["mu"], dtype=float))
        sigma_coef = float(np.ravel(np.asarray(py.coefficients["sigma"], dtype=float))[0])
        nu_coef = float(np.ravel(np.asarray(py.coefficients["nu"], dtype=float))[0])
        nl = chr(10)
        # NOTE: repr() on a numpy 2.x scalar renders np.float64(...), which R
        # cannot parse -- the D2.48 lesson; go through float() first.
        script = (
            "suppressPackageStartupMessages({library(gamlss.dist)})" + nl +
            "y <- " + _r_vector(FIT_Y) + nl +
            "x <- " + _r_vector(FIT_X) + nl +
            "mu <- exp(" + repr(float(mu_coef[0])) + " + " + repr(float(mu_coef[1])) + " * x)" + nl +
            "ll <- sum(dSI(y, mu = mu, sigma = " + repr(float(np.exp(sigma_coef))) + "," + nl +
            "              nu = " + repr(float(nu_coef)) + ", log = TRUE))" + nl +
            "cat(format(-2 * ll, digits = 17))"
        )
        out = _run_r(script.splitlines())
        np.testing.assert_allclose(
            float(out.strip()), float(py.deviance), rtol=1e-10, atol=1e-8,
            err_msg="R's own dSI at our point must reproduce our deviance",
        )
