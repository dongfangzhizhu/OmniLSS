"""LOGSHASH parity against live R via the SHASH transform (D2.52).

Why a transform and not a direct diff
-------------------------------------
The family is defined in the gamlss.dist *development* source
(R/gamlss.dist/R/LOGSHASH.R, Mikis 15-07-2025) and is **not** in the
installed gamlss.dist 6.1.1 -- verified live:
"LOGSHASH" %in% getNamespaceExports("gamlss.dist") is FALSE.  There is
therefore no live R function to diff against directly (the pre-existing
test_r_consistency_logshash.py SUBFAILED for exactly this reason and is
rewritten here).

R nevertheless defines LOGSHASH *literally* as the log-transform of SHASH,
and SHASH **is** installed, so every contract is checkable against live R
at the transformed arguments:

    dLOGSHASH(x) = dSHASH(log(x)) / x        (Jacobian of the log map)
    pLOGSHASH(q) = pSHASH(log(q))
    qLOGSHASH(p) = exp(qSHASH(p))

and the family object's ten score slots are SHASH's own slots at log(y).
The log-likelihood therefore differs from a SHASH fit on log(y) only by
the parameter-free constant sum(log(y)), which makes the end-to-end fit
checkable too:

    G.dev_LOGSHASH(y) = G.dev_SHASH(log(y)) + 2 * sum(log(y))

What the old implementation got wrong
-------------------------------------
* the log-density used a different (Jones-Pewsey style)
  sinh((asinh(z)-nu)/tau) parameterisation with no log(c) term --
  self-consistent enough to pass the AD deriv gate, but not R's
  distribution;
* the links were mu=log, nu=identity (R: mu=identity, nu=log);
* there were no d/p/q/r at all (nan placeholders -- the family was
  PLACEHOLDER status, rejected by production routes);
* there were no initial-value entries.

R boundary quirks pinned here (the support assignments come *after* the
log/tail transforms in R's source, so the logged value is 0, not -inf):

    dLOGSHASH: ft[x <= 0] <- 0                       (log mode gives 0)
    pLOGSHASH: cdf[q <= 0] <- 0 ; cdf[q == Inf] <- 1 (log.p gives 0)
    qLOGSHASH: q[p == 0] <- 0 ; q[p == 1] <- Inf ; p outside [0,1] -> NaN

Validation: R's d/p/q stop on sigma < 0, tau < 0 or nu < 0; we return NaN
(the dexGAUS convention).  The family object declares nu.valid = TRUE
-- R's own inconsistency, reproduced as-is.

Initial-value provenance: the family object does not exist in the installed
R, so the four start expressions are transcribed from the vendored
development source (recorded with file/line below) rather than deparsed
from a live object.  Everything else in this file is live-R verified.
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
from omnilss.dpqr_batch_f import dLOGSHASH, pLOGSHASH, qLOGSHASH, rLOGSHASH

FAMILY = "LOGSHASH"
RSCRIPT = shutil.which("Rscript")

#: (mu, sigma, nu, tau) profiles.  nu > 0 everywhere: R's dSHASH/dLOGSHASH
#: stop on nu < 0, and the scanner's D2.28 discipline puts probe parameters
#: inside their own legal domain.  Spans negative/positive mu (the log-scale
#: location), small/large sigma, small/large nu and tau.  See the exclusion
#: note below for the one profile R itself cannot answer.
# Deliberately excluded: (mu=1.2, sigma=0.8, nu=0.05, tau=2.0) -- live R's
# own qSHASH does not return on it (its uniroot interval-expansion while-loop
# does not terminate), so it is outside the practical domain for BOTH sides.
PROFILES = [
    dict(mu=0.0, sigma=1.0, nu=0.5, tau=0.5),
    dict(mu=0.3, sigma=0.5, nu=0.2, tau=1.5),
    dict(mu=-0.7, sigma=1.5, nu=1.0, tau=0.3),
    dict(mu=-1.5, sigma=0.4, nu=2.0, tau=0.8),
    dict(mu=0.8, sigma=2.0, nu=0.8, tau=0.6),
]

#: Positive response grid: near the support edge, interior, deep tail.
X_GRID = np.array([1e-3, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 20.0, 1e3])
P_GRID = np.array([0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99])

#: One embedded digits=17 sample (numpy default_rng(20261022):
#: x <- normal(60); u <- 0.5 + 0.3*x + 0.4*normal(60); y <- exp(u)).
FIT_Y = np.array([
    0.8918361167405889, 2.2723941697523613, 1.237383038338046, 1.4764112168349768, 3.607533983965031, 2.932022728856379,
    1.6936170655184513, 0.7458867885194218, 1.5407435409704409, 3.495750689001439, 0.8396747934845571, 1.4224749164576367,
    0.7619067004873336, 1.9966772799801815, 1.503460078323217, 0.6475842556395537, 1.0775351827724362, 1.9144991717853752,
    2.5762703815426304, 1.805405150214095, 1.123392385915151, 5.95699174534575, 1.0681981100158382, 1.5984558048237394,
    0.7807070854814053, 0.8219335637059323, 2.8267137486556906, 1.6583572599377172, 1.6617984935474546, 1.6216211928275683,
    2.5113200105503175, 2.602180016081026, 1.40150993026968, 4.013404164458629, 1.9831345207044917, 1.3889506607990783,
    1.295567387377681, 1.7833947047359069, 2.3377105839155377, 1.8867175867077624, 0.9060033646899261, 1.4211721948657647,
    2.7472551263318388, 0.7617899377778258, 1.1915527459064887, 1.1386893771383297, 2.466838083698327, 4.038219704579829,
    1.3496980089292108, 1.596075153412522, 1.274019641689378, 2.449292314798616, 1.1053186029134099, 1.1926259931414551,
    0.7914753039861553, 4.161851583921653, 2.3704430080853376, 0.6587966659684447, 2.768434324233642, 0.6735665555053004,
])
FIT_X = np.array([
    -0.22328003266808794, -0.03210875738847211, -0.5467463997451933, -0.1919413039394447, 0.30213226383465386, 0.37473299666173326,
    1.088839120942718, -1.1732035470647537, 0.32294808613096276, 1.3183111975830026, -0.5519890165543019, -0.8228698426434058,
    -0.4270909236388347, -0.531664134107571, 0.948441544330522, 0.027783170261051756, 0.32329658785378007, 0.23812300510489684,
    1.3591732562945975, 0.08058295121975718, -0.10549253514201101, 3.439138404232171, 0.22980686849911303, 1.1735409881468788,
    -0.747362200338692, -0.9557636824933192, 1.2026763310067605, 0.6314119976810105, 1.1096540619617945, -0.6260401842769959,
    -0.26502170420530685, 1.321164952897431, -1.9604007117755966, 1.177795349916604, -1.9691520879685322, -0.13980320730717594,
    -0.345398491728433, -0.6336254812232096, 0.8204196735373097, 0.26899096049460985, -1.1020560001140358, 0.5922680918775663,
    0.06801667347258508, -1.7673288467905541, 0.39484610446949814, -0.9352094130400525, 0.31056457166023155, 1.6662991504252436,
    -1.39608649694258, -0.7478660282706865, -0.6638279793678532, 2.035363499846621, 0.5206057530849415, -0.14910435339119882,
    -1.5662186623375411, 0.7038466178163314, -0.40043960414969093, -1.026132144561171, 1.5444204843223646, -1.3623542757267708,
])


def _r_vector(values) -> str:
    """Render a numeric grid as R c(...) code with plain float literals.

    numpy 2.x renders np.float64(0.01) inside f-strings, which R cannot
    parse; repr(float(v)) round-trips to the same double (D2.48 lesson).
    """
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
    """R program: SHASH at the transformed arguments + one SHASH fit on log(y).

    This is the live-R yardstick for the whole module -- the transform
    identities above mean these are exactly the LOGSHASH contracts.
    """
    profiles = ", ".join(
        f"list(mu={p['mu']}, sigma={p['sigma']}, nu={p['nu']}, tau={p['tau']})"
        for p in PROFILES
    )
    return f"""
suppressPackageStartupMessages({{library(gamlss); library(gamlss.dist); library(jsonlite)}})
out <- list()
profiles <- list({profiles})

# d/p/q at the transformed arguments.  For d the transform carries the
# Jacobian 1/x; for q it is the exp on the way back.
d_rows <- list()
for (i in seq_along(profiles)) {{
  p <- profiles[[i]]
  d_rows[[i]] <- list(
    profile = p, x = {_r_vector(X_GRID)},
    d = dSHASH(log({_r_vector(X_GRID)}), mu = p$mu, sigma = p$sigma,
               nu = p$nu, tau = p$tau) / {_r_vector(X_GRID)},
    p = pSHASH(log({_r_vector(X_GRID)}), mu = p$mu, sigma = p$sigma,
               nu = p$nu, tau = p$tau),
    q = exp(qSHASH({_r_vector(P_GRID)}, mu = p$mu, sigma = p$sigma,
                   nu = p$nu, tau = p$tau))
  )
}}
out$transform <- d_rows

# End-to-end: a SHASH fit on log(y) is the same likelihood as a LOGSHASH
# fit on y up to the constant 2*sum(log(y)).
y <- {_r_vector(FIT_Y)}
x <- {_r_vector(FIT_X)}
m <- gamlss(log(y) ~ x, family = SHASH(), sigma.formula = ~1,
            nu.formula = ~1, tau.formula = ~1,
            control = gamlss.control(n.cyc = 50, trace = FALSE))
out$fit <- list(
  deviance = m$G.deviance,
  mu_coef = as.numeric(coef(m, parameter = "mu")),
  sigma_coef = as.numeric(coef(m, parameter = "sigma")),
  nu_coef = as.numeric(coef(m, parameter = "nu")),
  tau_coef = as.numeric(coef(m, parameter = "tau")),
  converged = m$converged
)
cat(toJSON(out, auto_unbox = TRUE, digits = 17, na = "null"))
"""


@pytest.fixture(scope="module")
def r_probe():
    """Live-R contract dump via the SHASH transform."""
    if RSCRIPT is None:  # pragma: no cover
        pytest.skip("Rscript not on PATH")
    stdout = _run_r(_r_probe_script().splitlines())
    return json.loads(stdout)


# ---------------------------------------------------------------------------
# d/p/q against live R's SHASH at the transformed arguments
# ---------------------------------------------------------------------------


@pytest.mark.skipif(RSCRIPT is None, reason="Rscript not on PATH")
class TestDpqrAgainstLiveRTransform:
    def test_d_matches_transform(self, r_probe):
        for row in r_probe["transform"]:
            p = row["profile"]
            ours = np.asarray(dLOGSHASH(row["x"], p["mu"], p["sigma"],
                                        p["nu"], p["tau"]))
            np.testing.assert_allclose(
                ours, np.asarray(row["d"]), rtol=1e-12, atol=1e-300,
                err_msg=f"dLOGSHASH vs dSHASH(log x)/x at {p}",
            )

    def test_p_matches_transform(self, r_probe):
        for row in r_probe["transform"]:
            p = row["profile"]
            ours = np.asarray(pLOGSHASH(row["x"], p["mu"], p["sigma"],
                                        p["nu"], p["tau"]))
            np.testing.assert_allclose(
                ours, np.asarray(row["p"]), rtol=1e-12, atol=1e-300,
                err_msg=f"pLOGSHASH vs pSHASH(log q) at {p}",
            )

    def test_q_matches_transform(self, r_probe):
        # rtol 1e-4, not machine precision: R's qSHASH is a uniroot solve at
        # its default tolerance, so the reference itself carries ~1e-5
        # relative error in the far tail (measured max rel 1.6e-05 at
        # p = 0.99).  Our bisection solves the same monotone equation to
        # machine precision -- the residual is R's, the IG/SEP/GIG
        # disposition.
        for row in r_probe["transform"]:
            p = row["profile"]
            ours = np.asarray(qLOGSHASH(P_GRID, p["mu"], p["sigma"],
                                        p["nu"], p["tau"]))
            np.testing.assert_allclose(
                ours, np.asarray(row["q"]), rtol=1e-4, atol=1e-300,
                err_msg=f"qLOGSHASH vs exp(qSHASH(p)) at {p}",
            )

    def test_p_upper_tail_and_log_p(self, r_probe):
        for row in r_probe["transform"]:
            p = row["profile"]
            upper = np.asarray(pLOGSHASH(row["x"], p["mu"], p["sigma"],
                                         p["nu"], p["tau"], lower_tail=False))
            lower = np.asarray(pLOGSHASH(row["x"], p["mu"], p["sigma"],
                                         p["nu"], p["tau"]))
            np.testing.assert_allclose(upper + lower, 1.0, rtol=1e-12,
                                       atol=1e-14)
            log_p = np.asarray(pLOGSHASH(row["x"], p["mu"], p["sigma"],
                                         p["nu"], p["tau"], log_p=True))
            np.testing.assert_allclose(log_p, np.log(lower), rtol=1e-12,
                                       atol=1e-14)


# ---------------------------------------------------------------------------
# R's boundary quirks and validation semantics (pure Python)
# ---------------------------------------------------------------------------


class TestBoundaryQuirks:
    """The support assignments land after the log/tail transforms in R."""

    def test_d_nonpositive_is_zero_in_both_modes(self):
        p = dict(mu=0.3, sigma=1.2, nu=0.4, tau=0.8)
        for x in (0.0, -1.0, -50.0):
            assert float(np.asarray(dLOGSHASH(x, **p))) == 0.0
            # log mode gives 0, NOT -inf -- the dGIG/dNBII quirk shape.
            assert float(np.asarray(dLOGSHASH(x, log=True, **p))) == 0.0

    def test_p_boundaries(self):
        p = dict(mu=0.3, sigma=1.2, nu=0.4, tau=0.8)
        assert float(np.asarray(pLOGSHASH(0.0, **p))) == 0.0
        assert float(np.asarray(pLOGSHASH(-2.0, **p))) == 0.0
        assert float(np.asarray(pLOGSHASH(np.inf, **p))) == 1.0
        assert float(np.asarray(pLOGSHASH(0.0, log_p=True, **p))) == 0.0
        assert float(np.asarray(pLOGSHASH(-2.0, log_p=True, **p))) == 0.0

    def test_q_boundaries(self):
        p = dict(mu=0.3, sigma=1.2, nu=0.4, tau=0.8)
        assert float(np.asarray(qLOGSHASH(0.0, **p))) == 0.0
        assert float(np.asarray(qLOGSHASH(1.0, **p))) == np.inf
        assert np.isnan(float(np.asarray(qLOGSHASH(-0.1, **p))))
        assert np.isnan(float(np.asarray(qLOGSHASH(1.1, **p))))

    def test_invalid_parameters_return_nan(self):
        # R's d/p/q stop on sigma < 0, tau < 0, nu < 0; we answer NaN.
        assert np.isnan(float(np.asarray(dLOGSHASH(1.0, 0.0, -1.0, 0.5, 0.5))))
        assert np.isnan(float(np.asarray(dLOGSHASH(1.0, 0.0, 1.0, -0.5, 0.5))))
        assert np.isnan(float(np.asarray(dLOGSHASH(1.0, 0.0, 1.0, 0.5, -0.5))))
        assert np.isnan(float(np.asarray(pLOGSHASH(1.0, 0.0, 1.0, -0.5, 0.5))))
        assert np.isnan(float(np.asarray(qLOGSHASH(0.5, 0.0, 1.0, -0.5, 0.5))))

    def test_p_q_roundtrip_machine_precision(self):
        for p in PROFILES:
            u = np.linspace(0.01, 0.99, 40)
            q = np.asarray(qLOGSHASH(u, p["mu"], p["sigma"], p["nu"], p["tau"]))
            back = np.asarray(pLOGSHASH(q, p["mu"], p["sigma"], p["nu"], p["tau"]))
            np.testing.assert_allclose(back, u, rtol=1e-10, atol=1e-12,
                                       err_msg=f"roundtrip at {p}")

    def test_r_sampler_distribution(self):
        """Statistical check only (RNG streams differ by construction).

        Exact zeros are correct, not a defect: R's rLOGSHASH is
        qLOGSHASH(runif(n)) and runif returns [0, 1), so a draw can land on
        exactly 0 -- which R's q[p == 0] <- 0 maps to 0, and so do we (the
        _shasho_r precedent documents the same shape).
        """
        import jax.random as jr

        for p in [(0.0, 1.0, 0.5, 0.5), (0.3, 0.5, 0.2, 1.5)]:
            draws = np.asarray(rLOGSHASH(jr.PRNGKey(7), 20000, *p))
            assert np.all(np.isfinite(draws)) and np.all(draws >= 0.0)
            # Median of the log-scale SHASH mapped back through the transform.
            expect_med = float(np.asarray(qLOGSHASH(0.5, *p)))
            assert abs(np.median(draws) - expect_med) < 0.15 * max(expect_med, 1.0)


# ---------------------------------------------------------------------------
# Capability status and initial values
# ---------------------------------------------------------------------------


class TestCapabilityAndInitials:
    def test_status_left_placeholder(self):
        from omnilss.family_capabilities import implementation_status_for_family

        status = implementation_status_for_family(FAMILY)
        assert status.value == "experimental"

    def test_links_are_r_defaults(self):
        fam = resolve_family(FAMILY)
        assert fam.links == {"mu": "identity", "sigma": "log",
                             "nu": "log", "tau": "log"}

    def test_mu_initial_is_log_y_plus_mean_over_two(self):
        # R LOGSHASH.R (vendored development source): mu <- (log(y)+mean(log(y)))/2
        fam = resolve_family(FAMILY)
        w = np.ones_like(FIT_Y)
        target = _initial_mu_target(fam, FIT_Y, w)
        log_y = np.log(FIT_Y)
        np.testing.assert_allclose(target, (log_y + log_y.mean()) / 2.0,
                                   rtol=1e-12)

    def test_sigma_initial_is_fifth_of_log_sd(self):
        # R LOGSHASH.R: sigma <- rep(sd(log(y))/5, length(y)); sd is n-1.
        fam = resolve_family(FAMILY)
        mu = (np.log(FIT_Y) + np.log(FIT_Y).mean()) / 2.0
        w = np.ones_like(FIT_Y)
        sigma = _initial_sigma(fam, FIT_Y, mu, w)
        np.testing.assert_allclose(
            sigma, np.std(np.log(FIT_Y), ddof=1) / 5.0, rtol=1e-12)

    def test_nu_and_tau_initials_are_one_half(self):
        # R LOGSHASH.R: nu <- rep(.5, ...) ; tau <- rep(.5, ...)
        fam = resolve_family(FAMILY)
        mu = (np.log(FIT_Y) + np.log(FIT_Y).mean()) / 2.0
        w = np.ones_like(FIT_Y)
        assert _initial_parameter_value(fam, "nu", FIT_Y, mu, w) == 0.5
        assert _initial_parameter_value(fam, "tau", FIT_Y, mu, w) == 0.5


# ---------------------------------------------------------------------------
# End-to-end fit parity: our LOGSHASH(y ~ x) vs R's SHASH(log(y) ~ x)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(RSCRIPT is None, reason="Rscript not on PATH")
class TestFitParityAgainstLiveR:
    def test_rs_fit_matches_r_shash_fit_on_log_y(self, r_probe):
        """Same likelihood up to 2*sum(log(y)); deviance rtol 1e-3.

        Measured on the embedded sample at n.cyc = 50 on both sides:
        ours 53.7826562140556 vs R 53.772077160882141 (rel 1.97e-04).  The
        trajectories track from the first cycle (diff 0.006 at cycle 1) --
        same basin, slightly different per-cycle progress, the WEI2/PE2
        flat-surface disposition.  R's SHASH fit on this sample does not
        converge within 50 cycles (R itself reports "Algorithm RS has not
        yet converged"), which is why the budget is matched at 50.
        """
        r = r_probe["fit"]
        py = gamlss("y ~ x", family=FAMILY, data={"y": FIT_Y, "x": FIT_X},
                    method="RS", max_iter=50)
        mapped = float(py.deviance) - 2.0 * float(np.sum(np.log(FIT_Y)))
        np.testing.assert_allclose(
            mapped, float(r["deviance"]), rtol=1e-3, atol=1e-3,
            err_msg=f"mapped deviance {mapped} vs R {r['deviance']}",
        )
        np.testing.assert_allclose(
            np.ravel(np.asarray(py.coefficients["mu"], dtype=float)),
            np.ravel(np.asarray(r["mu_coef"], dtype=float)),
            rtol=1e-2, atol=1e-3, err_msg="mu coefficients vs R",
        )

    def test_long_budget_self_check_on_r_density(self):
        """Our converged point is genuine on R's own density.

        At n.cyc = 1000 both sides converge, and ours sits *below* R
        (53.484156174299699 vs 53.522983976260143, rel 7.25e-04).  The
        parity discipline's "ours is lower" direction requires the
        two-step adjudication; this is step 2: evaluate R's own dSHASH at
        our converged per-observation fitted values and check it reproduces
        our deviance.  Measured: bit-identical (53.484156174299699), so the
        lower point is real on R's likelihood -- R's RS stopped at an
        inferior stationary point, the PE2 disposition variant.
        """
        py = gamlss("y ~ x", family=FAMILY, data={"y": FIT_Y, "x": FIT_X},
                    method="RS", max_iter=1000)
        assert py.converged is True
        fitted = py.fitted_values
        script = f"""
suppressPackageStartupMessages({{library(gamlss.dist)}})
u <- {_r_vector(np.log(FIT_Y))}
ll <- sum(dSHASH(u,
                 mu = {_r_vector(fitted["mu"])},
                 sigma = {_r_vector(fitted["sigma"])},
                 nu = {_r_vector(fitted["nu"])},
                 tau = {_r_vector(fitted["tau"])}, log = TRUE))
cat(format(-2 * ll, digits = 17))
"""
        out = _run_r(script.splitlines())
        r_dev_at_our_point = float(out.strip())
        np.testing.assert_allclose(
            r_dev_at_our_point,
            float(py.deviance) - 2.0 * float(np.sum(np.log(FIT_Y))),
            rtol=1e-12, atol=1e-10,
            err_msg="R's own density at our point must reproduce our deviance",
        )
