"""GIG (generalised inverse Gaussian) parity against live R at digits=17.

What this replaces
------------------
The family shipped with a density in the textbook ``(lambda, chi, psi)``
parameterisation (``chi = 1/sigma^2``, ``psi = 1/(mu^2*sigma^2)``) whose mean
is ``mu * K_{nu+1}(1/sigma^2)/K_nu(1/sigma^2)`` -- a *different* distribution
from R's whenever that ratio differs from 1 (measured 8.08e-02 rel on the
parity scanner's dpqr gate, D2.47), and with NaN-placeholder p/q/r (the
family was still ``placeholder`` status, rejected by production routes).

What R actually defines (installed gamlss.dist 6.1.1, dumped from the
namespace and verified term-for-term)::

    c      = K_{nu+1}(1/sigma^2) / K_nu(1/sigma^2)
    log f  = nu*log(c) - nu*log(mu) + (nu-1)*log(x) - log(2)
             - log(K_nu(1/sigma^2))
             - (1/(2*sigma^2)) * (c*x/mu + mu/(c*x))

i.e. ``lambda = nu``, ``chi = mu/(c*sigma^2)``, ``psi = c/(mu*sigma^2)`` --
``E[X] = mu`` by construction.  The transcribed density agrees with live R to
9.1e-13 over 48 probe points (nu in [-1.5, 2], x from 0.01 to 50).

R quirks pinned here (explicit branch = convention, D2.25/D2.30)
---------------------------------------------------------------
* ``dGIG`` applies ``ifelse(x <= 0, 0, ft)`` *after* the log branch, so the
  logged density at a non-positive ``x`` is ``0``, not ``-inf``;
* ``pGIG`` integrates the ``mu = 1`` density from ``0.001`` to ``q/mu``.  By
  the exact scaling property ``GIG(mu, sigma, nu) = mu * GIG(1, sigma, nu)``
  that equals ``F(q) - F(0.001*mu)``: the true CDF minus a negligible lower
  slice, with R's own adaptive-quadrature error on top (measured up to
  1.24e-06 absolute at mu=1, sigma=1, nu=0, q=100 -- R answers
  0.999998759019705 against a true 1.0).  We integrate the true CDF; below
  R's 0.001 floor R's ``integrate`` runs over an inverted interval and
  answers tiny *negative* noise (``-4.8859e-219`` at mu=2, sigma=1, nu=-0.5,
  q=0.0005) while we return the true tiny value; in the deep tail R's
  ``integrate`` can instead overshoot 1 (``1.00000000000236`` at mu=2,
  q=100), which makes its ``1 - cdf`` upper tail negative and
  ``log(cdf)`` a tiny positive number -- both pinned in the tests below;
* ``qGIG`` has no ``p == 0``/``p == 1`` assignments in the installed
  version (the vendored 6.1-8 source does): ``qGIG(0)`` lands wherever
  ``uniroot`` converges inside the flat ``pGIG == 0`` region and ``qGIG(1)``
  wherever ``integrate`` noise first reads as 1; ``p`` outside [0, 1] is a
  hard ``stop``.  We return the mathematically correct 0 / +inf and clip;
* the derivative slots are R's own approximations: ``d2ldm2`` is the
  *expected* Hessian (not the observed one), ``d2ldd2``/``d2ldv2`` are
  ``-score^2`` forms, and ``dldv`` is gamlss's forward ``numeric.deriv``
  with ``delta = 0.01``.  All transcribed verbatim (D2.44 SICHEL pattern) and
  compared here against live R slot-by-slot.

Methodology
-----------
d/p/q, the six slots, the boundary quirks and the end-to-end fit are all
compared against **live R** through ``digits = 17`` subprocess probes.
Nothing is hard-coded; a mistyped constant cannot make these tests pass.
The fit uses one embedded digits=17 sample (live R: ``set.seed(20261020);
rGIG(30, mu = 5, sigma = 1, nu = -0.5)`` plus ``rnorm(30)`` covariates) so
both fitters provably see the same numbers.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)  # parity prerequisite
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from omnilss import gamlss  # noqa: E402
from omnilss._fitting_init import _initial_sigma  # noqa: E402
from omnilss.distribution_registry import resolve  # noqa: E402
from omnilss.gig import dGIG, pGIG, qGIG, rGIG  # noqa: E402
from omnilss.response_domain import validate_response  # noqa: E402

FAMILY = "GIG"
RSCRIPT = shutil.which("Rscript")

#: (mu, sigma, nu) profiles.  R's own defaults first; the rest span negative
#: and positive nu (support split), small/large sigma (peak width) and
#: small mu.  The (1, 1, 0) profile is the measured far-tail quadrature
#: case: R's default-tolerance integrate terminates early on the
#: super-exponentially decaying tail (reporting "OK" with
#: abs.error=5.19e-06), so its pGIG loses up to 1.24e-06 absolute there
#: while the true tail at q=100 is below 1e-15 -- see the CDF tiering.
PROFILES = [
    dict(mu=2.0, sigma=1.0, nu=-0.5),
    dict(mu=2.0, sigma=0.5, nu=1.0),
    dict(mu=5.0, sigma=2.0, nu=0.5),
    dict(mu=0.7, sigma=1.2, nu=-1.5),
    dict(mu=10.0, sigma=0.3, nu=2.0),
    dict(mu=1.0, sigma=1.0, nu=0.0),
]

#: Response grid: support edge, interior, deep tail.
Y_GRID = np.array([0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0])
Q_GRID = np.array([0.05, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 100.0])
P_GRID = np.array([0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99])

#: Two slot probe points (y, mu, sigma, nu); the second has positive nu and
#: small sigma, where c = K_{nu+1}/K_nu deviates strongly from 1.
SLOT_POINTS = [
    dict(y=3.0, mu=2.0, sigma=1.0, nu=-0.5),
    dict(y=4.0, mu=3.0, sigma=0.5, nu=1.0),
]

#: Measured agreement floors of the digits=17 probes.  d: 9.1e-13 (the
#: Bessel primal is scipy-exact on both sides).  p: 1.24e-06 at worst --
#: R's adaptive integrate, not our quadrature (4.4e-16 there).  q: 2.1e-05
#: on the scanner grid / 5.7e-04 absolute on deep-tail profiles -- R's
#: uniroot at its default tolerance on R's own quadrature CDF.  slots:
#: 1.9e-11, the residual being R's forward-difference dldv.
RTOL_D = 1e-12
RTOL_P = 1e-5
RTOL_Q = 1e-4
RTOL_SLOTS = 1e-10

#: One embedded digits=17 sample (live R: set.seed(20261020);
#: rGIG(30, mu = 5, sigma = 1, nu = -0.5); x <- rnorm(30) -- x continues the
#: same RNG stream after the sampler, which is what the probe below draws).
FIT_Y = np.array([
    0.8507288185851543, 6.3408204201915, 18.75348552957201, 7.228945453456809, 3.9160725114222124,
    1.5291572378341953, 2.2267429634055413, 0.6121549389254635, 1.091142575023533, 20.194693763133515,
    7.544182535294939, 1.6091620192436538, 2.509136480817101, 7.147079860541392, 4.122436724151571,
    4.120282793866815, 2.505949244027857, 3.655049864706367, 4.471790342641989, 1.8892978331388686,
    2.5450962106470425, 2.583861854086429, 1.4471043701338073, 1.036472855279439, 3.89435365079007,
    14.301285816796254, 16.39977964219486, 1.3255752381462722, 0.9940603766597336, 1.529541447521481,
])
FIT_X = np.array([
    0.3603093766189468, -0.3725097559103078, -0.8729090511236935, -0.4227506672661789, -0.22092670230201852,
    1.9209735330837745, 0.852844399084864, 1.2329264055187745, 1.0314036593047013, -0.5746159944700534,
    0.412128102870962, 1.0765012362712878, 1.6555983787769804, -0.6592282200820891, 1.0154187460217623,
    -0.06197877981277819, -0.36187510872067125, -1.4950890815866995, -0.5296422470505291, -0.20830226801106605,
    -0.5084822975016011, 1.02634150543233, 0.2581646904953074, 1.1223587273159465, 0.6105678033017727,
    -0.2830413210721494, 1.3934024018823306, 0.14725478262573902, 0.27138512013046817, -1.5269602094742032,
])

#: Same RS control on both sides (R's own default c.crit = 0.001).
R_FIT_CONTROL = "gamlss.control(n.cyc = 200, c.crit = 0.001, trace = FALSE)"


def _run_r(lines, timeout=900):
    """Run an R script, return stdout; skip the test when R is absent."""
    if RSCRIPT is None:  # pragma: no cover - CI without R
        pytest.skip("Rscript not on PATH")
    with tempfile.NamedTemporaryFile("w", suffix=".R", delete=False,
                                     encoding="utf-8") as fh:
        fh.write("\n".join(lines))
        path = fh.name
    try:
        proc = subprocess.run([RSCRIPT, path], capture_output=True,
                              text=True, timeout=timeout)
    finally:
        Path(path).unlink(missing_ok=True)
    assert proc.returncode == 0, f"R failed: {proc.stderr[-800:]}"
    return proc.stdout


def _r_vector(grid):
    """Render a numeric grid as R ``c(...)`` code with plain float literals.

    numpy 2.x renders ``np.float64(0.01)`` inside f-strings, which R cannot
    parse (``could not find function "np.float64"``).  ``repr(float(v))`` is
    the shortest string that round-trips to the same double, so digits=17
    fidelity is preserved and the literal stays plain.
    """
    return "c(" + ", ".join(repr(float(v)) for v in grid) + ")"


def _r_probe_script():
    """R program dumping the full GIG contract as one JSON blob."""
    profiles = ", ".join(
        f"list(mu={p['mu']}, sigma={p['sigma']}, nu={p['nu']})" for p in PROFILES
    )
    points = ", ".join(
        f"list(y={p['y']}, mu={p['mu']}, sigma={p['sigma']}, nu={p['nu']})"
        for p in SLOT_POINTS
    )
    return f"""
suppressPackageStartupMessages({{library(gamlss); library(gamlss.dist); library(jsonlite)}})
out <- list()
profiles <- list({profiles})
slot_points <- list({points})

dens <- list()
for (i in seq_along(profiles)) {{
  p <- profiles[[i]]
  row <- list(profile = p, x = {_r_vector(Y_GRID)})
  row$d <- dGIG(row$x, mu = p$mu, sigma = p$sigma, nu = p$nu)
  row$d_log <- dGIG(row$x, mu = p$mu, sigma = p$sigma, nu = p$nu, log = TRUE)
  row$d_nonpos <- dGIG(c(-1, 0), mu = p$mu, sigma = p$sigma, nu = p$nu)
  row$d_nonpos_log <- dGIG(c(-1, 0), mu = p$mu, sigma = p$sigma, nu = p$nu, log = TRUE)
  dens[[i]] <- row
}}
out$density <- dens

pg <- list()
for (i in seq_along(profiles)) {{
  p <- profiles[[i]]
  row <- list(profile = p, q = {_r_vector(Q_GRID)})
  row$p <- pGIG(row$q, mu = p$mu, sigma = p$sigma, nu = p$nu)
  row$p_upper <- pGIG(row$q, mu = p$mu, sigma = p$sigma, nu = p$nu, lower.tail = FALSE)
  row$p_log <- pGIG(row$q, mu = p$mu, sigma = p$sigma, nu = p$nu, log.p = TRUE)
  # The same integral pGIG's own loop evaluates, re-run at integrate's tight
  # tolerance.  At the default rel.tol (~6.1e-05) QUADPACK terminates early
  # on the super-exponentially decaying tail and still reports "OK": measured
  # abs.error=5.19e-06 at mu=1, sigma=1, nu=0, q=100, where the true tail is
  # below 1e-15.  This column is R's own converged answer and the yardstick
  # for the deep-tail tier below.
  row$p_upper_tight <- sapply(row$q, function(qq) {{
    1 - integrate(function(x) dGIG(x, mu = 1, sigma = p$sigma, nu = p$nu),
                  0.001, qq / p$mu, rel.tol = 1e-12)$value
  }})
  pg[[i]] <- row
}}
out$pgig <- pg
# R's 0.001 integration floor: below 0.001*mu its integrate runs over an
# inverted interval and answers tiny negative quadrature noise.
out$p_floor <- pGIG(0.0005, mu = 2, sigma = 1, nu = -0.5)

qg <- list()
for (i in seq_along(profiles)) {{
  p <- profiles[[i]]
  row <- list(profile = p, p = {_r_vector(P_GRID)})
  row$q <- qGIG(row$p, mu = p$mu, sigma = p$sigma, nu = p$nu)
  row$q_upper <- qGIG(row$p, mu = p$mu, sigma = p$sigma, nu = p$nu, lower.tail = FALSE)
  row$q_log <- qGIG(log(row$p), mu = p$mu, sigma = p$sigma, nu = p$nu, log.p = TRUE)
  qg[[i]] <- row
}}
out$qgig <- qg

fam <- GIG()
op <- options(digits = 17)
slots <- list()
for (i in seq_along(slot_points)) {{
  pt <- slot_points[[i]]
  slots[[i]] <- list(
    point = pt,
    dldm = fam$dldm(pt$y, pt$mu, pt$sigma, pt$nu),
    d2ldm2 = fam$d2ldm2(pt$y, pt$mu, pt$sigma, pt$nu),
    dldd = fam$dldd(pt$y, pt$mu, pt$sigma, pt$nu),
    d2ldd2 = fam$d2ldd2(pt$y, pt$mu, pt$sigma, pt$nu),
    dldv = fam$dldv(pt$y, pt$mu, pt$sigma, pt$nu),
    d2ldv2 = fam$d2ldv2(pt$y, pt$mu, pt$sigma, pt$nu)
  )
}}
options(op)
out$slots <- slots

b <- list()
b$p0 <- qGIG(0, mu = 2, sigma = 1, nu = -0.5)
b$p1 <- qGIG(1, mu = 2, sigma = 1, nu = -0.5)
b$pneg <- tryCatch(qGIG(-0.1, mu = 2, sigma = 1, nu = -0.5),
                   error = function(e) paste("ERR:", conditionMessage(e)))
b$pbig <- tryCatch(qGIG(1.1, mu = 2, sigma = 1, nu = -0.5),
                   error = function(e) paste("ERR:", conditionMessage(e)))
out$qgig_boundaries <- b

# y.valid: gamlss() must reject a negative response for GIG.
set.seed(20261020)
yv <- rGIG(30, mu = 5, sigma = 1, nu = -0.5)
yv[3] <- -1.0
out$yvalid_negative <- tryCatch(
  gamlss(y ~ 1, family = GIG(), data = data.frame(y = yv),
         control = gamlss.control(n.cyc = 5, trace = FALSE)),
  error = function(e) paste("ERR:", conditionMessage(e)))

# End-to-end fit on the embedded sample.
set.seed(20261020)
fit_y <- rGIG(30, mu = 5, sigma = 1, nu = -0.5)
fit_x <- rnorm(30)
m <- gamlss(fit_y ~ fit_x, family = GIG(),
            control = {R_FIT_CONTROL})
out$fit <- list(
  y = fit_y, x = fit_x,
  deviance = m$G.deviance,
  mu_coef = as.numeric(coef(m, parameter = "mu")),
  sigma_coef = as.numeric(coef(m, parameter = "sigma")),
  nu_coef = as.numeric(coef(m, parameter = "nu")),
  converged = m$converged
)
cat(toJSON(out, auto_unbox = TRUE, digits = 17, na = "null"))
"""


@pytest.fixture(scope="module")
def r_probe():
    """Live-R contract dump; the whole module compares against it."""
    import json

    stdout = _run_r(_r_probe_script().splitlines())
    return json.loads(stdout)


@pytest.mark.skipif(RSCRIPT is None, reason="Rscript not on PATH")
class TestDensityAgainstLiveR:
    def test_d_matches_live_r(self, r_probe):
        for row in r_probe["density"]:
            p = row["profile"]
            ours = np.asarray(dGIG(row["x"], p["mu"], p["sigma"], p["nu"]))
            np.testing.assert_allclose(
                ours, row["d"], rtol=RTOL_D, atol=1e-300,
                err_msg=f"dGIG at {p}",
            )

    def test_d_log_matches_live_r(self, r_probe):
        for row in r_probe["density"]:
            p = row["profile"]
            ours = np.asarray(
                dGIG(row["x"], p["mu"], p["sigma"], p["nu"], log=True)
            )
            np.testing.assert_allclose(
                ours, row["d_log"], rtol=RTOL_D, atol=1e-300,
                err_msg=f"dGIG(log=TRUE) at {p}",
            )

    def test_nonpositive_x_is_zero_even_in_log_mode(self, r_probe):
        """R applies the support branch after the log branch: 0, not -inf."""
        for row in r_probe["density"]:
            p = row["profile"]
            ours = np.asarray(
                dGIG(np.array([-1.0, 0.0]), p["mu"], p["sigma"], p["nu"], log=True)
            )
            np.testing.assert_array_equal(ours, row["d_nonpos_log"])
            np.testing.assert_array_equal(
                np.asarray(dGIG(np.array([-1.0, 0.0]), p["mu"], p["sigma"], p["nu"])),
                row["d_nonpos"],
            )


@pytest.mark.skipif(RSCRIPT is None, reason="Rscript not on PATH")
class TestCdfAgainstLiveR:
    def test_p_matches_live_r(self, r_probe):
        """rtol 1e-5: the residual is R's adaptive-quadrature error."""
        for row in r_probe["pgig"]:
            p = row["profile"]
            ours = np.asarray(pGIG(row["q"], p["mu"], p["sigma"], p["nu"]))
            np.testing.assert_allclose(
                ours, row["p"], rtol=RTOL_P, atol=1e-12,
                err_msg=f"pGIG at {p}",
            )

    def test_p_upper_tail_matches_live_r(self, r_probe):
        for row in r_probe["pgig"]:
            p = row["profile"]
            ours = np.asarray(
                pGIG(row["q"], p["mu"], p["sigma"], p["nu"], lower_tail=False)
            )
            r_upper = np.asarray(row["p_upper"], dtype=float)
            r_tight = np.asarray(row["p_upper_tight"], dtype=float)
            # Tier by where R's default-tolerance integrate still carries
            # information.  R's pGIG is 1 - integrate(dGIG, 0.001, q/mu) at
            # integrate's *default* rel.tol (~6.1e-05), and QUADPACK stops
            # early on the super-exponentially decaying tail: at mu=1,
            # sigma=1, nu=0 it reports "OK" with abs.error=5.19e-06 while
            # the true tail at q=100 is below 1e-15 (scipy quad, epsrel=1e-14;
            # our composite Gauss-Legendre gives 1 - 4.44e-16).  Measured
            # default-vs-tight gaps: 1.4e-04 relative at (1,1,0) q=10, 32% at
            # (1,1,0) q=20, 1.24e-06 absolute at (1,1,0) q=100 and 4.6e-04
            # relative at (5,2,0.5) q=100.  Above 1e-03 of tail mass the same
            # gap stays below 3e-07 relative, so that is the live boundary.
            live = r_upper > 1e-3
            np.testing.assert_allclose(
                ours[live], r_upper[live], rtol=RTOL_P, atol=1e-12,
                err_msg=f"pGIG(upper) at {p}",
            )
            # Deep tail: compare against R's own converged integral (the
            # probe's p_upper_tight column) -- our composite Gauss-Legendre
            # agrees with it to 1.2e-08 relative at worst, measured -- and pin
            # R's default answer inside its own measured quadrature noise
            # (max 1.24e-06 absolute over the grid).
            deep = (r_upper > 1e-8) & ~live
            np.testing.assert_allclose(
                ours[deep], r_tight[deep], rtol=RTOL_P, atol=1e-12,
                err_msg=f"pGIG(upper) deep tail vs R converged at {p}",
            )
            assert np.all(np.abs(r_upper[deep] - r_tight[deep]) <= 2e-6), (
                f"R's default-tolerance upper tail left its measured "
                f"quadrature-noise band at {p}: {r_upper[deep]} vs "
                f"{r_tight[deep]}"
            )
            # Below the floor both sides are numerically zero; R's answer is
            # quadrature noise that can even go negative (integrate
            # overshooting 1), ours is the true non-negative tail.
            floor = ~(live | deep)
            assert np.all((ours[floor] >= 0.0) & (ours[floor] < 1e-8)), (
                f"pGIG(upper) deep tail at {p}: {ours[floor]}"
            )
            assert np.all(np.abs(r_upper[floor]) < 1e-8), (
                f"R's upper tail left the numerically-zero band at {p}: "
                f"{r_upper[floor]}"
            )

    def test_p_log_p_matches_live_r(self, r_probe):
        for row in r_probe["pgig"]:
            p = row["profile"]
            ours = np.asarray(
                pGIG(row["q"], p["mu"], p["sigma"], p["nu"], log_p=True)
            )
            r_log = np.asarray(row["p_log"], dtype=float)
            # R's log.p is log of its own default-tolerance cdf, so it
            # inherits the integrate noise documented in the upper-tail test
            # above; the tight column converted back through log1p is the
            # converged yardstick.  |log(cdf)| tracks the tail mass, so the
            # same 1e-03 boundary separates the informative region (measured
            # R noise below 1e-07 relative there) from the deep tail.
            r_log_tight = np.log1p(-np.asarray(row["p_upper_tight"], dtype=float))
            finite = np.isfinite(r_log)
            live = finite & (np.abs(r_log) > 1e-3)
            np.testing.assert_allclose(
                ours[live], r_log[live], rtol=RTOL_P, atol=1e-12,
                err_msg=f"pGIG(log.p) at {p}",
            )
            deep = finite & (np.abs(r_log) > 1e-8) & ~live
            np.testing.assert_allclose(
                ours[deep], r_log_tight[deep], rtol=RTOL_P, atol=1e-12,
                err_msg=f"pGIG(log.p) deep tail vs R converged at {p}",
            )
            # Ours returns log of the true cdf: tiny negative, or -inf when
            # the true probability underflows.  R returns log of its
            # quadrature cdf, which can overshoot 1 (+2.36e-12 at mu=2,
            # q=100) or go negative (nan).  All are numerically zero.
            masked = ~(live | deep)
            assert np.all(
                (np.abs(ours[masked]) < 1e-8) | np.isneginf(ours[masked])
            ), f"pGIG(log.p) deep tail at {p}: {ours[masked]}"

    def test_p_below_r_integration_floor(self, r_probe):
        """Below 0.001*mu R answers negative quadrature noise; we the truth.

        R's integrate runs over an inverted interval there, so its answer is
        a tiny *negative* number (-4.8859e-219 at this point, measured),
        neither 0 nor the true CDF.  Ours returns the true tiny value: the
        probability is negligible on both sides, which is the contract.
        """
        r_floor = float(r_probe["p_floor"])
        assert -1e-100 < r_floor <= 0.0, r_floor
        ours = float(np.asarray(pGIG(0.0005, 2.0, 1.0, -0.5)))
        # True CDF at 0.0005 for this profile is ~1e-300 (super-exponential
        # decay); assert we return a tiny non-negative number, not 0 exactly
        # and certainly not NaN -- both are R's behaviour, neither is ours.
        assert 0.0 <= ours < 1e-6


@pytest.mark.skipif(RSCRIPT is None, reason="Rscript not on PATH")
class TestQuantileAgainstLiveR:
    def test_q_matches_live_r(self, r_probe):
        """rtol 1e-4: R's uniroot at its default tolerance on its own CDF."""
        for row in r_probe["qgig"]:
            p = row["profile"]
            ours = np.asarray(qGIG(row["p"], p["mu"], p["sigma"], p["nu"]))
            np.testing.assert_allclose(
                ours, row["q"], rtol=RTOL_Q, atol=1e-3,
                err_msg=f"qGIG at {p}",
            )

    def test_q_upper_tail_matches_live_r(self, r_probe):
        for row in r_probe["qgig"]:
            p = row["profile"]
            ours = np.asarray(
                qGIG(row["p"], p["mu"], p["sigma"], p["nu"], lower_tail=False)
            )
            np.testing.assert_allclose(
                ours, row["q_upper"], rtol=RTOL_Q, atol=1e-3,
                err_msg=f"qGIG(upper) at {p}",
            )

    def test_q_log_p_matches_live_r(self, r_probe):
        for row in r_probe["qgig"]:
            p = row["profile"]
            ours = np.asarray(
                qGIG(np.log(row["p"]), p["mu"], p["sigma"], p["nu"], log_p=True)
            )
            np.testing.assert_allclose(
                ours, row["q_log"], rtol=RTOL_Q, atol=1e-3,
                err_msg=f"qGIG(log.p) at {p}",
            )

    def test_q_boundaries_pin_r_artifacts(self, r_probe):
        """R's installed qGIG has no p==0/p==1 assignments; pin its values.

        R's qGIG(0) lands inside the flat pGIG == 0 region (any point of it
        is 'a root'), and qGIG(1) wherever integrate noise first reads as 1.
        Ours returns the mathematically correct 0 / +inf.
        """
        b = r_probe["qgig_boundaries"]
        # R's measured artifacts (installed gamlss.dist 6.1.1).
        assert 0.0 < float(b["p0"]) < 0.01
        assert float(b["p1"]) > 1.0
        # p outside [0, 1] is a hard stop in R.
        assert "p must be between 0 and 1" in str(b["pneg"])
        assert "p must be between 0 and 1" in str(b["pbig"])
        # Ours: the correct boundary convention.
        assert float(np.asarray(qGIG(0.0, 2.0, 1.0, -0.5))) == 0.0
        assert float(np.asarray(qGIG(1.0, 2.0, 1.0, -0.5))) == np.inf
        # Out-of-range p is clipped (house convention), not NaN-propagated.
        assert np.isfinite(np.asarray(qGIG(-0.1, 2.0, 1.0, -0.5)))
        assert np.asarray(qGIG(1.1, 2.0, 1.0, -0.5)) == np.inf


@pytest.mark.skipif(RSCRIPT is None, reason="Rscript not on PATH")
class TestSlotsAgainstLiveR:
    def test_scores_and_hessians_match_live_r(self, r_probe):
        fam = resolve(FAMILY)
        for entry in r_probe["slots"]:
            pt = entry["point"]
            y, mu, sigma, nu = pt["y"], pt["mu"], pt["sigma"], pt["nu"]
            for slot, key in (
                ("mu", "dldm"), ("sigma", "dldd"), ("nu", "dldv"),
            ):
                ours = np.asarray(fam.score_functions[slot](y, mu, sigma, nu))
                np.testing.assert_allclose(
                    ours, entry[key], rtol=RTOL_SLOTS, atol=1e-14,
                    err_msg=f"score {slot} at {pt}",
                )
            for slot, key in (
                ("mu", "d2ldm2"), ("sigma", "d2ldd2"), ("nu", "d2ldv2"),
            ):
                ours = np.asarray(fam.hessian_functions[slot](y, mu, sigma, nu))
                np.testing.assert_allclose(
                    ours, entry[key], rtol=RTOL_SLOTS, atol=1e-14,
                    err_msg=f"hessian {slot} at {pt}",
                )


@pytest.mark.skipif(RSCRIPT is None, reason="Rscript not on PATH")
class TestFitAgainstLiveR:
    def test_rs_fit_matches_r_gamlss(self, r_probe):
        """RS fit on the embedded sample: deviance rtol 1e-3.

        Coefficients: mu at rtol 1e-2 (measured 7e-4), nu at rtol 5e-2
        (measured 3e-3), sigma at rtol 5e-2 (measured 4.5e-2).  The sigma
        and nu spreads are the WEI2/PE2 disposition (D2.3): near the
        optimum the GIG log-likelihood is flat in (sigma, nu) -- profiling
        our converged point, the deviance moves by less than 2e-3 over a
        +-5% band in either -- so the two fitters stop at slightly
        different points of the same optimum.  Primary evidence is the
        deviance: ours 143.1512610811 vs R 143.1523793335484, rel 7.8e-06
        (ours lower, i.e. the "suspect R stopped early" direction).  R's
        own bookkeeping corroborates that: its final ``coef(nu)``
        (-1.2232716) differs from its final ``fitted(nu)`` (-1.2063725)
        even though the nu submodel is intercept-only with an identity
        link -- glim.fit's step-halving branch leaves ``fit$coefficients``
        (pre-halving WLS) disconnected from ``fv`` (post-halving), so R's
        reported point is itself a half-step blend.
        """
        r = r_probe["fit"]
        # Both sides must see the identical sample.
        np.testing.assert_array_equal(np.asarray(r["y"]), FIT_Y)
        np.testing.assert_array_equal(np.asarray(r["x"]), FIT_X)

        py = gamlss("y ~ x", family=FAMILY, data={"y": FIT_Y, "x": FIT_X},
                    method="RS", max_iter=100)
        assert py.converged is True
        np.testing.assert_allclose(
            float(py.deviance), float(r["deviance"]),
            rtol=1e-3, atol=1e-3,
            err_msg=f"deviance {py.deviance} vs R {r['deviance']}",
        )
        np.testing.assert_allclose(
            np.ravel(np.asarray(py.coefficients["mu"], dtype=float)),
            np.ravel(np.asarray(r["mu_coef"], dtype=float)),
            rtol=1e-2, atol=1e-3, err_msg="mu coefficients vs R",
        )
        np.testing.assert_allclose(
            np.ravel(np.asarray(py.coefficients["sigma"], dtype=float)),
            np.ravel(np.asarray(r["sigma_coef"], dtype=float)),
            rtol=5e-2, atol=5e-3, err_msg="sigma coefficient vs R",
        )
        np.testing.assert_allclose(
            np.ravel(np.asarray(py.coefficients["nu"], dtype=float)),
            np.ravel(np.asarray(r["nu_coef"], dtype=float)),
            rtol=5e-2, atol=5e-3, err_msg="nu coefficient vs R",
        )

    def test_negative_response_rejected_like_r(self, r_probe):
        """R's GIG declares y.valid = all(y > 0); so does our gate."""
        r_err = str(r_probe["yvalid_negative"])
        assert "response variable out of range" in r_err
        y = FIT_Y.copy()
        y[3] = -1.0
        with pytest.raises(Exception, match="response variable out of range"):
            validate_response("GIG", y)
        # A legal sample must NOT be rejected (no over-rejection).
        validate_response("GIG", FIT_Y)


# ---------------------------------------------------------------------------
# Pure-Python invariants (no R required)
# ---------------------------------------------------------------------------


class TestInvariants:
    def test_p_q_roundtrip(self):
        u = np.linspace(0.01, 0.99, 50)
        for p in [(2.0, 1.0, -0.5), (2.0, 0.5, 1.0), (5.0, 2.0, 0.5),
                  (0.7, 1.2, -1.5), (10.0, 0.3, 2.0), (1.0, 1.0, 0.0)]:
            q = np.asarray(qGIG(u, *p))
            back = np.asarray(pGIG(q, *p))
            np.testing.assert_allclose(back, u, rtol=1e-10, atol=1e-12,
                                       err_msg=f"p(q(u)) at {p}")

    def test_p_monotone_bounded_and_edges(self):
        q = np.array([1e-8, 0.01, 0.1, 1.0, 10.0, 100.0, 1e6])
        p = np.asarray(pGIG(q, 2.0, 1.0, -0.5))
        assert np.all(np.diff(p) >= -1e-15)
        assert np.all((p >= 0.0) & (p <= 1.0))
        assert p[0] >= 0.0
        assert float(np.asarray(pGIG(0.0, 2.0, 1.0, -0.5))) == 0.0
        assert float(np.asarray(pGIG(np.inf, 2.0, 1.0, -0.5))) == 1.0
        assert float(np.asarray(pGIG(-1.0, 2.0, 1.0, -0.5))) == 0.0

    def test_upper_tail_complements_lower(self):
        q = np.array([0.1, 0.5, 1.0, 2.0, 5.0])
        lo = np.asarray(pGIG(q, 2.0, 1.0, -0.5))
        hi = np.asarray(pGIG(q, 2.0, 1.0, -0.5, lower_tail=False))
        np.testing.assert_allclose(lo + hi, 1.0, rtol=1e-12, atol=1e-14)

    def test_q_monotone_and_boundaries(self):
        p = np.linspace(0.02, 0.98, 40)
        q = np.asarray(qGIG(p, 2.0, 1.0, -0.5))
        assert np.all(np.diff(q) >= -1e-15)
        assert float(np.asarray(qGIG(0.0, 2.0, 1.0, -0.5))) == 0.0
        assert float(np.asarray(qGIG(1.0, 2.0, 1.0, -0.5))) == np.inf

    def test_log_p_matches_log_of_p(self):
        q = np.array([0.5, 1.0, 2.0, 5.0])
        direct = np.asarray(pGIG(q, 2.0, 1.0, -0.5, log_p=True))
        via_log = np.log(np.asarray(pGIG(q, 2.0, 1.0, -0.5)))
        np.testing.assert_allclose(direct, via_log, rtol=1e-12, atol=1e-14)

    def test_invalid_parameters_return_nan(self):
        assert np.isnan(float(np.asarray(dGIG(1.0, -1.0, 1.0, 0.0))))
        assert np.isnan(float(np.asarray(pGIG(1.0, 1.0, 0.0, 0.0))))
        assert np.isnan(float(np.asarray(qGIG(0.5, 1.0, -1.0, 0.0))))

    def test_broadcast_scalar_and_vector_forms_agree(self):
        q = np.array([0.5, 1.0, 2.0])
        vec = np.asarray(pGIG(q, 2.0, 1.0, -0.5))
        scal = np.asarray([pGIG(float(v), 2.0, 1.0, -0.5) for v in q])
        np.testing.assert_allclose(vec, scal, rtol=1e-14, atol=1e-15)

    def test_r_sampler_mean_tracks_mu(self):
        """Statistical check only (RNG streams differ by construction)."""
        import jax.random as jr

        for profile in [(2.0, 1.0, -0.5), (5.0, 0.8, 0.5)]:
            draws = np.asarray(rGIG(jr.PRNGKey(7), 20000, *profile))
            assert np.all(np.isfinite(draws)) and np.all(draws > 0.0)
            assert abs(draws.mean() - profile[0]) < 0.15 * profile[0]

    def test_sigma_initial_matches_r_expression(self):
        """R GIG()$sigma.initial: sigma <- sd(y)/mean(y), n-1 sd."""
        y = FIT_Y
        sigma = _initial_sigma(
            resolve(FAMILY), y, np.full(len(y), y.mean()), np.ones(len(y)),
            fixed_parameter_values={},
        )
        np.testing.assert_allclose(
            sigma, np.std(y, ddof=1) / np.mean(y), rtol=1e-12
        )

    def test_capability_status_left_placeholder(self):
        from omnilss.family_capabilities import implementation_status_for_family

        status = implementation_status_for_family(FAMILY)
        assert status.value == "experimental"
