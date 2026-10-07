"""BCT parity, part 2: R's closed-form derivatives, links, and jit-safe density.

What D2.12 left open
--------------------
``test_bct_truncation_parity.py`` fixed the *density* (the missing truncation
term) but explicitly left two gaps:

1. the fitting path differentiated the log-density numerically
   (``step=1e-5`` for scores, ``1e-4`` for Hessians) where ``gamlss.dist::BCT``
   supplies closed forms;
2. the links were ``mu: log`` and ``tau: logshiftto2`` against R's
   ``identity`` and ``log``.

D2.12 had already recorded that swapping the link alone made things *worse*
(``non_finite_deviance``, ``mu ~ -1.5e7``), and named the numeric derivatives
as the binding constraint.  So the order is not negotiable: derivatives first,
links second.

The trap in the middle
----------------------
Doing only the first step makes the fit **worse**, which is worth stating
plainly because it looks like a failed fix:

    baseline (numeric scores, old links)   deviance 200.7269   (R: 199.5647)
    analytic scores, old links             deviance 209.5343   <-- worse
    analytic scores + R's links            deviance 199.5666   (rel 9.4e-06)

``logshiftto2`` is ``log(tau - 2)``, which imposes an implicit ``tau > 2``
floor.  With R's closed-form tau score RS drove the eta down to -35.35, i.e.
``tau = exp(-35.35) + 2 = 2.0`` -- parked exactly on that floor, 9.97 deviance
units adrift of R.  R's own ``dBCT`` only requires ``tau > 0``
(``if (any(tau <= 0)) stop(...)``) and its ``tau.link`` is plain ``log``, so the
tau direction can run free toward the ``tau -> infinity`` (Box-Cox Gaussian)
limit, which is where this dataset's deviance actually bottoms out: sweeping
tau gives 209.79 at 2.001, 204.43 at 4, 200.03 at 50, 199.66 at 1e3, and
199.637 as tau -> infinity, against R's reported 199.5647.

What "parity" means for the derivatives
---------------------------------------
``dldm``/``dldd``/``dldv``/``dldt`` are transcribed from R verbatim, not
derived, and two of them are coarse in R itself -- ``d2ldv2`` is literally
``-7*sigma^2/4``, and ``dldt`` differentiates ``log pt`` with a hard-coded 0.01
forward step.  Consequently "our score == derivative of our density" is *not*
an achievable assertion for ``tau``: R's own dldt misses that bar by 1.8e-05,
while our transcribed score sits 2.8e-14 from R.  These tests therefore
measure distance **to live R**, which is the actual contract.
"""

import math
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest


RSCRIPT = Path("C:/Program Files/R/R-4.6.0/bin/x64/Rscript.exe")
# Resolved from __file__, not from the cwd: pytest is routinely run from
# omnilss/, where a relative "tools/r_parity_batch.R" does not exist and Rscript
# fails with an access-violation exit code (3221225477) rather than a clean
# "file not found".
R_BATCH = Path(__file__).resolve().parents[2] / "tools" / "r_parity_batch.R"
Y = np.array([0.5, 1.0, 2.0, 3.0, 3.5, 4.0], dtype=np.float64)
MU = 0.8
SIGMA = 1.5
NU = 0.7
TAU = 4.0
PARAMS = ("mu", "sigma", "nu", "tau")
# The dataset used by the fit-parity and limit tests (seed fixed so R sees the
# same numbers).
Y_FULL = 3.0 + 1.5 * np.random.default_rng(11).normal(size=60)


@pytest.fixture(scope="module")
def bct():
    from omnilss.distributions import BCT

    return BCT()


@pytest.fixture(scope="module")
def r_available():
    if not RSCRIPT.exists():
        pytest.skip(f"Rscript not found at {RSCRIPT}")
    return True


def _r_eval(body: str) -> str:
    """Evaluate ``body`` in an R session with gamlss.dist loaded; return stdout."""
    script = (
        "suppressMessages(library(gamlss))\n"
        "suppressMessages(library(gamlss.dist))\n"
        f"{body}\n"
    )
    with tempfile.NamedTemporaryFile("w", suffix=".R", delete=False, encoding="utf-8") as fh:
        fh.write(script)
        path = fh.name
    try:
        proc = subprocess.run(
            [str(RSCRIPT), path], capture_output=True, text=True, timeout=600
        )
    finally:
        Path(path).unlink(missing_ok=True)
    assert proc.returncode == 0, f"R failed: {proc.stderr[-500:]}"
    return proc.stdout.strip()


# ---------------------------------------------------------------------------
# 1. Links -- transcribed from R, not guessed
# ---------------------------------------------------------------------------


def test_links_match_r(bct):
    """R: ``BCT()$mu.link``/``sigma.link``/``nu.link``/``tau.link`` = identity/log/identity/log."""
    assert bct.links == {
        "mu": "identity",
        "sigma": "log",
        "nu": "identity",
        "tau": "log",
    }


@pytest.mark.parametrize(
    ("param", "eta"),
    [("mu", 2.5), ("sigma", 0.4), ("nu", 1.6), ("tau", 3.0)],
)
def test_link_functions_match_r(bct, param, eta):
    """Compare each link/inverse against R numerically.

    The regression that motivated this was ``tau = logshiftto2`` = ``log(tau-2)``,
    which is wrong as a *link* (R's is plain ``log``) and silently imposes a
    ``tau > 2`` floor on the fit.
    """
    r_expr = {
        "mu": f"BCT()$mu.linkinv({eta})",
        "sigma": f"BCT()$sigma.linkinv({eta})",
        "nu": f"BCT()$nu.linkinv({eta})",
        "tau": f"BCT()$tau.linkinv({eta})",
    }[param]
    r_value = float(_r_eval(f'cat(sprintf("%.17g", {r_expr}))'))
    ours = float(np.asarray(bct.link_inverses[param](np.array([eta])), dtype=np.float64)[0])
    assert abs(ours - r_value) < 1e-12, f"{param}: ours {ours} vs R {r_value}"


def test_tau_link_has_no_two_offset(bct):
    """``logshiftto2`` would give ``log(tau-2)``; R's ``log`` gives ``log(tau)``."""
    eta = np.array([3.0], dtype=np.float64)
    ours = float(np.asarray(bct.link_inverses["tau"](eta), dtype=np.float64)[0])
    assert abs(ours - math.exp(3.0)) < 1e-12
    # The old (wrong) behaviour, kept as an explicit contrast.
    assert abs((math.exp(3.0) + 2.0) - ours) > 1.0


def test_tau_lower_bound_is_zero_not_two(bct):
    """R's ``dBCT`` only requires ``tau > 0``.

    The density must therefore stay finite for ``tau`` well below 2 -- under the
    old ``tau > 2`` clamp every such call silently returned the ``tau = 2`` value.
    """
    from omnilss.dpqr_functions import dBCT

    lo = np.asarray(dBCT(1.0, 1.0, 1.0, 0.5, 0.5, log=True), dtype=np.float64)
    at2 = np.asarray(dBCT(1.0, 1.0, 1.0, 0.5, 2.0, log=True), dtype=np.float64)
    assert np.isfinite(lo).all()
    assert abs(float(lo) - float(at2)) > 1e-6, "tau=0.5 collapsed onto the tau=2 clamp"


# ---------------------------------------------------------------------------
# 2. Analytic scores vs central differences of our own density
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("param", PARAMS)
def test_analytic_score_matches_density_derivative(bct, param):
    """``mu``/``sigma``/``nu`` are exact; ``tau`` inherits R's 0.01 forward step.

    ``tau`` is loosened to 1e-5 because R's own dldt sits 1.8e-5 from the exact
    derivative -- see ``test_analytic_scores_match_live_r`` for the assertion
    that actually settles the question.
    """
    import jax.numpy as jnp

    pos = PARAMS.index(param)
    h = 1e-5

    def log_density(delta: float) -> np.ndarray:
        args = [Y, MU + delta * (pos == 0), SIGMA + delta * (pos == 1),
                NU + delta * (pos == 2), TAU + delta * (pos == 3)]
        return np.log(np.asarray(bct.d(*args), dtype=np.float64))

    fd = (log_density(h) - log_density(-h)) / (2.0 * h)
    analytic = np.asarray(
        bct.score_functions[param](jnp.asarray(Y), MU, SIGMA, NU, TAU),
        dtype=np.float64,
    )
    rel = np.abs(analytic - fd) / np.maximum(np.abs(fd), 1.0)
    # Measured max rel for tau is 1.84e-05, and that entire gap is R's own
    # forward step: our score is 2.8e-14 from R's dldt while this finite
    # difference is 1.8e-05 from it.  The bound is set at 5e-05 to leave room
    # for platform variation in betainc, not because 1.8e-05 would be acceptable
    # in a derivative we were writing ourselves.
    tol = 5e-5 if param == "tau" else 1e-6
    assert np.max(rel) < tol, f"{param}: max rel {np.max(rel):.3e} (tol {tol:.0e})"


# ---------------------------------------------------------------------------
# 3. Analytic Hessians
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("param", PARAMS)
def test_analytic_hessian_is_strictly_negative(bct, param):
    """IRLS needs ``d2l/dtheta2 < 0``; the RS backend floors at -1e-15 itself.

    Also pins that dropping ``_BCT_HESS_FLOOR`` (the per-family clamp that used
    to mask an unstable finite-difference Hessian) did not reintroduce a
    non-negative second derivative.
    """
    import jax.numpy as jnp

    hess = np.asarray(
        bct.hessian_functions[param](jnp.asarray(Y), MU, SIGMA, NU, TAU),
        dtype=np.float64,
    )
    assert np.all(hess < 0.0), f"{param}: min hessian {hess.min():.3e}"


@pytest.mark.parametrize("param", PARAMS)
def test_analytic_hessian_matches_r(bct, param, r_available):
    """``d2ldm2``/``d2ldd2``/``d2ldv2``/``d2ldt2`` against live R.

    R supplies scalars for these (``d2ldm2(mu, sigma, nu, tau)`` etc.), so
    compare the first element only.
    """
    import jax.numpy as jnp

    call = {
        "mu": f"BCT()$d2ldm2({MU}, {SIGMA}, {NU}, {TAU})",
        "sigma": f"BCT()$d2ldd2({SIGMA}, {TAU})",
        "nu": f"BCT()$d2ldv2({SIGMA})",
        "tau": f"BCT()$d2ldt2({TAU})",
    }[param]
    r_value = float(_r_eval(f'cat(sprintf("%.17g", {call}))'))
    # These second derivatives depend only on the parameters, never on y, so the
    # result is a scalar (y broadcasts away).
    ours = float(
        np.asarray(
            bct.hessian_functions[param](jnp.asarray(Y), MU, SIGMA, NU, TAU),
            dtype=np.float64,
        )
    )
    assert abs(ours - r_value) <= 1e-9 * max(abs(r_value), 1.0), (
        f"{param}: ours {ours!r} vs R {r_value!r}"
    )


@pytest.mark.parametrize("param", PARAMS)
def test_analytic_scores_match_live_r(bct, param, r_available):
    """The actual parity contract: our score == R's score.

    For ``tau`` this is the assertion that settles the "which one is right?"
    question -- our transcribed score is ~9 orders of magnitude closer to R
    than a finite difference of our own density is.

    R is queried **one element at a time** on purpose.  These derivatives build
    ``z`` with ``ifelse(nu != 0, ...)``, which collapses to the *scalar* branch
    when ``nu`` has length 1, so a vectorised R call silently returns a single
    answer instead of a vector.  For ``dldv`` at mu=0.8, sigma=1.5, nu=0.7,
    tau=4.0 the two even disagree numerically:

        vectorised   -0.069174347774152767   (the scalar branch, applied to all y)
        per element  0.77975390179463977, 1.6286821513634322, ... (correct)

    The per-element form is what the derivative actually means -- and it is also
    what ``tools/r_parity_batch.R`` does for every other family, so using it here
    keeps BCT consistent with the rest of the parity suite.
    """
    import jax.numpy as jnp

    r_name = {"mu": "dldm", "sigma": "dldd", "nu": "dldv", "tau": "dldt"}[param]
    y_expr = ", ".join(repr(float(v)) for v in Y)
    r_values = [
        float(v)
        for v in _r_eval(
            f"y <- c({y_expr})\n"
            f"f <- BCT()\n"
            f"for (i in seq_along(y)) "
            f"cat(sprintf('%.17g\\n', f${r_name}(y[i], {MU}, {SIGMA}, {NU}, {TAU})))"
        ).splitlines()
        if v.strip()
    ]
    ours = np.asarray(
        bct.score_functions[param](jnp.asarray(Y), MU, SIGMA, NU, TAU),
        dtype=np.float64,
    )
    assert len(r_values) == len(Y), f"expected {len(Y)} R values, got {len(r_values)}"
    diffs = np.abs(ours - np.asarray(r_values, dtype=np.float64))
    scale = np.maximum(np.abs(np.asarray(r_values, dtype=np.float64)), 1e-3)
    assert np.max(diffs / scale) < 1e-9, (
        f"{param}: max rel to R {np.max(diffs / scale):.3e}\n"
        f"  ours = {ours}\n  R    = {r_values}"
    )


# ---------------------------------------------------------------------------
# 4. dBCT is jit- and grad-safe (the CG path depends on it)
# ---------------------------------------------------------------------------


def test_dBCT_is_jittable():
    """scipy's ``t.logcdf`` used to raise ``TracerArrayConversionError`` here.

    ``fitting_cg`` differentiates ``family.d(..., log=True)`` inside ``jit`` to
    score candidate steps, so a density that cannot be traced makes every BCT
    CG cycle fail.  Reachable only once the truncation term stopped calling
    ``np.asarray`` on a traced value.
    """
    import jax
    import jax.numpy as jnp

    from omnilss.dpqr_functions import dBCT

    fn = jax.jit(lambda y, mu, s, n, t: dBCT(y, mu, s, n, t, log=True))
    eager = float(dBCT(2.0, 2.0, 0.7, 1.3, 4.0, log=True))
    traced = float(fn(2.0, 2.0, 0.7, 1.3, 4.0))
    assert abs(eager - traced) < 1e-14


def test_dBCT_gradient_wrt_tau():
    """The incomplete beta has no gradient w.r.t. ``a``/``b``, and ``a = tau/2``.

    Without an explicit tangent rule this raises ``ValueError: Betainc gradient
    with respect to a and b not supported`` -- again fatal to CG.  The custom JVP
    supplies d/darg exactly and d/ddf by central difference (which is what R's
    own ``dldt`` does with a 0.01 step).
    """
    import jax
    import jax.numpy as jnp

    from omnilss.dpqr_functions import dBCT

    analytic = float(jax.grad(lambda t: dBCT(2.0, 2.0, 0.7, 1.3, t, log=True))(4.0))
    h = 1e-6
    fd = (
        float(dBCT(2.0, 2.0, 0.7, 1.3, 4.0 + h, log=True))
        - float(dBCT(2.0, 2.0, 0.7, 1.3, 4.0 - h, log=True))
    ) / (2.0 * h)
    assert abs(analytic - fd) < 1e-7 * max(abs(fd), 1.0)


def test_log_pt_t_matches_scipy():
    """The incomplete-beta construction must stay inside the 12-digit parity."""
    from scipy.stats import t as t_dist

    from omnilss.dpqr_functions import _log_pt_t

    worst = 0.0
    for sigma in (0.15, 0.4, 0.8, 1.5, 3.0):
        for nu in (0.2, 0.7, 1.0, 2.5, 8.0):
            for df in (2.5, 4.0, 10.0, 30.0):
                arg = 1.0 / (sigma * nu)
                ours = float(_log_pt_t(arg, df))
                worst = max(worst, abs(ours - float(t_dist.logcdf(arg, df))))
    assert worst < 5e-13, f"worst |ours - scipy| = {worst:.3e}"


# ---------------------------------------------------------------------------
# 5. End-to-end fit parity against live R
# ---------------------------------------------------------------------------


def test_rs_fit_matches_r_deviance(r_available):
    """RS deviance against live R on identical data, 500 cycles."""
    from omnilss import gamlss
    from omnilss.controls import gamlss_control

    rng = np.random.default_rng(11)
    y = 3.0 + 1.5 * rng.normal(size=60)

    from omnilss.distributions import BCT

    model = gamlss(
        formula="y ~ 1",
        family=BCT(),
        data={"y": y},
        control=gamlss_control(n_cyc=500),
        method="RS",
    )
    ours = float(model.deviance)

    import json

    job = {
        "kind": "fit",
        "data": {"y": [float(v) for v in y]},
        "formula": "y ~ 1",
        "family": "BCT",
        "n_cyc": 500,
    }
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as fh:
        # r_parity_batch.R does lapply(jobs, ...) and needs a JSON *array*.
        json.dump([job, job], fh)
        path = fh.name
    try:
        proc = subprocess.run(
            [str(RSCRIPT), str(R_BATCH), path],
            capture_output=True,
            text=True,
            timeout=900,
        )
    finally:
        Path(path).unlink(missing_ok=True)
    assert proc.returncode == 0, f"R failed: {proc.stderr[-500:]}"
    r_dev = float(json.loads(proc.stdout)[0]["deviance"])

    assert abs(ours - r_dev) / r_dev < 1e-4, (
        f"RS deviance {ours:.6f} vs R {r_dev:.6f} "
        f"(rel {abs(ours - r_dev) / r_dev:.3e})"
    )


def test_tau_reaches_the_gaussian_limit_in_the_right_direction(bct):
    """Sanity-check the mechanism behind the deviance parity.

    On this dataset the deviance decreases monotonically as ``tau`` grows and
    bottoms out in the ``tau -> infinity`` (Box-Cox Gaussian) limit, which is
    where R lands too.  A link that cannot express large ``tau`` (the old
    ``logshiftto2``, floored at 2) therefore cannot reach the optimum -- this is
    the geometric reason the fix works, not a coincidence.
    """
    import jax.numpy as jnp

    def dev(tau: float) -> float:
        return float(
            jnp.sum(
                jnp.asarray(
                    bct.g_dev_inc(jnp.asarray(Y_FULL), 2.75, 0.55, 1.45, tau),
                    dtype=np.float64,
                )
            )
        )

    assert dev(50.0) < dev(4.0) < dev(2.001)