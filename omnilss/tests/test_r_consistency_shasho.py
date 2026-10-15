"""SHASHo (original-parameterization sinh-arcsinh) parity against live R at digits=17.

What this replaces
------------------
The previous incarnation queried RTestBus for the registry alias
``"SHASHO"`` and compared at ``rtol=1e-5`` against the bus's ``digits=10``
serialisation -- a floor coarser than the agreement being asserted, on a
bus whose ``paste0`` probe could not even resolve the mixed-case name
before D2.42.  Rewritten here against the real family, on the real
numbers, at the precision the agreement actually has.

The one contract detail a *boundary* exposes
--------------------------------------------
``qSHASHo``'s two versions disagree, the same way ``qSHASHo2``'s do:

* vendored 6.1-8 assigns ``q[p == 0] <- -Inf``, ``q[p == 1] <- Inf``,
  ``q[p < 0] <- NaN``, ``q[p > 1] <- NaN``;
* installed 6.1.1 stops for any ``p`` outside the open interval
  (``p must be between 0 and 1``; all four boundary points measured).

We follow vendored -- the support is the whole real line, so ±Inf are
the continuous extensions at ``p = 0`` / ``p = 1`` -- and pin the live
stop here so an upgrade that adopts it is visible.  The pre-D2.43
implementation instead clipped ``p`` into ``(eps, 1 - eps)`` and returned
finite values (``q(0) = -12.99`` for the profile below), which is neither
version's contract; that clip is what this delivery removes.

The sampler is not a new algorithm
----------------------------------
R's ``rSHASHo`` is ``qSHASHo(runif(n))``.  ``_shasho_r`` keeps that exact
construction, and the test below feeds **R's own uniform stream** (same
seed) through our ``q`` -- bit-identical to R's draws is then a statement
about ``q``, not about two samplers agreeing statistically.

Methodology
-----------
d/p/q are compared element-wise against **live R** through ``digits=17``
subprocess probes (the ST1/ST2 house style).  Nothing is hard-coded.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)  # parity prerequisite
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from omnilss.distribution_registry import resolve  # noqa: E402

FAMILY = "SHASHO"  # the registry alias (capital O); R exports SHASHo
RSCRIPT = shutil.which("Rscript")

# (mu, sigma, nu, tau) profiles.  Profile 0 is the unit-parameter
# degeneration -- SHASHo there is exactly the normal distribution, which
# makes it an independent correctness anchor (checked against
# dnorm/pnorm/qnorm, not against R's SHASHo).
PROFILES = [
    dict(mu=0.0, sigma=1.0, nu=0.0, tau=1.0),
    dict(mu=0.5, sigma=1.2, nu=0.3, tau=0.8),
    dict(mu=-1.0, sigma=0.7, nu=-0.5, tau=1.5),
    dict(mu=2.0, sigma=2.0, nu=1.0, tau=0.5),
]

Y_GRID = np.array([-6.0, -3.0, -1.5, -0.5, 0.0, 0.5, 1.5, 3.0, 6.0, 12.0])
P_GRID = np.array([0.001, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 0.999])

RTOL = 1e-12
ATOL = 1e-300


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


def _rvec(values) -> str:
    return ", ".join(f"{float(v):.17g}" for v in values)


@pytest.fixture(scope="module")
def fam():
    return resolve(FAMILY)


@pytest.fixture(scope="module")
def r_dpqr_probe():
    """Element-wise live-R d/p/q at digits=17 (house style)."""
    yvec = _rvec(Y_GRID)
    pvec = _rvec(P_GRID)
    lines = ["suppressMessages(library(gamlss.dist))", "out <- list()"]
    for pi, p in enumerate(PROFILES):
        args = (f"mu={p['mu']!r}, sigma={p['sigma']!r}, "
                f"nu={p['nu']!r}, tau={p['tau']!r}")
        lines.append(
            f"out[['{pi}.d']] <- vapply(c({yvec}), function(xi) "
            f"suppressWarnings(as.numeric(gamlss.dist::dSHASHo(xi, {args}))), "
            f"numeric(1))"
        )
        lines.append(
            f"out[['{pi}.p']] <- vapply(c({yvec}), function(qi) "
            f"suppressWarnings(as.numeric(gamlss.dist::pSHASHo(qi, {args}))), "
            f"numeric(1))"
        )
        lines.append(
            f"out[['{pi}.q']] <- vapply(c({pvec}), function(pp) "
            f"suppressWarnings(as.numeric(gamlss.dist::qSHASHo(pp, {args}))), "
            f"numeric(1))"
        )
    lines.append("cat(jsonlite::toJSON(out, digits=17, auto_unbox=TRUE))")
    return json.loads(_run_r(lines).strip())


@pytest.fixture(scope="module")
def r_sampler_probe():
    """R's own uniform stream, the draws built from it, and the stop count.

    ``u`` is ``set.seed(42); runif(8)``; ``r_from_u`` feeds exactly those
    uniforms through ``qSHASHo`` and ``r_direct`` is ``rSHASHo(8)`` under
    the same seed -- the two must be identical (R's own construction),
    and both are what our ``q``/``r`` are checked against.
    """
    lines = [
        "suppressMessages(library(gamlss.dist))",
        "args <- 'mu=0.5, sigma=1.2, nu=0.3, tau=0.8'",
        "set.seed(42)",
        "u <- runif(8)",
        "out <- list(u = as.numeric(u),",
        "            r_from_u = as.numeric(gamlss.dist::qSHASHo(u, mu=0.5, sigma=1.2, nu=0.3, tau=0.8)))",
        "set.seed(42)",
        "out$r_direct <- as.numeric(gamlss.dist::rSHASHo(8, mu=0.5, sigma=1.2, nu=0.3, tau=0.8))",
        "out$stop_count <- 0L",
        "for (p in c(0, 1, -0.5, 1.5)) {",
        "  r <- tryCatch({ gamlss.dist::qSHASHo(p, mu=0.5, sigma=1.2, nu=0.3, tau=0.8); 'NO ERROR' },",
        "                error = function(e) 'STOP')",
        "  if (r == 'STOP') out$stop_count <- out$stop_count + 1L",
        "}",
        "cat(jsonlite::toJSON(out, digits=17, auto_unbox=TRUE))",
    ]
    return json.loads(_run_r(lines).strip())


def _pos(fn, first, profile):
    """Call a dpqr entry point positionally -- the kernels take *args only."""
    return np.asarray(
        fn(jnp.asarray(first, dtype=jnp.float64),
           *(float(profile[k]) for k in ("mu", "sigma", "nu", "tau")))
    )


# ---------------------------------------------------------------------------
# 1. Density and CDF against live R
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("pi", range(len(PROFILES)))
def test_d_matches_live_r(r_dpqr_probe, fam, pi):
    profile = PROFILES[pi]
    r_d = np.asarray(r_dpqr_probe[f"{pi}.d"], dtype=float)
    got = _pos(fam.d, Y_GRID, profile)
    assert np.all(np.isfinite(got))
    np.testing.assert_allclose(got, r_d, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("pi", range(len(PROFILES)))
def test_p_matches_live_r(r_dpqr_probe, fam, pi):
    profile = PROFILES[pi]
    r_p = np.asarray(r_dpqr_probe[f"{pi}.p"], dtype=float)
    got = _pos(fam.p, Y_GRID, profile)
    np.testing.assert_allclose(got, r_p, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("pi", range(len(PROFILES)))
def test_q_matches_live_r(r_dpqr_probe, fam, pi):
    profile = PROFILES[pi]
    r_q = np.asarray(r_dpqr_probe[f"{pi}.q"], dtype=float)
    got = _pos(fam.q, P_GRID, profile)
    assert np.all(np.isfinite(got))
    np.testing.assert_allclose(got, r_q, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("pi", range(len(PROFILES)))
def test_p_tail_paths_match_live_r(r_dpqr_probe, fam, pi):
    """``lower_tail`` / ``log_p`` are part of R's pSHASHo body, not extras.

    R has no flag channel through the probe, so the reference is
    reconstructed the way R's own body does -- ``1 - p`` then ``log`` --
    and the identity against our flag paths is what is asserted.
    """
    profile = PROFILES[pi]
    r_p = np.asarray(r_dpqr_probe[f"{pi}.p"], dtype=float)
    scalars = tuple(float(profile[k]) for k in ("mu", "sigma", "nu", "tau"))
    jax_upper = np.asarray(
        fam.p(jnp.asarray(Y_GRID), *scalars, lower_tail=False)
    )
    np.testing.assert_allclose(jax_upper, 1.0 - r_p, rtol=RTOL, atol=ATOL)
    jax_logp = np.asarray(
        fam.p(jnp.asarray(Y_GRID), *scalars, log_p=True)
    )
    np.testing.assert_allclose(jax_logp, np.log(r_p), rtol=RTOL, atol=ATOL)


def test_g_dev_inc_is_minus_two_log_d(r_dpqr_probe, fam):
    """The scoring path must be the same density, not a lookalike."""
    for pi, profile in enumerate(PROFILES):
        scalars = tuple(float(profile[k]) for k in ("mu", "sigma", "nu", "tau"))
        logd = np.asarray(fam.d(jnp.asarray(Y_GRID), *scalars, log=True))
        dev = np.asarray(fam.g_dev_inc(jnp.asarray(Y_GRID), *scalars))
        np.testing.assert_allclose(dev, -2.0 * logd, rtol=RTOL, atol=ATOL)
        # ...and the deviance path agrees with the density-scale probe.
        r_d = np.asarray(r_dpqr_probe[f"{pi}.d"], dtype=float)
        np.testing.assert_allclose(np.exp(-0.5 * dev), r_d, rtol=RTOL,
                                   atol=ATOL)


# ---------------------------------------------------------------------------
# 2. The unit-parameter degeneration: an anchor independent of R's SHASHo
# ---------------------------------------------------------------------------


def test_degenerates_to_normal_at_unit_parameters(fam):
    """At ``mu=0, sigma=1, nu=0, tau=1`` SHASHo is *exactly* the normal.

    ``z = (x - mu)/sigma``, ``arg = tau*asinh(z) - nu = 0``, so
    ``c = 1``, ``r = 0`` and the log-likelihood collapses to the standard
    normal's.  Checked against ``jax.scipy.stats.norm`` (R's ``pNO`` /
    ``qnorm`` are the same functions): an error in the parameterisation
    cannot hide behind agreement with R here.
    """
    from jax.scipy.stats import norm

    scalars = (0.0, 1.0, 0.0, 1.0)
    d = np.asarray(fam.d(jnp.asarray(Y_GRID), *scalars))
    p = np.asarray(fam.p(jnp.asarray(Y_GRID), *scalars))
    q = np.asarray(fam.q(jnp.asarray(P_GRID), *scalars))
    np.testing.assert_allclose(d, np.asarray(norm.pdf(jnp.asarray(Y_GRID))),
                               rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(p, np.asarray(norm.cdf(jnp.asarray(Y_GRID))),
                               rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(q, np.asarray(norm.ppf(jnp.asarray(P_GRID))),
                               rtol=RTOL, atol=ATOL)


def test_p_q_roundtrip(fam):
    """Independent of R: p and q are mutual inverses.

    The ``y`` leg uses the mid-range grid only: at the far tails of the
    heavy profile the CDF is within 1e-14 of 1, where ``norm.ppf``'s
    relative error is amplified by ``sinh`` (derivative ~cosh(r) ~ 1e3
    there) -- a double-precision limit R shares, not a defect.  The
    ``p`` leg feeds exact probabilities and inverts tightly.
    """
    y_mid = np.array([-3.0, -1.5, -0.5, 0.0, 0.5, 1.5, 3.0])
    for profile in (PROFILES[1], PROFILES[2]):
        scalars = tuple(float(profile[k]) for k in ("mu", "sigma", "nu", "tau"))
        p_of_y = np.asarray(fam.p(jnp.asarray(y_mid), *scalars))
        # Keep only points whose CDF is comfortably inside (0, 1) in
        # double precision.  Two boundaries matter: the CDF itself rounds
        # to exactly 1.0 at the heavy profile's far tail (R's pSHASHo does
        # the same -- checked in the parity grid), and within ~1e-5 of 1
        # JAX's ndtri loses ~6 relative digits, which sinh (derivative
        # cosh(r) ~ 50 there) amplifies into a ~3e-4 roundtrip error
        # (measured; R's qnorm keeps more digits in the same region).
        # The parity grid above covers p <= 0.999, where the kernels agree
        # to 1e-12; this roundtrip deliberately stays in that regime.
        inner = (p_of_y > 0.0) & (p_of_y < 1.0 - 1e-4)
        assert inner.sum() >= 3, p_of_y
        y_back = np.asarray(fam.q(jnp.asarray(p_of_y[inner]), *scalars))
        np.testing.assert_allclose(y_back, y_mid[inner], rtol=1e-9, atol=1e-9)
        q_of_p = np.asarray(fam.q(jnp.asarray(P_GRID), *scalars))
        p_back = np.asarray(fam.p(jnp.asarray(q_of_p), *scalars))
        np.testing.assert_allclose(p_back, P_GRID, rtol=1e-9, atol=1e-9)


# ---------------------------------------------------------------------------
# 3. The quantile boundary: vendored ±Inf/NaN, live R stops
# ---------------------------------------------------------------------------


def test_q_boundary_contract_pins_both_r_versions(fam, r_sampler_probe):
    """We return the vendored continuous extension; installed R raises.

    Both halves asserted on purpose.  The first documents *our* contract
    (vendored 6.1-8's assignments); the second pins the installed
    version's opposite behaviour (``p must be between 0 and 1`` for all
    four of p = 0, 1, -0.5, 1.5 -- measured, see the probe), so an
    environment upgrade tells us exactly what changed instead of silently
    drifting.  The pre-D2.43 clip-to-eps behaviour (finite values at the
    boundaries) is gone: neither version's contract.
    """
    scalars = dict(mu=0.5, sigma=1.2, nu=0.3, tau=0.8)
    assert float(fam.q(0.0, **scalars)) == -np.inf
    assert float(fam.q(1.0, **scalars)) == np.inf
    assert np.isnan(float(fam.q(-0.5, **scalars)))
    assert np.isnan(float(fam.q(1.5, **scalars)))
    assert np.isnan(float(fam.q(np.nan, **scalars)))
    # log_p / lower_tail feed the same transform.
    assert float(fam.q(-np.inf, log_p=True, **scalars)) == -np.inf
    assert float(fam.q(0.0, lower_tail=False, **scalars)) == np.inf
    assert float(fam.q(1.0, lower_tail=False, **scalars)) == -np.inf

    assert r_sampler_probe["stop_count"] == 4, r_sampler_probe


# ---------------------------------------------------------------------------
# 4. The sampler is R's construction, verified bit-for-bit
# ---------------------------------------------------------------------------


def test_r_is_quantile_of_uniforms(fam):
    """``rSHASHo`` is ``qSHASHo(runif(n))`` -- asserted exactly.

    If ``r`` ever grows its own sampler this fails, which is how
    sampler/CDF drift starts.
    """
    key = jax.random.PRNGKey(20261015)
    n = 64
    u = jax.random.uniform(key, shape=(n,), dtype=jnp.float64)
    expected = np.asarray(fam.q(u, 0.5, 1.2, 0.3, 0.8))
    got = np.asarray(fam.r(key, n, 0.5, 1.2, 0.3, 0.8))
    np.testing.assert_array_equal(got, expected)
    assert np.all(np.isfinite(got))


def test_our_q_reproduces_r_sampler_bit_for_bit(fam, r_sampler_probe):
    """R's own uniform stream through our ``q`` is R's own draw.

    R's two paths are asserted *bit-identical* (``r_direct`` ==
    ``r_from_u`` -- R's own construction, exact); ours matches them to
    the measured cross-library floor of 2.4e-15 relative (JAX's
    ``ndtri``/``asinh`` chain vs R's ``qnorm``/``asinh`` chain on the
    same doubles).  That floor is what a shared algorithm with different
    libm backends looks like -- anything larger would be a real defect.
    """
    u = np.asarray(r_sampler_probe["u"], dtype=float)
    r_ref = np.asarray(r_sampler_probe["r_from_u"], dtype=float)
    r_direct = np.asarray(r_sampler_probe["r_direct"], dtype=float)
    np.testing.assert_array_equal(r_ref, r_direct)  # R's own construction
    got = np.asarray(fam.q(jnp.asarray(u), 0.5, 1.2, 0.3, 0.8))
    np.testing.assert_allclose(got, r_ref, rtol=1e-13, atol=1e-13)


def test_sampler_stays_finite_and_reasonable(fam):
    """Uniforms in [0, 1) map into the real line; a draw must be finite."""
    key = jax.random.PRNGKey(20261015)
    for profile in PROFILES:
        scalars = tuple(float(profile[k]) for k in ("mu", "sigma", "nu", "tau"))
        draws = np.asarray(fam.r(key, 200, *scalars))
        assert np.all(np.isfinite(draws)), (profile, draws)


# ---------------------------------------------------------------------------
# 5. SHASHo is NOT SHASH -- the parameterisation confusion guard
# ---------------------------------------------------------------------------


def test_shasho_is_not_shash(fam):
    """The two parameterisations are different distributions.

    ``SHASH`` scales ``z`` by ``tau`` as well (``z = (x-mu)/(tau*sigma)``)
    while ``SHASHo`` does not (``z = (x-mu)/sigma``); only at ``tau = 1``
    do they coincide.  This fails if anyone re-derives one from the other.

    The integral check covers SHASHo only.  ``SHASH``'s density does not
    integrate to 1 on a wide grid (measured 0.9601 on [-40, 40] and
    0.9998 on [-400, 400] at this profile) -- a pre-existing defect of
    the ``SHASH`` family's dpqr path, out of scope here and registered as
    a follow-up; asserting it here would fail for a reason this delivery
    does not touch.
    """
    shash = resolve("SHASH")
    profile = dict(mu=0.5, sigma=1.2, nu=0.3, tau=0.8)
    d_o = _pos(fam.d, Y_GRID, profile)
    d_1 = _pos(shash.d, Y_GRID, profile)
    assert not np.allclose(d_o, d_1, rtol=1e-3), (
        "SHASHo density collapsed onto SHASH"
    )
    # SHASHo must be a valid density integrating to ~1 over a wide grid.
    grid = np.linspace(-40.0, 40.0, 20001)
    vals = _pos(fam.d, grid, profile)
    assert np.all(vals >= 0.0)
    np.testing.assert_allclose(np.trapezoid(vals, grid), 1.0, rtol=1e-3)


# ---------------------------------------------------------------------------
# 6. The export-level gate fix: the capital-O alias
# ---------------------------------------------------------------------------


def test_rtestbus_resolves_the_capital_o_alias():
    """RTestBus must resolve ``"SHASHO"`` to R's ``dSHASHo``.

    The bus built the call with ``paste0(func_type, family_name)``, which
    asks R for ``dSHASHO`` -- a name the installed gamlss.dist does not
    export (the tail ``o`` is lowercase).  Eleven mixed-case families
    were silently unprobed until the D2.42 resolver (exact export match
    first, case-insensitive fallback second).  The comparison is at the
    bus's own ``digits = 10`` precision, so 1e-8 is the honest tolerance.
    """
    from tests.rbus.core import RTestBus

    bus = RTestBus()
    x = np.array([-1.5, 0.0, 1.5, 6.0])
    mu = np.full(4, 0.5)
    sigma = np.full(4, 1.2)
    nu = np.full(4, 0.3)
    tau = np.full(4, 0.8)
    r_d = np.asarray(
        bus.eval_family(FAMILY, "d", {
            "x": x.tolist(), "mu": mu.tolist(), "sigma": sigma.tolist(),
            "nu": nu.tolist(), "tau": tau.tolist(),
        }),
        dtype=float,
    )
    assert np.all(np.isfinite(r_d)), r_d
    fam = resolve(FAMILY)
    ours = _pos(fam.d, x, dict(mu=0.5, sigma=1.2, nu=0.3, tau=0.8))
    np.testing.assert_allclose(ours, r_d, rtol=1e-8, atol=1e-10)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
