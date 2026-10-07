"""BCTo / BCPEo parity: the two families that could not be fitted at all.

The defect
----------
``BCTo`` and ``BCPEo`` raised on **every** RS cycle and so were unusable:

    BCTo    ValueError: Betainc gradient with respect to a and b not supported.
    BCPEo   NotImplementedError: Differentiation rule for 'igamma_grad_a'
            not implemented.

Both come from the same place.  Each family carried its own hand-rolled copy
of the log-density, and each copy called a special function that JAX cannot
differentiate twice:

* ``_bcto_log_pdf`` reached ``betainc(tau/2, 1/2, x)``.  ``betainc`` has no
  derivative in ``a``/``b``, and ``a`` *is* ``df/2``.  D2.13 added a
  ``custom_jvp`` so the **first** derivative works -- but the RS Hessian
  differentiates the JVP again, which re-enters ``betainc`` and raises.
* ``_bcpeo_log_pdf`` reached ``gammainc(1/tau, s)``; differentiating in the
  first argument raises ``igamma_grad_a``.

``custom_jvp`` cannot fix this class of problem: it patches one derivative,
and the second derivative re-enters the special function.  The fix has to
remove the AD path for those parameters.

The fix
-------
1. R's ``dBCTo`` body is **byte-identical** to ``dBCPE``'s / ``dBCT``'s
   (``identical(deparse(body(dBCTo)), deparse(body(dBCT)))`` is ``TRUE`` in
   ``gamlss.dist``) -- the ``o`` families differ only in ``mu.link``.  So both
   log-pdfs now delegate to ``dpqr_functions.dBCT`` / ``dBCPE``: one family,
   one density, and the ``_log_pt_t`` ``custom_jvp`` from D2.13 comes for free.
2. ``build_ad_family`` grew ``score_overrides`` / ``hessian_overrides``, and
   both families hand in R's closed-form derivatives borrowed from ``BCT()`` /
   ``BCPE()``.  These are not merely a workaround: for this cluster R's
   formulas *are* the parity target, and D2.15 showed that "improving" R's
   rough second derivatives makes parity worse.

Measured parity (40 strictly-positive points, ``y ~ x``, ``c_crit=0.001``
matching R's default -- see the convergence note below)
    BCTo   ours 106.819450   R 106.822650   rel 3.00e-05
    BCPEo  ours 106.764103   R 107.149897   rel 3.60e-03

Convergence criterion: absolute, not relative
---------------------------------------------
R's RS loop is ``while (abs(G.dev.old - G.dev) > c.crit && iter < n.cyc)`` --
a purely **absolute** deviance change, default ``c.crit = 0.001``.  Parity
comparisons must use the same criterion.

Note that ``gamlss.control`` has **no** ``delta`` argument; its signature ends
at ``save, ...``, so ``gamlss.control(delta=1e-8)`` is swallowed by ``...``
and R silently runs at the default.  An earlier round of this diagnosis
paired that silent no-op with our own ``c_crit=1e-8`` and concluded that our
optimiser had found lower deviance in five of six families.  Both readings were
artifacts of comparing two different stopping rules.

At ``c_crit = 1e-8`` our deviance does come out lower than R's (BCT
``y ~ 1``: 107.972605 after 466 cycles, against 108.107153 at ``c_crit=1e-3``
and R's 108.106595 after 111), and that is **not** a defect:

* evaluating R's coefficients on our density reproduces R's reported deviance
  to float precision (rel 7.1e-11; the residual is summation order, since R
  computes ``sum(w * G.dev.incr)``), so the objectives are identical;
* the per-parameter score sums at R's point are far from zero
  (BCT: mu -0.145, sigma +0.116, nu +0.028 at ``y ~ x``), so **R's reported
  fit is not a stationary point** -- R stopped early;
* the ``tau`` direction is numerically flat there (max |score| 3.9e-9), which
  is the physical reason R's own trace was still descending at its stopping
  iteration.

``test_rs_deviance_beats_r_only_by_going_further`` pins that reasoning so it
is not silently "fixed" later.
"""

import json
import subprocess
import tempfile
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from omnilss.controls import GAMLSSControl
from omnilss.fitting import gamlss

RSCRIPT = Path("C:/Program Files/R/R-4.6.0/bin/x64/Rscript.exe")

# Byte-identical to the D2.16 dataset so the two modules can be compared.
FIT_Y = np.abs(2.0 + 1.2 * np.random.default_rng(2026).normal(size=40)) + 0.15
FIT_X = np.arange(len(FIT_Y), dtype=float)

PROFILES = {
    "BCTo": {"mu": 2.4, "sigma": 0.7, "nu": 1.3, "tau": 3.5},
    "BCPEo": {"mu": 2.4, "sigma": 0.7, "nu": 1.3, "tau": 2.6},
}

# R's default is c.crit = 0.001 and its test is on the ABSOLUTE deviance
# change.  Using 1e-8 on our side while R stays at 0.001 compares two
# different stopping rules -- see the module docstring.
R_CRITER = 0.001

R_DEV = {
    "BCTo": 106.82264971300128,
    "BCPEo": 107.14989680868845,
}


# ---------------------------------------------------------------------------
# 1. The defect itself: both families must fit at all.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["BCTo", "BCPEo"])
def test_o_variant_fits_without_raising(name):
    """Regression: every RS cycle used to raise inside a special function.

    ``pytest.raises`` used to be the *expected* outcome here, which is how a
    completely unusable family sat in the tree for this long.  If this test
    starts raising again, the log-density has re-grown its own copy of the
    Box-Cox kernel instead of delegating.
    """
    fit = gamlss(
        formula="y ~ x",
        family=name,
        data={"y": FIT_Y, "x": FIT_X},
        control=GAMLSSControl(n_cyc=200, c_crit=R_CRITER, trace=False),
    )
    assert np.isfinite(fit.deviance)
    assert fit.convergence_status() is True, fit.termination_reason
    assert fit.iter > 0


@pytest.mark.parametrize("name", ["BCTo", "BCPEo"])
def test_all_fitted_values_finite(name):
    """The old failures produced NaN/Inf deviance rather than a clean error.

    A family that "fits" with non-finite parameters is worse than one that
    raises, so finiteness is asserted separately from the no-raise check.
    """
    fit = gamlss(
        formula="y ~ x",
        family=name,
        data={"y": FIT_Y, "x": FIT_X},
        control=GAMLSSControl(n_cyc=200, c_crit=R_CRITER, trace=False),
    )
    for par in fit.parameters:
        v = np.asarray(fit.fitted_values[par], dtype=float)
        assert v.shape == FIT_Y.shape, (par, v.shape)
        assert np.all(np.isfinite(v)), (par, v)


# ---------------------------------------------------------------------------
# 2. The delegating log-pdf is the same density, not a lookalike.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name,base,d_name", [
    ("BCTo", "BCT", "dBCT"),
    ("BCPEo", "BCPE", "dBCPE"),
])
def test_o_log_pdf_is_bit_identical_to_base_family(name, base, d_name):
    """``_bcto_log_pdf`` now *is* ``dBCT``; assert the identity, do not re-derive it.

    R's ``dBCTo`` body is byte-identical to ``dBCT``'s, so any difference here
    means we have introduced a second, drifting implementation -- exactly the
    bug this change removes.
    """


    from omnilss import resolve_family
    from omnilss.distributions_b14 import BCPEo, BCTo

    p = PROFILES[name]
    y = jnp.asarray(np.linspace(0.4, 4.0, 15))
    args = (y, p["mu"], p["sigma"], p["nu"], p["tau"])

    o_log = float(jnp.asarray(resolve_family(name).d(*args, log=True))[3])
    b_log = float(jnp.asarray(resolve_family(base).d(*args, log=True))[3])
    assert o_log == b_log

    # And the module-level helper the RS path actually calls.
    helper = BCTo() if name == "BCTo" else BCPEo()
    from omnilss.distributions_b14 import _bcto_log_pdf, _bcpeo_log_pdf

    fn = _bcto_log_pdf if name == "BCTo" else _bcpeo_log_pdf
    assert float(jnp.asarray(fn(*args))[3]) == b_log
    assert helper is not None


@pytest.mark.parametrize("name,base", [("BCTo", "BCT"), ("BCPEo", "BCPE")])
def test_o_family_derivatives_are_r_closed_forms_not_ad(name, base):
    """Derivatives must be R's transcriptions, shared with the base family.

    Matching by value catches a subtler regression than "it fits": if someone
    re-enables AD for ``tau`` the fit still works (the first derivative is
    patchable) but the numbers drift away from R.

    Identity (``is``) cannot be used -- ``BCT()`` is a factory, so each call
    builds fresh closures.  ``__qualname__`` is the stable discriminator: it
    still names the base family's enclosing scope (``BCT.<locals>._score``)
    rather than an AD closure built inside ``build_ad_family``.
    """
    from omnilss import resolve_family

    o = resolve_family(name)
    b = resolve_family(base)
    assert set(o.score_functions) == set(b.score_functions)
    assert set(o.hessian_functions) == set(b.hessian_functions)

    p = PROFILES[name]
    args = (FIT_Y, p["mu"], p["sigma"], p["nu"], p["tau"])
    for par in ("mu", "sigma", "nu", "tau"):
        assert par in o.score_functions
        # Same numeric function as the base family's R transcription.
        np.testing.assert_allclose(
            np.asarray(o.score_functions[par](*args), dtype=float),
            np.asarray(b.score_functions[par](*args), dtype=float),
            rtol=1e-14, atol=1e-300,
        )
        np.testing.assert_allclose(
            np.asarray(o.hessian_functions[par](*args), dtype=float),
            np.asarray(b.hessian_functions[par](*args), dtype=float),
            rtol=1e-14, atol=1e-300,
        )
        # ... and it came from the base family, not from a fresh AD closure.
        assert base in o.score_functions[par].__qualname__, par
        assert base in o.hessian_functions[par].__qualname__, par


def test_ad_family_still_derives_where_no_override_is_given():
    """``score_overrides`` must be additive, not a switch that disables AD.

    A family with no overrides must still get genuinely AD-derived functions;
    otherwise the new keyword would silently freeze every AD family onto
    whatever its first call happened to produce.
    """


    from omnilss.ad import build_ad_family
    from omnilss.distributions_b14 import BCToFamily

    f = build_ad_family(
        family_class=BCToFamily,
        name="D217_SANITY",
        parameters=("mu", "sigma"),
        log_pdf_func=lambda y, mu, sigma: -0.5 * ((y - mu) / sigma) ** 2
        - jnp.log(sigma),
    )
    assert set(f.score_functions) == {"mu", "sigma"}
    y = jnp.asarray([1.0, 2.0, 3.0])
    mu = jnp.asarray([1.5, 2.5, 3.5])
    sigma = jnp.asarray([1.0, 1.2, 0.8])
    # d/dmu of -0.5((y-mu)/sigma)^2 - log sigma is (y-mu)/sigma^2
    got = f.score_functions["mu"](y, mu, sigma)
    assert np.allclose(np.asarray(got), np.asarray((y - mu) / sigma**2), rtol=1e-12)


def test_ad_family_overrides_bypass_the_hessian_trace():
    """The warm-up must skip overridden parameters.

    The warm-up traces ``_vmap_hess[par]`` for *every* parameter even when the
    result is discarded.  For a family whose log-density touches ``betainc``
    that trace is exactly what raises, so skipping is the whole fix -- if it
    regresses, ``BCTo``/``BCPEo`` go back to raising during construction.

    ``_log_pt_t`` is the **real** primitive, not a synthetic stand-in, because
    the failure mode is specific: D2.13's ``custom_jvp`` makes ``grad`` succeed
    (0.0845 at tau=2) while differentiating that JVP again re-enters
    ``betainc`` and raises.  A cheaper substitute was tried first --
    ``jnp.round``, whose first derivative is 0 and whose second derivative is
    *also* 0 rather than an error -- and it reproduces nothing, so it would
    have made this test pass for the wrong reason.  Bare ``betainc`` fails in
    the other direction: its *first* derivative already raises, which would
    not distinguish "custom_jvp helps grad1" from "nothing works here".

    ``jnp.sum`` is needed around ``_log_pt_t`` because ``z`` is a fixed
    length-3 vector: the AD path vmaps over the varying argument, so a
    length-3 return would trip "Gradient only defined for scalar-output
    functions" long before reaching ``betainc``.
    """
    import jax

    from omnilss.ad import build_ad_family
    from omnilss.distributions_b14 import BCToFamily
    from omnilss.dpqr_functions import _log_pt_t

    z = jnp.asarray([1.0, 1.2, 0.8])

    def log_pdf(y, mu, sigma, tau):
        # `tau` reaches betainc(a=tau/2, b=1/2, ...): grad1 works, grad2 raises.
        return (
            -0.5 * ((y - mu) / sigma) ** 2
            - jnp.log(sigma)
            + jnp.sum(_log_pt_t(z, tau))
        )

    def _only_tau(a):
        return jnp.sum(_log_pt_t(z, a))

    # Confirm the premise against the real primitive.
    assert float(jax.grad(_only_tau)(jnp.asarray(2.0))) != 0.0
    with pytest.raises(ValueError, match="Betainc gradient"):
        jax.grad(lambda a: jnp.sum(jax.grad(_only_tau)(a)))(jnp.asarray(2.0))

    def ok_score(y, mu, sigma, tau):
        return (y - mu) / sigma**2

    built = build_ad_family(
        family_class=BCToFamily,
        name="D217_OVERRIDE",
        parameters=("mu", "sigma", "tau"),
        log_pdf_func=log_pdf,
        score_overrides={"tau": ok_score},
        hessian_overrides={
            "tau": lambda y, mu, sigma, tau: -jnp.ones_like(y) / sigma**2
        },
    )
    # Only `tau` is overridden, so `mu`/`sigma` must still be AD-derived.
    assert set(built.score_functions) == {"mu", "sigma", "tau"}
    assert built.score_functions["tau"] is ok_score
    assert built.score_functions["mu"] is not ok_score

    y = jnp.asarray([1.0, 2.0, 3.0])
    mu = jnp.asarray([1.5, 2.5, 3.5])
    sigma = jnp.asarray([1.0, 1.2, 0.8])
    tau = jnp.asarray([2.0, 2.0, 2.0])
    assert np.allclose(
        np.asarray(built.score_functions["tau"](y, mu, sigma, tau)),
        np.asarray((y - mu) / sigma**2), rtol=1e-12,
    )
    # The AD path for the un-overridden parameters is intact and correct.
    assert np.allclose(
        np.asarray(built.score_functions["mu"](y, mu, sigma, tau)),
        np.asarray((y - mu) / sigma**2), rtol=1e-12,
    )


def test_ad_family_without_override_still_raises_on_betainc():
    """The premise above is not a contrivance: unpatched AD really does raise.

    Without this, ``test_ad_family_overrides_bypass_the_hessian_trace`` could
    pass because ``score_overrides`` is ignored entirely.  Built with no
    overrides, the same family must still produce a tau Hessian that raises --
    that is the pre-D2.17 state, and it is what the fix exists to remove.
    """
    import jax

    from omnilss.ad import build_ad_family
    from omnilss.distributions_b14 import BCToFamily
    from omnilss.dpqr_functions import _log_pt_t

    z = jnp.asarray([1.0, 1.2, 0.8])

    def log_pdf(y, mu, sigma, tau):
        return (
            -0.5 * ((y - mu) / sigma) ** 2
            - jnp.log(sigma)
            + jnp.sum(_log_pt_t(z, tau))
        )

    # The AD Hessian is exactly grad-of-grad.
    with pytest.raises(ValueError, match="Betainc gradient"):
        jax.grad(lambda a: jnp.sum(jax.grad(
            lambda b: jnp.sum(_log_pt_t(z, b))
        )(a)))(jnp.asarray(2.0))

    # Construction still succeeds -- the warm-up is wrapped in try/except --
    # but the resulting tau Hessian is unusable when actually called.
    built = build_ad_family(
        family_class=BCToFamily,
        name="D217_NO_OVERRIDE",
        parameters=("mu", "sigma", "tau"),
        log_pdf_func=log_pdf,
    )
    assert built.score_functions["tau"] is not None
    y = jnp.asarray([1.0, 2.0, 3.0])
    mu = jnp.asarray([1.5, 2.5, 3.5])
    sigma = jnp.asarray([1.0, 1.2, 0.8])
    tau = jnp.asarray([2.0, 2.0, 2.0])
    with pytest.raises(Exception):
        np.asarray(built.hessian_functions["tau"](y, mu, sigma, tau))


# ---------------------------------------------------------------------------
# 3. Links: the `o` families differ from their base only here.
# ---------------------------------------------------------------------------


def test_o_families_use_log_mu_link_and_base_uses_identity():
    from omnilss import resolve_family

    assert resolve_family("BCTo").links["mu"] == "log"
    assert resolve_family("BCPEo").links["mu"] == "log"
    assert resolve_family("BCT").links["mu"] == "identity"
    assert resolve_family("BCPE").links["mu"] == "identity"


@pytest.mark.parametrize("name", ["BCTo", "BCPEo"])
def test_o_family_non_mu_links_match_r(name):
    """Only ``mu.link`` differs; sigma/nu/tau must match R exactly."""
    from omnilss import resolve_family

    links = resolve_family(name).links
    assert links["sigma"] == "log"
    assert links["nu"] == "identity"
    if "tau" in links:
        assert links["tau"] == "log"


# ---------------------------------------------------------------------------
# 4. Live-R parity.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def r_probe():
    """One element-wise call into live R per y.

    Two R-side facts this encodes, both found the hard way:

    * ``G.dev.incr`` bodies open with
      ``z <- ifelse(nu != 0, ..., log(y/mu)/sigma)``.  With a scalar ``nu``
      that ``ifelse`` collapses to length 1, so a *vectorised* call evaluates
      every observation's kernel at ``y[1]``.  Element-wise only.
    * A ``gamlss.family`` object carries **no** ``d``/``p``/``q`` slots -- the
      full slot list ends at ``y.valid``.  Calling ``o$p(...)`` gives
      "attempt to apply non-function" even though ``!is.null(o$p)`` is TRUE
      (``$`` partial-matches the ``parameters`` list).  The quantile functions
      are namespace-level functions ``pBCTo``/``qBCTo``, whose first argument
      is named ``q``.
    """
    if not RSCRIPT.exists():
        pytest.skip(f"Rscript not found at {RSCRIPT}")

    y_path = Path(tempfile.gettempdir()) / "omnilss_o_y.txt"
    y_path.write_text("\n".join(f"{v:.17g}" for v in FIT_Y), encoding="utf-8")

    lines = [
        "suppressMessages(library(gamlss.dist))",
        f'y <- as.numeric(scan("{y_path.as_posix()}", quiet=TRUE))',
        "out <- list()",
    ]
    for name, p in PROFILES.items():
        args = f"mu={p['mu']}, sigma={p['sigma']}, nu={p['nu']}, tau={p['tau']}"
        lines += [
            f'o <- suppressWarnings(get("{name}", envir=asNamespace("gamlss.dist")))()',
            # Element-wise: vectorised calls collapse z to length 1.
            f"out[['{name}.gdev']] <- vapply(y, function(yi) "
            f"suppressWarnings(as.numeric(o$G.dev.incr(yi, {args}))), numeric(1))",
            # dpqr are namespace functions, not family slots.
            f"out[['{name}.d']] <- vapply(y, function(yi) "
            f"suppressWarnings(as.numeric(gamlss.dist::d{name}(yi, {args}))), numeric(1))",
            f"out[['{name}.p']] <- as.numeric(gamlss.dist::p{name}"
            f"(c(0.25, 0.5, 0.75, 0.9), {args}))",
            f"out[['{name}.q']] <- as.numeric(gamlss.dist::q{name}"
            f"(c(0.25, 0.5, 0.75, 0.9), {args}))",
        ]
    lines.append("cat(jsonlite::toJSON(out, digits=17, auto_unbox=TRUE))")

    with tempfile.NamedTemporaryFile("w", suffix=".R", delete=False, encoding="utf-8") as fh:
        fh.write("\n".join(lines))
        path = fh.name
    try:
        proc = subprocess.run(
            [str(RSCRIPT), path], capture_output=True, text=True, timeout=900
        )
    finally:
        Path(path).unlink(missing_ok=True)
        y_path.unlink(missing_ok=True)
    assert proc.returncode == 0, f"R failed: {proc.stderr[-800:]}"
    return json.loads(proc.stdout.strip())


@pytest.mark.parametrize("name", ["BCTo", "BCPEo"])
def test_g_dev_inc_matches_live_r_elementwise(name, r_probe, families_by_name):
    fam = families_by_name[name]
    p = PROFILES[name]
    y = jnp.asarray(FIT_Y)
    got = np.asarray(
        fam.g_dev_inc(y, p["mu"], p["sigma"], p["nu"], p["tau"]), dtype=float
    )
    exp = np.asarray(r_probe[f"{name}.gdev"], dtype=float)
    np.testing.assert_allclose(got, exp, rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize("name", ["BCTo", "BCPEo"])
def test_density_matches_live_r(name, r_probe, families_by_name):
    fam = families_by_name[name]
    p = PROFILES[name]
    got = np.asarray(
        fam.d(jnp.asarray(FIT_Y), p["mu"], p["sigma"], p["nu"], p["tau"]),
        dtype=float,
    )
    exp = np.asarray(r_probe[f"{name}.d"], dtype=float)
    np.testing.assert_allclose(got, exp, rtol=1e-9, atol=1e-300)


@pytest.mark.parametrize("name", ["BCTo", "BCPEo"])
def test_p_matches_live_r(name, r_probe, families_by_name):
    fam = families_by_name[name]
    p = PROFILES[name]
    for pp, exp in zip((0.25, 0.5, 0.75, 0.9), r_probe[f"{name}.p"]):
        got = float(np.atleast_1d(
            fam.p(pp, p["mu"], p["sigma"], p["nu"], p["tau"])
        )[0])
        assert got == pytest.approx(exp, rel=1e-9), (name, pp)


@pytest.mark.parametrize("name", ["BCTo", "BCPEo"])
def test_q_matches_live_r(name, r_probe, families_by_name):
    fam = families_by_name[name]
    p = PROFILES[name]
    for pp, exp in zip((0.25, 0.5, 0.75, 0.9), r_probe[f"{name}.q"]):
        got = float(np.atleast_1d(
            fam.q(pp, p["mu"], p["sigma"], p["nu"], p["tau"])
        )[0])
        assert got == pytest.approx(exp, rel=1e-9), (name, pp)


@pytest.mark.parametrize("name", ["BCTo", "BCPEo"])
def test_rs_deviance_matches_live_r(name):
    """End-to-end: RS deviance against R's reported deviance.

    Uses R's default ``c.crit = 0.001`` (absolute change).  See the module
    docstring for why a tighter criterion is not a stricter test.
    """
    fit = gamlss(
        formula="y ~ x",
        family=name,
        data={"y": FIT_Y, "x": FIT_X},
        control=GAMLSSControl(n_cyc=200, c_crit=R_CRITER, trace=False),
    )
    assert fit.deviance == pytest.approx(R_DEV[name], rel=5e-3)


def test_rs_deviance_beats_r_only_by_going_further():
    """Pin *why* our deviance is lower at a tight criterion.

    R's reported fit is not a stationary point of the shared objective: the
    per-parameter score sums at R's coefficients are far from zero.  If a
    future change makes those sums vanish, R's answer became a genuine
    stationary point and this test should fail -- at which point "our
    deviance is lower" would be a real disagreement rather than R stopping
    early, and the parity tolerances above would need revisiting.

    Uses a ``y ~ 1`` fit so every R coefficient is a genuine scalar.  With
    ``y ~ x`` the only handle on ``mu`` is ``fit[["mu.fv"]]``, a 40-element
    vector of fitted values -- reading its first two entries as intercept and
    slope and rebuilding ``mu(x) = b0 + b1*x`` produces ``mu(40) ~ 71`` against
    ``y`` in [0.4, 4] and inflates the deviance to 464.  That mistake was made
    once already; a constant-mu fit removes the ambiguity entirely.
    """
    from omnilss import resolve_family

    # Live R: gamlss(y ~ 1, family="BCT", control = gamlss.control(n.cyc = 5000))
    R_BCT = dict(
        dev=108.10659521714604, mu=1.8333488126235054,
        sigma=0.81923731211310713, nu=1.7263944249800565,
        tau=210772.94932637957,
    )

    y = jnp.asarray(FIT_Y)
    fam = resolve_family("BCT")
    r_mu = jnp.full_like(y, R_BCT["mu"])
    r_sigma = jnp.full_like(y, R_BCT["sigma"])
    r_nu = jnp.full_like(y, R_BCT["nu"])
    r_tau = jnp.full_like(y, R_BCT["tau"])

    # Same density: evaluating R's point must reproduce R's reported deviance.
    # Tolerance is float-level, not exact: R computes `sum(w * G.dev.incr)` in
    # its own accumulation order (measured 108.10659522484534 against R's
    # 108.10659521714604, rel 7.1e-11).  A `rel=1e-12` "bit-for-bit" claim
    # only held on the `y ~ x` dataset by luck of ordering and is not a
    # property worth asserting.
    r_dev_on_ours = float(jnp.sum(
        jnp.asarray(fam.g_dev_inc(y, r_mu, r_sigma, r_nu, r_tau))
    ))
    assert r_dev_on_ours == pytest.approx(R_BCT["dev"], rel=1e-9)

    # ... and the mu score does not vanish there.
    score_mu = float(jnp.sum(jnp.asarray(
        fam.score_functions["mu"](y, r_mu, r_sigma, r_nu, r_tau)
    )))
    assert abs(score_mu) > 1e-3, (
        "R's reported fit now looks stationary; the 'we beat R' explanation "
        "for the lower deviance no longer holds"
    )

    # The tau direction is numerically flat -- the physical reason R's own
    # trace was still descending when it stopped.
    score_tau = jnp.asarray(
        fam.score_functions["tau"](y, r_mu, r_sigma, r_nu, r_tau)
    )
    assert float(jnp.max(jnp.abs(score_tau))) < 1e-8


def test_our_tighter_criterion_reaches_a_lower_deviance():
    """The corollary: at ``c_crit=1e-8`` we go past R and keep descending.

    Not a defect and not a claim of superiority -- the point is that it is
    *explained*.  Both are valid fits of the same objective; R simply stops at
    its default tolerance.  If a future change made the tight-criterion
    deviance go *up*, the optimiser would have regressed.
    """
    tight = gamlss(
        formula="y ~ 1", family="BCT",
        data={"y": FIT_Y},
        control=GAMLSSControl(n_cyc=3000, c_crit=1e-8, trace=False),
    )
    loose = gamlss(
        formula="y ~ 1", family="BCT",
        data={"y": FIT_Y},
        control=GAMLSSControl(n_cyc=3000, c_crit=1e-3, trace=False),
    )
    assert tight.convergence_status() is True
    assert loose.convergence_status() is True
    assert tight.deviance < loose.deviance
    # R's default-tolerance answer sits between the two.
    assert tight.deviance < 108.10659521714604 <= loose.deviance


@pytest.fixture(scope="module")
def families_by_name():
    from omnilss import resolve_family

    return {name: resolve_family(name) for name in ("BCTo", "BCPEo")}
