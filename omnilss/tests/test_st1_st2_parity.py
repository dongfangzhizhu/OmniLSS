"""ST1/ST2 parity: the two skew-t families that could not be fitted at all.

The defect
----------
``tools/rs_family_parity_scan.py`` reported, for both families::

    rsdev  fail  python raised ValueError: Betainc gradient with respect to
                  a and b not supported.
    deriv  fail  tau: raised ValueError

Every RS cycle raised, so neither family had ever been fitted.

Root cause
----------
``_t_cdf_jax`` reaches ``betainc(df/2, 0.5, x)`` and ``betainc`` has no
derivative in ``a``/``b``.  This is *worse* than the Box-Cox cluster fixed in
D2.17: there, D2.13's ``custom_jvp`` made the score work and only the Hessian
re-entered ``betainc``; here even the **first** derivative raises, so
``custom_jvp`` has nothing to attach to.

The fix
-------
Transcribe R's ``dldm``/``dldd``/``dldv``/``dldt`` and the four diagonal
second derivatives for both families, and hand them to ``build_ad_family``
through the ``score_overrides`` / ``hessian_overrides`` added in D2.17.  The
density itself was already correct and is untouched by this change (measured
rel <= 1.8e-15 across six profiles).

Kept deliberately
-----------------
* **Every second derivative is ``-dldX**2`` clamped at -1e-15.**  That is a
  rough observed-information approximation, not a true Hessian.  D2.15
  established that the target is *what R feeds the IRLS*; a better second
  derivative makes parity worse.
* **``dldt`` keeps R's hard-coded forward step** -- 0.001 on ``tau`` (ST1) and
  0.0005 on ``df = tau+1`` (ST2).  These are different step sizes for what is
  nominally the same derivative, which is R's inconsistency, not ours.
* **``dldv`` divides by ``nu`` unguarded.**  At ``nu = 0`` that is 0/0 and R
  does return ``NaN`` (``ST1(nu=0)$dldv`` is ``NaN``, ``d2ldv2`` is ``NA``).
  Adding a guard would diverge from the reference; R's own ``nu.initial`` is
  0.1, so the optimiser does not sit on that boundary.

ST1 vs ST2
----------
Not just a different argument: ST2's ``w = nu*sqrt(lam)*z`` carries ``lam``
and the whole density uses ``df = tau + 1``, which brings an extra
``dwdz = nu*sqrt(lam)*(1 - lam*z^2/(tau+1))`` term into ``dldm``/``dldd``.
R's ``lam`` also appears in **ST1's** ``dldm``/``dldd`` even though ST1's
``w = nu*z`` does not need it -- that is R's actual code, reproduced rather
than tidied up.

ST2's ``dldt`` is knowingly wrong *in R* -- see
``KNOWN_ST2_TAU_DEFECT`` below and the tolerance note in
``tools/rs_family_parity_scan.py``.  It is transcribed as-is.
"""

import json
import subprocess
import tempfile
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from omnilss.controls import GAMLSSControl
from omnilss.distributions_st_compat import _t_cdf_jax
from omnilss.fitting import gamlss

RSCRIPT = Path("C:/Program Files/R/R-4.6.0/bin/x64/Rscript.exe")

FAMILIES = ("ST1", "ST2")

# R names its second derivatives d2ldm2/d2ldd2/d2ldv2/d2ldt2 -- the trailing
# letter is the *suffix* (m, d=sigma, v=nu, t=tau), not the parameter name.
R_SUFFIX = {"mu": "m", "sigma": "d", "nu": "v", "tau": "t"}

# nu = 0.1 is R's own `nu.initial`; nu = 0 is excluded because R returns NaN
# for dldv and NA for d2ldv2 there (0/0), which is asserted separately below.
PROFILES = {
    "ST1": [
        dict(mu=1.0, sigma=1.0, nu=0.1, tau=4.0),
        dict(mu=1.0, sigma=1.0, nu=0.8, tau=4.0),
        dict(mu=1.0, sigma=1.0, nu=-0.8, tau=4.0),
        dict(mu=1.0, sigma=1.0, nu=0.5, tau=2.0),
        dict(mu=1.0, sigma=1.0, nu=0.5, tau=1.2),
        dict(mu=1.0, sigma=1.0, nu=0.5, tau=8.0),
    ],
    "ST2": [
        dict(mu=1.0, sigma=1.0, nu=0.1, tau=4.0),
        dict(mu=1.0, sigma=1.0, nu=0.8, tau=4.0),
        dict(mu=1.0, sigma=1.0, nu=-0.8, tau=4.0),
        dict(mu=1.0, sigma=1.0, nu=0.5, tau=2.0),
        dict(mu=1.0, sigma=1.0, nu=0.5, tau=1.2),
        dict(mu=1.0, sigma=1.0, nu=0.5, tau=8.0),
    ],
}

PROBE_Y = np.array([0.35, 1.0, 1.7, 2.4, -0.6, 0.2, 3.1, 1.3])

# --- the tau >= 1e6 normal-limit switch (D2.18, second pass) ----------------
#
# R's dST1/dST2 compute a t-kernel log-likelihood AND a normal-limit one, then
# select with `ifelse(tau < 1e+06, <t kernel>, <normal>)`.  ST3/ST4 carry the
# same switch; ST1/ST2 were the holdouts, and it mattered: R's own ST1 optimum
# on the parity sample sits at tau = 1.4e56, i.e. inside the normal limit, so
# omitting the branch made our density wrong *at R's solution*.
#
# 999999.0 sits just below the threshold and is included on purpose -- it is
# the t kernel, and its residual ~1e-9 (against R) is `_t_cdf_jax`'s
# betainc losing precision as df grows, not a branch defect.
GAUSS_LIMIT_TAU = [
    dict(mu=1.2, sigma=0.8, nu=0.5, tau=999999.0),
    dict(mu=1.2, sigma=0.8, nu=0.5, tau=1.4e56),
]

# Measured on live R: R's own ST2 dldt vs a central difference of log dST2
# (y=0.3, mu=sigma=1, nu=0.5).  R is 6-12% off because its j term holds w
# fixed while differencing pt(w, df=tau+1), omitting dw/dtau.
KNOWN_ST2_TAU_DEFECT = {
    0.5: (0.929927, 0.990414),
    1.0: (0.298691, 0.328777),
    2.0: (0.083972, 0.096027),
}

# Same 40-point strictly-positive dataset as the Box-Cox modules.
FIT_Y = np.abs(2.0 + 1.2 * np.random.default_rng(2026).normal(size=40)) + 0.15

# Live R: gamlss(y ~ 1, family=<nm>, control = gamlss.control(n.cyc = 5000))
# with c_crit left at R's default 0.001 (see the D2.17 note on absolute vs
# relative deviance change -- tightening ours alone is not a stricter test).
R_DEV = {"ST1": 109.573585738715, "ST2": 109.57550620607012}
R_CRIT = 0.001


@pytest.fixture(scope="module")
def r_probe():
    """Element-wise live-R probe for the eight derivatives plus the density."""
    if not RSCRIPT.exists():
        pytest.skip(f"Rscript not found at {RSCRIPT}")

    yvec = ", ".join(f"{float(v):.17g}" for v in PROBE_Y)
    lines = ["suppressMessages(library(gamlss.dist))", "out <- list()"]
    for nm in FAMILIES:
        lines.append(
            f'o <- suppressWarnings(get("{nm}", envir=asNamespace("gamlss.dist")))()'
        )
        for pi, p in enumerate(PROFILES[nm]):
            args = f"mu={p['mu']}, sigma={p['sigma']}, nu={p['nu']}, tau={p['tau']!r}"
            # Element-wise: R's bodies branch on scalars, so a vectorised call
            # collapses to the first element's branch decision.  The response
            # is passed positionally -- dST1's first formal is `x`, not `y`.
            lines.append(
                f"out[['{nm}.{pi}.d']] <- vapply(c({yvec}), function(xi) "
                f"suppressWarnings(as.numeric(gamlss.dist::d{nm}(xi, {args}, "
                f"log=TRUE))), numeric(1))"
            )
            for base in ("dld", "d2ld"):
                for key, suf in R_SUFFIX.items():
                    fn = f"{base}{suf}2" if base == "d2ld" else f"{base}{suf}"
                    lines.append(
                        f"out[['{nm}.{pi}.{fn}']] <- vapply(c({yvec}), function(yi) "
                        f"suppressWarnings(as.numeric(o[['{fn}']]("
                        f"y=yi, {args}))), numeric(1))"
                    )
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
    assert proc.returncode == 0, f"R failed: {proc.stderr[-800:]}"
    return json.loads(proc.stdout.strip())


@pytest.fixture(scope="module")
def families():
    from omnilss import resolve_family

    return {nm: resolve_family(nm) for nm in FAMILIES}


@pytest.fixture(scope="module")
def r_gauss_probe():
    """Live-R densities on both sides of the `tau >= 1e6` normal-limit switch.

    Queried rather than hard-coded: a transcribed 16-digit constant that I
    mistyped would turn into a *passing* test, which is the one failure mode a
    parity test must not have.
    """
    if not RSCRIPT.exists():
        pytest.skip(f"Rscript not found at {RSCRIPT}")

    yvec = ", ".join(f"{float(v):.17g}" for v in PROBE_Y)
    lines = ["suppressMessages(library(gamlss.dist))", "out <- list()"]
    for nm in FAMILIES:
        for pi, p in enumerate(GAUSS_LIMIT_TAU):
            args = f"mu={p['mu']}, sigma={p['sigma']}, nu={p['nu']}, tau={p['tau']!r}"
            lines.append(
                f"out[['{nm}.{pi}']] <- vapply(c({yvec}), function(xi) "
                f"suppressWarnings(as.numeric(gamlss.dist::d{nm}(xi, {args}, "
                f"log=TRUE))), numeric(1))"
            )
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
    assert proc.returncode == 0, f"R failed: {proc.stderr[-800:]}"
    return json.loads(proc.stdout.strip())


# ---------------------------------------------------------------------------
# 1. The defect itself: both families must fit at all.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", FAMILIES)
def test_st_variant_fits_without_raising(name):
    """Regression: every RS cycle used to raise inside betainc.

    This is the exact check the parity scanner failed.  Keep it cheap so it
    runs without R: a family that raises here is unusable regardless of parity.
    """
    fit = gamlss(
        formula="y ~ 1", family=name, data={"y": FIT_Y},
        control=GAMLSSControl(n_cyc=200, c_crit=R_CRIT, trace=False),
    )
    assert np.isfinite(fit.deviance)
    assert fit.convergence_status() is True, fit.termination_reason
    for par in fit.parameters:
        v = np.asarray(fit.fitted_values[par], dtype=float)
        assert v.shape == FIT_Y.shape, (par, v.shape)
        assert np.all(np.isfinite(v)), (par, v)


def test_betainc_itself_is_undifferentiable_here():
    """Pin the premise: `_t_cdf_jax` has no usable first derivative.

    Without this, the override tests below could be passing because
    ``score_overrides`` is ignored rather than because AD genuinely fails.
    ``betainc`` needs no `custom_jvp` here (unlike D2.13's `_log_pt_t`), so
    even grad-of-grad raises on the *first* derivative.
    """
    import jax

    w = jnp.asarray(0.5)
    df = jnp.asarray(3.0)
    with pytest.raises(ValueError, match="Betainc gradient"):
        jax.grad(lambda d: jnp.sum(_t_cdf_jax(w, d)))(df)


@pytest.mark.parametrize("name", FAMILIES)
def test_derivatives_are_r_transcriptions_not_ad(name, families):
    """Derivatives must come from the R tables, not from AD.

    ``__qualname__`` is the discriminator: the overrides are built by
    ``_make_st_derivative_tables``, while AD closures are named after
    ``build_ad_family``'s ``make_grad`` / ``make_hess`` (verified for ST3).
    """
    fam = families[name]
    for par in ("mu", "sigma", "nu", "tau"):
        assert "_make_st_derivative_tables" in fam.score_functions[par].__qualname__, par
        assert "_make_st_derivative_tables" in fam.hessian_functions[par].__qualname__, par
        assert "make_grad" not in fam.score_functions[par].__qualname__, par
        assert "make_hess" not in fam.hessian_functions[par].__qualname__, par


def test_ad_family_override_is_additive_for_st():
    """The transcription override is applied per family, not per module.

    It must reach exactly the families that were transcribed and leave the
    rest on AD.  ST3/ST4 are the AD half; ST5 moved to the transcribed half in
    D2.62 ("ST5 从 deviance=inf 到 rsdev pass"), so it belongs with ST1/ST2
    here -- this test used to pin all three to AD and silently forbade that
    fix.
    """
    from omnilss import resolve_family

    for nm in ("ST3", "ST4"):
        fam = resolve_family(nm)
        assert "make_grad" in fam.score_functions["mu"].__qualname__, nm

    st5 = resolve_family("ST5")
    assert "_st5_dldm" in st5.score_functions["mu"].__qualname__
    assert "make_grad" not in st5.score_functions["mu"].__qualname__


# ---------------------------------------------------------------------------
# 2. The transcribed formulas, against live R.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", FAMILIES)
def test_density_matches_live_r(name, r_probe, families):
    """The density was already correct; assert it stayed that way.

    Compared element-wise with a *scalar* response.  `build_ad_family`'s
    `_prepare_jax_args` collapses a length-1 array to a scalar (its
    `target_size` is 1 whenever every argument has size 1), so passing
    `y` as `jnp.asarray([v])` returns a 0-d result -- correct value, wrong
    shape to index.  R's `dST1` is likewise called per element, since its body
    branches on scalar `tau` and `length(tau) > 1` would take the vector path.
    """
    fam = families[name]
    for pi, p in enumerate(PROFILES[name]):
        exp = np.asarray(r_probe[f"{name}.{pi}.d"], dtype=float)
        got = np.asarray([
            float(np.asarray(
                fam.d(float(v), p["mu"], p["sigma"], p["nu"], p["tau"], log=True),
                dtype=float,
            ))
            for v in PROBE_Y
        ])
        np.testing.assert_allclose(got, exp, rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize("name", FAMILIES)
def test_density_normal_limit_branch_matches_live_r(name, r_gauss_probe, families):
    """`tau >= 1e6` must switch the density to the normal limit, as R does.

    Two tolerances, deliberately.  Above the switch both sides are closed-form
    normal evaluations, so agreement must be at machine precision (we get
    4.4e-16).  Just below it the t kernel is in play and `_t_cdf_jax`'s
    `betainc(df/2, 0.5, x)` has lost precision by df ~ 1e6 (we sit 1.7e-9 from
    R) -- a pre-existing limitation of the Student-t CDF, unrelated to the
    switch.
    """
    fam = families[name]
    for pi, p in enumerate(GAUSS_LIMIT_TAU):
        exp = np.asarray(r_gauss_probe[f"{name}.{pi}"], dtype=float)
        got = np.asarray([
            float(np.asarray(
                fam.d(float(v), p["mu"], p["sigma"], p["nu"], p["tau"], log=True),
                dtype=float,
            ))
            for v in PROBE_Y
        ])
        tol = 1e-7 if p["tau"] < 1e6 else 1e-12
        np.testing.assert_allclose(got, exp, rtol=0.0, atol=tol)


def test_normal_limit_branch_is_not_inert():
    """Guard against a vacuous pass.

    If the two branches coincided, `test_density_normal_limit_branch_matches_live_r`
    would still be green after the branch was deleted -- it compares against R,
    but only probes values R itself never visits on both sides.  Measured on
    live R at (y=0.35, mu=1.2, sigma=0.8, nu=0.5): tau=999999 gives
    -1.779029676131917 and tau=1.4e56 gives -1.779029378512898, a 3e-7 split.
    """
    from omnilss import resolve_family

    fam = resolve_family("ST1")
    p = GAUSS_LIMIT_TAU[0]
    t_side = float(np.asarray(
        fam.d(0.35, p["mu"], p["sigma"], p["nu"], 999999.0, log=True), dtype=float))
    n_side = float(np.asarray(
        fam.d(0.35, p["mu"], p["sigma"], p["nu"], 1.4e56, log=True), dtype=float))
    assert abs(t_side - n_side) > 1e-9, (
        "ST1's two tau branches are indistinguishable -- the normal-limit "
        "assertions would pass even with the branch removed"
    )


def test_st1_and_st2_agree_above_the_switch():
    """Above `tau >= 1e6` both families reduce to the same normal limit.

    R's dST1 and dST2 print bit-identical values there, so any divergence is
    ours alone -- a cheap cross-check that the branch was wired into both
    families rather than just one.
    """
    from omnilss import resolve_family

    y = np.array([0.35, 1.0, 1.7, 2.4, -0.6, 0.2, 3.1, 1.3])
    p = GAUSS_LIMIT_TAU[1]
    d1 = np.asarray(
        resolve_family("ST1").d(y, p["mu"], p["sigma"], p["nu"], p["tau"], log=True),
        dtype=float,
    )
    d2 = np.asarray(
        resolve_family("ST2").d(y, p["mu"], p["sigma"], p["nu"], p["tau"], log=True),
        dtype=float,
    )
    np.testing.assert_allclose(d1, d2, rtol=0.0, atol=1e-14)


@pytest.mark.parametrize("name", FAMILIES)
def test_scores_match_live_r(name, r_probe, families):
    """R's forward-difference `dldt` is copied bit-for-bit.

    Tolerance 5e-6: R's own step (0.001 / 0.0005) is the limiting factor, and
    our value sits 1.2e-6 from R's -- the same order D2.13 measured for the
    Box-Cox cluster.  Widening R's step would move us *away* from parity.
    """
    fam = families[name]
    worst = 0.0
    for pi, p in enumerate(PROFILES[name]):
        args = (jnp.asarray(PROBE_Y), p["mu"], p["sigma"], p["nu"], p["tau"])
        for key, suf in R_SUFFIX.items():
            got = np.asarray(fam.score_functions[key](*args), dtype=float)
            exp = np.asarray(r_probe[f"{name}.{pi}.dld{suf}"], dtype=float)
            rel = np.abs(got - exp) / np.maximum(np.abs(exp), 1e-300)
            worst = max(worst, float(np.max(rel)))
    assert worst < 5e-6, f"{name} worst score rel {worst:.2e}"


@pytest.mark.parametrize("name", FAMILIES)
def test_diagonal_hessians_match_live_r(name, r_probe, families):
    """All four are `-dldX^2` floored at -1e-15, exactly as R writes them."""
    fam = families[name]
    worst = 0.0
    for pi, p in enumerate(PROFILES[name]):
        args = (jnp.asarray(PROBE_Y), p["mu"], p["sigma"], p["nu"], p["tau"])
        for key, suf in R_SUFFIX.items():
            got = np.asarray(fam.hessian_functions[key](*args), dtype=float)
            exp = np.asarray(r_probe[f"{name}.{pi}.d2ld{suf}2"], dtype=float)
            rel = np.abs(got - exp) / np.maximum(np.abs(exp), 1e-300)
            worst = max(worst, float(np.max(rel)))
    assert worst < 5e-6, f"{name} worst hessian rel {worst:.2e}"


def test_second_derivative_is_negative_square_with_r_floor(families):
    """The `-1e-15` floor is load-bearing: without it RS gets a positive weight."""
    fam = families["ST1"]
    y = jnp.asarray(PROBE_Y)
    for key in ("mu", "sigma", "nu", "tau"):
        score = jnp.asarray(fam.score_functions[key](y, 1.0, 1.0, 0.5, 3.0))
        hess = jnp.asarray(fam.hessian_functions[key](y, 1.0, 1.0, 0.5, 3.0))
        assert np.all(hess <= -1e-15 + 1e-25), key
        # Where |score| is large enough for the square to dominate, the
        # identity must hold exactly.
        big = np.abs(np.asarray(score)) > 1e-6
        if big.any():
            np.testing.assert_allclose(
                np.asarray(hess)[big], -(np.asarray(score)[big] ** 2), rtol=1e-12
            )


# ---------------------------------------------------------------------------
# 3. ST1 vs ST2 are genuinely different, not one cloned into the other.
# ---------------------------------------------------------------------------


def test_st1_and_st2_differ_in_tau_kernel():
    """ST2's `w` carries `sqrt(lam)` and its `df` is tau+1; ST1's does not."""
    from omnilss.distributions_st_compat import _st1_log_pdf, _st2_log_pdf

    y = jnp.asarray([1.0, 1.7, 2.4])
    nu, tau = 0.8, 4.0
    a = np.asarray(_st1_log_pdf(y, 1.0, 1.0, nu, tau), dtype=float)
    b = np.asarray(_st2_log_pdf(y, 1.0, 1.0, nu, tau), dtype=float)
    assert not np.allclose(a, b)


def test_st2_dldm_carries_the_dwdz_term_st1_lacks():
    """R's ST2 `dldm` has `dwdz`; ST1's has the bare `nu`.

    R: ST2  `-(dt(w,df=tau+1)/pt(w,df=tau+1)) * dwdz/sigma + lam*z/sigma`
       ST1  `-(dt(w,df=tau)  /pt(w,df=tau))   * nu/sigma   + lam*z/sigma`
    """
    from omnilss import resolve_family

    y = jnp.asarray([1.5, 2.0, 3.0])
    st1 = resolve_family("ST1")
    st2 = resolve_family("ST2")
    s1 = np.asarray(st1.score_functions["mu"](y, 1.0, 1.0, 0.5, 2.0), dtype=float)
    s2 = np.asarray(st2.score_functions["mu"](y, 1.0, 1.0, 0.5, 2.0), dtype=float)
    assert not np.allclose(s1, s2)


def test_dldv_is_nan_at_nu_zero_like_r():
    """R gives NaN for dldv and NA for d2ldv2 at nu = 0. So do we.

    `dldv = hazard * (w / nu)` and `w = nu * ...`, so this is 0/0.  Adding a
    guard would look like a fix and would be a divergence from the reference.

    The second derivative is the subtle part.  R writes
    ``ifelse(-dldv^2 < -1e-15, ., -1e-15)``, and R's ``ifelse(NA, a, b)``
    returns **NA** rather than ``b`` -- so `d2ldv2` is NA, verified on live R
    as `is.nan(...) == FALSE` with the value printing as NA.  A plain
    `jnp.where` would return -1e-15 because the comparison against NaN is
    False; the implementation propagates NaN explicitly to match.
    """
    from omnilss import resolve_family

    y = jnp.asarray([1.0, 2.0])
    for nm in FAMILIES:
        fam = resolve_family(nm)
        s = np.asarray(fam.score_functions["nu"](y, 1.0, 1.0, 0.0, 4.0), dtype=float)
        h = np.asarray(fam.hessian_functions["nu"](y, 1.0, 1.0, 0.0, 4.0), dtype=float)
        assert np.all(np.isnan(s)), (nm, s)
        assert np.all(np.isnan(h)), (nm, h)

    # The other three parameters stay finite at nu = 0: their scores do not
    # divide by nu, and d2ldm2 correctly floors to -1e-15 (R agrees).
    fam = resolve_family("ST1")
    for par in ("mu", "sigma", "tau"):
        s = np.asarray(fam.score_functions[par](y, 1.0, 1.0, 0.0, 4.0), dtype=float)
        h = np.asarray(fam.hessian_functions[par](y, 1.0, 1.0, 0.0, 4.0), dtype=float)
        assert np.all(np.isfinite(s)), (par, s)
        assert np.all(np.isfinite(h)) and np.all(h <= -1e-15 + 1e-25), (par, h)


# ---------------------------------------------------------------------------
# 4. ST2's tau score is wrong in R, and knowingly so.
# ---------------------------------------------------------------------------


def test_known_st2_tau_defect_is_still_present_and_bounded():
    """Document R's own ST2 `dldt` error instead of quietly "fixing" it.

    R's `j` term differentiates ``pt(w, df=tau+1)`` while holding ``w`` fixed,
    but ``w = nu*sqrt(lam)*z`` depends on ``tau`` through ``lam`` -- so the
    dw/dtau contribution is missing.  Measured against a central difference
    of ``log dST2`` on live R, R is 6-12% low.

    The point of this test is that a future change cannot silently replace R's
    value with a "correct" one: parity means matching R, and D2.15 recorded
    that tightening a derivative away from R makes parity worse.  If R ever
    fixes this, ``test_scores_match_live_r`` will fail and *that* is the
    signal to revisit.
    """
    from omnilss import resolve_family

    fam = resolve_family("ST2")
    for tau, (r_val, exact) in KNOWN_ST2_TAU_DEFECT.items():
        got = float(np.asarray(
            fam.score_functions["tau"](jnp.asarray([0.3]), 1.0, 1.0, 0.5, tau)
        )[0])
        assert got == pytest.approx(r_val, rel=1e-5), (tau, got, r_val)
        # R is low, and stays low by roughly this much.
        assert exact > got > 0.8 * exact, (tau, got, exact)


def test_st2_tau_score_is_not_close_to_the_exact_derivative():
    """The defect is real, not a rounding artefact -- measured, not assumed.

    The parity scanner's `deriv` gate flags ST2 for exactly this: its central
    difference of ``log d`` is fully converged (identical from h=1e-3 to 1e-7)
    and still disagrees with R by 34%.  A tolerance entry in
    ``tools/rs_family_parity_scan.py`` records the same finding.
    """
    from omnilss import resolve_family

    fam = resolve_family("ST2")
    y = jnp.asarray([0.3])
    analytic = np.atleast_1d(np.asarray(
        fam.score_functions["tau"](y, 1.0, 1.0, 0.5, 2.0), dtype=float
    ))
    # Central difference of log(d) with the converged step.  `fam.d` collapses
    # to a scalar for a length-1 response, hence atleast_1d.
    best = None
    for h in (1e-3, 1e-4, 1e-5, 1e-6, 1e-7):
        args_hi = [1.0, 1.0, 0.5, 2.0 + h]
        args_lo = [1.0, 1.0, 0.5, 2.0 - h]
        d_hi = np.atleast_1d(np.asarray(fam.d(y, *args_hi), dtype=float))
        d_lo = np.atleast_1d(np.asarray(fam.d(y, *args_lo), dtype=float))
        fd = (np.log(d_hi) - np.log(d_lo)) / (2.0 * h)
        best = fd if best is None else best
        np.testing.assert_allclose(fd, best, rtol=1e-6)
    rel = abs(analytic[0] - best[0]) / abs(best[0])
    assert rel > 0.1, f"ST2 dldt suddenly matches the exact derivative ({rel:.3e})"


# ---------------------------------------------------------------------------
# 5. Links and end-to-end RS deviance.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", FAMILIES)
def test_links_match_r(name, families):
    """R: mu.link=identity, sigma.link=log, nu.link=identity, tau.link=log."""
    links = families[name].links
    assert links["mu"] == "identity"
    assert links["sigma"] == "log"
    assert links["nu"] == "identity"
    assert links["tau"] == "log"


@pytest.mark.parametrize("name", FAMILIES)
def test_rs_deviance_matches_live_r(name):
    """End-to-end at R's default ``c.crit = 0.001`` (absolute change).

    Tightening only our side is not a stricter test -- see the D2.17 note on
    R's RS criterion being an absolute deviance difference.
    """
    fit = gamlss(
        formula="y ~ 1", family=name, data={"y": FIT_Y},
        control=GAMLSSControl(n_cyc=5000, c_crit=R_CRIT, trace=False),
    )
    assert fit.convergence_status() is True, fit.termination_reason
    assert fit.deviance == pytest.approx(R_DEV[name], rel=5e-4)
