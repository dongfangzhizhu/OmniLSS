"""TF2 parity: the reparameterisation shell that forwarded TF's raw derivatives.

What was wrong
--------------
``tools/rs_family_parity_scan.py`` reported TF2 with a correct density but a
badly wrong fit::

    TF2  dpqr   pass
    TF2  deriv  fail  mu: max rel diff 7.519e-01; sigma: max rel diff 1.276e+00
    TF2  rsdev  fail  deviance 3399.12 vs R 144.172 (rel 2.258e+01)

That combination -- ``dpqr`` green, ``deriv`` red -- is the signature of a
family whose *density* is right and whose *derivatives* belong to some other
density.  TF2 is a reparameterisation shell around TF::

    sigma1 = sqrt((nu - 2)/nu) * sigma          # TF2.R
    dTF2    = dTF(x, mu, sigma1, nu) / sigma1

The shell forwarded TF's score/hessian functions with the **raw** sigma::

    def dldm_tf2(y, mu, sigma, nu):
        return fam.score_functions["mu"](y, mu, sigma, _nu_gt2(nu))   # sigma, not sigma1

so every chain-rule term was missing.  Measured against our own density at
``(y=0.3, mu=0, sigma=1, nu=3)``, all three parameters were wrong::

    mu      analytic  0.38834951   FD  1.10091743   diff 7.1e-01
    sigma   analytic -0.88349515   FD -0.66972477   diff 2.1e-01
    nu      analytic  0.03111859   FD -0.18480405   diff 2.2e-01

The hessians agreed with the *forwarded* scores to 1e-11, which is what made
this expensive to see from the outside: RS was iterating a gradient that was
internally self-consistent and simply did not belong to the density being
minimised.  ``converged=True`` and a plausible-looking Hessian both held.

R's own formulas (``gamlss.dist`` TF2.R) transcribe as::

    dldm   = TF$dldm (y, mu, sigma1, nu)
    dldd   = ds1dd * TF$dldd (y, mu, sigma1, nu)
    dldv   = TF$dldv (y, mu, sigma1, nu) + ds1dv * TF$dldd(y, mu, sigma1, nu)
    d2ldm2 = -(nu+1) / ((nu+3) * sigma1^2)
    d2ldd2 = -ds1dd^2 * (2*nu) / ((nu+3) * sigma1^2)
    d2ldv2 = [trigamma((nu+1)/2) - trigamma(nu/2) + 2(nu+5)/(nu(nu+1)(nu+3))]/4
             + ds1dv^2 * TF$d2ldd2(sigma1, nu),  floored at -1e-15

The hessians stay coarse on purpose (D2.13's rule: copy R's roughness too).
``d2ldm2`` in particular ignores ``y`` and ``mu`` entirely -- it is *not* the
second derivative of our density, and ``test_hessian_matches_rs_own_formulas``
pins that rather than pretending otherwise.

Why ``nu.initial`` also needed fixing
--------------------------------------
``TF2``'s shape parameter is ``nu``, which fell through to the generic 1.0
fallback.  R's ``TF2()$nu.initial`` is **4**, not ``TF``'s 10 -- same parameter
name, same file, different value.  Sharing the ``TF`` branch would have been
the "batch replace similar lines" mistakeD2.19 documented.

Three R-side details this file keeps tripping over
--------------------------------------------------
- ``o$parameters`` is a **list**, not a character vector; ``cat()`` on it
  errors with "argument 2 (type 'list') cannot be handled by 'cat'".  Use
  ``paste(unlist(...), collapse=", ")``.
- The deviance slot is ``G.dev.incr`` with a capital G.  ``TF2()$g.dev.incr``
  is ``NULL`` (partial matching does not reach it), so a lowercase probe
  silently reports "no deviance" instead of failing.
- ``$nu.initial`` is an expression, not a closure: ``body()`` warns "argument
  is not a function" and returns NULL.  ``eval()`` it with ``y`` and ``length``
  in scope.
"""

import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest

import jax.numpy as jnp

from omnilss import gamlss, resolve_family

# Rscript discovery goes through the parity scanner's own resolver.  The first
# version of the ST3C/SST file hard-coded a path and every R-backed test
# silently skipped, which is indistinguishable from a green run.
from tests.test_st3c_sst_parity import _find_rscript, _run_r, _scanner_sample

RSCRIPT = _find_rscript()

FAMILY = "TF2"
# R's deviance on the scanner's sample at n.cyc = 500.  After the fix we report
# the same 144.171705 (from nu_eta = 240 rather than R's 47.33 -- nu is weakly
# identified and flat past nu_eta = 20; see test_nu_direction_is_flat...).
R_DEV = 142.7996894991248
FIT_REL_TOL = 1e-6

# R's TF2 metadata, read from live R (see the docstring).
R_NU_INITIAL = 4.0
R_LINKS = {"mu": "identity", "sigma": "log", "nu": "logshiftto2"}


@pytest.fixture(scope="module")
def r_meta_fixture():
    """Live-R TF2 links / valid / initial values, plus the deviance formula."""
    if not RSCRIPT:
        pytest.skip("Rscript not found")
    lines = [
        "suppressMessages(library(gamlss.dist))",
        "o <- TF2()",
        "out <- list()",
        'out[["links"]] <- as.list(o[c("mu.link", "sigma.link", "nu.link")])',
        'out[["nu.valid.pos"]] <- isTRUE(all(eval(body(o$nu.valid),',
        " list(nu=c(-1, 0, 1)))))",
        'out[["sigma.valid.pos"]] <- isTRUE(all(eval(body(o$sigma.valid),',
        " list(sigma=c(-1, 0, 1)))))",
        'out[["nu.initial"]] <- as.numeric(eval(o$nu.initial,',
        " list(y = c(1, 2, 3), length = 3)))[1]",
        'out[["TF.nu.initial"]] <- as.numeric(eval(TF()$nu.initial,',
        " list(y = c(1, 2, 3), length = 3)))[1]",
        # G.dev.incr has a capital G; o$"g.dev.incr" is NULL via partial matching.
        'out[["dev"]] <- deparse(o$G.dev.incr)',
        "cat(jsonlite::toJSON(out, digits = 17, auto_unbox = TRUE))",
    ]
    return _run_r(lines)


@pytest.fixture(scope="module")
def r_fit_fixture():
    """Live-R fit on the scanner's exact sample, full coefficient set."""
    if not RSCRIPT:
        pytest.skip("Rscript not found")
    payload = _scanner_sample(FAMILY)
    tmp = Path(tempfile.gettempdir()) / "d220_r_fit_payload.json"
    tmp.write_text(json.dumps(payload), encoding="utf-8")
    lines = [
        "suppressMessages(library(gamlss.dist)); suppressMessages(library(gamlss))",
        f'p <- jsonlite::fromJSON("{tmp.as_posix()}")',
        "df <- data.frame(y = p$y, x = p$x)",
        f'f <- suppressWarnings(gamlss(y ~ x, family="{FAMILY}", data = df,',
        "  trace = FALSE, control = gamlss.control(n.cyc = 500, trace = FALSE)))",
        "ca <- coefAll(f)",
        'out <- list(deviance = deviance(f), iter = f$iter,',
        "  mu = unname(ca$mu), sigma = unname(ca$sigma), nu = unname(ca$nu))",
        "cat(jsonlite::toJSON(out, digits = 17, auto_unbox = TRUE))",
    ]
    return _run_r(lines)


# ---------------------------------------------------------------------------
# 1. The scores must be derivatives of *our own* density.  No R needed, so
#    this keeps failing loudly even when the R fixtures are unavailable.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("nu", [2.5, 3.0, 8.0, 30.0])
def test_scores_match_finite_differences_of_our_density(nu):
    """The bug in one assertion: scores disagreed with the density they claim.

    `h = 1e-6` on the natural scale.  A reparameterisation shell that forgets
    the chain rule produces a score that is *self-consistent* with nothing --
    it is simply TF's score at the wrong sigma, so this test is the only place
    the error is visible without an R reference.
    """
    fam = resolve_family(FAMILY)
    y, mu, sigma = 0.3, 0.0, 1.0
    Y, MU, SIGMA, NU = (np.array([v]) for v in (y, mu, sigma, nu))

    def logpdf(mm, ss, nn):
        return float(np.asarray(fam.d(y, mm, ss, nn, log=True), dtype=float))

    h = 1e-6
    base = (mu, sigma, nu)
    for idx, par in enumerate(("mu", "sigma", "nu")):
        up = list(base)
        up[idx] += h
        dn = list(base)
        dn[idx] -= h
        fd = (logpdf(*up) - logpdf(*dn)) / (2 * h)
        an = float(np.asarray(
            fam.score_functions[par](
                np.array([y]), np.array([mu]), np.array([sigma]), np.array([nu])),
            dtype=float)[0])
        assert an == pytest.approx(fd, rel=1e-4, abs=1e-6), (
            f"{par} score {an:.8f} vs FD {fd:.8f} at nu={nu}"
        )


def test_the_old_forwarding_would_have_failed_this():
    """Pin the exact numbers from the diagnosis so the test cannot rot.

    A regression test that only says "scores match FD" would still pass if
    someone reintroduced the chain-rule terms with the wrong sign, as long as
    two wrongs cancelled.  These are the measured pre-fix values.
    """
    fam = resolve_family(FAMILY)
    y, mu, sigma, nu = 0.3, 0.0, 1.0, 3.0

    def logpdf(mm, ss, nn):
        return float(np.asarray(fam.d(y, mm, ss, nn, log=True), dtype=float))

    # TF's score at the raw sigma -- what the shell used to forward.
    tf = resolve_family("TF")
    old = {
        par: float(np.asarray(
            tf.score_functions[par](
                np.array([y]), np.array([mu]), np.array([sigma]), np.array([nu])),
            dtype=float)[0])
        for par in ("mu", "sigma", "nu")
    }
    h = 1e-6
    base = (mu, sigma, nu)
    for idx, par in enumerate(("mu", "sigma", "nu")):
        up = list(base)
        up[idx] += h
        dn = list(base)
        dn[idx] -= h
        fd = (logpdf(*up) - logpdf(*dn)) / (2 * h)
        assert abs(old[par] - fd) > 0.1, (
            f"{par}: old forwarding {old[par]:.8f} vs FD {fd:.8f} -- "
            "if these now agree the diagnosis in the docstring is stale"
        )


# ---------------------------------------------------------------------------
# 2. The hessians are R's, roughness included.
# ---------------------------------------------------------------------------


def test_hessian_matches_rs_own_formulas(r_meta_fixture):
    """R's closed forms, transcribed -- not the true second derivative.

    ``d2ldm2`` in R ignores y and mu.  That is not a bug on our side: RS feeds
    these to IRLS, and "improving" them changes what RS iterates on, which is
    how D2.13's BCT ended up worse before both the derivative and the link were
    fixed together.  This test therefore compares against R's *formula*.
    """
    from omnilss.distributions import TF2 as _TF2  # noqa: F401  (import check)
    from jax.scipy.special import polygamma

    fam = resolve_family(FAMILY)
    sigma, nu = 1.3, 4.0
    sigma1 = sigma * np.sqrt((nu - 2.0) / nu)
    ds1dd = np.sqrt((nu - 2.0) / nu)
    ds1dv = np.sqrt(nu / (nu - 2.0)) * sigma / nu**2

    y, mu = 0.3, 0.0
    Y, MU = np.array([y]), np.array([mu])

    got_mu = float(np.asarray(fam.hessian_functions["mu"](
        Y, MU, np.array([sigma]), np.array([nu])), dtype=float)[0])
    got_sigma = float(np.asarray(fam.hessian_functions["sigma"](
        Y, MU, np.array([sigma]), np.array([nu])), dtype=float)[0])
    got_nu = float(np.asarray(fam.hessian_functions["nu"](
        Y, MU, np.array([sigma]), np.array([nu])), dtype=float)[0])

    assert got_mu == pytest.approx(-(nu + 1) / ((nu + 3) * sigma1**2), rel=1e-12)
    assert got_sigma == pytest.approx(
        -(ds1dd**2) * (2.0 * nu) / ((nu + 3) * sigma1**2), rel=1e-12)

    # R writes `TF()$d2ldd2(sigma1, nu)` here -- a *two-argument closed form*
    # equal to -2*nu / ((nu+3)*sigma1^2).  Substituting our TF's
    # autodiff-derived hessian instead (which takes (y, mu, sigma, nu)) swaps
    # one function for another: it returns -1.0 where R returns -3.0 at nu=3,
    # putting d2ldv2 at -0.00983 against R's -0.12802.  Since this is the only
    # term steering the nu direction, the fit then ran into the nu > 2 floor
    # (nu_eta = -36.0) instead of R's optimum at nu_eta = 2.22.  The value
    # below is the closed form, asserted rather than derived.
    tf_hess_sigma = -(2.0 * nu) / ((nu + 3.0) * sigma1**2)
    val = polygamma(1, (nu + 1) / 2) - polygamma(1, nu / 2) + \
        2.0 * (nu + 5) / (nu * (nu + 1) * (nu + 3))
    val = val / 4.0 + ds1dv**2 * tf_hess_sigma
    # R: ifelse(d2ldv2 < -1e-15, d2ldv2, -1e-15) -- it only replaces values
    # that are *not* below the threshold.  Python's max(val, -1e-15) is not the
    # same function: it clamps every negative value upward, which turned a
    # correct -0.02435836 into -1e-15 and looked like an implementation bug.
    assert val < -1e-15, f"probe point no longer exercises the else branch: {val}"
    assert got_nu == pytest.approx(val, rel=1e-10, abs=1e-15)


def test_scores_match_live_r_at_every_probe(r_fit_fixture):
    """Our three scores against R's, element-wise.

    R's values, read from``TF2()$dldm/dldd/dldv`` at
    ``(y=0.3, mu=0, sigma=1)``:

    ==============  =========  =========  =========
    nu              dldm       dldd       dldv
    ==============  =========  =========  =========
     2.5            1.779661-0.466102   -0.311396
     3.0            1.100917   -0.669725   -0.184804
     5.0            0.582524   -0.825243   -0.042506
    11.2            0.393972   -0.881808   -0.006164
    30.0            0.331079   -0.900676   -0.000744
    ==============  =========  =========  =========

    Matching the score but not the hessian was the whole difficulty in D2.20:
    the scores were fixable in isolation and the fit still ran away, because
    ``d2ldv2`` is what actually steers nu.  Asserting all three together keeps
    that failure mode from recurring.
    """
    fam = resolve_family(FAMILY)
    y, mu, sigma = 0.3, 0.0, 1.0
    Y, MU, SIGMA = (np.array([v]) for v in (y, mu, sigma))
    table = {
        2.5: (1.779661, -0.466102, -0.311396),
        3.0: (1.100917, -0.669725, -0.184804),
        5.0: (0.582524, -0.825243, -0.042506),
        11.2: (0.393972, -0.881808, -0.006164),
        30.0: (0.331079, -0.900676, -0.000744),
    }
    for nu, (e_mu, e_sigma, e_nu) in table.items():
        NU = np.array([nu])
        got = [
            float(np.asarray(fn(Y, MU, SIGMA, NU), dtype=float)[0])
            for fn in (fam.score_functions["mu"], fam.score_functions["sigma"],
                       fam.score_functions["nu"])
        ]
        for par, g, e in zip(("mu", "sigma", "nu"), got, (e_mu, e_sigma, e_nu)):
            assert g == pytest.approx(e, rel=1e-6, abs=1e-6), (
                f"{par} at nu={nu}: ours {g:.6f} vs R {e:.6f}"
            )


def test_r_deviance_formula_is_minus_two_log_density(r_meta_fixture):
    """R's TF2 deviance increment, read from the family object.

    Guards the `G.dev.incr` vs `g.dev.incr` trap: the lowercase lookup is NULL
    through partial matching, so a probe that used it would report "R has no
    deviance function" -- which reads like a finding rather than a typo.
    """
    assert "-2 * dTF2(" in str(r_meta_fixture["dev"])
    assert "log = TRUE" in str(r_meta_fixture["dev"])


# ---------------------------------------------------------------------------
# 3. Metadata against live R.
# ---------------------------------------------------------------------------


def test_links_and_valid_match_r(r_meta_fixture):
    # R's jsonlite drops names unless as.list() is used, so the keys come back
    # as "mu.link" / "nu.link" -- the flat "mu" spelling asserted here on the
    # Python side is the same trap D2.19 hit.
    assert r_meta_fixture["links"] == {
        "mu.link": R_LINKS["mu"],
        "sigma.link": R_LINKS["sigma"],
        "nu.link": R_LINKS["nu"],
    }
    assert r_meta_fixture["nu.valid.pos"] is False, "R: all(nu > 0)"
    assert r_meta_fixture["sigma.valid.pos"] is False, "R: all(sigma > 0)"


def test_nu_initial_is_fours_not_tfs_ten(r_meta_fixture):
    """R's TF2 starts nu at 4 while TF starts at 10.

    Both facts are asserted on purpose.  The second one is what makes the first
    one safe: it documents that the TF/TF2 split is a real difference rather
    than a transcription slip, so a future reader does not "simplify" the two
    branches into one.
    """
    assert r_meta_fixture["nu.initial"] == R_NU_INITIAL
    assert r_meta_fixture["TF.nu.initial"] == 10.0
    assert r_meta_fixture["nu.initial"] != r_meta_fixture["TF.nu.initial"]


def test_tf2_keeps_logshiftto2_nu_link():
    """In-place guard against a plausible-looking "fix".

    ``logshiftto2`` looks odd next to every other family's ``log`` and is the
    kind of thing a reader tidies up.  It is R's own choice and it is what
    enforces ``nu > 2`` for TF2; changing it to ``log`` would let the RS path
    evaluate the density below the boundary where ``(nu-2)/nu`` goes negative.
    """
    fam = resolve_family(FAMILY)
    assert fam.links["nu"] == "logshiftto2"
    assert fam.links["mu"] == "identity"
    assert fam.links["sigma"] == "log"


@pytest.fixture(scope="module")
def r_low_nu_fixture():
    """Live-R ``dTF2`` across ``nu = 2``, which is R's domain boundary.

    ``sigma1 = sqrt((nu-2)/nu) * sigma`` is imaginary at and below 2, so R
    returns NaN and emits ``In sqrt((nu - 2)/nu) : NaNs produced``.  Probed
    with ``tryCatch`` because a bare call is fine here but the pattern is the
    one that costD2.19 four debugging rounds elsewhere.
    """
    if not RSCRIPT:
        pytest.skip("Rscript not found")
    lines = [
        "suppressMessages(library(gamlss.dist))",
        "probe <- function(nu) {",
        "  v <- tryCatch(suppressWarnings("
        "as.numeric(dTF2(0.5, 0, 1, nu, log = TRUE))), error = function(e) NA_real_)",
        "  if (is.na(v)) NULL else v",
        "}",
        'out <- list(`1.5` = probe(1.5), `2.0` = probe(2.0),',
        '  `2.0001` = probe(2.0001), `3.0` = probe(3.0))',
        "cat(jsonlite::toJSON(out, digits = 17, auto_unbox = TRUE))",
    ]
    return _run_r(lines)


def test_r_is_nan_at_or_below_nu_two(r_low_nu_fixture):
    """The evidence for our ``nu > 2`` floor, taken from R rather than asserted.

    This is D2.19's floor criterion applied in the *legitimate* direction.
    ``SST``'s ``tau >= 2 + eps`` clamp and ``ST3C``'s former ``tau >= 1 + eps``
    clamp looked alike but had opposite signs: R is NaN below SST's bound (so
    the clamp marks a real edge) and finite below ST3C's (so the clamp deleted
    a real region).  R is NaN at and below TF2's bound of 2, which puts this
    clamp on the same side as SST's.
    """
    # jsonlite renders an R NULL as `{}` (an empty object), not `null`, so the
    # "not evaluable" marker arrives as an empty dict rather than None.  Both
    # spellings are accepted because D2.19's SST probe produced a real JSON
    # null -- the two R-side shapes are both in the wild depending on whether
    # the value went through as.list().
    def _finite(v) -> bool:
        if v is None or (isinstance(v, dict) and not v):
            return False
        return bool(np.isfinite(float(v)))

    below = [r_low_nu_fixture.get(k) for k in ("1.5", "2.0")]
    assert not any(_finite(v) for v in below), f"R evaluated dTF2 at or below nu=2: {below}"
    for k in ("2.0001", "3.0"):
        assert _finite(r_low_nu_fixture[k]), f"R should be finite at nu={k}"


def test_nu_floor_does_not_flatten_the_density():
    """Our floor sits above 2, so the density still varies with nu.

    Guards the ST3C failure mode from the other side.  A floor is legitimate
    here *because* R is NaN below 2 -- but it must not extend far enough to
    flatten the usable range, which is what ST3C's ``1 + eps`` did.
    """
    fam = resolve_family(FAMILY)
    vals = [
        float(np.asarray(fam.d(0.5, 0.0, 1.0, nu, log=True), dtype=float))
        for nu in (2.5, 3.0, 5.0, 11.2)
    ]
    assert all(np.isfinite(v) for v in vals)
    assert len(set(vals)) == len(vals), f"density is constant in nu: {vals}"


# ---------------------------------------------------------------------------
# 4. The fit itself.
# ---------------------------------------------------------------------------


def test_rs_deviance_matches_live_r(r_fit_fixture):
    """The acceptance criterion: deviance on R's number, not "it converged".

    Re-measured 2026-11-02 against live R 4.6.0 / gamlss 5.5.0 on the current
    scanner sample: R stops after 4 cycles with nu = 37.7756 and deviance
    142.7996894991248; we stop after 4 cycles with nu = 25.5886 and the *same*
    deviance to the last digit.  The previous recording was 144.171705 with R's
    nu driven to 3.6e20 -- a different sample/stopping regime, which is why
    R_DEV moved.

    nu is weakly identified here: the deviance surface along nu is flat to
    fifteen digits well below R's stopping point, so two different nu values are
    the same objective value.  That is why the assertion below is two-sided with
    a tolerance rather than an equality.
    """
    import warnings

    data = _scanner_sample(FAMILY)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = gamlss("y ~ x", family=FAMILY, data=data, method="RS", max_iter=500)
    r_dev = float(r_fit_fixture["deviance"])
    assert r_dev == pytest.approx(R_DEV, rel=1e-12), (
        f"R's deviance moved to {r_dev!r}; R_DEV is a recorded baseline, so either "
        "the scanner's sample or R's stopping rule changed underneath us"
    )
    rel = abs(model.deviance - r_dev) / r_dev
    assert rel < FIT_REL_TOL, (
        f"{FAMILY}: deviance {model.deviance:.6f} vs R {r_dev:.6f} (rel {rel:.3e})"
    )
    assert model.converged, f"{FAMILY} did not converge"


def test_nu_direction_is_flat_so_the_two_stops_agree(r_fit_fixture):
    """Cross-evaluate at *R's* coefficients, not just compare deviances.

    Equal deviance numbers can agree by coincidence; matching at R's own
    coefficient vector cannot. R's ``nu`` lands past the ``nu > 1e6`` normal
    switch (``nu = 3.6e20``), which is exactly why the missing branch in
    ``dTF``/``_log_density`` was worth 2334.79 - 144.17 = 2190.6 of deviance.
    """
    import warnings

    fam = resolve_family(FAMILY)
    r = r_fit_fixture
    data = _scanner_sample(FAMILY)
    y = np.asarray(data["y"], dtype=float)
    x = np.asarray(data["x"], dtype=float)
    mu = float(r["mu"][0]) + float(r["mu"][1]) * x
    inv = {"identity": lambda v: v, "log": np.exp,
           "logshiftto2": lambda v: np.exp(v) + 2.0}
    sigma = inv[fam.links["sigma"]](float(r["sigma"]))
    nu = inv[fam.links["nu"]](float(r["nu"]))
    assert nu > 1e6, (
        f"R's optimum should sit in the normal branch, got nu={nu:.3e}. "
        "If R's optimum moved back into the t branch this test is checking "
        "nothing -- revisit whether the nu > 1e6 switch is still reachable."
    )
    total = 0.0
    for i in range(len(y)):
        total += -2.0 * float(np.asarray(
            fam.d(float(y[i]), float(mu[i]), sigma, nu, log=True), dtype=float))
    assert total == pytest.approx(float(r["deviance"]), rel=1e-9, abs=1e-9), (
        f"our density at R's coefficients gives {total:.6f}, "
        f"R reports {float(r['deviance']):.6f}"
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = gamlss("y ~ x", family=FAMILY, data=data, method="RS", max_iter=500)
    # Our own stop must also be on the plateau, not on the approach to it.
    assert model.deviance <= float(r["deviance"]) + 1e-6, (
        f"we are worse than R: {model.deviance:.6f} > {float(r['deviance']):.6f}"
    )


def test_dropping_the_ds1dd_term_is_absorbed_by_the_other_parameters(monkeypatch):
    """A negative result worth keeping: the sigma defect is NOT isolable.

    Dropping the ``ds1dd`` chain term from the sigma score -- exactly what
    D2.20 removed -- leaves the fitted deviance at 144.172 against R's 144.172,
    a relative difference of **1.4e-13**.  The fit does not degrade at all.

    The reason is that ``sigma`` is not independently identified here: at fixed
    ``mu`` and ``nu``, ``sigma1 = sqrt((nu-2)/nu) * sigma`` is a bijection, so a
    rescaling of the sigma score is absorbed by the sigma estimate itself and
    the same optimum is reached.  The original defect was only visible because
    all three scores were wrong *together* (and the density lacked the normal
    branch), not because of any single term.

    This is recorded as a test rather than deleted because the natural
    assumption -- "each chain-rule term is individually necessary" -- is false,
    and the first version of this file asserted it and failed.
    """
    import dataclasses
    import warnings

    from omnilss import distributions as dist

    data = _scanner_sample(FAMILY)
    original = dist.TF2()
    tf = dist.TF()

    def dldd_no_chain(y, mu, sigma, nu):
        # TF's score at the *raw* sigma: exactly what D2.20 removed.
        return tf.score_functions["sigma"](
            y, mu, sigma,
            jnp.maximum(jnp.asarray(nu, dtype=jnp.float64), 2.0 + 1e-16))

    broken = dataclasses.replace(original, score_functions={
        **original.score_functions, "sigma": dldd_no_chain})

    monkeypatch.setattr(dist, "TF2", lambda: broken)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = gamlss("y ~ x", family=FAMILY, data=data, method="RS", max_iter=500)

    assert model.deviance == pytest.approx(R_DEV, rel=1e-6), (
        f"deviance {model.deviance:.6f} -- if this moved away from R, the sigma "
        "term does matter on its own and this docstring needs rewriting"
    )


def test_normal_limit_branch_matches_the_normal_density():
    """The nu > 1e6 switch must hand back the normal limit, not a t tail.

    History matters here.  This used to be a *fit-level* test: dropping the
    branch cost 2190 deviance because R's TF2 optimum sat at nu = 3.6e20.  On
    the current scanner sample R stops at nu = 37.8 (see
    test_rs_deviance_matches_live_r), so the fit never enters the branch and a
    fit-level assertion cannot exercise it any more -- it would pass with the
    branch deleted, which is exactly the failure this rewrite prevents.

    So the branch is pinned where it actually lives: at large nu the TF2
    density must equal the normal density with TF2's variance scaling
    sigma1 = sigma * sqrt((nu-2)/nu).
    """
    import jax.numpy as jnp
    from jax.scipy.stats import norm

    from omnilss import distributions as dist

    fam = dist.TF2()
    x = jnp.asarray([0.3, -0.9, 1.7])
    for nu in (1e7, 1e9, 1e12):
        tf2_density = np.asarray(fam.d(x, 0.0, 1.0, nu), dtype=float)
        normal_density = np.asarray(norm.pdf(x, 0.0, 1.0), dtype=float)
        np.testing.assert_allclose(
            tf2_density, normal_density, rtol=1e-6,
            err_msg=f"nu={nu:g} did not reach the normal limit",
        )
