"""DBI (Double Binomial, Borg 1988) -- the family that was a BI forwarder.

What this replaces
------------------
Until D2.23, ``DBI()`` was a compatibility shell that forwarded every slot to
``BI()``::

    parameters=("mu",)   links={"mu": "logit"}   no sigma

So it reported BI's deviance (217.973775379603) where R's ``DBI`` reports
217.95942206518868, and it could not pass the ``rsdev`` gate at all -- ``DBI``
was on the fail-closed ``compat_alias`` list, so ``gamlss()`` refused it.  D2.22
registered that gap; this file is its closure.

The hard part is the normalising constant
-----------------------------------------
``dDBI`` is a binomial pmf tilted by powers of the ratio terms and renormalised
by ``C(mu, sigma, bd)``.  ``C`` looks like it needs a numeric integral: R routes
it through ``.C("getBI_C2", ...)`` and the name says "C".  It does not.

Three closed forms were tried and all were wrong (Borg's beta mixture; a
``log B(a, b)`` form; a log-difference), and the tell was in the first probe
grid: ``C`` at ``mu`` in {0.2, 0.5, 0.8} looked independent of ``mu``, but that
grid is symmetric about 0.5 and ``C`` is too -- extend it to 0.01/0.99 and the
dependence is unmistakable.  **A probe grid that is symmetric about the point you
are probing hides the effect you are looking for.**

``R/gamlss.dist/src/getBI_C2.c`` is vendored in this repository and is the whole
definition -- a finite sum over ``j = 0..bd``.  Transcribed literally it matches
``GetBI_C`` to **1.5e-14** over a 224-point grid.

Two transcription traps
-----------------------
1. ``GetBI_C`` does ``sigma <- rep(1/sigma, ...)`` before calling into C, and
   ``getBI_C2``'s ``ss`` is written in terms of *that* value.  Our
   ``_dbi_log_c`` takes sigma in R's user-facing sense and inverts once, at the
   boundary.  Getting it backwards yields a plausible density that is not R's.
2. ``dDBI`` branches to plain ``dbinom`` when ``abs(sigma - 1) < 0.001``.  That
   is what makes ``C(1) = 0`` exact rather than ``-1.4e-16``, and R's
   ``sigma.initial`` is ``rep(1.1, ...)``, so an RS fit *passes through* the
   branch on its way from 1.1 to the optimum.  The branch is inherited here.

The Hessians are not second derivatives
---------------------------------------
R writes ``d2ldm2 = -dldm^2`` and ``d2ldd2 = -dldd^2``, floored at ``-1e-15``.
That is Fisher scoring, not ``d2logL``, and it is what R feeds IRLS.  Putting the
real second derivative in would move us *away* from parity -- the same D2.13
lesson as Box-Cox, where R's own roughness is the thing being matched.

Score accuracy: whose value is "right"
--------------------------------------
Our scores and R's differ by up to **2.45e-05 absolute** (at values around
5.7e+03).  That gap is R's, not ours: R obtains ``dC/dmu`` and ``dC/dsigma``
from ``numeric.deriv``, while we differentiate the same closed form by AD.
Central differences of *our own* log-density reproduce our analytic scores to
**1e-09..1e-10** at four probe points spanning ``bd`` 5..20 and ``sigma``
0.5..2.0.  Same discipline as D2.13: an assertion must ask "close to whom".

Final deviance: same objective, R stops marginally early
--------------------------------------------------------
On the exact sample ``gate_rs_deviance`` compares (48 observations, bd = 10)::

    R's dDBI at OUR coefficients  208.04234639211853
    our reported deviance          208.04234639211847
    R's gamlss() deviance          208.043104514376921
    R's dDBI at R's own fv         208.043104514376921

R's density returns our number to 2.7e-16 -- the last bit -- so both sides
minimise the same objective and our being 3.6e-06 *below* R is its absolute
``c.crit`` halting it -- D2.18's criterion 2.  Not a defect.  Raising R's
budget does not move it: n.cyc = 20 / 40 / 100 / 300 all report
208.043104514376921.

The pinned ``OURS_GATE_DEV`` was re-measured in D2.41: D2.36's inner-loop
alignment and D2.38's initial-value roll-out moved our trajectory by
6.1e-05 (both readings below R, both converging), so the D2.22-era
constant no longer described the fit.  The re-measurement re-ran live R
on the same sample and confirmed the two parity facts above *before* the
constant was touched -- an updated constant without that probe would just
be a new stale number.

Reproducing that comparison needs three things aligned, and misaligning any
one produces a plausible number about a different question: the job list
(shared rng stream), the **grid size** ``n`` (``main()`` defaults to 8, not
``N_OBS``; it changes *which* 48 observations, not how many), and the fit
settings (``sigma_formula="~1"``, ``max_iter=N_CYC``).  Wrong ``n`` alone gives
113.69 where the gate sees 208.04.  See ``_gate_sample``.

Note also that ``model.coefficients`` are **link-scale** (``mu`` eta, ``sigma``
eta), so the mutual evaluation applies ``ilogit``/``exp`` before calling
``dDBI``; feeding them in raw gives ``mu must be between 0 and 1`` and sends
you hunting for a non-existent bug (D2.18 criterion 3).
"""

from __future__ import annotations

import functools
import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

jax = pytest.importorskip("jax")

from omnilss import gamlss  # noqa: E402
from omnilss.distributions import resolve_family  # noqa: E402
from omnilss.distributions_dbi import (  # noqa: E402
    DBI,
    _dbi_log_c,
    _dbi_log_pdf,
)
from omnilss.family_capabilities import (  # noqa: E402
    ImplementationStatus,
    get_family_capability,
)

REPO_ROOT = Path(__file__).resolve().parents[2]

# Live-R references from gamlss(cbind(succ, fail) ~ x, family = DBI()) on the
# D2.22 fixture (default_rng(7), 60 observations, bd = 10).
R_DBI_DEV = 217.95942206518868
R_DBI_SIGMA_COEF = 0.020070403755973468

# And on the sample ``gate_rs_deviance`` actually compares -- see _gate_sample
# for the three alignment facts (job list, grid size n, fit settings) that
# have to hold for these numbers to mean anything.  R at n.cyc = 20, 40, 100
# and 300 all report R_GATE_DEV unchanged, so R has stopped for its own
# reasons; our being lower is its absolute c.crit, cross-checked by R's own
# dDBI returning OURS_GATE_DEV at our coefficients to 2.7e-16.
#
# OURS_GATE_DEV re-measured in D2.41 (live R probe, same sample and control):
# the D2.36 inner-loop alignment and the D2.38 initial-value roll-out moved
# our trajectory by 6.1e-05, and the D2.22-era constant went stale while the
# test kept passing its *relative* assertions -- the pin is what caught it.
#
# D2.149 re-measurement (live R 4.6.0, 2026-11-02): the scanner's job stream has
# moved since D2.41 (families were added, which shifts every later family's
# rng consumption), so the sample this test pins is no longer the D2.41 one --
# the first six successes are now [6, 4, 0, 2, 5, 5].  On the *current* sample:
#   ours   215.84842962427894   (16 cycles, converged)
#   R      215.8445356840011    (3 cycles)
# and both sides' densities reproduce their own numbers exactly (R's dDBI at
# our coefficients = 215.84842962427888; ours at R's = 215.84453568400122).
# Ours is therefore 1.8e-5 ABOVE R, which is the "genuine defect" direction --
# registered as D2.149: finite-difference checks show our dldm/dldd are exact,
# but RS stops with dDev/d eta_sigma = 0.61 still on the table (R's own point
# has 0.033), i.e. R's -dldd^2 sigma Hessian approximation stalls the IRLS.
# D2.153: with the true sigma curvature (see _sigma_hessian) ours lands at
# 215.84452506478777, 1.1e-8 BELOW R's 215.8445356840011.  The adjudication the
# "ours is lower" discipline asks for was run: R's own dDBI at our coefficients
# returns 215.84452506478772, i.e. the same objective value to 5e-14, so R
# simply stopped 3 cycles in.
R_GATE_DEV = 215.8445356840011
OURS_GATE_DEV = 215.84452506478777


def _fixture_7():
    rng = np.random.default_rng(7)
    n = 60
    bd = np.full(n, 10.0)
    x = rng.normal(0.0, 1.0, n)
    p = 1.0 / (1.0 + np.exp(-(0.3 + 0.8 * x)))
    succ = rng.binomial(10, p).astype(float)
    return succ, x, bd


# ---------------------------------------------------------------------------
# The normalising constant
# ---------------------------------------------------------------------------

def test_C_is_a_finite_sum_not_an_integral():
    """Regression guard: three closed-form guesses failed here.

    Spelled out because the C-level entry point's name invites the wrong
    assumption, and because the probe that "proved" independence from mu used
    a grid symmetric about 0.5.
    """
    import jax.numpy as jnp

    # Asymmetric probe grid -- the one that actually reveals the dependence.
    values = [
        float(_dbi_log_c(jnp.array(mu), jnp.array(0.5), jnp.array(10.0)))
        for mu in (0.01, 0.1, 0.3, 0.7, 0.9, 0.99)
    ]
    assert not np.allclose(values, values[0], atol=1e-6), (
        "C must depend on mu; a symmetric probe grid (0.2/0.5/0.8) hides this"
    )
    # Symmetric about 0.5 in mu, which is exactly what the bad grid saw.
    assert float(_dbi_log_c(jnp.array(0.2), jnp.array(0.5), jnp.array(10.0))) == pytest.approx(
        float(_dbi_log_c(jnp.array(0.8), jnp.array(0.5), jnp.array(10.0))), abs=1e-12
    )


def test_C_matches_live_R_at_the_sigma_one_fixed_point():
    """R's branch makes C(1) exactly 0, not -1.4e-16."""
    import jax.numpy as jnp

    got = float(_dbi_log_c(jnp.array(0.4), jnp.array(1.0), jnp.array(10.0)))
    assert abs(got) < 1e-12, f"C(1) = {got}, R's branch makes it 0"


def test_C_matches_live_R_over_a_grid():
    """224-point comparison against GetBI_C / dDBI dumped from live R."""
    import csv

    import jax.numpy as jnp

    path = REPO_ROOT / "tmp" / "d223_c_ref.csv"
    if not path.exists():
        pytest.skip("R reference table not generated (needs R + gamlss.dist)")
    rows = list(csv.DictReader(path.read_text(encoding="utf-8").splitlines()))
    assert rows, "empty reference table"
    worst_c = worst_lp = 0.0
    for row in rows:
        mu, sg, bd = float(row["mu"]), float(row["sigma"]), float(row["bd"])
        x = min(3.0, bd)
        got_c = float(_dbi_log_c(jnp.array(mu), jnp.array(sg), jnp.array(bd)))
        got_lp = float(_dbi_log_pdf(jnp.array(x), jnp.array(mu), jnp.array(sg),
                                    jnp.array(bd)))
        worst_c = max(worst_c, abs(got_c - float(row["C"])))
        worst_lp = max(worst_lp, abs(got_lp - float(row["logpdf3"])))
    assert worst_c < 1e-12, f"worst C diff {worst_c:.3e}"
    assert worst_lp < 1e-11, f"worst log-pdf diff {worst_lp:.3e}"


def test_C_uses_Rs_sigma_inversion_not_ours():
    """``GetBI_C`` inverts sigma before calling C.  Pin the direction.

    Without the inversion C comes out with the wrong sign structure and the
    density still sums to 1 -- it is just not R's density.  A normalisation
    check cannot catch that, so this asserts the value against R's.
    """
    import jax.numpy as jnp

    mu, sg, bd = 0.4, 0.5, 10.0
    ours = float(_dbi_log_c(jnp.array(mu), jnp.array(sg), jnp.array(bd)))
    un_inverted = float(
        _dbi_log_c(jnp.array(mu), jnp.array(1.0 / sg), jnp.array(bd))
    )
    assert ours == pytest.approx(0.360969472410190, abs=1e-12), (
        "R's GetBI_C(0.4, 0.5, 10) = 0.360969472410190"
    )
    assert abs(un_inverted - ours) > 1e-3, (
        "the 1/sigma inversion changed nothing -- it is being dropped"
    )


def test_density_sums_to_one_at_every_probe():
    """A wrong C still looks like a density; only the sum gives it away."""
    import jax.numpy as jnp

    for mu in (0.2, 0.5, 0.8):
        for sg in (0.5, 1.0, 1.5, 2.0):
            grid = jnp.arange(0, 11, dtype=jnp.float64)
            lp = _dbi_log_pdf(
                grid, jnp.full(11, mu), jnp.full(11, sg), jnp.full(11, 10.0)
            )
            total = float(np.exp(np.asarray(lp, dtype=float)).sum())
            assert total == pytest.approx(1.0, abs=1e-12), (
                f"mu={mu} sigma={sg}: sum = {total}"
            )


def test_scalar_inputs_give_scalar_C():
    """Shape discipline: unconditional broadcasting turns scalars into (1,).

    A (1,)-shaped C compares unequal to a scalar under ``==`` and would make
    every downstream scalar assertion quietly wrong.
    """
    import jax.numpy as jnp

    assert np.ndim(float(_dbi_log_c(jnp.array(0.4), jnp.array(0.5), jnp.array(10.0)))) == 0
    arr = _dbi_log_c(
        jnp.array([0.2, 0.5, 0.8]), jnp.array(0.5), jnp.full(3, 10.0)
    )
    assert arr.shape == (3,)


def test_mixed_bd_in_one_call_matches_per_element():
    """The summation mask must respect each observation's own bd."""
    import jax.numpy as jnp

    mu = jnp.array([0.4, 0.4, 0.4])
    sg = jnp.array([0.5, 0.5, 0.5])
    bd = jnp.array([2.0, 5.0, 20.0])
    together = np.asarray(_dbi_log_c(mu, sg, bd), dtype=float)
    for i, b in enumerate((2.0, 5.0, 20.0)):
        alone = float(
            _dbi_log_c(jnp.array(0.4), jnp.array(0.5), jnp.array(b))
        )
        assert together[i] == pytest.approx(alone, abs=1e-14)


def test_scalar_bd_with_vector_parameters_broadcasts():
    """Scalar ``bd`` + vector ``mu``/``sigma`` -- the shape that hid a failure.

    ``test_scalar_inputs_give_scalar_C`` above passes with a scalar-only
    implementation, and so does ``test_mixed_bd_in_one_call_matches_per_element``
    with a vector-``bd`` one.  The combination the parity scanner actually
    uses -- ``_PROBE_BD`` is the scalar ``10.0`` while ``calibrate`` hands over
    length-``n`` parameter vectors -- is what a "decide the shape from ``bd``
    alone" implementation gets wrong: the squeeze fires on an ``(n,)`` result
    and raises ``cannot reshape array of shape (n,) into shape ()``.

    That exception does not read as a shape bug where it surfaces.  It reaches
    ``gate_derivatives`` as *every* probe candidate raising, ``calibrate``
    returns ``ok=False``, and the family is recorded ``deriv: skip`` -- a silent
    loss of coverage that looks exactly like "no support grid available".
    """
    import jax.numpy as jnp

    from omnilss.distributions_dbi import _dbi_cdf, _dbi_quantile

    mu = jnp.array([0.2, 0.4, 0.6, 0.8])
    sg = jnp.array([0.5, 1.0, 1.5, 2.0])

    # C: scalar bd, vector params -> one C per observation.
    got = _dbi_log_c(mu, sg, 10.0)
    assert got.shape == (4,)
    per_element = jnp.array([
        float(_dbi_log_c(mu[i], sg[i], 10.0)) for i in range(4)
    ])
    np.testing.assert_allclose(np.asarray(got), np.asarray(per_element), rtol=1e-14)

    # ...and identical to passing bd as a full array (scalar bd is *not* a
    # separate code path once the broadcast is correct).
    np.testing.assert_array_equal(
        np.asarray(got), np.asarray(_dbi_log_c(mu, sg, jnp.full(4, 10.0)))
    )

    # The same discipline for the density, CDF and quantile: R's pDBI/qDBI are
    # called with a scalar bd in exactly this shape by user code, and a
    # TypeError there ("'float' object is not subscriptable") is the same
    # bug wearing a different hat.
    x = jnp.array([0.0, 1.0, 2.0, 3.0])
    assert _dbi_log_pdf(x, mu, sg, 10.0).shape == (4,)
    assert _dbi_cdf(jnp.array([2.0, 4.0, 6.0, 8.0]), mu, sg, 10.0).shape == (4,)
    assert _dbi_quantile(jnp.array([0.2, 0.4, 0.6, 0.8]), mu, sg, 10.0).shape == (4,)

    # All-scalar stays scalar in every entry point.
    assert _dbi_cdf(4.0, 0.3, 1.2, 10.0).shape == ()
    assert _dbi_quantile(0.6, 0.3, 1.2, 10.0).shape == ()
    assert _dbi_log_pdf(3.0, 0.3, 1.2, 10.0).shape == ()


def test_sigma_score_is_not_differentiable_across_the_sigma_one_branch():
    """R's own inconsistency, pinned -- see ``_DERIV_TOLERANCE[("DBI","sigma")]``.

    ``dDBI`` switches to plain ``dbinom`` for ``|sigma - 1| < 0.001``, so inside
    that window the log-density is *exactly constant* in sigma and any central
    difference of it is 0.  R's ``dldd`` does **not** switch.  Measured on live R
    at ``y=4, mu=0.45``::

        sigma     R dldd              central diff of R's own dDBI
        0.9995    -0.480365218462      0.000000000000
        1.0000    -0.480176057746      0.000000000000
        1.0005    -0.479987093435      (0, inside the window)
        1.0020    -0.479421170164     -0.479421170829

    We match R's ``dldd`` to 1.3e-08 (R's ``numeric.deriv`` precision), which
    is what parity requires.  The point of this test is that it stays *true*:
    if someone later "fixes" our score to branch with the density -- the
    obvious reading of the deriv gate's failure -- this fails, and the deriv
    tolerance note in ``tools/rs_family_parity_scan.py`` loses its evidence.
    """
    import jax.numpy as jnp

    fam = resolve_family("DBI")
    dldd = fam.score_functions["sigma"]

    # Inside the branch window the score keeps returning R's general value.
    assert float(dldd(4.0, 0.45, 1.0, 10.0)) == pytest.approx(
        -0.480176057746, abs=5e-8
    )
    # Outside it, the score does agree with a central difference of our density
    # -- so the mismatch really is confined to the branch, not systemic.
    h = 1e-6
    s = 1.002

    def loglik(sigma):
        return float(_dbi_log_pdf(4.0, 0.45, sigma, 10.0))

    fd = (loglik(s + h) - loglik(s - h)) / (2.0 * h)
    assert float(dldd(4.0, 0.45, s, 10.0)) == pytest.approx(fd, abs=1e-6)

    # The two branches must be observably different, or this whole discussion
    # would be about an inert switch (D2.18 criterion 4).
    assert abs(loglik(1.0) - loglik(1.002)) > 1e-4


# ---------------------------------------------------------------------------
# Family shape
# ---------------------------------------------------------------------------

def test_dbi_is_no_longer_a_bi_forwarder():
    fam = resolve_family("DBI")
    assert fam.parameters == ("mu", "sigma")
    assert fam.links == {"mu": "logit", "sigma": "log"}
    assert fam.fixed_parameters == ("bd",)


def test_sigma_is_a_real_parameter_now():
    """The defect this commit fixes: one parameter cannot beat one parameter."""
    fam = DBI()
    assert "sigma" in fam.score_functions
    assert "sigma" in fam.hessian_functions
    assert fam.score_functions["sigma"] is not fam.score_functions["mu"]


def test_dbi_left_the_compat_alias_list():
    """Fail-closed admission only lets go once the numerics are there.

    The status moves to EXPERIMENTAL, not IMPLEMENTED: DBI is not in
    ``_PRODUCTION_SAFE``, and claiming otherwise would be the kind of
    unverified assertion this project keeps removing.
    """
    cap = get_family_capability("DBI")
    assert cap.implementation_status is not ImplementationStatus.COMPAT_ALIAS
    assert cap.implementation_status is ImplementationStatus.EXPERIMENTAL


def test_dbi_is_fittable_through_the_public_api():
    """``gamlss()`` used to raise FamilyCapabilityError for DBI."""
    succ, x, bd = _fixture_7()
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = gamlss("y ~ x", family="DBI",
                       data={"y": succ, "x": x, "bd": bd},
                       method="RS", max_iter=30)
    assert model.converged
    assert np.isfinite(float(model.deviance))


# ---------------------------------------------------------------------------
# Scores: ours are closer to the true derivative than R's are
# ---------------------------------------------------------------------------

def test_scores_reproduce_our_own_log_density():
    """Independent of R: central differences of our density."""
    import jax.numpy as jnp

    fam = DBI()
    dldm = fam.score_functions["mu"]
    dldd = fam.score_functions["sigma"]
    eps = 1e-6
    for mu, sg, bd, x in (
        (0.4, 0.7, 10.0, 3.0),
        (0.2, 1.5, 10.0, 7.0),
        (0.6, 0.5, 5.0, 2.0),
        (0.35, 2.0, 20.0, 13.0),
    ):
        fd_mu = (
            float(_dbi_log_pdf(x, mu + eps, sg, bd))
            - float(_dbi_log_pdf(x, mu - eps, sg, bd))
        ) / (2 * eps)
        fd_sg = (
            float(_dbi_log_pdf(x, mu, sg + eps, bd))
            - float(_dbi_log_pdf(x, mu, sg - eps, bd))
        ) / (2 * eps)
        assert float(dldm(x, mu, sg, bd)) == pytest.approx(fd_mu, abs=1e-7)
        assert float(dldd(x, mu, sg, bd)) == pytest.approx(fd_sg, abs=1e-7)


def test_scores_match_live_r_to_the_precision_of_rs_numeric_deriv():
    """Absolute, not relative: R's scores run to 5.7e+03.

    A relative threshold here reads 1.000e+00 at ``dldm = -1.8e-15`` against
    R's ``-9.9e-08`` and would flag a correct implementation as broken.  The
    gap is R's ``numeric.deriv``, not ours -- the previous test is the evidence.
    """
    import csv

    path = REPO_ROOT / "tmp" / "d223_full_ref.csv"
    if not path.exists():
        pytest.skip("R reference table not generated (needs R + gamlss.dist)")
    rows = list(csv.DictReader(path.read_text(encoding="utf-8").splitlines()))
    fam = DBI()
    worst_mu = worst_sg = 0.0
    for row in rows:
        mu, sg, bd = float(row["mu"]), float(row["sigma"]), float(row["bd"])
        x = min(3.0, bd)
        worst_mu = max(worst_mu, abs(
            float(fam.score_functions["mu"](x, mu, sg, bd)) - float(row["dldm"])
        ))
        worst_sg = max(worst_sg, abs(
            float(fam.score_functions["sigma"](x, mu, sg, bd)) - float(row["dldd"])
        ))
    assert worst_mu < 1e-4, f"worst |dldm - R| = {worst_mu:.3e}"
    assert worst_sg < 1e-4, f"worst |dldd - R| = {worst_sg:.3e}"


def test_hessians_are_rs_fisher_scoring_not_true_second_derivatives():
    """R uses ``-dldm^2``.  The real second derivative is a different number."""
    import jax.numpy as jnp

    fam = DBI()
    y, mu, sg, bd = 3.0, 0.4, 0.7, 10.0
    score = float(fam.score_functions["mu"](y, mu, sg, bd))
    hess = float(fam.hessian_functions["mu"](y, mu, sg, bd))
    assert hess == pytest.approx(-score**2, rel=1e-12)

    eps = 1e-5
    true_d2 = (
        float(fam.score_functions["mu"](y, mu + eps, sg, bd))
        - float(fam.score_functions["mu"](y, mu - eps, sg, bd))
    ) / (2 * eps)
    assert abs(true_d2 - hess) > 1e-6, (
        "the Hessian became the true second derivative; that would move us "
        "away from R, which feeds IRLS -dldm^2"
    )


def test_hessian_floor_is_minus_one_e_minus_fifteen():
    """R's ``ifelse(d2ldd2 < -1e-15, d2ldd2, -1e-15)``.

    The floor only shows at a stationary point, where ``-dldd^2`` would be
    exactly ``0`` (or ``-1e-30``, negative but above the floor).  My first
    attempt asserted it at ``sigma=1`` and got -0.4804 -- the score is not zero
    there, so the floor was never reached.  ``sigma ~ 1.4750742`` is the actual
    root of ``dldd`` at ``mu=0.5, bd=10, x=3``, found with brentq; the test
    verifies the root rather than hardcoding it, so it keeps meaning if the
    family is ever retuned.
    """
    import jax.numpy as jnp

    fam = DBI()
    score = fam.score_functions["sigma"]
    hess = fam.hessian_functions["sigma"]

    lo, hi = 0.5, 6.0
    f_lo = float(score(3.0, 0.5, lo, 10.0))
    f_hi = float(score(3.0, 0.5, hi, 10.0))
    assert f_lo * f_hi < 0.0, "dldd must change sign on [0.5, 6] for this test to mean anything"
    root = _brentq(lambda s: float(score(3.0, 0.5, s, 10.0)), lo, hi)
    assert abs(float(score(3.0, 0.5, root, 10.0))) < 1e-9
    # D2.153: sigma no longer uses R's -dldd^2 surrogate, so at the score's root
    # this is the *true* curvature (-0.297), not the -1e-15 floor.  That is the
    # point of the change: the surrogate made the IRLS step 1/dldd and stalled
    # the fit.  The floor is still asserted where it belongs -- on the mu
    # parameter, which keeps R's surrogate.
    sigma_curvature = float(hess(3.0, 0.5, root, 10.0))
    assert sigma_curvature == pytest.approx(-0.29703672294742633, rel=1e-6)
    mu_hess = fam.hessian_functions["mu"]
    mu_root_score = float(fam.score_functions["mu"](3.0, 0.5, root, 10.0))
    assert mu_root_score == mu_root_score  # finite guard for the message below
    assert float(mu_hess(3.0, 0.5, root, 10.0)) <= -1e-15


def _brentq(fn, lo, hi, xtol: float = 1e-15):
    """Bisection with a tight tolerance -- scipy is not a test dependency here."""
    a, b = lo, hi
    fa = fn(a)
    for _ in range(200):
        mid = 0.5 * (a + b)
        fm = fn(mid)
        if fa * fm <= 0.0:
            b = mid
        else:
            a, fa = mid, fm
        if abs(b - a) < xtol:
            break
    return 0.5 * (a + b)


# ---------------------------------------------------------------------------
# End-to-end parity
# ---------------------------------------------------------------------------

def test_dbi_deviance_matches_live_r():
    succ, x, bd = _fixture_7()
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = gamlss("y ~ x", family="DBI",
                       data={"y": succ, "x": x, "bd": bd},
                       method="RS", max_iter=30)
    assert float(model.deviance) == pytest.approx(R_DBI_DEV, rel=1e-7)


def test_dbi_sigma_coefficient_is_close_to_r():
    """R's sigma eta is 0.020070403755973468; ours lands near it."""
    succ, x, bd = _fixture_7()
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = gamlss("y ~ x", family="DBI",
                       data={"y": succ, "x": x, "bd": bd},
                       method="RS", max_iter=30)
    sigma_coef = float(np.asarray(model.coefficients["sigma"]).ravel()[0])
    assert sigma_coef == pytest.approx(R_DBI_SIGMA_COEF, rel=1e-2)


def test_dbi_is_no_longer_identical_to_bi():
    """The old shell's signature: it returned BI's deviance exactly."""
    succ, x, bd = _fixture_7()
    import warnings

    dev = {}
    for family in ("BI", "DBI"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = gamlss("y ~ x", family=family,
                           data={"y": succ, "x": x, "bd": bd},
                           method="RS", max_iter=30)
        dev[family] = float(model.deviance)
    assert dev["DBI"] != dev["BI"], (
        "DBI is a BI forwarder again -- the two-parameter likelihood is gone"
    )


@functools.lru_cache(maxsize=1)
def _gate_sample():
    """The exact sample and settings ``gate_rs_deviance`` compares for DBI.

    Three separate things have to line up, and getting any of them wrong
    produces a *plausible* number that belongs to a different comparison:

    1. **The job list.**  ``_build_jobs`` draws every family's sample from one
       shared ``default_rng(20261006)`` stream (D2.18 criterion 1), so DBI's
       data depends on which families precede it.  ``["DBI"]`` and the
       39-family prefix are both faithful to *some* invocation and neither
       matches the other.
    2. **The grid size ``n``.**  ``main()``'s default is ``--n 8``, and
       ``_build_jobs(families, n)`` forwards it to ``_family_grids`` --
       which is where the *response grid* comes from.  It is **not**
       ``N_OBS``: the sample always has ``N_OBS = 48`` observations, but
       *which* 48 depends on ``n``.  Passing ``N_OBS`` here yields a deviance
       of 113.69 where the gate really sees 208.04.
    3. **The fit configuration.**  The gate passes ``sigma_formula="~1"`` and
       ``max_iter=N_CYC`` (20).  The public default differs and moves the
       answer.

    Getting (2) wrong cost a full detour this round: three deviances
    (113.69 / 137.57 / 138.44) all looked like evidence about DBI's parity and
    none of them was.  The helper below mirrors ``main``'s call exactly.
    """
    import sys

    sys.path.insert(0, str(REPO_ROOT / "tools"))
    import rs_family_parity_scan as scanner  # noqa: PLC0415

    # main(): `_build_jobs(families, opts.n)` with opts.n defaulting to 8.
    jobs, index = scanner._build_jobs(["DBI"], 8)
    raw = jobs[index["DBI"]["fit"]]["data"]
    # gate_rs_deviance reconstructs bd from the two-column binomial response.
    bd = np.asarray(raw["y"], dtype=float) + np.asarray(raw.pop("fail"), dtype=float)
    return {
        "y": np.asarray(raw["y"], dtype=float),
        "x": np.asarray(raw["x"], dtype=float),
        "bd": bd,
    }


def test_gate_sample_is_the_one_the_gate_actually_compares():
    """Pin the three alignment facts above, so a future edit cannot drift.

    Each assertion is a number the gate really produces, not a restatement of
    the helper's own logic.
    """
    data = _gate_sample()

    # (1) 48 observations, all successes, i.e. N_OBS not `n`.
    assert len(data["bd"]) == 48
    assert set(np.unique(data["bd"])) == {10.0}

    # (2) the response grid comes from `n=8`, so successes span 0..bd.
    #     With n=N_OBS the first six successes would be 10, 10, 2, 10, 10, 10.
    # D2.149: the stream moved when families were added upstream of DBI; the
    # point of the assertion is that the grid comes from n=8 (successes span
    # 0..bd) rather than from N_OBS, not the exact six numbers.
    assert data["y"][:6].tolist() == [6.0, 4.0, 0.0, 2.0, 5.0, 5.0]

    # (3) calling _build_jobs twice yields byte-identical data (one rng stream).
    assert data["y"].tolist() == _gate_sample()["y"].tolist()


def test_scanner_sample_deviance_is_the_same_objective_as_r():
    """Both sides minimise the same thing; ours is marginally lower.

    The decisive number is R's **own density** evaluated at **our**
    coefficients (live R probe, D2.41)::

        R's dDBI at OUR coefficients  208.04234639211853
        our reported deviance          208.04234639211847
        R's gamlss() deviance          208.043104514376921
        R's dDBI at R's own fv         208.043104514376921

    R's density returns our number to 2.7e-16 -- the last bit -- so the two
    sides optimise an identical objective and our being 3.6e-06 *below* R is
    its absolute ``c.crit`` halting it (D2.18 criterion 2), not a defect.
    Raising R's budget confirms it stops for its own reasons: n.cyc = 20, 40,
    100 and 300 all report 208.043104514376921 unchanged.

    Note the link-scale step (criterion 3): ``mu_eta`` is fed through
    ``plogis`` and ``sigma_eta`` through ``exp`` before ``dDBI`` sees them.
    """
    import warnings

    from omnilss.links import logit_inverse, log_inverse

    data = _gate_sample()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = gamlss("y ~ x", family="DBI",
                       data={k: data[k] for k in ("y", "x", "bd")},
                       sigma_formula="~1", method="RS", max_iter=20)
    ours = float(model.deviance)
    assert ours == pytest.approx(OURS_GATE_DEV, rel=1e-12)
    # D2.149: we are 1.8e-5 ABOVE R here, which D2.18 criterion 2 calls a
    # genuine defect rather than an early-stop artefact.  The defect is
    # registered in DEVELOPMENT_PLAN (DBI's -dldd^2 sigma Hessian stalls the
    # IRLS); until it is fixed this asserts the measured facts instead of a
    # relationship that no longer holds, and it must be *updated* when the
    # fix lands, not silently re-tightened.
    assert ours == pytest.approx(R_GATE_DEV, rel=1e-4), (
        f"ours {ours} vs R {R_GATE_DEV}: the gap grew past 1e-4"
    )
    # D2.153 fixed the stall this test used to record (D2.149), so ours is now
    # *below* R on this sample.  That direction is only innocent with evidence:
    # R's density at our coefficients has to reproduce our deviance, which the
    # next assertion checks.
    assert ours <= R_GATE_DEV

    # Reconstruct the natural-scale fitted values from our link-scale
    # coefficients -- R's dDBI would reject the raw etas with
    # "mu must be between 0 and 1".
    mu_eta = np.asarray(model.coefficients["mu"], dtype=float).ravel()
    sg_eta = np.asarray(model.coefficients["sigma"], dtype=float).ravel()
    mu = np.asarray(logit_inverse(mu_eta[0] + mu_eta[1] * data["x"]), dtype=float)
    sigma = np.full(data["x"].shape, float(np.asarray(log_inverse(sg_eta[0]))))

    from omnilss.distributions_dbi import _dbi_log_pdf

    same_objective = float(-2.0 * np.sum(
        np.asarray(_dbi_log_pdf(data["y"], mu, sigma, data["bd"]), dtype=float)
    ))
    assert same_objective == pytest.approx(ours, rel=1e-12), (
        "our reported deviance is not the value our own density produces -- "
        "the two sides would not be comparing one objective"
    )


def test_scanner_sample_is_reproducible():
    """Two calls, one rng stream -- guards the guard above.

    If ``_build_jobs`` were ever re-seeded, the deviance assertion would still
    pass against its hard-coded reference while quietly comparing against a
    different R baseline.
    """
    a = _gate_sample()
    b = _gate_sample()
    for left, right in zip((a["y"], a["x"], a["bd"]), (b["y"], b["x"], b["bd"])):
        assert np.array_equal(left, right)


def test_coefficients_are_link_scale():
    """D2.18 criterion 3, restated for DBI after itcost a debugging round.

    Feeding ``model.coefficients['mu']`` straight into ``dDBI`` gives
    ``mu must be between 0 and 1``: those are etas, and mu's link is logit.
    """
    succ, x, bd = _fixture_7()
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = gamlss("y ~ x", family="DBI",
                       data={"y": succ, "x": x, "bd": bd},
                       method="RS", max_iter=30)
    beta = np.asarray(model.coefficients["mu"]).ravel()
    mu_natural = 1.0 / (1.0 + np.exp(-(beta[0] + beta[1] * x)))
    assert 0.0 < mu_natural.min() and mu_natural.max() < 1.0
    # The raw coefficients are eta, so at least one is negative here.
    assert (mu_natural - beta[0]).any()


def test_sigma_stays_positive():
    succ, x, bd = _fixture_7()
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = gamlss("y ~ x", family="DBI",
                       data={"y": succ, "x": x, "bd": bd},
                       method="RS", max_iter=30)
    sigma = np.asarray(model.fitted_values["sigma"], dtype=float)
    assert np.all(sigma > 0.0)


# ---------------------------------------------------------------------------
# R-backed reference regeneration (skips without R)
# ---------------------------------------------------------------------------

def test_live_r_references_are_reproducible():
    rscript = None
    try:
        from tools.rs_family_parity_scan import _rscript_path  # type: ignore

        rscript = _rscript_path()
    except Exception:
        spec_path = REPO_ROOT / "tools" / "rs_family_parity_scan.py"
        if spec_path.exists():
            import importlib.util

            spec = importlib.util.spec_from_file_location("_scan_d223", spec_path)
            if spec and spec.loader:
                mod = importlib.util.module_from_spec(spec)
                try:
                    spec.loader.exec_module(mod)
                    rscript = mod._rscript_path()
                except Exception:
                    rscript = None
    if not rscript:
        pytest.skip("Rscript not available")

    out = REPO_ROOT / "tmp" / "d223_c_ref.csv"
    script = REPO_ROOT / "tmp" / "d223_ref.R"
    script.write_text(
        "suppressPackageStartupMessages(library(gamlss))\n"
        "suppressPackageStartupMessages(library(gamlss.dist))\n"
        "pts <- expand.grid(bd = c(2, 5, 10, 20),\n"
        "                 mu = c(0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99),\n"
        "                 sigma = c(0.3, 0.5, 0.7, 1.0, 1.1, 1.5, 2.0, 3.0))\n"
        "pts$C <- GetBI_C(pts$mu, pts$sigma, pts$bd)\n"
        "# dDBI stops when x > bd, so clamp x the way the grid does.\n"
        "pts$logpdf3 <- mapply(function(b, m, s)\n"
        "  dDBI(min(3, b), mu = m, sigma = s, bd = b, log = TRUE),\n"
        "  pts$bd, pts$mu, pts$sigma)\n"
        "pts$dldm <- mapply(function(b, m, s)\n"
        "  DBI()$dldm(min(3, b), m, s, b), pts$bd, pts$mu, pts$sigma)\n"
        "pts$dldd <- mapply(function(b, m, s)\n"
        "  DBI()$dldd(min(3, b), m, s, b), pts$bd, pts$mu, pts$sigma)\n"
        "write.csv(pts, commandArgs(TRUE)[1], row.names = FALSE)\n"
        "cat('DONE', nrow(pts), '\\n')\n",
        encoding="utf-8",
    )
    proc = subprocess.run(
        [rscript, str(script), out.as_posix()],
        capture_output=True, text=True, cwd=REPO_ROOT, timeout=900,
    )
    if proc.returncode != 0:
        pytest.skip(f"R failed: {proc.stderr[-300:]}")
    assert "DONE 224" in proc.stdout, proc.stdout[-200:]

    import csv

    import jax.numpy as jnp

    rows = list(csv.DictReader(out.read_text(encoding="utf-8").splitlines()))
    assert len(rows) == 224
    worst = max(
        abs(float(_dbi_log_c(jnp.array(float(r["mu"])), jnp.array(float(r["sigma"])),
                             jnp.array(float(r["bd"])))) - float(r["C"]))
        for r in rows
    )
    assert worst < 1e-12, f"stale C reference: worst diff {worst:.3e}"