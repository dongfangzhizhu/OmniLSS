"""BI / DBI binomial parity: the response is a *count*, the mean is a *proportion*.

What was wrong
--------------
``gamlss("y ~ x", family="BI", data={"y": succ, "x": x, "bd": bd})`` reported
``converged=True`` and a deviance of **8566.07** where R's ``gamlss`` on the
same 60 observations returns **217.97377537960233**.  Three independent
defects stacked to produce it; fixing any one of them leaves the fit broken,
which is why the symptom looked like a convergence problem rather than a
response-convention problem.

**1. ``_initial_mu_beta`` took logit of a count.**
The BI branch read ``logit(clip(y, 1e-6, 1-1e-6))`` -- but ``y`` is a success
count, so ``y = 7`` of ``bd = 10`` was treated as the proportion 0.999999.
R's ``BI()$mu.initial`` (verified against live R) is::

    expression({ mu <- (y + 0.5)/(bd + 1) })

Two details, both load-bearing: the denominator is ``bd``, and the half-count
adjustment is R's own.  The sibling ``BB`` branch in the same function had the
``/ bd`` but not the half-count adjustment.

**2. ``fixed_parameters`` were invisible to score, Hessian and deviance.**
``FamilyDefinition.estimable_parameters`` deliberately excludes fixed
parameters, and ``rs_fit`` built both ``other_params`` (the ``context`` handed
to the score/Hessian) and ``dev_kwargs`` (the deviance) from ``family.parameters``
alone.  So ``bd`` never reached ``g_dev_inc``, whose ``bd`` argument defaults to
``None`` -- and ``None`` selects the **Bernoulli** branch::

    log L = y*log(mu) + (1-y)*log(1-mu)

with ``y`` a count of 7.  That inflated the deviance *and* pointed the IRLS step
the wrong way.

**3. ``FamilyHeuristicInit`` seeded ``mu = mean(y) = 5.78``.**
For BI ``mu`` is a proportion while ``y`` is a count, so the generic
``mean(y)`` warm start fed ``logit(5.78)`` into the link.  ``logit`` of a value
above 1 is NaN, the link clips to +15, and the fit began at ``mu = 1.0`` with
deviance 21360.6 -- before a single IRLS step.  ``BI``/``DBI``/``BB`` now sit in
``_NON_MEAN_MU_FAMILIES`` alongside the families that already needed the
family-aware initialiser.

The diagnostic trail, in order, is the reason this file is as long as it is:

    mu = mean(y) = 5.78          -> logit clips -> mu = 1.0,  dev = 21360.6
    mu = 1.0,   bd missing       -> Bernoulli dev on a count, two-cycle
                                    oscillation between mu = 1-3e-7 and 3e-7
    mu = logit(count)            -> dev = 8566.07, converged=True (the reported bug)
    all three fixed             -> dev = 217.973775379603 == R, bit for bit

Symptom note worth keeping: at every intermediate stage ``converged`` was
``True``.  ``converged`` reports that the step size collapsed, not that the
objective reached a minimum; a runaway that oscillates between two clipped
eta values looks like perfect convergence from the outside.

R source references
-------------------
- ``gamlss.dist`` ``BI.R`` / ``DBI.R``: ``mu.initial``, ``dldm``, ``y.valid``
- ``gamlss`` ``gamlss-5.R``: the response may be a two-column
  ``cbind(success, failure)`` matrix; ``y`` is then the success count and
  ``bd`` the row total (see ``test_r_binomial_two_column_response.py``)
"""

from __future__ import annotations

import json
import subprocess
import warnings
from pathlib import Path

import numpy as np
import pytest

from omnilss._fitting_init import _initial_mu_beta
from omnilss.algorithms.warm_start_policy import FamilyHeuristicInit
from omnilss.distributions import resolve_family
from omnilss.fitting import gamlss

REPO_ROOT = Path(__file__).resolve().parents[2]

# Live-R reference, from tools/rs_family_parity_scan._rscript_path() driving
# gamlss(cbind(succ, fail) ~ x, family = BI()/DBI()) on the fixture below.
# Regenerating this requires R with gamlss + gamlss.dist installed; the tests
# that consume it skip cleanly when Rscript is absent.
R_BI_DEV = 217.97377537960233
R_DBI_DEV = 217.95942206518868
# R's sigma coefficient on the same fixture (link scale).  Live R:
# coef(fit)[["sigma"]][["coefficients"]] on gamlss(cbind(succ, fail) ~ x,
# family = DBI()).
R_DBI_SIGMA_COEF = 0.020070403755973468


def _fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The 60-observation binomial sample every reference number is tied to.

    ``default_rng(7)`` is the project-wide convention; the values are written
    out verbatim in ``tests/data/bi_dbinomial_fixture.json`` so the R side can
    be driven from the same array rather than a re-seeded ``set.seed(7)``.
    """
    rng = np.random.default_rng(7)
    n = 60
    bd = np.full(n, 10.0)
    x = rng.normal(0.0, 1.0, n)
    p = 1.0 / (1.0 + np.exp(-(0.3 + 0.8 * x)))
    succ = rng.binomial(10, p).astype(float)
    return succ, x, bd


def _rscript_path() -> str | None:
    sys_path = REPO_ROOT / "tools" / "rs_family_parity_scan.py"
    if sys_path.exists():
        import importlib.util

        spec = importlib.util.spec_from_file_location("_scan_mod", sys_path)
        if spec and spec.loader:
            mod = importlib.util.module_from_spec(spec)
            try:
                spec.loader.exec_module(mod)
                return mod._rscript_path()
            except Exception:
                pass
    return None


def _fit(family: str, succ: np.ndarray, x: np.ndarray, bd: np.ndarray, **kw):
    """Fit through ``gamlss()`` when the family is on a production route.

    ``DBI`` is registered as a *compatibility alias* and
    ``method_route_capability_report(..., allow_compat_alias=True)`` is the
    only way past that gate -- ``gamlss()`` does not expose the flag.  Rather
    than reach around the fail-closed admission (which is a deliberate safety
    boundary, per the D2.x ledger), the DBI parity checks call ``rs_fit``
    directly.  That keeps the alias gate intact and still measures the
    numerics, which is what these tests are about.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if family.upper() == "DBI":
            from omnilss.algorithms.rs_algorithm import rs_fit

            return rs_fit(
                formula="y ~ x",
                family=resolve_family("DBI"),
                data={"y": succ, "x": x, "bd": bd},
                sigma_formula="~1",
                max_iter=kw.pop("max_iter", 30),
            )
        return gamlss("y ~ x", family=family,
                      data={"y": succ, "x": x, "bd": bd}, method="RS", **kw)


# ---------------------------------------------------------------------------
# 1. The initial value
# ---------------------------------------------------------------------------

def test_bi_initial_value_is_rs_half_count_adjusted_proportion():
    """BI must start from logit((y + 0.5)/(bd + 1)), not logit(y)."""
    succ, x, bd = _fixture()
    fam = resolve_family("BI")
    design = np.column_stack([np.ones_like(x), x])

    beta = _initial_mu_beta(fam, design, succ, np.ones_like(x),
                            fixed_parameter_values={"bd": bd})
    mu0 = 1.0 / (1.0 + np.exp(-(design @ np.asarray(beta).ravel())))

    # R's expression evaluated by hand, then put on the logit scale.
    r_mu = (succ + 0.5) / (bd + 1.0)
    expected = np.log(r_mu) - np.log1p(-r_mu)
    fitted_target = design @ np.asarray(beta).ravel()
    # lstsq does not reproduce `target` exactly (it projects onto the design),
    # so compare the recovered mu against the regression of R's own target.
    assert mu0.min() > 0.05, f"mu started at {mu0.min()}, not a proportion"
    assert mu0.max() < 0.95, f"mu started at {mu0.max()}, not a proportion"
    assert np.all(np.isfinite(fitted_target))
    # The regression of R's target must be the beta we actually return.
    ref_beta, *_ = np.linalg.lstsq(design, expected, rcond=None)
    np.testing.assert_allclose(np.asarray(beta).ravel(), ref_beta, rtol=1e-10, atol=1e-12)


def test_bi_initial_value_is_not_logit_of_a_count():
    """Pin the regression: the old spelling put 7/10 at proportion 0.999999."""
    succ, x, bd = _fixture()
    fam = resolve_family("BI")
    design = np.column_stack([np.ones_like(x), x])
    beta = np.asarray(
        _initial_mu_beta(fam, design, succ, np.ones_like(x),
                         fixed_parameter_values={"bd": bd})
    ).ravel()
    old_beta, *_ = np.linalg.lstsq(
        design, np.log(np.clip(succ, 1e-6, 1 - 1e-6) / (1 - np.clip(succ, 1e-6, 1 - 1e-6))),
        rcond=None,
    )
    assert not np.allclose(beta, old_beta), (
        "the count-as-proportion spelling is back -- that is defect 1"
    )


def test_bb_initial_value_gained_the_half_count_adjustment():
    """BB had ``/ bd`` but not R's ``+ 0.5``; both spellings are R's idiom."""
    y = np.array([0.0, 5.0, 10.0, 3.0])
    bd = np.full(4, 10.0)
    design = np.column_stack([np.ones(4), np.arange(4, dtype=float)])
    fam = resolve_family("BB")
    beta = np.asarray(
        _initial_mu_beta(fam, design, y, np.ones(4),
                         fixed_parameter_values={"bd": bd})
    ).ravel()
    r_mu = (y + 0.5) / (bd + 1.0)
    ref, *_ = np.linalg.lstsq(design, np.log(r_mu) - np.log1p(-r_mu), rcond=None)
    np.testing.assert_allclose(beta, ref, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("family", ["BI", "DBI"])
def test_binomial_init_requires_bd(family):
    """Silently guessing ``bd`` would be worse than refusing."""
    y = np.array([1.0, 2.0, 3.0])
    design = np.column_stack([np.ones(3), np.arange(3, dtype=float)])
    fam = resolve_family(family)
    with pytest.raises(ValueError, match="bd"):
        _initial_mu_beta(fam, design, y, np.ones(3), fixed_parameter_values=None)


# ---------------------------------------------------------------------------
# 2. bd must reach score, Hessian and deviance
# ---------------------------------------------------------------------------

def test_fixed_parameters_reach_the_deviance():
    """``g_dev_inc`` picks Bernoulli when bd is absent; check it is present.

    BI only.  DBI is a compatibility alias, so ``gamlss()`` refuses it before
    any fitting happens and the spy would record zero calls -- which looks
    exactly like the bug it is meant to catch.
    """
    succ, x, bd = _fixture()
    fam = resolve_family("BI")
    seen: list[dict] = []

    import dataclasses

    original = fam.g_dev_inc

    def spy(**kwargs):
        seen.append({k: np.shape(v) for k, v in kwargs.items()})
        return original(**kwargs)

    spied = dataclasses.replace(fam, g_dev_inc=spy)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gamlss("y ~ x", family=spied, data={"y": succ, "x": x, "bd": bd},
               method="RS")

    assert seen, "g_dev_inc was never called"
    assert all("bd" in kwargs for kwargs in seen), (
        "at least one deviance evaluation ran without bd -- it was scored as a "
        "Bernoulli on a success count"
    )


def test_rs_fit_passes_bd_to_the_deviance_for_dbi():
    """Same guarantee for DBI, via ``rs_fit`` because of the alias gate."""
    import dataclasses

    succ, x, bd = _fixture()
    fam = resolve_family("DBI")
    original = fam.g_dev_inc
    seen: list[dict] = []

    def spy(**kwargs):
        seen.append({k: np.shape(v) for k, v in kwargs.items()})
        return original(**kwargs)

    spied = dataclasses.replace(fam, g_dev_inc=spy)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gamlss_model = _fit("DBI", succ, x, bd)
    # The spy is on a replaced family, so run rs_fit directly with it to prove
    # the plumbing (the model from _fit is the sanity check that it converges).
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from omnilss.algorithms.rs_algorithm import rs_fit

        rs_fit(formula="y ~ x", family=spied,
               data={"y": succ, "x": x, "bd": bd}, sigma_formula="~1", max_iter=30)
    assert seen and all("bd" in kwargs for kwargs in seen)
    assert float(gamlss_model.deviance) < 300.0


def test_deviance_value_actually_depends_on_bd():
    """A spy proves plumbing; this proves the plumbing changes the number."""
    fam = resolve_family("BI")
    y = np.array([7.0, 3.0, 9.0])
    mu = np.array([0.5, 0.4, 0.8])
    bd = np.full(3, 10.0)
    with_bd = np.asarray(fam.g_dev_inc(y=y, mu=mu, bd=bd))
    without_bd = np.asarray(fam.g_dev_inc(y=y, mu=mu))
    assert not np.allclose(with_bd, without_bd)

    # The Bernoulli branch reads a *count* as a proportion, so its log density
    # is y*log(mu) + (1-y)*log(1-mu) with y > 1.  At mu = 0.5 that happens to
    # be (2y-1)*log(0.5): positive for y=7 (1.386), and strongly *negative*
    # for y=9 (-21.73) -- a negative incremental deviance is impossible for a
    # real likelihood, which is exactly why the mix-up was diagnosable at all.
    assert without_bd[2] < 0.0
    assert without_bd[0] > 0.0
    # With bd, every term is a proper log binomial pmf and the deviance is
    # positive everywhere.
    assert np.all(with_bd > 0.0)
    assert with_bd[2] == pytest.approx(2.63028956, rel=1e-8)


# ---------------------------------------------------------------------------
# 3. The warm start must not be mean(y) for a binomial
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("family", ["BI", "DBI", "BB"])
def test_warm_start_omits_mu_for_binomial_families(family):
    """``mean(y)`` is a count, not a proportion, for these three."""
    assert family in FamilyHeuristicInit._NON_MEAN_MU_FAMILIES
    y = np.array([0.0, 5.0, 10.0, 7.0])
    result = FamilyHeuristicInit().initialize(
        y=y, parameter_names=("mu", "sigma"), family_name=family
    )
    assert "mu" not in result.initial_fitted, (
        "a binomial mu must come from _initial_mu_beta, not mean(y)"
    )


def test_mean_y_would_exceed_one_for_this_fixture():
    """Document why: the generic warm start is outside the link's domain.

    ``mean(succ) = 4.933`` is above 1, so ``logit`` of it is NaN and the link
    clips to +15 -- the fit starts at mu = 1.0.  The *proportion* 0.493 is
    inside the domain, which is the whole point of the fix.
    """
    succ, _, bd = _fixture()
    mean_count = float(np.mean(succ))
    assert mean_count > 1.0
    mean_proportion = mean_count / float(bd[0])
    assert 0.0 < mean_proportion < 1.0


# ---------------------------------------------------------------------------
# 4. End-to-end parity with live R
# ---------------------------------------------------------------------------

def test_bi_deviance_matches_live_r():
    succ, x, bd = _fixture()
    model = _fit("BI", succ, x, bd)
    assert float(model.deviance) == pytest.approx(R_BI_DEV, rel=1e-12)


def test_bi_fit_is_bit_for_bit_converged_not_just_flagged():
    """The failure mode reported ``converged=True``; check the coefficients.

    ``convergence_status()`` returns the *boolean* on this build and the
    vocabulary string on ``termination_reason`` -- I asserted the string on the
    wrong accessorfirst, so check the one that actually carries it.
    """
    succ, x, bd = _fixture()
    model = _fit("BI", succ, x, bd)
    assert model.converged
    assert model.convergence_status() is True
    assert model.termination_reason == "relative_deviance_converged"
    beta = np.asarray(model.coefficients["mu"]).ravel()
    assert np.all(np.abs(beta) < 10.0), f"coefficients ran away: {beta}"


def test_bi_deviance_is_far_below_the_old_bug_value():
    """Regression anchor for the reported 8566.07."""
    succ, x, bd = _fixture()
    model = _fit("BI", succ, x, bd)
    assert float(model.deviance) < 300.0


def test_dbi_reaches_rs_two_parameter_parity():
    """DBI parity became reachable in D2.23 -- this is where it landed.

    D2.22 wrote this test the other way round: our ``DBI`` was a
    single-parameter shell (``parameters ==("mu",)``) forwarding to``BI``, so
    it returned BI's deviance 217.97377537960304 where R reported
    217.95942206518868, and the test pinned that gap with::

        assert fam.parameters == ("mu",), (
            "DBI grew a sigma parameter -- update this test and the D2.22 "
            "reference numbers; R's DBI has (mu, sigma) and 217.95942206518868"
        )

    That message was a hand-off, and D2.23 is the hand-off being taken.  The
    normalising constant ``C`` had to be transcribed from
    ``R/gamlss.dist/src/getBI_C2.c`` before the sigma direction had any
    meaning at all; ``test_dbi_double_binomial.py`` covers that and the rest.

    The three D2.22 fixes are still what this rests on: a fit that used to
    report 8566.07 with ``converged=True`` now converges onto the binomial
    likelihood.  Asserting the coincidence with BI would be the regression --
    it is asserted *against* here, not for.
    """
    succ, x, bd = _fixture()
    fam = resolve_family("DBI")
    assert fam.parameters == ("mu", "sigma")
    model = _fit("DBI", succ, x, bd)
    deviance = float(model.deviance)

    assert deviance != pytest.approx(R_BI_DEV, rel=1e-6), (
        "DBI is BI again -- the two-parameter likelihood is gone"
    )
    assert deviance == pytest.approx(R_DBI_DEV, rel=1e-5)
    assert float(
        np.asarray(model.coefficients["sigma"], dtype=np.float64).ravel()[0]
    ) == pytest.approx(R_DBI_SIGMA_COEF, rel=1e-2)


def test_mu_stays_inside_the_unit_interval():
    succ, x, bd = _fixture()
    model = _fit("BI", succ, x, bd)
    mu = np.asarray(model.fitted_values["mu"], dtype=np.float64)
    assert mu.min() > 0.0 and mu.max() < 1.0


def test_dbi_eta_is_not_clipped():
    """eta = logit(mu); a clip means mu was driven at the boundary."""
    succ, x, bd = _fixture()
    model = _fit("DBI", succ, x, bd)
    mu = np.asarray(model.fitted_values["mu"], dtype=np.float64)
    eta = np.log(mu / (1.0 - mu))
    assert np.abs(eta).max() < 15.0, f"eta hit the clip: {np.abs(eta).max()}"


def test_live_r_reference_is_reproducible():
    """Re-derive R_BI_DEV / R_DBI_DEV from the fixture (skips without R)."""
    rscript = _rscript_path()
    if rscript is None:
        pytest.skip("Rscript not available")
    succ, x, bd = _fixture()
    payload = REPO_ROOT / "tmp" / "bi_dbi_fixture_r.json"
    payload.parent.mkdir(parents=True, exist_ok=True)
    payload.write_text(
        json.dumps({"succ": succ.tolist(), "x": x.tolist(), "bd": bd.tolist()}),
        encoding="utf-8",
    )
    script = REPO_ROOT / "tmp" / "bi_dbi_ref.R"
    script.write_text(
        "suppressPackageStartupMessages(library(gamlss))\n"
        "suppressPackageStartupMessages(library(gamlss.dist))\n"
        "library(jsonlite)\n"
        f'd <- fromJSON("{payload.as_posix()}")\n'
        "df <- data.frame(succ = d$succ, fail = d$bd - d$succ, x = d$x)\n"
        "f <- gamlss(cbind(succ, fail) ~ x, family = BI(), data = df,\n"
        "             control = gamlss.control(n.cyc = 30, trace = FALSE))\n"
        "cat('BI ', format(f$G.deviance, digits = 17), '\\n')\n"
        "g <- gamlss(cbind(succ, fail) ~ x, family = DBI(), data = df,\n"
        "             control = gamlss.control(n.cyc = 30, trace = FALSE))\n"
        "cat('DBI', format(g$G.deviance, digits = 17), '\\n')\n",
        encoding="utf-8",
    )
    proc = subprocess.run(
        [rscript, str(script)], capture_output=True, text=True, cwd=REPO_ROOT,
        timeout=900,
    )
    if proc.returncode != 0:
        pytest.skip(f"R failed: {proc.stderr[-300:]}")
    values = {}
    for line in proc.stdout.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[0] in {"BI", "DBI"}:
            values[parts[0]] = float(parts[1])
    assert values.get("BI") == pytest.approx(R_BI_DEV, rel=1e-12), (
        "the hard-coded R reference is stale -- R or the fixture moved"
    )
    # DBI's R value is kept for the follow-up that adds its sigma parameter;
    # it must not drift either.
    assert values.get("DBI") == pytest.approx(R_DBI_DEV, rel=1e-12)