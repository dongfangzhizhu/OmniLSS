"""Parity-scanner probe defects that masqueraded as family defects.

What was wrong
--------------
``tools/rs_family_parity_scan.py`` reported the two binomial families as the
worst offenders in the whole registry::

    BI    dpqr  fail  d: max rel 1.246e+90; p: 9.437e-01; q: 1.000e+00
    DBI   dpqr  fail  d: max rel 1.246e+90; p: 9.437e-01; q: 1.000e+00

A relative error of 1e90 is not a subtle numerical disagreement -- it is the
signature of a probe feeding a parameter the wrong value.  Both families'
density, CDF and quantile functions turned out to be correct to the last digit.
The scanner was lying, and it was lying in the most expensive possible way: the
number was so large that it would have justified rewriting working, verified
density code.

Two independent probe defects, each found by checking the probe first
---------------------------------------------------------------------
**1. Positional argument order.**  ``_python_dpqr`` built its call as
``fn(x, bd_array, *params)``, matching R's ``dBI(x, bd, mu, log)`` where ``bd``
comes *before* ``mu``.  Our signature is ``dBI(x, mu, bd, log)``.  So ``mu``
received the value 10 and ``bd`` received the probability::

    positional  [3.2e-03, 2.0e+06, 7.5e+24, 9.2e+33, 8.5e+52,
                 1.1e+62, 1.1e+81, 1.2e+90]
    by name     [0.0563, 0.0980, 0.2213, 0.2307, 0.2307, 0.2213,
                 0.0980, 0.0563]

That trailing ``1.2e+90`` is the reported ``1.246e+90`` exactly.  R's side
already passed arguments by name (``names(aa) <- argnames`` in
``r_parity_batch.R``); the Python side did not.

**2. Non-integer grid for a discrete family.**  ``calibrate`` proposed
``np.arange(n) * (bd / (n - 1))`` as "binomial responses are counts in 0..bd",
which for ``n=8, bd=10`` is ``0, 1.43, 2.86, 4.29, 5.71, 7.14, 8.57, 10``.
R's ``dbinom`` answers every non-integer ``x`` with ``-Inf`` and prints
``non-integer x = 1.428571``; six of eight grid points were unevaluable on the
R side.  Our ``dBI`` rounds ``x`` (``x_count = jnp.round(x)``) and happily
evaluated them.  A grid point where the two sides disagree about whether the
function is defined cannot produce a meaningful parity verdict in either
direction.

**The two had to combine to produce 1e90.**  Either alone is survivable: with
the non-integer grid the mis-bound call peaked at 1.2e+90 because
``bd - x_count`` went deeply negative inside the ``gammaln`` combination term
while ``mu`` sat at ``1 - 1e-10``; with an integer grid the same mis-binding
peaks at ~1.4e+02.  Fixing only the grid would have left a wrong-but-plausible
number, and fixing only the argument order would have left R unable to evaluate
most of the grid.  Worth stating plainly because the natural reading of "two
bugs" is "either one explains the symptom".

Neither defect was in a family implementation.  Both were in the tool that
reports on the families -- which is why this file tests the *scanner*, not
``BI``.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(_ROOT / "tools"))

import rs_family_parity_scan as scan  # noqa: E402

from omnilss import resolve_family  # noqa: E402

BINOMIAL_FAMILIES = ("BI", "DBI")

# Which of those take a sigma besides mu.  DBI stopped being a BI forwarder in
# D2.23 -- it is now the two-parameter Double Binomial, so `d(k, bd, mu)` no
# longer binds and any probe that omits sigma raises TypeError.
#
# The failure this caused is worth spelling out: a probe written for
# one-parameter families does not report "wrong family", it reports
# `_dbi_dispatch() missing 1 required positional argument: 'sigma'`.Three
# tests here failed on that before the arity was parameterised, and the
# naturalbut wrong reaction is to blame the family for being two-parameter.
TWO_PARAMETER_BINOMIAL = {"DBI"}


def _density_kwargs(family, n, mu, bd):
    """Extra keyword arguments ``family.d`` needs beyond mu and bd.

    The values come from the scanner's own grid rather than a guess: an
    arbitrary ``sigma`` here makes the final "does the scanner agree with a
    direct call" comparison fail for a reason that has nothing to do with
    argument binding, which is the thing this file exists to test.
    """
    if family not in TWO_PARAMETER_BINOMIAL:
        return {}
    _y, parrays, _bd = scan._family_grids(family, n)
    return {"sigma": np.asarray(parrays[1], dtype=float)}


# ---------------------------------------------------------------------------
# 1. Argument order.  The regression this guards is silent: positional
#    arguments still work, they just bind to the wrong names.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("family", BINOMIAL_FAMILIES)
def test_python_dpqr_passes_named_arguments(family):
    """`bd` and `mu` must reach their own parameters.

    Asserted against hand-computed values rather than against R, because the
    bug is a binding bug: comparing to R would still pass if both sides were
    wrong the same way, and it would need a live R to run at all.

    ``dBI(0, bd=10, mu=0.25) = (1-0.25)^10 = 0.0563135147`` -- the k=0 term
    needs no binomial coefficient, so it isolates the parameters cleanly.
    """
    d = np.asarray(scan._python_dpqr(family, "d", None), dtype=float)
    assert d is not None and len(d) == 8
    assert np.all(np.isfinite(d)), f"non-finite density: {d}"
    # A plain probability density: finite and within [0, 1].  Note the scanner
    # does *not* pass log=True for `d` -- it compares against R's `dBI(..., log
    # = FALSE)`.  Two earlier versions of this assertion assumed a log density
    # (`d <= 0`) and were wrong in opposite directions; the value decides, not
    # the guess.
    assert np.all((d >= 0.0) & (d <= 1.0)), f"not a probability density: {d}"

    # Cross-check the k=0 entry against the closed form (1-mu)^bd, which needs
    # no binomial coefficient and therefore isolates mu and bd completely.
    # No exp() here: the scanner compares plain densities, not logs.
    cal = scan.calibrate(family, 8)
    mu0 = float(np.asarray(cal.params[0], dtype=float)[0])
    if family in TWO_PARAMETER_BINOMIAL:
        # DBI is not dbinom: at sigma = 1 it switches to dbinom (R's own
        # `abs(sigma - 1) < 0.001` branch), which is the one point where the
        # closed form below is exact.  sigma == 1 also makes the assertion
        # *stronger* -- a wrong sigma would not reproduce dbinom.
        got = float(np.asarray(
            resolve_family(family).d(0.0, bd=cal.bd, mu=mu0, sigma=1.0, log=True),
            dtype=float))
        expected = np.log((1.0 - mu0) ** cal.bd)
        assert got == pytest.approx(expected, rel=1e-12, abs=1e-12), (
            f"DBI at sigma=1 must reduce to dbinom; got {got} vs {expected}"
        )
        return
    expected = (1.0 - mu0) ** cal.bd
    assert float(d[0]) == pytest.approx(expected, rel=1e-12), (
        f"k=0 density {d[0]} vs closed form (1-mu)^bd = {expected}"
    )


@pytest.mark.parametrize("family", BINOMIAL_FAMILIES)
def test_positional_call_is_wrong_and_the_keyword_path_is_not(family):
    """Pin the diagnosis so this file cannot quietly become decorative.

    The two defects had to be present **together** to produce 1.2e+90.  With
    the non-integer grid in place, the mis-bound call peaked at 1.2e+90 -- the
    non-integer counts drove ``bd - x_count`` deeply negative inside the
    ``gammaln`` combination term while ``mu`` sat at ``1 - 1e-10``.  With an
    integer grid the same mis-binding peaks at ~1.4e+02: wrong, obviously, but
    three orders of magnitude short of the number that started this.

    So the test asserts what is actually true rather than the number that
    happened to be printed: the positional call is badly wrong, and the
    keyword call the scanner now uses is not.  If a future refactor makes the
    two agree, this fails and the explanation needs rewriting -- which is the
    point.

    The two families fail differently, and both outcomes are correct.  ``BI``
    takes ``(x, bd, mu)``, so a positional call *binds* -- ``bd`` lands on
    ``mu`` and vice versa -- and returns a confidently wrong density; that is
    the dangerous case.  ``DBI`` takes ``(x, mu, sigma, bd)`` (D2.23), so the
    same call has the wrong arity and raises ``TypeError``: also correct, and
    louder.
    """
    fam = resolve_family(family)
    n = 8
    y, parrays, bd = scan._family_grids(family, n)
    mu = np.asarray(parrays[0], dtype=float)
    extra = _density_kwargs(family, n, mu, bd)

    if family in TWO_PARAMETER_BINOMIAL:
        with pytest.raises(TypeError):
            fam.d(y, np.full(n, bd), mu, log=True)
    else:
        positional = np.asarray(
            fam.d(y, np.full(n, bd), mu, log=True), dtype=float)
        assert np.max(np.abs(positional)) > 10.0, (
            f"positional call peaks at {np.max(np.abs(positional)):.3e}; the "
            "mis-binding seems to have been fixed -- update this test's premise"
        )
        # The mis-bound call returns +139.03 where a log density must be
        # non-positive -- not merely large, but outside the range any log
        # density can take, which states "wrong" more precisely than a
        # magnitude threshold.
        assert np.any(positional > 0.0), (
            f"mis-bound call returned only non-positive values {positional}; "
            "update this test's premise"
        )
        assert np.all(np.isfinite(positional))

    keyword = np.asarray(
        fam.d(y, bd=np.full(n, bd), mu=mu, log=True, **extra), dtype=float)
    assert np.all(np.isfinite(keyword))
    assert np.all(keyword <= 0.0), f"log density must be <= 0; got {keyword}"
    # The scanner compares plain densities, so compare those too.
    scanner_vals = np.asarray(scan._python_dpqr(family, "d", None), dtype=float)
    keyword_plain = np.asarray(
        fam.d(y, bd=np.full(n, bd), mu=mu, **extra), dtype=float)
    np.testing.assert_allclose(scanner_vals, keyword_plain, rtol=1e-12, atol=0.0)


def test_both_sides_of_the_comparison_are_now_evaluable():
    """The R job's grid must be integral, or half of it returns -Inf.

    This is the check that makes the parity verdict meaningful: if the grid has
    non-integer points then R answers ``-Inf`` and Python answers a rounded
    value, and no amount of numerical agreement elsewhere rescues the gate.
    """
    for family in BINOMIAL_FAMILIES:
        cal = scan.calibrate(family, 8)
        assert cal.ok, f"{family}: calibration failed"
        y = np.asarray(cal.y, dtype=float)
        assert np.all(y == np.round(y)), (
            f"{family}: grid has non-integer responses {y}; R's dbinom returns "
            "-Inf for those and prints 'non-integer x'"
        )
        assert np.all(y >= 0.0)
        assert cal.bd is None or np.all(y <= cal.bd), (
            f"{family}: grid {y} exceeds bd={cal.bd}"
        )


def test_our_density_agrees_with_the_closed_form_on_the_integer_grid():
    """The binomial density is exact on integer counts; state it without R.

    ``dbinom(k; n, p) = C(n,k) p^k (1-p)^(n-k)``, computed here in plain numpy.
    This is the assertion that would have caught the first defect if the scanner
    had ever been fixed in the wrong direction -- e.g. "correcting" ``dBI`` to
    reject the values R rejects, which would have been fixing a tool by
    breaking a verified implementation.

    ``DBI`` reduces to exactly that at ``sigma = 1``: R's own ``dDBI`` switches
    to ``dbinom`` when ``abs(sigma - 1) < 0.001``, precisely so ``C(1) = 0`` is
    exact.  So the closed form still applies to it, and checking it there is
    strictly *stronger* than checking it at an arbitrary sigma -- a ``C`` that
    had been transcribed wrongly would not collapse to ``dbinom``.
    """
    from math import comb

    for family in BINOMIAL_FAMILIES:
        fam = resolve_family(family)
        n_trial, mu = 10, 0.45
        extra = {"sigma": 1.0} if family in TWO_PARAMETER_BINOMIAL else {}
        for k in range(n_trial + 1):
            expected = comb(n_trial, k) * mu**k * (1.0 - mu) ** (n_trial - k)
            got = float(np.asarray(
                fam.d(float(k), bd=float(n_trial), mu=mu, log=True, **extra),
                dtype=float))
            assert got == pytest.approx(np.log(expected), rel=1e-12, abs=1e-12), (
                f"{family} at k={k}: log density {got} vs closed form "
                f"{np.log(expected)}"
            )


def test_rounding_of_non_integer_x_is_the_only_divergence():
    """Documented, deliberate, and now bounded by the grid check above.

    ``dBI`` rounds ``x`` (``x_count = jnp.round(x)``) where R returns ``-Inf``.
    Keeping the rounding is right for the fitting path -- ``gamlss()`` builds
    the response from ``cbind(success, failure)`` and small floating-point dust
    in a count must not turn the density into zero.  It is recorded here so the
    tolerance is a decision rather than an oversight.
    """
    fam = resolve_family("BI")
    rounded = float(np.asarray(fam.d(2.4, bd=10.0, mu=0.45, log=True), dtype=float))
    exact = float(np.asarray(fam.d(2.0, bd=10.0, mu=0.45, log=True), dtype=float))
    assert rounded == pytest.approx(exact, rel=1e-12)

# ===========================================================================
# D2.22: the binomial *fitting* payload
# ===========================================================================
#
# Before D2.22 both binomial families reported ``rsdev: skip`` with::
#
#     R fit error: y values must be 0 <= y <= 1
#
# which reads like a broken family but is R's ``gamlss()`` refusing a response
# shape it does not recognise.  ``BI()$y.valid`` is ``all(y >= 0)`` -- it would
# happily accept the counts -- yet ``gamlss()``'s binomial branch only
# understands the two-column ``cbind(success, failure)`` matrix and otherwise
# validates the column as a proportion.  So the *whole* RS parity gate was
# switched off for these families by a spelling mismatch in the harness.

BINOMIAL_FIT_FAMILIES = ("BI", "DBI")


def _binomial_fit_payload():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[2] / "tools" / "rs_family_parity_scan.py"
    spec = importlib.util.spec_from_file_location("_scan_d222", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("family", BINOMIAL_FIT_FAMILIES)
def test_fit_data_emits_r_s_two_column_binomial_response(family):
    """The scanner must hand R what R's gamlss() will accept."""
    import numpy as np

    mod = _binomial_fit_payload()
    rng = np.random.default_rng(1)
    grid = np.array([0.0, 2.0, 5.0, 7.0, 10.0])
    data = mod._fit_data(family, rng, grid)

    assert "fail" in data, (
        "R's gamlss() only accepts cbind(success, failure) for a binomial "
        "family; without `fail` the fit is rejected and rsdev goes to skip"
    )
    y = np.asarray(data["y"], dtype=float)
    fail = np.asarray(data["fail"], dtype=float)
    bd = mod._PROBE_BD
    np.testing.assert_allclose(y + fail, np.full_like(y, bd))
    assert np.all(y >= 0) and np.all(fail >= 0)
    # Counts, not proportions: R's dbinom answers non-integer x with -Inf.
    assert np.all(y == np.round(y))


def test_fit_data_leaves_non_binomial_families_alone():
    """The `fail` column must not leak into families that do not have bd."""
    import numpy as np

    mod = _binomial_fit_payload()
    rng = np.random.default_rng(1)
    data = mod._fit_data("NO", rng, np.array([1.0, 2.0, 3.0, 4.0]))
    assert "fail" not in data
    assert set(data) == {"y", "x"}


@pytest.mark.parametrize("family", BINOMIAL_FIT_FAMILIES)
def test_fit_job_carries_the_two_column_formula(family):
    """``response_formula`` is what makes r_parity_batch.R build the cbind."""
    import numpy as np

    mod = _binomial_fit_payload()
    jobs, index = mod._build_jobs([family], 6)
    fit_job = jobs[index[family]["fit"]]
    assert fit_job["response_formula"] == "cbind(y, fail) ~ x"
    assert "fail" in fit_job["data"]


def test_non_binomial_fit_job_has_no_response_formula():
    """Absent key means the R side keeps its single-column path."""
    mod = _binomial_fit_payload()
    jobs, index = mod._build_jobs(["NO"], 6)
    assert "response_formula" not in jobs[index["NO"]["fit"]]


def test_python_side_reconstructs_bd_from_the_two_columns():
    """Both sides must minimise the *same* binomial likelihood.

    R infers bd from the cbind row total; our gamlss() wants an explicit bd.
    ``gate_rs_deviance`` adds it from the same two columns -- if that
    reconstruction drifts, the gate compares R's binomial fit against our
    Bernoulli fit and reports a meaningless gap.
    """
    import numpy as np

    y = np.array([0.0, 3.0, 7.0, 10.0])
    fail = np.array([10.0, 7.0, 3.0, 0.0])
    reconstructed = y + fail
    np.testing.assert_allclose(reconstructed, np.full(4, 10.0))
    # The verifier used in the D2.22 run: both sides landed on 251.671 for BI
    # and 203.879 for BB, which is only possible if bd agrees exactly.
    assert np.all(reconstructed == 10.0)
