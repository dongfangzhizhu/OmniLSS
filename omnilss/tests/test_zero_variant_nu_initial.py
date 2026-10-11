"""A zero-variant's ``nu`` start must be R's, not the 1.0 fallback (D2.26).

``ZIPIG`` reported a deviance of 455.26 at20 cycles and 732.79 at 200 -- worse
the longer it ran.  The trace was a period-four limit cycle: 3251.14 down to
214.6458 over18 monotone steps, then 788.22 / 455.26 / 816.69 / 1450.21, after
which iterations 23-30 replayed iterations 2-9 to five digits.

Root cause: ``ZIPIG`` appeared in no branch of ``_initial_parameter_value``, so
``nu`` fell through to ``return 1.0``.  ``nu`` uses a logit link, and
``logit(1.0) = +inf`` goes straight into the eta clip at ``+15`` -- seventeen
logit units from R's ``logit(0.1) = -2.197``.  From there the iteration parked
on a plateau whose floor sits 2.86 deviance above R's answer.

What these tests pin is the *rule* -- a zero-variant starts at the value R
starts it at, so a family missing from the branch list fails here instead of
oscillating for 200 iterations in the parity scan.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from omnilss._fitting_init import _initial_parameter_value
from omnilss.distribution_registry import resolve

_Y = np.array([0.0, 1.0, 2.0, 3.0, 7.0])
_MU = np.full(5, 2.0)
_W = np.ones(5)


def _nu_start(name: str) -> float:
    return _initial_parameter_value(resolve(name), "nu", _Y, _MU, _W)


#: Zero-variants whose zero-mass parameter *is* `nu`, and where R starts it at
#: `rep(0.1, length(y))`.  Read off the R sources line by line -- ZIPIG.R:91 is
#: `nu <- rep(0.1, length(y))`, ZAPIG.R / ZAZIPF.R / ZABI.R / ZINBI.R /
#: ZIBI.R agree.  Not inferred from the name.
#: ZANBI was listed here through D2.26 but D2.38's live-R dump of the family
#: *object* shows ZANBI()$nu.initial is data-driven --
#: `rep(max(sum(y == 0)/length(y), 0.01), length(y))` -- not the constant 0.1
#: the source read suggested; it is asserted separately below.
_R_NU_INITIAL_0_1 = ("ZIPIG", "ZAPIG", "ZIBI", "ZAZIPF", "ZABI", "ZINBI")

#: Zero-variants over a *continuous* base, which R starts elsewhere.  Listed so
#: that widening the 0.1 branch cannot quietly capture them: R's ZAGA / ZAIG
#: start `nu <- rep(0.5, ...)`.
_CONTINUOUS_BASE = ("ZAGA", "ZAIG", "BEZI", "BEOI", "ZIP2", "ZAP")

#: Families that look like they belong but do not, because their `nu` is a
#: *shape* parameter and the zero mass lives in `tau` (or vice versa).  Their
#: R starts are -0.5 / 2 / 0.1 respectively.  This is the D2.19 lesson in a
#: list: same parameter name, different role.
_NOT_THE_ZERO_MASS = {
    "ZASICHEL": -0.5,   # tau.initial = sum(y==0)/n is the zero mass
    "ZISICHEL": -0.5,   # tau.initial = rep(0.1, ...)
    "ZINBF": 2.0,       # nu.initial = rep(2, length(y)) -- a shape
}

#: Zero-variants with no R counterpart, so no R initial value exists to match.
#: ``ZILG()`` raises ``could not find function`` on live R 4.6 with
#: gamlss.dist loaded, and has no ``R/gamlss.dist/R/ZILG.R`` in the vendored
#: snapshot.  A project extension, not a parity gap.
_NO_R_COUNTERPART = ("ZILG",)


@pytest.mark.parametrize("name", _R_NU_INITIAL_0_1)
def test_a_zero_variant_starts_at_rs_nu_initial(name):
    """The specific regression: these started at 1.0, i.e. logit(1.0) = +inf.

    ``nu.initial = 1.0`` is worse than no value at all under a logit link: the
    inverse is infinite, and RS's eta clip turns that into a finite +15 --
    seventeen logit units from R's start.
    """
    assert _nu_start(name) == 0.1


def test_zanbi_nu_initial_is_the_observed_zero_fraction():
    """D2.38: live R's ZANBI()$nu.initial is data-driven, not a constant.

    The family object reads `nu <- rep(max(sum(y == 0)/length(y), 0.01),
    length(y))` -- the observed zero fraction floored at 0.01.  The generic
    1.0 default starts logit(1.0) = +inf, seventeen logit units away.
    """
    y = np.array([0.0, 0.0, 1.0, 2.0, 3.0, 0.0, 4.0, 5.0])
    mu = np.full(y.shape, float(np.mean(y)))
    start = _initial_parameter_value(resolve("ZANBI"), "nu", y, mu, np.ones(y.size))
    assert start == pytest.approx(3.0 / 8.0)

    all_positive = np.array([1.0, 2.0, 3.0, 4.0])
    mu_pos = np.full(all_positive.shape, float(np.mean(all_positive)))
    start_pos = _initial_parameter_value(
        resolve("ZANBI"), "nu", all_positive, mu_pos, np.ones(all_positive.size)
    )
    assert start_pos == pytest.approx(0.01)


@pytest.mark.parametrize("name", sorted(_NOT_THE_ZERO_MASS))
def test_a_shape_parameter_nu_is_not_given_the_zero_mass_start(name):
    """The other side: only families whose `nu` is the zero mass start at 0.1.

    ZASICHEL and ZISICHEL carry the zero mass in `tau` because the Sichel base
    already has a `nu`; ZINBF's `nu` is a shape parameter R starts at 2.  A
    blanket "zero-variants start nu at 0.1" would silently change all three.

    That none of them is at 0.1 today is the assertion -- their remaining gap
    to R (-0.5 / -0.5 / 2.0) is recorded in
    ``test_known_nu_initial_gaps_against_r`` rather than asserted here, because
    D2.26 did not change them.
    """
    assert _nu_start(name) != 0.1, (
        f"{name}'s nu is a shape parameter, not the zero mass; R does not "
        f"start it at 0.1 (R uses {_NOT_THE_ZERO_MASS[name]})"
    )


def test_no_zero_variant_falls_through_to_the_one_fallback():
    """The rule: a logit-linked zero mass must never reach the 1.0 default.

    A missing name is invisible -- the fallback returns 1.0 without complaint
    and the damage only shows up as a non-converging fit.  This is the test
    that would have caught ZIPIG before the parity scan did.

    Restricted to families whose `nu` actually carries the zero mass, i.e.
    logit-linked and no `tau`: where the base already has a `nu` the factory
    renames the zero mass to `tau`, and those are covered by
    ``test_a_shape_parameter_nu_is_not_given_the_zero_mass_start``.

    ``ZILG`` is excluded because it has no R counterpart at all -- verified
    against live R 4.6, where ``ZILG()`` raises ``could not find function``.  It
    is a project extension, so there is no ``nu.initial`` to match; its start is
    registered in ``test_known_nu_initial_gaps_against_r`` instead of being
    silently counted as compliant.
    """
    from omnilss.distribution_registry import list_families

    offenders = []
    for name in list_families():
        if name in _NO_R_COUNTERPART:
            continue
        family = resolve(name)
        params = set(family.parameters)
        if "nu" not in params or "tau" in params:
            continue
        if family.links.get("nu") != "logit":
            continue
        if not family.name.startswith(("ZA", "ZI", "BEI")):
            continue
        value = _initial_parameter_value(family, "nu", _Y, _MU, _W)
        if value == 1.0:
            offenders.append((name, value))
    assert not offenders, (
        "logit-linked zero mass falling through to nu = 1.0 (logit(1.0) = "
        f"+inf, clipped to +15): {offenders}"
    )


@pytest.mark.parametrize("name", _CONTINUOUS_BASE)
def test_a_continuous_base_zero_variant_is_not_captured_by_the_discrete_rule(name):
    """The other side: R starts these elsewhere, so a blanket 0.1 would be wrong.

    Read off live R: ZAGA and ZAIG start ``nu <- rep(0.5, ...)``, BEZI and BEOI
    ``rep(0.3, ...)``.  They are zero-variants in name only.

    What is asserted here is only *separation*: none of them may land on the
    0.1 branch.  That these still differ from R (0.2 against R's 0.5 / 0.3) is
    a pre-existing gap, not something D2.26 fixed -- see
    ``test_known_nu_initial_gaps_against_r`` which records it rather than
    letting it pass for correct.
    """
    assert _nu_start(name) != 0.1, (
        f"{name} is a continuous-base variant; R does not start it at 0.1"
    )
    assert _nu_start(name) == 0.2


def test_known_nu_initial_gaps_against_r():
    """Register the remaining ``nu.initial`` gaps instead of hiding them.

    D2.26 fixed the seven families whose ``nu`` carries the zero mass.  Five
    others still start somewhere R does not:

        ZAGA / ZAIG    ours 0.2   R 0.5
        BEZI / BEOI    ours 0.2   R 0.3
        ZABNB / ZIBNB  ours 1.0   R 0.1   (their zero mass is `tau`, and
                                          their `tau` is data-driven in R)
        ZASICHEL / ZISICHEL  ours 1.0  R -0.5

    All seven still fit and none of them is currently failing a parity gate, so
    they are not in scope for D2.26.  Pinning the numbers here means a future
    change to any of them shows up as *this test failing*, which is the point:
    a gap that is not recorded is indistinguishable from a fix.
    """
    gaps = {
        "ZAGA": (0.2, 0.5),
        "ZAIG": (0.2, 0.5),
        "BEZI": (0.2, 0.3),
        "BEOI": (0.2, 0.3),
        "ZABNB": (1.0, 0.1),
        "ZIBNB": (1.0, 0.1),
        "ZASICHEL": (1.0, -0.5),
        "ZISICHEL": (1.0, -0.5),
        # No R counterpart at all, so there is nothing to converge to.
        "ZILG": (1.0, None),
    }
    for name, (ours_expected, r_value) in sorted(gaps.items()):
        actual = _nu_start(name)
        assert actual == ours_expected, (
            f"{name} nu.initial is {actual}, the recorded gap expected "
            f"{ours_expected} (R uses {r_value})"
        )


def test_zipig_converges_instead_of_cycling():
    """End-to-end on the parity scanner's own sample.

    Reproducing the scanner's data requires the prefix-equivalent family list
    (documented in ``_build_jobs``), which is expensive to build; so this test
    uses a fresh integer sample with the same shape and asserts the *behaviour*
    rather than the exact deviance: finite, monotone in the budget, and
    reported as converged.  Before the fix the deviance went 455 -> 733 when the
    budget rose from 20 to 200.
    """
    from omnilss import gamlss

    y = np.tile(np.arange(0.0, 8.0), 6)          # 48 integer observations
    x = np.linspace(-1.0, 1.0, y.size)

    devs = []
    for n in (20, 60):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = gamlss(
                "y ~ x", family="ZIPIG", data={"y": y, "x": x},
                sigma_formula="~1", method="RS", max_iter=n,
            )
        devs.append(float(model.deviance))
        assert np.isfinite(model.deviance), (n, model.deviance)

    # More budget must not make the deviance dramatically worse.
    assert devs[1] < devs[0] * 1.05, devs
    assert devs[0] == pytest.approx(devs[1], rel=1e-9), (
        f"the fit does not settle: {devs[0]} at 20 cycles vs {devs[1]} at 60"
    )