"""R gamlss.tr trun() / gen.trun(): truncated family objects.

The d/p closures behind the family object are the trun.d / trun.p helpers that
test_gamlss_tr.py already checks against live R (maxabsdiff <= 7e-16 for NO and
PO, left/right/both), so these tests pin the object contract: naming, support,
y_valid, g_dev_inc and delegation.
"""
from __future__ import annotations

import numpy as np
import pytest

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402

from omnilss.families import FamilyDefinition  # noqa: E402
from omnilss.gamlss_tr import (  # noqa: E402
    TruncatedFamily,
    gen_trun,
    trun,
    trun_d,
    trun_p,
    trun_q,
)


def test_trun_builds_a_family_definition() -> None:
    family = trun(0.0, "NO", "left")
    assert isinstance(family, FamilyDefinition)
    assert isinstance(family, TruncatedFamily)
    assert family.name == "NOtr"
    assert family.parameters == ("mu", "sigma")
    assert family.truncation_type == "left"
    assert family.truncation_parameters == (0.0,)
    assert family.base_name == "NO"


def test_custom_extra_name_changes_the_generated_name() -> None:
    assert trun(0.0, "NO", "left", extra_name="cut").name == "NOcut"


def test_support_for_each_truncation_type() -> None:
    assert trun(0.0, "NO", "left").support() == (0.0, None)
    assert trun(1.0, "NO", "right").support() == (None, 1.0)
    assert trun([-1.0, 1.0], "NO", "both").support() == (-1.0, 1.0)


def test_y_valid_respects_the_truncated_support() -> None:
    left = trun(0.0, "NO", "left")
    assert list(np.asarray(left.y_valid([-1.0, 0.0, 1.0]))) == [False, True, True]
    right = trun(1.0, "NO", "right")
    assert list(np.asarray(right.y_valid([0.5, 1.0, 2.0]))) == [True, True, False]
    both = trun([-1.0, 1.0], "NO", "both")
    assert list(np.asarray(both.y_valid([-2.0, 0.0, 2.0]))) == [False, True, False]


def test_discrete_support_is_inclusive_at_the_lower_bound() -> None:
    family = trun(2.0, "PO", "left")
    assert list(np.asarray(family.y_valid([1.0, 2.0, 3.0]))) == [False, False, True]


def test_density_and_cdf_delegate_to_the_trun_helpers() -> None:
    family = trun(0.0, "NO", "left")
    x = jnp.asarray([0.5, 1.0, 2.0], dtype=jnp.float64)
    np.testing.assert_allclose(
        np.asarray(family.d(x, mu=0.0, sigma=1.0)),
        np.asarray(trun_d(0.0, "NO", "left")(x, mu=0.0, sigma=1.0)),
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(family.p(x, mu=0.0, sigma=1.0)),
        np.asarray(trun_p(0.0, "NO", "left")(x, mu=0.0, sigma=1.0)),
        rtol=1e-12,
    )


def test_quantile_stays_the_base_family_quantile() -> None:
    # R's trun() regenerates only d and p, so q and r remain the base ones.
    from omnilss.distribution_registry import resolve

    family = trun(0.0, "NO", "left")
    probs = jnp.asarray([0.2, 0.5, 0.8], dtype=jnp.float64)
    np.testing.assert_allclose(
        np.asarray(family.q(probs, mu=0.0, sigma=1.0)),
        np.asarray(resolve("NO").q(probs, 0.0, 1.0)),
        rtol=1e-12,
    )
    # The truncation-aware quantile is available explicitly through trun_q.
    truncated = np.asarray(trun_q(0.0, "NO", "left")(probs, mu=0.0, sigma=1.0))
    assert np.all(truncated >= 0.0)


def test_g_dev_inc_is_elementwise_minus_two_log_density() -> None:
    family = trun(0.0, "NO", "left")
    y = jnp.asarray([0.5, 1.0, 2.0], dtype=jnp.float64)
    log_density = np.asarray(family.d(y, mu=0.0, sigma=1.0, log=True), dtype=float)
    got = np.asarray(family.g_dev_inc(y=y, mu=0.0, sigma=1.0), dtype=float)
    np.testing.assert_allclose(got, -2.0 * log_density, rtol=1e-12)
    # The truncated density integrates to one over the truncated support.
    grid = np.linspace(0.0, 40.0, 40001)
    dens = np.asarray(family.d(jnp.asarray(grid), mu=0.0, sigma=1.0), dtype=float)
    assert np.trapezoid(dens, grid) == pytest.approx(1.0, abs=1e-5)


def test_gen_trun_returns_r_style_names() -> None:
    generated = gen_trun(0.0, "NO", "left")
    for key in ("d", "p", "q", "r", "family", "dNOtr", "pNOtr", "qNOtr", "rNOtr", "NOtr"):
        assert key in generated
    assert generated["family"] is generated["NOtr"]
    assert generated["dNOtr"] is generated["family"].d


def test_gen_trun_honours_extra_name() -> None:
    generated = gen_trun([-1.0, 1.0], "NO", "both", extra_name="cut")
    assert "NOcut" in generated
    assert "dNOcut" in generated


def test_par_length_validation_is_enforced() -> None:
    with pytest.raises(ValueError, match="length of par should be 2"):
        trun(0.0, "NO", "both")
    with pytest.raises(ValueError, match="length of par should be 1"):
        trun([0.0, 1.0], "NO", "left")


def test_varying_truncation_is_not_ported() -> None:
    with pytest.raises(NotImplementedError, match="varying"):
        trun(0.0, "NO", "left", varying=True)
