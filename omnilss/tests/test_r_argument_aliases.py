# SPDX-License-Identifier: GPL-3.0-or-later
"""R-compatible argument aliases: `n.cyc` and `trace` (D2.133).

R's `gamlss()` takes the outer cycle budget as `n.cyc` (a
`gamlss.control` field users pass through `...`) and prints the per-cycle
deviance when `trace = TRUE`.  Porting an R call site should not require
rewriting those two names, so `omnilss.gamlss` accepts them as aliases of
`max_iter` / `verbose`.  Conflicting values are rejected rather than
silently resolved: picking one would hide whichever the caller meant.
"""
import numpy as np
import pytest

from omnilss import NO, gamlss


@pytest.fixture
def data() -> dict:
    rng = np.random.default_rng(11)
    x = rng.standard_normal(40)
    y = 1.5 + 0.8 * x + rng.standard_normal(40)
    return {"y": y, "x": x}


def test_n_cyc_is_max_iter(data):
    """`n.cyc` must produce exactly the `max_iter` fit -- same trajectory,
    same coefficients, not merely a similar optimum."""
    alias = gamlss("y ~ x", family=NO(), data=data, method="RS", n_cyc=4)
    explicit = gamlss("y ~ x", family=NO(), data=data, method="RS", max_iter=4)
    assert float(alias.g_dev) == pytest.approx(float(explicit.g_dev), rel=0, abs=0)
    np.testing.assert_array_equal(
        np.asarray(alias.coefficients["mu"]), np.asarray(explicit.coefficients["mu"])
    )
    assert int(alias.iter) == int(explicit.iter)


def test_n_cyc_sets_the_cycle_budget(data):
    """A smaller budget really stops earlier (otherwise the alias is a no-op)."""
    short = gamlss("y ~ x", family=NO(), data=data, method="RS", n_cyc=1)
    long = gamlss("y ~ x", family=NO(), data=data, method="RS", n_cyc=20)
    assert int(short.iter) < int(long.iter)


def test_conflicting_n_cyc_and_max_iter_are_rejected(data):
    with pytest.raises(ValueError, match="conflicts with max_iter"):
        gamlss("y ~ x", family=NO(), data=data, method="RS", n_cyc=5, max_iter=7)


def test_agreeing_n_cyc_and_max_iter_are_accepted(data):
    model = gamlss("y ~ x", family=NO(), data=data, method="RS", n_cyc=5, max_iter=5)
    assert np.isfinite(float(model.g_dev))


@pytest.mark.parametrize("value", [0, -3, True, "10", 2.5])
def test_invalid_n_cyc_is_rejected(data, value):
    with pytest.raises(ValueError, match="n_cyc must be a positive integer"):
        gamlss("y ~ x", family=NO(), data=data, method="RS", n_cyc=value)


def test_trace_turns_verbose_on(data, capsys):
    """R's `trace=TRUE` prints the outer cycles; we print the same progress."""
    gamlss("y ~ x", family=NO(), data=data, method="RS", n_cyc=3, trace=True)
    captured = capsys.readouterr().out
    assert "Iteration" in captured


def test_trace_off_is_quiet(data, capsys):
    gamlss("y ~ x", family=NO(), data=data, method="RS", n_cyc=3)
    assert capsys.readouterr().out == ""


def test_non_logical_trace_is_rejected(data):
    with pytest.raises(ValueError, match="trace should be logical"):
        gamlss("y ~ x", family=NO(), data=data, method="RS", trace="yes")
