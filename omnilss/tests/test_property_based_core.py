import numpy as np
import pytest

hypothesis = pytest.importorskip("hypothesis")
from hypothesis import given, settings, strategies as st

# JAX trace/compile latency dominates the first call of each property, so the
# default 200 ms per-example deadline fires under load.  The properties are
# about finiteness and monotonicity, not about wall time (D2.145).
NO_DEADLINE = settings(deadline=None)

from omnilss.distributions import resolve_family


@NO_DEADLINE
@given(
    y=st.floats(min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False),
    mu=st.floats(min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False),
    sigma=st.floats(min_value=1e-6, max_value=1e3, allow_nan=False, allow_infinity=False),
)
def test_no_logpdf_finite(y, mu, sigma):
    """The NO log-density must stay finite for every in-support (y, mu, sigma).

    D2.145: this called fam.logpdf(...), which does not exist on
    FamilyDefinition (the API is d/p/q/r plus a pdf() wrapper), so the
    property was never actually checked -- it failed on the attribute, not on
    the mathematics.
    """
    fam = resolve_family("NO")
    val = fam.d(np.array([y]), np.array([mu]), np.array([sigma]), log=True)
    assert np.isfinite(np.asarray(val)).all()


@NO_DEADLINE
@given(
    x1=st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False),
    x2=st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False),
)
def test_no_p_monotone(x1, x2):
    fam = resolve_family("NO")
    lo, hi = sorted([x1, x2])
    # R's p functions take the quantile as the first formal "q"; passing y=
    # raised TypeError, so the monotonicity property never ran (D2.145).
    p_lo = np.asarray(fam.p(np.array([lo]), np.array([0.0]), np.array([1.0])))[0]
    p_hi = np.asarray(fam.p(np.array([hi]), np.array([0.0]), np.array([1.0])))[0]
    assert p_lo <= p_hi
