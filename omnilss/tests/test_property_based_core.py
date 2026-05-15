import numpy as np
import pytest

hypothesis = pytest.importorskip("hypothesis")
from hypothesis import given, strategies as st

from omnilss.distributions import resolve_family


@given(
    y=st.floats(min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False),
    mu=st.floats(min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False),
    sigma=st.floats(min_value=1e-6, max_value=1e3, allow_nan=False, allow_infinity=False),
)
def test_no_logpdf_finite(y, mu, sigma):
    fam = resolve_family("NO")
    val = fam.logpdf(y=np.array([y]), mu=np.array([mu]), sigma=np.array([sigma]))
    assert np.isfinite(np.asarray(val)).all()


@given(
    x1=st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False),
    x2=st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False),
)
def test_no_p_monotone(x1, x2):
    fam = resolve_family("NO")
    lo, hi = sorted([x1, x2])
    p_lo = np.asarray(fam.p(y=np.array([lo]), mu=np.array([0.0]), sigma=np.array([1.0])))[0]
    p_hi = np.asarray(fam.p(y=np.array([hi]), mu=np.array([0.0]), sigma=np.array([1.0])))[0]
    assert p_lo <= p_hi
