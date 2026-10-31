"""DELAPORT must fit exactly like DEL (D2.109).

DELAPORT is DEL under another registry name -- its d/p/q match R's dDEL/pDEL/
qDEL to ~1e-15 (see test_delaport_is_r_del.py).  But the initial-value table
keyed on ``family.name``, so DELAPORT had no branch at all and fell through to
the generic defaults.  Same density, different starting point, different
answer: on the scanner sample DELAPORT fitted 223.87 where R reports 213.346,
while DEL -- the same distribution -- passed.

The fix wires DELAPORT into DEL's three initial-value branches.  This test
pins the consequence that matters: fitting the same data with either name must
give the same deviance and the same coefficients.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from omnilss import gamlss  # noqa: E402


def _data() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(20261006)
    return {
        "y": np.abs(rng.normal(2.0, 1.0, 48)).round(),
        "x": rng.normal(size=48),
    }


def _fit(family: str, data: dict[str, np.ndarray]):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return gamlss(
            "y ~ x",
            family=family,
            data=data,
            sigma_formula="~1",
            method="RS",
            max_iter=20,
            allow_compat_alias=True,
        )


def test_delaport_and_del_land_on_the_same_point():
    data = _data()
    del_model = _fit("DEL", data)
    alias_model = _fit("DELAPORT", dict(data))
    np.testing.assert_allclose(
        float(del_model.deviance), float(alias_model.deviance), rtol=1e-12
    )
    assert set(del_model.coefficients) == set(alias_model.coefficients)
    for parameter, values in del_model.coefficients.items():
        np.testing.assert_allclose(
            np.asarray(values, dtype=float),
            np.asarray(alias_model.coefficients[parameter], dtype=float),
            rtol=1e-10,
            atol=1e-12,
            err_msg=parameter,
        )