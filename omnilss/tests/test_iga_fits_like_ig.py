"""IGA must fit exactly like IG (D2.111).

IGA is IG under another registry name -- R's qIGA is literally qIG, which is
why the dpqr gate already carries a shared ``("IGA", "q")`` tolerance (D2.77).
But the initial-value table keys on ``family.name``, so IGA had no branch at
all and fell through to the generic defaults while IG had two.  Same density,
different starting point, different answer -- the DELAPORT defect (D2.109)
in a second family.

The fix wires IGA into both IG branches.  This test pins the consequence that
matters: fitting the same data with either name must give the same deviance
and the same coefficients.
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
        "y": np.abs(rng.normal(3.0, 1.0, 48)) + 0.1,
        "x": rng.normal(size=48),
    }


def _fit(family: str) -> object:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return gamlss(
            "y ~ x",
            family=family,
            data=_data(),
            sigma_formula="~1",
            method="RS",
            max_iter=20,
            allow_compat_alias=True,
        )


def test_iga_and_ig_land_on_the_same_point():
    ig_model = _fit("IG")
    iga_model = _fit("IGA")
    np.testing.assert_allclose(
        float(ig_model.deviance), float(iga_model.deviance), rtol=1e-12
    )
    assert set(ig_model.coefficients) == set(iga_model.coefficients)
    for parameter, values in ig_model.coefficients.items():
        np.testing.assert_allclose(
            np.asarray(values, dtype=float),
            np.asarray(iga_model.coefficients[parameter], dtype=float),
            rtol=1e-10,
            atol=1e-12,
            err_msg=parameter,
        )