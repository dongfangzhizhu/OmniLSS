# SPDX-License-Identifier: GPL-3.0-or-later
"""P3-3: mu/sigma/nu/tau .fix across every fitting route, and categorical
prediction after CG/MIXED (D2.244).

The fix flags used to raise for anything but method="RS".  They are
method-independent: the override rewrites the family's fixed set and freezes the
value in a data column that every backend already reads.  The R reference below
is live R 4.6.0 / gamlss 5.5.0 on the same sample:

    gamlss(y ~ x, family = NO(), sigma.fix = TRUE, sigma.start = 0.5,
           method = RS(n.cyc=100))   -> dev 67.132413859573532
    the same with CG(n.cyc=100) and mixed(1,100) -> identical
    without sigma.start -> dev 76.631645584083216, sigma fixed at 0.598641360014006

Three defects were found while extending it and are pinned here:
  * NO's sigma.initial was missing, so sigma.fix without sigma.start froze
    sigma on the eps floor (deviance ~1e13 against R's 76.63);
  * cg_fit_v2 emitted a fake coefficient for every fixed parameter, so MIXED
    reported sigma.coefficients instead of R's NULL;
  * cg_fit_v2 recorded "intercept" as "intercept-only", which made MIXED
    prediction rebuild 3 columns against a 4-column fit.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from omnilss import NO, gamlss
from omnilss.prediction import predict_params

CONTRASTS_GOLDEN = (
    Path(__file__).parent / "fixtures" / "r_golden" / "contrasts_2026-11-01.json"
)

METHOD_KWARGS = {
    "RS": {"max_iter": 100},
    "CG": {"max_iter": 100},
    "MIXED": {"mixed_n1": 1, "mixed_n2": 100},
}


def _data():
    rng = np.random.default_rng(5)
    n = 60
    x = rng.normal(size=n)
    y = 1.0 + 0.4 * x + rng.normal(scale=0.5, size=n)
    return {"y": y, "x": x}


# Live R values (see module docstring).
R_DEV_FIXED = 67.132413859573532
R_MU_FIXED = (0.93689395167610423, 0.4565896665555827)
R_DEV_DEFAULT = 76.631645584083216
R_SIGMA_DEFAULT = 0.598641360014006


@pytest.mark.parametrize("method", ("RS", "CG", "MIXED"))
def test_sigma_fix_honours_sigma_start_on_every_route(method):
    model = gamlss(
        "y ~ x", family=NO(), data=_data(), method=method,
        sigma_fix=True, sigma_start=0.5, **METHOD_KWARGS[method],
    )
    assert float(model.g_dev) == pytest.approx(R_DEV_FIXED, rel=1e-9)
    # MIXED lands on the same optimum through its CG stage to ~1e-10.
    np.testing.assert_allclose(
        np.ravel(np.asarray(model.coefficients["mu"], dtype=float)),
        np.asarray(R_MU_FIXED),
        rtol=1e-8,
        atol=1e-10,
    )
    # R returns NULL for sigma.coefficients when sigma.fix = TRUE.
    assert "sigma" not in model.coefficients
    np.testing.assert_allclose(
        np.asarray(model.fitted_values["sigma"], dtype=float),
        np.full(model.n, 0.5),
        rtol=0.0,
        atol=1e-12,
    )


@pytest.mark.parametrize("method", ("RS", "CG", "MIXED"))
def test_sigma_fix_without_start_uses_r_sigma_initial(method):
    """R NO()$sigma.initial is rep(sd(y)); the eps floor gave deviance ~1e13."""
    model = gamlss(
        "y ~ x", family=NO(), data=_data(), method=method,
        sigma_fix=True, **METHOD_KWARGS[method],
    )
    assert float(model.g_dev) == pytest.approx(R_DEV_DEFAULT, rel=1e-9)
    np.testing.assert_allclose(
        np.asarray(model.fitted_values["sigma"], dtype=float),
        np.full(model.n, R_SIGMA_DEFAULT),
        rtol=1e-10,
        atol=0.0,
    )


def test_joint_route_honours_fixed_sigma():
    """JOINT used to return its last Adam iterate (dev 67.1638); best-iterate
    tracking lands on the RS optimum and still leaves sigma unestimated."""
    model = gamlss(
        "y ~ x", family=NO(), data=_data(), method="JOINT",
        sigma_fix=True, sigma_start=0.5, max_iter=300,
    )
    assert float(model.g_dev) == pytest.approx(R_DEV_FIXED, rel=1e-8)
    assert "sigma" not in model.coefficients


def test_fixed_flag_still_validates_its_type():
    with pytest.raises(ValueError, match="should be logical"):
        gamlss(
            "y ~ x", family=NO(), data=_data(), method="CG",
            sigma_fix="yes", max_iter=10,
        )


def _contrasts_case():
    golden = json.loads(CONTRASTS_GOLDEN.read_text(encoding="utf-8"))
    data = {key: np.asarray(value) for key, value in golden["data"].items()}
    if data["g"].dtype.kind in "UO" or data["g"].dtype == object:
        data["g"] = data["g"].astype(str)
    newdata = {"g": np.array(["a", "c"]), "x": np.array([0.0, 1.0])}
    return golden, data, newdata


@pytest.mark.parametrize("method", ("CG", "MIXED"))
@pytest.mark.parametrize(("scheme", "key"), [
    ("contr.treatment", "treatment"),
    ("contr.sum", "sum"),
])
def test_categorical_prediction_after_cg_and_mixed(method, scheme, key):
    golden, data, newdata = _contrasts_case()
    kwargs = {"max_iter": 30} if method == "CG" else {"mixed_n1": 1, "mixed_n2": 30}
    model = gamlss(
        "y ~ g + x", family=NO(), data=data, method=method,
        contrasts=scheme, **kwargs,
    )
    assert float(model.g_dev) == pytest.approx(
        float(golden[key]["deviance"]), rel=1e-6
    )
    params = predict_params(model, newdata)
    np.testing.assert_allclose(
        np.asarray(params["mu"], dtype=float),
        np.asarray(golden[key]["pred_mu"], dtype=float),
        rtol=1e-8,
        atol=1e-8,
    )
