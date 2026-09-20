"""Direct CG must fail closed when its backend cannot consume smooth penalties."""
import numpy as np
import pytest

from omnilss import gamlss


@pytest.mark.parametrize("kwargs", [
    {"formula": "y ~ s(x)"},
    {"formula": "y ~ x", "sigma_formula": "~ pb(x)"},
    {"formula": "y ~ x", "parameter_formulas": {"nu": "~ ps(x)"}},
    {"formula": "y ~ x", "parameter_formulas": {"tau": "~ cs(x)"}},
])
def test_cg_rejects_smooth_terms_before_fit(kwargs):
    with pytest.raises(ValueError, match="method='CG' does not support"):
        gamlss(
            family="NO",
            data={"y": np.ones(8), "x": np.linspace(0, 1, 8)},
            method="CG",
            max_iter=2,
            **kwargs,
        )


def test_cg_without_smooth_terms_remains_available():
    model = gamlss(
        "y ~ x",
        family="NO",
        data={"y": np.arange(8, dtype=float), "x": np.linspace(0, 1, 8)},
        method="CG",
        max_iter=2,
    )
    assert model.additional_slots["method"] == "CG"
