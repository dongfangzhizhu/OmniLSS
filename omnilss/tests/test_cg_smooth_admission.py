"""CG must fail closed for smooth terms before building an unpenalized backend."""
import numpy as np
import pytest

from omnilss import gamlss


DATA = {"y": np.linspace(0.1, 1.0, 8), "x": np.linspace(-1.0, 1.0, 8), "z": np.linspace(1.0, 2.0, 8)}


@pytest.mark.parametrize("backend", ["full_hessian", "irls_cross", "CG_FULL_HESSIAN", "CG_IRLS_CROSS"])
@pytest.mark.parametrize(
    "formula,sigma_formula,parameter_formulas",
    [
        ("y ~ s(x)", "~ 1", None),
        ("y ~ x", "~ pb (x)", None),
        ("y ~ x", "~ 1", {"nu": "~ ps(x)"}),
        ("y ~ x", "~ 1", {"tau": "~ cs(z)"}),
    ],
)
def test_cg_rejects_every_smooth_formula_location(
    backend, formula, sigma_formula, parameter_formulas
):
    with pytest.raises(ValueError, match="method='CG' does not support"):
        gamlss(
            formula,
            family="NO",
            data=DATA,
            sigma_formula=sigma_formula,
            parameter_formulas=parameter_formulas,
            method="CG",
            cg_backend=backend,
        )


def test_algorithm_alias_checks_smooth_before_dispatch():
    with pytest.raises(ValueError, match="method='CG' does not support"):
        gamlss(
            "y ~ lo(x)",
            family="NO",
            data=DATA,
            method="RS",
            algorithm="CG",
        )


def test_mapping_overrides_legacy_smooth_formula():
    # The effective mapping is linear, so the legacy smooth nu formula must
    # not trigger the CG admission rejection.
    with pytest.raises(Exception) as caught:
        gamlss(
            "y ~ x",
            family="NO",
            data=DATA,
            method="CG",
            nu_formula="~ pb(x)",
            parameter_formulas={"nu": "~ 1"},
            max_iter=1,
        )
    assert "does not support penalized smooth terms" not in str(caught.value)
