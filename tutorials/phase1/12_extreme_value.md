# Pareto and Extreme Value - PARETO, PARETO2, GU, RG

[中文版本](12_extreme_value_cn.md)

> Status: scaffold aligned with the current OmniLSS API. Expand this page with
> distribution-specific derivations, R comparisons, and exercises as content is
> authored.

## Scope

This tutorial follows the `tutorials/README.md` plan for **Pareto and Extreme Value — PARETO, PARETO2, GU, RG**.

Covered topics:

- Extreme value theory
- Risk modeling
- Tail analysis

## Current OmniLSS API pattern

```python
import numpy as np
from omnilss import gamlss, gamlss_control

# Replace these arrays/formulas with the variables used in this article.
data = {
    "y": np.asarray(y),
    "x": np.asarray(x),
}

model = gamlss(
    "y ~ x",
    family="PARETO",
    data=data,
    method="RS",
    control=gamlss_control(n_cyc=20, c_crit=1e-4),
)

print(f"Global deviance: {model.g_dev:.4f}")
print(f"AIC: {model.additional_slots['aic']:.2f}")
print(model.coefficients["mu"])
```

## Cole-Green cross-derivative check

Use the full-Hessian CG backend when the tutorial needs an auditable
Cole-Green reference path rather than the routine RS path.

```python
model_cg = gamlss(
    "y ~ x",
    family="PARETO",
    data=data,
    method="CG",
    cg_backend="full_hessian",
)

print(model_cg.additional_slots["cg_backend"])
print(model_cg.additional_slots["cg_cross_derivatives"])
```

The experimental eta-scale backend is available for comparison:

```python
model_irls = gamlss(
    "y ~ x",
    family="PARETO",
    data=data,
    method="CG",
    cg_backend="irls_cross",
)
```

## Migration notes

- R's `sigma.formula` maps to `sigma_formula`.
- Coefficients are stored in `model.coefficients[parameter]`.
- Fitted parameter values are stored in `model.fitted_values[parameter]`.
- AIC/BIC diagnostics are stored in `model.additional_slots["aic"]` and
  `model.additional_slots["sbc"]`.
- Parameter predictions use `model.predict_params(newdata)[parameter]`.

## Next authoring tasks

- Add a small simulated dataset for the families in this article.
- Add side-by-side R and OmniLSS output checks.
- Add diagnostics and interpretation specific to **Pareto and Extreme Value — PARETO, PARETO2, GU, RG**.
