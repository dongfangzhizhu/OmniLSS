# Survival Analysis

> Status: scenario scaffold aligned with the current OmniLSS API. Expand with a
> full dataset walkthrough, business interpretation, and deployment notes.

## Scenario goal

This scenario follows the `tutorials/README.md` plan for **Survival Analysis**.

Focus areas:

- Weibull, Exponential

## End-to-end API skeleton

```python
import numpy as np
from omnilss import gamlss, gamlss_control

data = {
    "y": np.asarray(y),
    "x": np.asarray(x),
}

model = gamlss(
    "y ~ x",
    data=data,
    family="WEI",
    method="RS",
    control=gamlss_control(n_cyc=30, c_crit=1e-4),
)

params = model.predict_params({"x": np.asarray(x_new)})
print(params["mu"])
print(model.additional_slots["aic"])
```

## When to compare CG backends

For correctness-sensitive scenario writeups, add a CG reference fit and record
the backend diagnostics.

```python
reference = gamlss(
    "y ~ x",
    data=data,
    family="WEI",
    method="CG",
    cg_backend="full_hessian",
)

print(reference.additional_slots["cg_backend"])
print(reference.additional_slots["cg_cross_derivatives"])
```

## Scenario checklist

- Define the response, covariates, and distribution family choice.
- Fit a baseline RS model and report deviance/AIC/BIC.
- Add parameter-specific formulas (`sigma_formula`, `parameter_formulas`) when
  the scenario needs scale/shape modeling.
- Use `model.predict_params()` for distribution-parameter predictions.
- Add a `method="CG"` reference fit only when cross-derivative diagnostics are
  relevant to the story.
