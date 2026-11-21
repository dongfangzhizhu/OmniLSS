# Prediction & Serialization

[中文版本](tutorial-prediction_cn.md)

After fitting, OmniLSS rebuilds parameter design matrices (including categorical
contrasts and smooth bases) for arbitrary `newdata` without requiring the
response column or matching training row counts.

## Core Prediction Methods

```python
import numpy as np
from omnilss import GA, gamlss

model = gamlss("y ~ x", family=GA(), data=train)
newdata = {"x": np.linspace(-2.0, 2.0, 5)}

mu_hat = model.predict(newdata=newdata)                            # response scale mu
eta_mu = model.predict(newdata=newdata, type="link")               # linear predictor
params = model.predict_params(newdata=newdata)                     # {"mu": ..., "sigma": ...}
quantiles = model.predict_quantiles(newdata, quantiles=[0.05, 0.5, 0.95])
```

| Method / Function | Output |
|---|---|
| `model.predict(newdata)` | Conditional location parameter `mu` |
| `model.predict(newdata, type="link")` | Linear predictor `eta_mu` |
| `model.predict_params(newdata)` | All distribution parameters (`mu`, `sigma`, `nu`, `tau`) |
| `model.predict_quantiles(newdata, quantiles=...)` | Conditional quantiles via family `q()` |
| `predictAll(model, newdata=...)` | R `predictAll()` structured multi-parameter output |
| `centiles_pred(model, xname=..., xvalues=...)` | R `centiles.pred()` centiles, z-scores, and standard centiles |
| `lpred(model, what="mu", type="terms", se_fit=True)` | Centered term contributions and standard errors |

## Smoothers & Categorical Contrasts

Models with `pb()`, `ps()`, `cs()`, or categorical contrasts (`contr.treatment`,
`contr.sum`, `contr.helmert`, `contr.poly`) save their basis and coding state
so `newdata` prediction matches training semantics across `RS`, `CG`, and
`MIXED`:

```python
model = gamlss("y ~ pb(x)", family="NO", data=train)
pred = model.predict(newdata={"x": x_new})
```

## JSON Serialization & Round-Trip Verification

All 22 admitted families are verified for `<= 1e-10` prediction drift across
JSON save/load round-trips (`docs/development/m2-assessment-2026-11-15.json`):

```python
from omnilss import load_model_json, save_model_json

save_model_json(model, "model.omnilss.json")
restored = load_model_json("model.omnilss.json")
assert np.allclose(
    model.predict_params(newdata)["mu"],
    restored.predict_params(newdata)["mu"],
    atol=1e-10,
)
```

## Next Steps

- [Model Selection & Diagnostics](tutorial-model-selection.md)
