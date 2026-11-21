# Quickstart

[中文版本](quickstart_cn.md)

OmniLSS is a Python/JAX implementation of R's `gamlss` ecosystem for
Generalized Additive Models for Location, Scale, and Shape (GAMLSS): every
parameter of a probability distribution (`mu`, `sigma`, `nu`, `tau`) can be
modeled simultaneously with linear terms, smoothers, and random effects.

## Installation & Environments

OmniLSS uses `uv` to manage virtual environments (`.venv` on Windows CPU,
`debianjaxgpu` on WSL Debian GPU):

```bash
# Install editable core package
uv pip install -e omnilss

# Optional extras: gRPC / FastAPI serving
uv pip install -e "omnilss[server]"
```

Always enable 64-bit floating point (`JAX_ENABLE_X64=true`) when matching R
`gamlss` numerically.

## Minimal Example

```python
import numpy as np
from omnilss import GA, gamlss

rng = np.random.default_rng(0)
n = 120
x = rng.normal(size=n)
# Simulate mu = exp(1.0 + 0.5 * x), sigma = 0.3
mu_true = np.exp(1.0 + 0.5 * x)
y = rng.gamma(shape=1.0 / 0.09, scale=mu_true * 0.09)

model = gamlss("y ~ x", family=GA(), data={"y": y, "x": x})
print("converged:", model.converged, "deviance:", float(model.g_dev))
print("mu coef:", model.coefficients["mu"])
print("sigma:", float(np.exp(model.coefficients["sigma"][0])))
```

## Prediction & Quantiles

```python
newdata = {"x": np.linspace(-2.0, 2.0, 5)}
mu_hat = model.predict(newdata=newdata)
params = model.predict_params(newdata=newdata)          # {"mu": ..., "sigma": ...}
quantiles = model.predict_quantiles(newdata=newdata, quantiles=[0.1, 0.5, 0.9])
```

## Current Scope & Highlights

1. **Distribution families**: 146 registered families (0 placeholders); 22
   admitted `production_safe` families backed by live-R golden fixtures,
   prediction round-trips, and layered route parity (`BE`, `BEINF`, `BI`,
   `EXP`, `GA`, `GEOM`, `GU`, `IG`, `IGAMMA`, `LO`, `LOGNO`, `LOGNO2`,
   `NBI`, `NBII`, `NET`, `NO`, `NO2`, `PE`, `PO`, `RG`, `WEI`, `ZAGA`).
2. **Algorithms**: `method="RS"` (default), `"CG"` (18 live-R validated
   families), `"MIXED"` (R `mixed(n1, n2)` two-stage RS -> CG), `"HYBRID"`,
   and experimental `"RS_JAX"`.
3. **Smoothers & Random Effects**: `pb(x)`, `ps(x)`, `cs(x)`, `cy(x)`,
   `lo(x)`, `re(random=...)`, and `random(g)` directly inside formulas.
4. **In-scope R API coverage**: 100% of the 1,036 in-scope R symbols across 11
   packages (`gamlss`, `gamlss.dist`, `gamlss.cens`, `gamlss.tr`, `gamlss.mx`,
   `gamlss.add`, `gamlss.spatial`, `gamlss.foreach`, `gamlss.prepdata`,
   `gamlss.ggplots`, `gamlss.data`) with zero unverified symbols. `gamlss2` is
   explicitly out of scope.

## Next Steps

- [Distribution Families & Capability Gates](tutorial-families.md)
- [Fitting Algorithms (RS / CG / MIXED / HYBRID)](tutorial-fitting-algorithms.md)
- [Smoothers & Random Effects](tutorial-smoothers.md)
