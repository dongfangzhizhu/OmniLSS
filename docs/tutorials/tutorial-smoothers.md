# Smoothers & Random Effects

[中文版本](tutorial-smoothers_cn.md)

Nonparametric smooth terms and random effects are specified directly inside
model formulas using R `gamlss` syntax and solved inside the RS backfitting
loop.

## Penalized B-Splines (`pb`, `ps`, `cy`)

```python
model_pb = gamlss("y ~ pb(x)", family="NO", data=data)
model_ps = gamlss("y ~ ps(x, df=5)", family="NO", data=data)
model_cy = gamlss("y ~ cy(x, df=4)", family="NO", data=data)
```

- `pb(x)` selects the smoothing parameter automatically (ML / GAIC / GCV) or
  targets a fixed `df` / `lambda`.
- `ps(x)` aligns with R's equidistant B-spline basis + difference penalty.
- `cy(x)` enforces periodic (cyclic) boundary conditions.
- Live-R consistency for `pb()` is validated on `NO` and `GA`
  (`smooth_terms=validated`).

## Cubic Splines (`cs`) and Local Regression (`lo`)

```python
from omnilss import gamlss, gamlss_lo, lo, lo_control

model_cs = gamlss("y ~ cs(x, df=4)", family="NO", data=data)
lo_fit = gamlss_lo(x, y, w=np.ones_like(y), span=0.5, degree=1)
```

- `cs(x)` stores spline state so `newdata` prediction rebuilds the exact curve.
- `lo()` / `gamlss_lo()` implement exact weighted local polynomial regression
  (degree 0/1/2, tricube weights, biweight robustness, and exact hat-matrix
  `trace_hat` / `nl_df`), verified to `1e-10` against `stats::loess`.

## Random Effects (`re`, `random`)

```python
from omnilss import gamlss_re, re

re_fit = gamlss_re(x_design, y, w=weights, groups=group_ids, method="REML")
```

- `re()` / `gamlss_re()` solve exact marginal ML/REML weighted linear mixed
  models (random intercepts and random slopes) via Woodbury + Cholesky,
  supporting both population (`level=0`) and group-specific (`level=1`)
  predictions.

## Parameter-Specific Smoothers

Every distribution parameter can carry its own smooth terms:

```python
model = gamlss("y ~ pb(x)", sigma_formula="~ pb(x, df=4)", family="NO", data=data)
```

## Next Steps

- [Prediction & Serialization](tutorial-prediction.md)
