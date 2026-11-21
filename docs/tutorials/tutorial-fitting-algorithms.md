# Fitting Algorithms: RS / CG / MIXED / HYBRID / RS_JAX

[中文版本](tutorial-fitting-algorithms_cn.md)

OmniLSS implements R `gamlss`'s core estimation algorithms plus an experimental
JAX-compiled RS backend, selected via `method=...`.

## RS (Rigby & Stasinopoulos, Default)

- Alternates over distribution parameters, updating each parameter via modified
  IRLS with R `glim.fit` inner-loop semantics while holding other parameters
  fixed.
- Validated (`rs_fit=validated`) for all 22 `production_safe` families plus
  `ZIP`.

```python
model = gamlss("y ~ x", family="GA", data=data)              # default RS
model = gamlss("y ~ x", family="GA", data=data, method="RS")
```

## CG (Cole & Green)

- Joint Newton-type updates using cross-parameter expected information blocks.
- Validated (`cg_fit=validated`) for 18 admitted families whose CG cell passes
  directly against live R in `docs/development/layered-parity-2026-11-15.json`
  (`BE`, `BEINF`, `BI`, `EXP`, `GA`, `GEOM`, `GU`, `IG`, `IGAMMA`, `LO`,
  `LOGNO`, `LOGNO2`, `NET`, `NO`, `NO2`, `PO`, `WEI`, `ZAGA`).

```python
model = gamlss("y ~ x", family="BE", data=data, method="CG")
```

## MIXED (R `mixed(n1, n2)` Two-Stage Algorithm)

- Runs `mixed_n1` RS cycles followed by `mixed_n2` CG cycles warm-started from
  the full RS parameter state.
- Final `iter` and `converged` follow CG semantics; both stages are recorded in
  `model.additional_slots`.
- Smooth terms are rejected explicitly on CG/MIXED; use `RS` or `HYBRID` when
  fitting penalized smoothers.

```python
model = gamlss(
    "y ~ x",
    family="NBI",
    data=data,
    method="MIXED",
    mixed_n1=1,
    mixed_n2=20,
)
```

## HYBRID & Experimental RS_JAX

- `method="HYBRID"` automatically chooses between `RS` and `L-BFGS` according to
  sample size, family, and smoother presence (not R's `mixed()`).
- `method="RS_JAX"` compiles the RS loop with XLA for 6 core families (`NO`,
  `GA`, `PO`, `BI`, `WEI`, `TF`; currently `experimental` tier).

```python
model_h = gamlss("y ~ pb(x)", family="NO", data=data, method="HYBRID")
model_j = gamlss("y ~ x", family="NO", data=data, method="RS_JAX")
```

## Convergence Controls & Numerical Safeguards

```python
from omnilss.controls import gamlss_control

control = gamlss_control(n_cyc=100, c_crit=1e-3, trace=False)
model = gamlss("y ~ x", family="GA", data=data, control=control)
```

- All parameter-fixing flags (`mu_fix`, `sigma_fix`, `nu_fix`, `tau_fix`) and
  categorical contrasts (`contr.treatment`, `contr.sum`, `contr.helmert`,
  `contr.poly`) work across `RS`, `CG`, and `MIXED`.
- Numerical safeguards (linear predictor clipping, Hessian ridge stabilization)
  are logged in `model.additional_slots["safeguard_events"]` for auditability.

## Next Steps

- [Smoothers & Random Effects](tutorial-smoothers.md)
