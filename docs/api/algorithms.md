# Algorithms API

[中文版本](algorithms_cn.md)

## Method routing

`gamlss()` accepts a `method` argument that selects the fitting backend.

| Method | Backend | Recommended use |
|---|---|---|
| `"RS"` | NumPy Rigby-Stasinopoulos IRLS | Default path; fastest on CPU; supports all families and smooth terms. |
| `"RS_JAX"` | JAX JIT-compiled RS | Experimental accelerator path for supported families (NO, GA, PO, BI, WEI, TF). |
| `"CG"` | `CG_FULL_HESSIAN` observed-information backend | Cole-Green backend with full cross-parameter Hessian blocks and auditable diagnostics. |
| `"CG"` + `cg_backend="irls_cross"` | `CG_IRLS_CROSS` eta-level backend | Experimental IRLS outer loop with eta-scale cross-derivative corrections and global line search. |
| `"auto"` | Device-aware router | Resolves to `RS` or `RS_JAX` using `omnilss.config.auto_select_method()`. |

Current package defaults keep GPU and TPU crossover thresholds at `math.inf`, so `method="auto"` behaves like `method="RS"` until hardware-specific benchmarks are recorded.

Configure thresholds with `omnilss.config.set_crossover()`:

```python
import omnilss.config as cfg

cfg.set_crossover("gpu", n=50_000, family="NO")
model = gamlss("y ~ x", family="NO", data=data, method="auto")
```

For temporary threshold experiments, use `omnilss.config.crossover_config()`:

```python
with cfg.crossover_config(gpu={"NO": 50_000}):
    model = gamlss("y ~ x", family="NO", data=data, method="auto")
```

See [`config.md`](config.md) for the full configuration API and YAML/JSON file format.

## Cole-Green diagnostics

`method="CG"` defaults to the full-Hessian correctness backend.  Fitted models
record the backend and cross-derivative status in `additional_slots`:

```python
model = gamlss("y ~ x", sigma_formula="~ x", family="NO", data=data, method="CG")
print(model.additional_slots["cg_backend"])             # CG_FULL_HESSIAN
print(model.additional_slots["cg_cross_derivatives"])  # full_hessian
```

For the eta-level IRLS cross-derivative backend, opt in explicitly:

```python
model = gamlss(
    "y ~ x",
    sigma_formula="~ x",
    family="NO",
    data=data,
    method="CG",
    cg_backend="irls_cross",
)
print(model.additional_slots["cg_backend"])             # CG_IRLS_CROSS
print(model.additional_slots["cg_cross_derivatives"])  # eta_correction
```

The IRLS-cross backend is useful for development and larger-design experiments;
use the default full-Hessian backend as the correctness reference.
