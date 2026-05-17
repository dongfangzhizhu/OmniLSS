# Algorithms API

## Method routing

`gamlss()` accepts a `method` argument that selects the fitting backend.

| Method | Backend | Recommended use |
|---|---|---|
| `"RS"` | NumPy Rigby-Stasinopoulos IRLS | Default path; fastest on CPU; supports all families and smooth terms. |
| `"RS_JAX"` | JAX JIT-compiled RS | Experimental accelerator path for supported families (NO, GA, PO, BI, WEI, TF). |
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
