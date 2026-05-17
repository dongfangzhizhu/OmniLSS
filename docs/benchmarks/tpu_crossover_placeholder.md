# TPU Crossover Benchmark Placeholder

OmniLSS currently keeps all TPU automatic JAX crossover thresholds at `math.inf`, so `method="auto"` does not switch to `RS_JAX` on TPU until maintainers record target-hardware measurements.

## Expected test environments

Benchmark at least one representative host for each supported TPU generation before changing package defaults:

| TPU generation | Minimum target | Notes |
|---|---:|---|
| TPU v2 | single core and full slice | Legacy baseline for compatibility. |
| TPU v3 | single core and full slice | Common research environment. |
| TPU v4 | single core and full slice | Preferred production baseline when available. |

Record the JAX version, `jaxlib` version, XLA runtime, TPU topology, host CPU/RAM, and whether data transfer time is included.

## Pending benchmark matrix

Use the same matrix as the GPU crossover sweep so GPU and TPU results can be compared directly:

| Variable | Values |
|---|---|
| `n` observations | 100 · 500 · 1,000 · 5,000 · 10,000 · 50,000 · 100,000 · 500,000 |
| `p` design columns | 2 · 5 · 10 · 20 · 50 |
| Families | NO · GA · PO · BI · WEI · TF |
| Methods | `RS` vs `RS_JAX` after JIT warm-up |

Timing rules:

- Run at least three warm-up calls for the JAX path and call `jax.block_until_ready()` before measuring.
- Use median timing rather than the fastest run.
- Run each `(n, p, family)` combination five times and report p50/p95.
- Record OOMs or TPU compilation failures explicitly instead of dropping those rows.

## Manual update procedure

After measurements are reviewed, update the thresholds either in source or in a local YAML configuration file.

Source-level defaults live in `omnilss/src/omnilss/config.py`:

```python
TPU_CROSSOVER_N: dict[str, float] = {
    "NO": 10_000,
    "GA": math.inf,
    "PO": math.inf,
    "BI": math.inf,
    "WEI": math.inf,
    "TF": math.inf,
    "default": math.inf,
}
```

For local experiments, prefer `~/.omnilss/config.yaml`:

```yaml
tpu_crossover_n:
  NO: 10000
  default: .inf
```

Environment variables can override either source defaults or YAML settings:

```bash
OMNILSS_TPU_CROSSOVER_N='NO=10000,GA=20000,default=inf'
```
