# Configuration API

`omnilss.config` controls device-aware method routing for `gamlss(..., method="auto")`.

## Runtime switches

| Setting | Meaning |
|---|---|
| `AUTO_METHOD_ENABLED` | When `False`, `method="auto"` always resolves to NumPy `RS`. |
| `FORCE_JAX` | When `True`, supported families resolve to `RS_JAX` regardless of device. |
| `GPU_CROSSOVER_N` | Per-family GPU threshold table. |
| `TPU_CROSSOVER_N` | Per-family TPU threshold table. |

A threshold of `math.inf` means “never automatically switch to JAX”.

## `set_crossover()`

Use `set_crossover(device, n, family="default")` to update a threshold for the current Python process:

```python
import math
import omnilss.config as cfg

cfg.set_crossover("gpu", n=50_000, family="NO")
cfg.set_crossover("gpu", n=100_000)      # default for unlisted families
cfg.set_crossover("tpu", n=math.inf)     # disable TPU auto-switching
```

Validation rules:

- `device` must be `"gpu"` or `"tpu"`.
- `n` must be non-negative.
- Unknown families emit `UserWarning` but are accepted so users can preconfigure future families.

## `crossover_summary()`

Use `crossover_summary(verbose=False)` to print the active routing configuration:

```python
import omnilss.config as cfg
cfg.crossover_summary(verbose=True)
```

The summary includes the detected backend, auto-routing switches, and GPU/TPU threshold tables.

## `crossover_config()`

Use `crossover_config()` for temporary experiments that should not mutate global thresholds after the `with` block exits:

```python
import omnilss.config as cfg

with cfg.crossover_config(gpu={"NO": 50_000, "default": 100_000}):
    model = gamlss("y ~ x", family="NO", data=data, method="auto")
```

## YAML / JSON configuration files

At import time, OmniLSS loads the first existing file from:

1. `OMNILSS_CONFIG_FILE`
2. `./omnilss_config.yaml`
3. `~/.omnilss/config.yaml`

Example YAML:

```yaml
auto_method_enabled: true
force_jax: false

gpu_crossover_n:
  NO: 50000
  GA: 80000
  default: .inf

tpu_crossover_n:
  NO: 10000
  default: .inf
```

Environment variables are applied after configuration files and therefore take precedence:

```bash
OMNILSS_AUTO_METHOD=1
OMNILSS_FORCE_JAX=0
OMNILSS_GPU_CROSSOVER_N='NO=50000,GA=80000,default=inf'
OMNILSS_TPU_CROSSOVER_N='NO=10000,default=inf'
```
