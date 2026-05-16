# ADR-006: Device-Aware Automatic Method Selection

**Status**: Accepted  
**Date**: 2026-05-16  
**Deciders**: OmniLSS maintainers

---

## Context

OmniLSS has two RS fitting backends:

| Backend | Implementation | When faster |
|---------|---------------|-------------|
| `method='RS'` | NumPy IRLS (`algorithms/rs_algorithm.py`) | Always on CPU; also on GPU for small n |
| `method='RS_JAX'` | JAX `while_loop` + `jnp.linalg.lstsq` (`algorithms/jax_rs_core.py`) | GPU at large n (crossover TBD) |

Users should not need to know which backend to use.  The system should
choose automatically based on the available device and dataset size.

---

## Benchmark Results

### CPU (Intel i7-12700K, 12P/20L cores, 34 GB RAM, JAX 0.6.2)

JAX RS is **always slower** than NumPy RS on CPU.  No crossover found up to
n = 1,000,000.  The JAX `while_loop` overhead dominates for all tested n.

**Decision: CPU always uses `method='RS'`.**

### GPU (NVIDIA RTX 3060 12 GB, WSL2 Debian, JAX 0.10.0, CUDA 12)

Two measurements were taken:

**1. Full `method='RS_JAX'` (includes NumPy RS warm-start):**

| n | JAX warm (ms) | NumPy warm (ms) | JAX faster? |
|---|---:|---:|:---:|
| 1,000 | 799 | 141 | no |
| 10,000 | 561 | 88 | no |
| 100,000 | 859 | 186 | no |
| 500,000 | 1,448 | 419 | no |

**2. Pure JAX core only (no warm-start, JIT already compiled):**

| n | JAX core (ms) | NumPy RS (ms) | JAX faster? |
|---|---:|---:|:---:|
| 10,000 | 454 | 127 | no |
| 100,000 | 620 | 183 | no |
| 500,000 | 817 | 477 | no |

**Root cause**: For small design matrices (p=2: intercept + one predictor),
GPU kernel launch overhead dominates.  The `jnp.linalg.lstsq` call on a
`[n, 2]` matrix does not benefit from GPU parallelism.

**Decision: GPU crossover is set to `math.inf` (never auto-switch) until
benchmarked with larger design matrices or batch workloads.**

### TPU

Not yet benchmarked.  Crossover set to `math.inf`.

---

## Decision

Implement a three-tier automatic selection system in `omnilss/config.py`:

```
method='auto'
    │
    ├── FORCE_JAX=True → 'RS_JAX' (if family supported)
    ├── AUTO_METHOD_ENABLED=False → 'RS'
    ├── family not JAX-supported → 'RS'
    ├── backend == 'cpu' → 'RS'  (never faster)
    ├── backend == 'gpu' → 'RS_JAX' if n >= GPU_CROSSOVER_N[family]
    ├── backend == 'tpu' → 'RS_JAX' if n >= TPU_CROSSOVER_N[family]
    └── else → 'RS'
```

Default crossover values:
- CPU: `math.inf` (never)
- GPU: `math.inf` (pending benchmark with larger p)
- TPU: `math.inf` (not yet benchmarked)

---

## Configuration API

```python
import omnilss.config as cfg

# Check current config
print(cfg.get_config_summary())

# Override GPU crossover after your own benchmarking
cfg.GPU_CROSSOVER_N["NO"] = 50_000
cfg.GPU_CROSSOVER_N["default"] = 100_000

# Disable auto-switching (always use NumPy RS)
cfg.AUTO_METHOD_ENABLED = False

# Force JAX everywhere (for testing)
cfg.FORCE_JAX = True
```

Environment variables (set before importing omnilss):
```bash
OMNILSS_AUTO_METHOD=0          # disable auto-switching
OMNILSS_FORCE_JAX=1            # force JAX everywhere
OMNILSS_GPU_CROSSOVER_N=50000  # single value for all families
OMNILSS_TPU_CROSSOVER_N=10000  # single value for all families
```

---

## Consequences

### Positive
- Users get the fastest backend automatically without knowing implementation details.
- `method='auto'` is forward-compatible: as crossover points are established,
  the config values can be updated without changing user code.
- Per-family and per-device granularity allows fine-tuned control.
- Environment variables allow deployment-time configuration without code changes.

### Negative / Risks
- Currently `method='auto'` == `method='RS'` on all tested hardware.
  Users who explicitly want JAX must use `method='RS_JAX'`.
- Crossover values are hardware-specific.  A value that works on RTX 3060
  may not be optimal on A100 or H100.

---

## How to Update Crossover Values

After benchmarking on your hardware:

1. Run `benchmarks/_gpu_benchmark_runner.py` (or `benchmarks/jax_rs_benchmark.py`)
2. Find the n where `jax_warm_ms < numpy_warm_ms` for each family
3. Update `omnilss/config.py`:
   ```python
   GPU_CROSSOVER_N: dict[str, float] = {
       "NO":  50_000,   # example
       "GA":  80_000,   # example
       ...
       "default": 100_000,
   }
   ```
4. Add a benchmark artifact to `docs/benchmarks/` citing hardware, JAX version,
   and CUDA version
5. Update this ADR with the new values

---

## Related

- `omnilss/config.py` — implementation
- `omnilss/algorithms/jax_rs_core.py` — JAX RS core
- `omnilss/algorithms/jax_family_specs.py` — family specs
- `docs/benchmarks/jax_rs_*.md` — CPU benchmark results
- `docs/benchmarks/gpu_crossover_*.md` — GPU benchmark results
- ADR-005: JAX functional design
