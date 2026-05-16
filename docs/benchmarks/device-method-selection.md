# Device-Aware Method Selection

## TL;DR

| Your hardware | What `method='RS'` does | What `method='auto'` does |
|---------------|------------------------|--------------------------|
| CPU (any) | NumPy IRLS — fastest | Same as `'RS'` |
| GPU (RTX 3060, tested) | NumPy IRLS — fastest | Same as `'RS'` (no crossover found) |
| GPU (untested) | NumPy IRLS | Same as `'RS'` until you set crossover |
| TPU | NumPy IRLS | Same as `'RS'` until you set crossover |

**Bottom line**: `method='RS'` and `method='auto'` are currently equivalent
on all tested hardware.  `method='RS_JAX'` is available for experimentation
but is not faster in practice yet.

---

## Why JAX RS is Slower on CPU and GPU

OmniLSS has two RS fitting backends:

**`method='RS'`** — NumPy IRLS  
Uses `np.linalg.lstsq` (CPU LAPACK) for the weighted least squares step.
For typical GAMLSS design matrices (p = 2–10 columns), CPU LAPACK is
extremely fast because the matrices are small.

**`method='RS_JAX'`** — JAX `while_loop` + `jnp.linalg.lstsq`  
Uses `jax.lax.while_loop` for the outer RS loop and `jnp.linalg.lstsq`
(cuSOLVER on GPU) for WLS.  Includes a NumPy RS warm-start for numerical
stability.

**Why JAX is slower for small p**:

1. **GPU kernel launch overhead** — each `jnp.linalg.lstsq` call on a
   `[n, 2]` matrix incurs GPU kernel launch latency (~100–500 µs).
   For p=2, this overhead dominates the actual computation.

2. **Warm-start cost** — `method='RS_JAX'` runs a full NumPy RS fit first
   to get stable initial values, then refines with JAX.  This doubles the
   NumPy RS cost before JAX even starts.

3. **Sequential IRLS** — IRLS is inherently sequential (each iteration
   depends on the previous).  GPU parallelism helps with the matrix
   operations but not the sequential structure.

**When JAX would be faster**:
- Very large design matrices (p >> 10), where GPU matrix operations dominate
- Batch fitting across many datasets simultaneously
- Future: fully JAX-native IRLS without NumPy warm-start

---

## Benchmark Results

### CPU: Intel i7-12700K (12P/20L cores, 34 GB RAM)

JAX 0.6.2, CPU backend.  No crossover found up to n = 1,000,000.

| Family | n | JAX warm (ms) | NumPy warm (ms) | Speedup vs R |
|--------|---|---:|---:|---:|
| NO | 100,000 | 250 | 43 | 1.88x vs R |
| GA | 100,000 | 564 | 221 | 2.34x vs R |
| WEI | 100,000 | 566 | 312 | 3.68x vs R |
| TF | 100,000 | 5,946 | 5,830 | 1.57x vs R |

Note: JAX is faster than **R gamlss** at large n, even on CPU.

### GPU: NVIDIA RTX 3060 12 GB (WSL2, JAX 0.10.0, CUDA 12)

No crossover found up to n = 500,000 for p=2 design matrices.

| Family | n | JAX warm (ms) | NumPy warm (ms) |
|--------|---|---:|---:|
| NO | 500,000 | 1,448 | 419 |
| GA | 500,000 | 3,232 | 1,331 |
| WEI | 500,000 | 4,296 | 3,255 |

---

## Configuring Auto-Selection

After benchmarking on your hardware, update the crossover thresholds:

```python
import omnilss.config as cfg

# Check current state
print(cfg.get_config_summary())

# Set GPU crossover after your benchmarking
# (n at which JAX warm time < NumPy warm time)
cfg.GPU_CROSSOVER_N["NO"]  = 50_000   # example
cfg.GPU_CROSSOVER_N["GA"]  = 80_000   # example
cfg.GPU_CROSSOVER_N["default"] = 100_000  # fallback for other families

# Now method='auto' will use JAX for NO when n >= 50,000
model = gamlss("y ~ x", family=NO(), data=data, method="auto")
```

Via environment variables (before importing omnilss):
```bash
# Use JAX for all families when n >= 50,000 on GPU
export OMNILSS_GPU_CROSSOVER_N=50000

# Disable auto-switching entirely
export OMNILSS_AUTO_METHOD=0

# Force JAX everywhere (for testing/benchmarking)
export OMNILSS_FORCE_JAX=1
```

### TPU Configuration

TPU has not been benchmarked yet.  To configure after testing:

```python
import omnilss.config as cfg
cfg.TPU_CROSSOVER_N["NO"]      = 10_000   # example
cfg.TPU_CROSSOVER_N["default"] = 20_000
```

---

## Running Your Own Benchmark

```bash
# GPU crossover benchmark (run in GPU environment)
source /path/to/jax-gpu-env/bin/activate
PYTHONPATH=omnilss/src python benchmarks/_gpu_benchmark_runner.py \
    --n-values 1000 5000 10000 50000 100000 200000 500000 \
    --families NO GA PO BI WEI TF

# Full benchmark with R comparison (CPU)
PYTHONPATH=omnilss/src python benchmarks/jax_rs_benchmark.py \
    --n-values 1000 10000 100000 500000
```

Results are written to `docs/benchmarks/`.

---

## Explicit Method Override

You can always override auto-selection:

```python
# Always use NumPy RS (fastest on CPU, safe everywhere)
model = gamlss("y ~ x", family=NO(), data=data, method="RS")

# Always use JAX RS (for testing or GPU workloads)
model = gamlss("y ~ x", family=NO(), data=data, method="RS_JAX")

# Auto-select based on device and n (currently == 'RS' everywhere)
model = gamlss("y ~ x", family=NO(), data=data, method="auto")
```

Supported families for `method='RS_JAX'`: NO, GA, PO, BI, WEI, TF.  
Other families fall back to `method='RS'` automatically.
