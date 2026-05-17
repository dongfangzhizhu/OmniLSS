# JAX RS Core — Benchmark Notes

[中文版本](jax_rs_notes_cn.md)

## CPU Performance Interpretation

The smoke benchmark (`--smoke`) runs on CPU only.  On CPU, the JAX RS path
includes a **NumPy RS warm-start** (full convergence) before handing off to the
JAX `while_loop`.  This means the reported `warm_ms` includes:

1. NumPy RS fitting (full convergence, same as `method='RS'`)
2. JAX `while_loop` refinement (typically 0–2 additional iterations)
3. JAX `jnp.linalg.lstsq` for final beta extraction

On CPU, the JAX path is therefore **not faster** than NumPy RS for small n
(n=1000).  The warm-start overhead dominates.

## Where JAX Wins

The JAX RS core is designed for:

1. **GPU/TPU acceleration** — the `while_loop` + `jnp.linalg.lstsq` runs on
   GPU via cuSOLVER.  Expected speedup: 5–50x for n ≥ 10,000.
2. **Large n** — for n ≥ 100,000, the JAX WLS solve dominates and GPU
   acceleration is significant.
3. **Differentiability** — the JAX path is fully differentiable, enabling
   gradient-based meta-learning and hyperparameter optimisation.

## Architecture

```
method='RS_JAX'
    │
    ├── NumPy RS warm-start (full convergence)
    │       └── Provides stable initial values for JAX core
    │
    └── JAX while_loop (jax.lax.while_loop)
            ├── IRLS step per parameter (jax.lax.fori_loop, max_inner=1)
            ├── Step-halving (jax.lax.cond) if deviance increases
            └── Convergence check (|Δdeviance| < tol)
```

## Correctness Gate

All benchmark runs must pass the correctness gate:
`|JAX deviance − NumPy deviance| < 0.1` for all families.

Run the gate:
```bash
cd omnilss
python -m pytest tests/test_jax_rs_core.py -q
```

Run the benchmark:
```bash
# Smoke (CPU, no R, n=1000)
python benchmarks/jax_rs_benchmark.py --smoke --no-r

# Full (requires GPU for GPU columns)
python benchmarks/jax_rs_benchmark.py
```
