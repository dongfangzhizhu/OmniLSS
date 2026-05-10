# OmniLSS

**Distributional modeling for the accelerator era** — OmniLSS is a JAX-native framework for differentiable distributional modeling — learning full conditional distributions, not point predictions. Built for GPUs and TPUs, it is designed to be the probabilistic foundation layer for machine learning and scientific computing.

## Features

- **80+ distribution families** — complete d/p/q/r functions, verified against R gamlss
- **30–120× faster than R gamlss** on CPU (steady-state, after JIT warm-up)
- **Numerical consistency with R** — 100% pass rate, deviance diff < 1×10⁻⁴
- **Smoothing** — P-splines (`pb`), cubic splines (`ps`, `cs`) with automatic GCV/REML
- **Multiple algorithms** — RS, CG, Mixed, Adam, L-BFGS
- **Neural GAMLSS** — distribution parameters output by neural networks
- **scikit-learn compatible** — optional `GAMLSSRegressor` for Pipeline integration
- **Probabilistic scoring** — CRPS, log score, DSS, interval score

## Installation

```bash
# From source
cd omnilss
pip install -e .

# Using uv (recommended)
uv sync --active --project ./omnilss
```

## Quick Start

```python
import numpy as np
from omnilss import gamlss, NO

np.random.seed(42)
n = 500
x = np.random.randn(n)
y = 2 + 3*x + np.random.randn(n)
data = {"y": y, "x": x}

# Fit model
model = gamlss("y ~ x", family=NO(), data=data)
print(f"Deviance: {model.g_dev:.2f}")

# Predict
newdata = {"x": np.array([0.0, 1.0, 2.0])}
params = model.predict_params(newdata)
print(params["mu"])    # predicted mean
print(params["sigma"]) # predicted std dev

# Quantile prediction
q = model.predict_quantiles(newdata, quantiles=[0.05, 0.5, 0.95])
print(q[0.5])  # median
```

## Performance

Benchmarked against R gamlss on CPU (Intel Core i7-12700K). Times are **steady-state**
(after JAX JIT warm-up). Each R call is a fresh Rscript process.

| Distribution | Mean speedup | Range |
|-------------|-------------|-------|
| NO (Normal) | **104×** | 88–122× |
| LOGNO (Log-Normal) | **110×** | 100–148× |
| PO (Poisson) | **62×** | 32–84× |
| GA (Gamma) | **34×** | 29–43× |
| BI (Binomial) | **65×** | 29–119× |
| NBI (Neg. Binomial) | **35×** | 13–55× |
| BE (Beta) | **27×** | 22–33× |
| ZIP | **22×** | 12–27× |
| ZAGA | **16×** | 14–20× |

> **Note on JAX JIT**: The first call to `gamlss()` triggers JAX JIT compilation
> (~0.5–2s overhead). Subsequent calls with the same model structure are 30–120× faster
> than R. For scripts that call `gamlss()` only once, add a warm-up call or use
> `jax.jit` explicitly. See [docs/performance_guide.md](docs/performance_guide.md).

## Algorithms

| Algorithm | Status | Description |
|-----------|--------|-------------|
| `RS` | ✅ Full | Rigby-Stasinopoulos — default, most stable |
| `CG` | ✅ Full | Cole-Green — includes cross-derivative corrections |
| `Mixed` | ✅ Full | Auto-selects RS or CG |
| `Adam` | ✅ | Gradient descent via Optax |
| `L-BFGS` | ✅ | Quasi-Newton method |

```python
# RS (default)
model = gamlss("y ~ x", family=NO(), data=data, algorithm="RS")

# CG with cross-derivatives
model = gamlss("y ~ x", family=NO(), data=data, algorithm="CG")

# Adam optimizer
model = gamlss("y ~ x", family=NO(), data=data, algorithm="Adam",
               learning_rate=0.01)
```

## Smoothing

```python
# P-spline with automatic lambda selection
model = gamlss("y ~ pb(x)", family=NO(), data=data)

# Explicit REML selection
model = gamlss("y ~ pb(x, lambda_method='REML')", family=NO(), data=data)

# Centile curves
curves = model.centiles(xvar="x", cent=[5, 50, 95])
```

## Neural GAMLSS

```python
from omnilss.deep import fit_deep_gamlss
import jax.numpy as jnp

model, params, history = fit_deep_gamlss(
    jnp.array(X), jnp.array(y),
    family=NO(),
    hidden_dims=(64, 32),
    n_epochs=100,
)
```

## Comparison with Similar Tools

| Tool | Backend | GPU | Differentiable | GAMLSS structure | Online learning |
|------|---------|-----|---------------|-----------------|----------------|
| **OmniLSS** | **JAX** | **✅** | **✅** | **✅** | ✗ |
| R gamlss | R | ✗ | ✗ | ✅ | ✗ |
| ondil | NumPy+Numba | ✗ | ✗ | ✅ | ✅ |
| NumPyro | JAX | ✅ | ✅ | ✗ | ✗ |
| pyGAM | NumPy | ✗ | ✗ | ✗ (mean only) | ✗ |

**vs R gamlss**: R gamlss is the gold standard for statistical modeling. OmniLSS is for
when you need Python, GPU, or ML pipeline integration.

**vs ondil**: ondil is designed for online/incremental learning on streaming data
(scikit-learn API). OmniLSS is for batch differentiable modeling with hardware acceleration.

## Benchmarks

```bash
# Performance benchmark — 81 tests, auto-generates report
python benchmarks/comprehensive_performance_test.py

# Quick test (3 distributions, ~5 minutes)
python benchmarks/comprehensive_performance_test.py --quick

# Consistency test vs R
python benchmarks/comprehensive_r_consistency_test.py

# Python only (no R required)
python benchmarks/comprehensive_performance_test.py --no-r
```

## Testing

```bash
# Full test suite
./run_tests.sh          # Linux/macOS
./run_tests.ps1 -All    # Windows

python -m pytest omnilss/tests -q
```

587 tests, 100% pass rate (92 R-validation tests).

## Documentation

- [omnilss/README.md](omnilss/README.md) — package documentation
- [docs/algorithm_guide.md](docs/algorithm_guide.md) — RS, CG, Mixed algorithms
- [docs/user_guide/distributions.md](docs/user_guide/distributions.md) — distribution families
- [docs/prediction_guide.md](docs/prediction_guide.md) — prediction API
- [docs/smoothing_parameter_guide.md](docs/smoothing_parameter_guide.md) — GCV/REML
- [benchmarks/README.md](benchmarks/README.md) — benchmark guide
- [tutorials/](tutorials/) — learning path and R migration guide
- [examples/colab/](examples/colab/) — Google Colab notebooks (CPU/GPU/TPU)

## Project Structure

```
omnilss/          ← Python package (pip install -e .)
  src/omnilss/   ← source code
  tests/         ← test suite
docs/            ← documentation
benchmarks/      ← performance & consistency benchmarks
tutorials/       ← learning path
examples/        ← example scripts and Colab notebooks
```

## License

GNU General Public License v3 or later (GPL-3.0+)

## Citation

```bibtex
@software{omnilss2026,
  title  = {OmniLSS: Differentiable Distributional Regression in JAX},
  author = {OmniLSS Team},
  year   = {2026},
  url    = {https://github.com/dongfangzhizhu/OmniLSS}
}
```
