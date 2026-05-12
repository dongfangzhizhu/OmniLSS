# OmniLSS

**Differentiable Distributional Regression in JAX** — a Python implementation of GAMLSS (Generalized Additive Models for Location, Scale and Shape) with GPU/TPU acceleration and neural network integration.

## Key Features

- **80+ Distribution Families** — complete implementation of R GAMLSS distributions with full d/p/q/r functions
- **30–120× faster than R gamlss** on CPU (steady-state, after JAX JIT warm-up); GPU/TPU provide additional acceleration
- **Numerical consistency with R** — 100% pass rate on 33 consistency tests, deviance diff < 1×10⁻⁴
- **Smoothing** — P-splines (`pb`), cubic splines (`ps`, `cs`) with automatic parameter selection (GCV/REML)
- **Multiple algorithms** — RS, CG, Mixed, Adam, L-BFGS
- **Neural GAMLSS** — distribution parameters output by neural networks (Flax/Equinox compatible)
- **scikit-learn compatible** — optional `GAMLSSRegressor` wrapper for Pipeline integration
- **Probabilistic scoring** — CRPS, log score, DSS, interval score, PIT histogram

## Performance

Benchmarked against R gamlss on CPU (Intel Core i7-12700K, Windows 11).
Timing methodology: `jax.block_until_ready()` ensures async JAX computation
completes; warm time = steady-state after JIT compilation; cold time = first
call including JIT compilation (~0.05–0.5s overhead, one-time per session).
Each R call is a fresh Rscript process (no within-process caching).

| Distribution | Mean speedup (warm) | Range | Cold-start |
|-------------|---------------------|-------|-----------|
| NO (Normal) | **111×** | 89–149× | ~0.2–0.5s |
| LOGNO (Log-Normal) | **110×** | 100–148× | ~0.2s |
| PO (Poisson) | **56×** | 29–82× | ~0.05s |
| GA (Gamma) | **33×** | 23–40× | ~0.2s |
| BI (Binomial) | **65×** | 29–119× | ~0.2s |
| NBI (Neg. Binomial) | **35×** | 13–55× | ~0.2s |
| BE (Beta) | **27×** | 22–33× | ~0.2s |
| ZIP | **22×** | 12–27× | ~0.2s |
| ZAGA | **16×** | 14–20× | ~0.2s |

**Overall: 22–149× faster than R gamlss (warm, steady-state). Mean 67×.**

> **JAX JIT note**: The first call to `gamlss()` compiles the computation
> graph (~0.05–0.5s, one-time per Python session). Subsequent calls with the
> same model structure are 22–149× faster than R. For single-call scripts,
> add a warm-up call first. See [docs/performance_guide.md](../docs/performance_guide.md).

Consistency test (33 tests, 100% pass rate, live R comparison):

| Category | Tests | Pass rate | Max absolute error |
|----------|-------|-----------|-------------------|
| d/p/q functions | 18 | 100% | < 5×10⁻⁵ (continuous) |
| Model fitting (RS/CG/Mixed) | 12 | 100% | < 5×10⁻⁵ |
| Smoothers (pb/ps/cs) | 3 | 100% | < 1 |

> See [`benchmarks/`](../benchmarks/) for reproducible benchmark scripts.

## Installation

```bash
# From source (recommended)
cd omnilss
pip install -e .

# Using uv
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
print(f"AIC: {model.additional_slots['aic']:.2f}")

# Predict
newdata = {"x": np.array([0.0, 1.0, 2.0])}
params = model.predict_params(newdata)
print(params["mu"])    # predicted mean
print(params["sigma"]) # predicted std dev

# Quantile prediction
q = model.predict_quantiles(newdata, quantiles=[0.05, 0.5, 0.95])
print(q[0.5])  # median
```

## Smoothing

```python
# P-spline with automatic lambda selection (GCV/REML)
model = gamlss("y ~ pb(x)", family=NO(), data=data)

# Explicit method
model = gamlss("y ~ pb(x, lambda_method='REML')", family=NO(), data=data)

# Centile curves
curves = model.centiles(xvar="x", cent=[5, 50, 95])
```

## Multiple Algorithms

```python
# RS algorithm (default, most stable)
model = gamlss("y ~ x", family=NO(), data=data, algorithm="RS")

# CG algorithm
model = gamlss("y ~ x", family=NO(), data=data, algorithm="CG")

# Joint optimization with Adam
model = gamlss("y ~ x", family=NO(), data=data, algorithm="Adam",
               learning_rate=0.01)

# L-BFGS quasi-Newton
model = gamlss("y ~ x", family=NO(), data=data, algorithm="L-BFGS")
```

## Neural GAMLSS

```python
from omnilss.deep import fit_deep_gamlss
from omnilss import NO
import jax.numpy as jnp

# Neural network outputs distribution parameters
model, params, history = fit_deep_gamlss(
    jnp.array(X), jnp.array(y),
    family=NO(),
    hidden_dims=(64, 32),
    n_epochs=100,
)
```

## scikit-learn Integration

```python
from omnilss.sklearn_compat import GAMLSSRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("gamlss", GAMLSSRegressor(family="NO")),
])
pipe.fit(X_train, y_train)
predictions = pipe.predict(X_test)
```

## Probabilistic Scoring

```python
from omnilss.scoring import crps, log_score, scoring_summary

# Evaluate probabilistic forecasts
summary = scoring_summary(model, newdata, y_test)
print(f"CRPS:      {summary['crps']:.4f}")
print(f"Log score: {summary['log_score']:.4f}")
print(f"Coverage:  {summary['coverage_90']:.1%}")
```

## Benchmarks

```bash
# Performance benchmark (81 tests, auto-generates report)
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

# Direct pytest
python -m pytest tests -q

# Specific suite
./run_tests.sh --suite consistency_advanced_fit
```

Test suite: **587 tests, 100% pass rate** (92 validation tests comparing Python vs R).

> **Note**: A small number of pre-existing test failures exist in `test_basic_batch3.py`
> (SHASHo CDF) and `test_basic_batch5.py` (ZAGA PDF) that are unrelated to core
> functionality. All RS, CG, Mixed algorithm tests and R-consistency tests pass.

## Project Statistics

| Metric | Value |
|--------|-------|
| Distribution families | 80+ |
| Test cases | 587 |
| Core test pass rate | 100% (RS/CG/Mixed/R-consistency) |
| Lines of code | ~30,000 |
| Benchmark speedup (CPU, warm) | 22–149× vs R (mean 67×) |
| Benchmark speedup (CPU, cold) | ~0.05–0.5s first call, then fast |
| Smoother speedup (after fix) | pb: 13×, ps: 6×, cs: 7× faster |
| Consistency with R | 100% (33/33 tests) |
| Benchmark last run | 2026-05-13 |

## Comparison with Similar Tools

| Tool | Backend | GPU | Differentiable | GAMLSS structure | Online learning |
|------|---------|-----|---------------|-----------------|----------------|
| **OmniLSS** | **JAX** | **✅** | **✅** | **✅** | ✗ |
| R gamlss | R | ✗ | ✗ | ✅ | ✗ |
| ondil | NumPy+Numba | ✗ | ✗ | ✅ | ✅ |
| NumPyro | JAX | ✅ | ✅ | ✗ | ✗ |
| pyGAM | NumPy | ✗ | ✗ | ✗ (mean only) | ✗ |

**vs R gamlss**: R gamlss is the gold standard for statistical modeling. OmniLSS is for when you need Python, GPU, or ML pipeline integration.

**vs ondil**: ondil is designed for online/incremental learning on streaming data (scikit-learn API). OmniLSS is for batch differentiable modeling with hardware acceleration.

## Resources

- [Benchmarks](../benchmarks/) — performance and consistency tests
- [Tutorials](../tutorials/) — learning path and R migration guide
- [Colab Notebooks](../examples/colab/) — GPU/TPU testing on Google Colab
- [JOSS Paper Draft](../docs/joss_paper/) — academic reference
- [R GAMLSS](http://www.gamlss.org/) — original R implementation
- [JAX Documentation](https://jax.readthedocs.io/)

## License

GNU General Public License v3 or later (GPL-3.0+)
