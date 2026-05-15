# OmniLSS

**Differentiable Distributional Regression in JAX** — a Python implementation of GAMLSS (Generalized Additive Models for Location, Scale and Shape) with GPU/TPU acceleration and neural network integration.

## Key Features

- **80+ Distribution Families** — broad R GAMLSS distribution coverage with ongoing d/p/q/r migration tracking
- **Benchmark validation gate** — consistency with native R `gamlss` is checked before performance comparisons
- **Transparent performance reporting** — benchmark reports separate cold time, warm time, deviance difference, and Python heap peak memory
- **Smoothing** — P-splines (`pb`), cubic splines (`ps`, `cs`) with automatic parameter selection (GCV/REML)
- **Multiple algorithms** — RS, CG, Mixed, Adam, L-BFGS
- **Neural GAMLSS** — distribution parameters output by neural networks (Flax/Equinox compatible)
- **scikit-learn compatible** — optional `GAMLSSRegressor` wrapper for Pipeline integration
- **Probabilistic scoring** — CRPS, log score, DSS, interval score, PIT histogram

## Architecture Freeze and Benchmark Policy

OmniLSS is currently in a 30-day architecture stabilization phase. The active
priority is to reduce complexity, stabilize protocol boundaries, and improve
long-term maintainability rather than adding new distributions, optimizers,
formula syntax, wrappers, or smoothing methods.

Current status and documentation-maintenance tasks are tracked in
[`../docs/maintenance/30-day-refactor-status.md`](../docs/maintenance/30-day-refactor-status.md).

Benchmark claims must be generated from the validation gate in `benchmarks/`:

```bash
python benchmarks/run_local_validation.py --quick
```

That command requires native R `gamlss` and runs numerical consistency before
performance. In environments without R, `--allow-python-only` is a smoke check
only and must not be used to claim R equivalence.

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

Run the validation gate from the repository root. It fails fast when R is
required but unavailable, runs consistency first, and only then runs performance.

```bash
python benchmarks/run_local_validation.py --quick
python benchmarks/run_local_validation.py --quick --allow-python-only --no-fit --no-smooth  # smoke only
```

See [`../benchmarks/README.md`](../benchmarks/README.md) for methodology and reporting rules.

## Testing

```bash
# Architecture-contract smoke tests
PYTHONPATH=src python -m pytest tests/test_core_architecture_contracts.py -q

# Full package tests
python -m pytest tests -q
```

R-backed consistency tests remain required for release and benchmark claims, but
they skip in Python-only environments when native R, `gamlss`, or `gamlss.dist`
are unavailable. Use the repository-level benchmark validation gate before
publishing equivalence or speed claims.

## Project Status

| Area | Current policy |
|--------|-------|
| Architecture mode | 30-day stabilization: Architecture > Features |
| Distribution families | 80+ legacy families, with protocol migration in progress |
| Distributions with full d/p/q/r | Tracked by generated consistency reports, not static README claims |
| Canonical protocols | Distribution, optimizer, parameter, link, and constraint boundaries exist |
| Benchmark claims | Must cite generated validation artifacts with hardware/backend/R availability |
| R equivalence | Requires native R `gamlss` validation; Python-only smoke checks are not proof |
| License | GPL-3.0-or-later |

## Comparison with Similar Tools

| Tool | Backend | GPU | Differentiable | GAMLSS structure | Online learning |
|------|---------|-----|---------------|-----------------|----------------|
| **OmniLSS** | **JAX** | **✅** | **✅** | **✅** | ✗ |
| R gamlss | R | ✗ | ✗ | ✅ | ✗ |
| ondil | NumPy+Numba | ✗ | ✗ | ✅ | ✅ |
| NumPyro | JAX | ✅ | ✅ | ✗ | ✗ |
| pyGAM | NumPy | ✗ | ✗ | ✗ (mean only) | ✗ |

**vs R gamlss**: R gamlss is the gold standard for statistical modeling. OmniLSS is for when you need Python, GPU, or ML pipeline integration. Any performance comparison must cite a generated benchmark report.

**vs ondil**: ondil is designed for online/incremental learning on streaming data (scikit-learn API). OmniLSS is for batch differentiable modeling with hardware acceleration. Both implement distributional regression — different design goals.

## Resources

- [Benchmarks](../benchmarks/) — performance and consistency scripts
- [R consistency coverage](../docs/testing/r-consistency-coverage.md) — current R-backed d/p/q coverage map
- [Benchmark methodology](../docs/benchmarks/benchmarking-principles.md) — reporting rules for generated artifacts
- [Architecture freeze](../docs/architecture/30-day-feature-freeze.md) — current stabilization policy
- [Maintenance status](../docs/maintenance/30-day-refactor-status.md) — active refactor tasks and audit links
- [R GAMLSS](http://www.gamlss.org/) — original R implementation
- [JAX Documentation](https://jax.readthedocs.io/)

## License

GNU General Public License v3 or later (GPL-3.0+)
