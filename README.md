# OmniLSS

**Differentiable Distributional Regression in JAX** — a Python implementation of GAMLSS (Generalized Additive Models for Location, Scale and Shape) with GPU/TPU acceleration and neural network integration.

## Key Features

- **80+ Distribution Families** — broad R GAMLSS distribution coverage with ongoing d/p/q/r migration tracking
- **Benchmark validation gate** — consistency with native R `gamlss` is checked before performance comparisons
- **Transparent performance reporting** — benchmark reports separate cold time, warm time, deviance difference, and Python heap peak memory
- **Smoothing** — P-splines (`pb`), cubic splines (`ps`, `cs`) with automatic parameter selection (GCV/REML)
- **Multiple algorithms** — RS, CG v2 (explicit outer-loop with cross-derivative correction), Mixed, Adam, L-BFGS
- **Neural GAMLSS** — distribution parameters output by neural networks (Flax/Equinox compatible)
- **scikit-learn compatible** — optional `GAMLSSRegressor` wrapper for Pipeline integration
- **Probabilistic scoring** — CRPS, log score, DSS, interval score, PIT histogram

## Architecture Freeze and Benchmark Policy

OmniLSS is currently in a 30-day architecture stabilization phase. The active
priority is to reduce complexity, stabilize protocol boundaries, and improve
long-term maintainability rather than adding new distributions, optimizers,
formula syntax, wrappers, or smoothing methods.

Current status and documentation-maintenance tasks are tracked in
[`docs/maintenance/30-day-refactor-status.md`](docs/maintenance/30-day-refactor-status.md).

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
model = gamlss("y ~ x", family=NO(), data=data, method="RS")

# Joint optimizer path (historical method="CG" route)
model = gamlss("y ~ x", family=NO(), data=data, method="CG")

# Joint optimization with Adam
model = gamlss("y ~ x", family=NO(), data=data, method="joint",
               optimizer="adam", learning_rate=0.01)

# L-BFGS quasi-Newton
model = gamlss("y ~ x", family=NO(), data=data, method="lbfgs")
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

## Experimental JIT RS Core

- Added `fitting_jit.create_jit_rs_no_core()` as an isolated experiment for TASK-01-C.
- It uses `jax.lax.while_loop` (outer loop) and `jax.lax.fori_loop` (inner IRLS) for NO family core updates without changing the production RS path.

## Benchmarks

Run the validation gate from the repository root. It fails fast when R is
required but unavailable, runs consistency first, and only then runs performance.

```bash
python benchmarks/run_local_validation.py --quick
python benchmarks/run_local_validation.py --quick --allow-python-only --no-fit --no-smooth  # smoke only
```

See [`benchmarks/README.md`](benchmarks/README.md) for methodology and reporting rules.

## Testing

```bash
# Architecture-contract smoke tests
PYTHONPATH=omnilss/src python -m pytest omnilss/tests/test_core_architecture_contracts.py -q

# Full package tests from the package directory
cd omnilss
python -m pytest tests -q
```

R-backed consistency tests remain required for release and benchmark claims, but
they skip in Python-only environments when native R, `gamlss`, or `gamlss.dist`
are unavailable. Use the benchmark validation gate before publishing equivalence
or speed claims.

## Project Status

| Area | Current policy |
|--------|-------|
| Architecture mode | 30-day stabilization: Architecture > Features |
| Distribution families | 80+ legacy families, with protocol migration in progress |
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

**vs R gamlss**: R gamlss is the gold standard for statistical modeling. OmniLSS is for when you need Python, GPU, or ML pipeline integration.

**vs ondil**: ondil is designed for online/incremental learning on streaming data (scikit-learn API). OmniLSS is for batch differentiable modeling with hardware acceleration.

## Resources

- [Benchmarks](benchmarks/) — performance and consistency scripts
- [R consistency coverage](docs/testing/r-consistency-coverage.md) — current R-backed d/p/q coverage map
- [Benchmark methodology](docs/benchmarks/benchmarking-principles.md) — reporting rules for generated artifacts
- [Architecture freeze](docs/architecture/30-day-feature-freeze.md) — current stabilization policy
- [Maintenance status](docs/maintenance/30-day-refactor-status.md) — active refactor tasks and audit links
- [R GAMLSS](http://www.gamlss.org/) — original R implementation
- [JAX Documentation](https://jax.readthedocs.io/)

## License

GNU General Public License v3 or later (GPL-3.0+)


## CG Algorithm（Cole-Green）

- ✅ **完整实现**：基于完整观测信息矩阵（`jax.hessian`），包含跨参数 Hessian 块
- ✅ 同步更新所有分布参数（非坐标下降）
- ✅ 适用于参数高度相关的场景
- ⚠️ 比 RS 更慢（每次迭代需计算完整 Hessian）
- 🔧 大规模数据集（n > 5000）建议使用 RS

## 性能说明

| Distribution | Speedup vs R | Test conditions |
|---|---|---|
| NO, LOGNO | ~30-100x | n=200, intercept-only, JAX JIT warm |
| GA, WEI | ~20-50x | n=200, single predictor |
| BE | ~2x | n=200（见注1） |
| ZAGA | ~15x | n=200, hand-optimized derivatives |

> **注1**：BE 分布使用了手写导数，加速比相对较低是因为 R 侧 BE 本身有高度优化的 C 实现。  
> **注2**：以上数据基于单次 JIT warm-up 后的测量，首次调用含编译开销（通常 2-5x 更慢）。  
> **注3**：性能基准脚本位于 `benchmarks/comprehensive_performance_test.py`，可本地复现。


## Service APIs
- REST scaffold: `omnilss-server/` (FastAPI) with `/fit`, `/predict`, `/diagnostics/{model_id}`, `/distributions/select`.
- gRPC boundary: protobuf files under `omnilss/src/omnilss/api/grpc/proto/` and runtime server wiring in `omnilss.api.grpc.serve`.
- Generate gRPC stubs locally with `cd omnilss && python tools/generate_grpc_stubs.py` (requires `omnilss[grpc]`).
