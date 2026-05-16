# Why OmniLSS Runs 30–120× Faster Than R GAMLSS

*A technical deep-dive into JAX-accelerated distributional regression*

---

Distributional regression — fitting separate models for the mean, variance, skewness,
and shape of a response variable — is one of the most powerful tools in modern
statistics. R's `gamlss` package has been the gold standard for this since 2005.
But it has a problem: it's slow.

We built **OmniLSS**, a Python/JAX reimplementation of GAMLSS, to fix that.
This post explains how we achieved 30–120× speedups over R on CPU, what the
actual bottlenecks were, and what we had to fix along the way.

---

## What is GAMLSS?

GAMLSS (Generalized Additive Models for Location, Scale and Shape) extends
classical regression by modeling all parameters of a response distribution,
not just the mean:

```
g₁(μ)   = η₁ = X₁β₁ + s₁(x)   # mean
g₂(σ)   = η₂ = X₂β₂            # variance
g₃(ν)   = η₃ = X₃β₃            # skewness
g₄(τ)   = η₄ = X₄β₄            # kurtosis
```

This lets you model heteroscedasticity, skewness, and heavy tails as functions
of covariates — something standard GLMs cannot do.

The fitting algorithm (RS: Rigby-Stasinopoulos) iterates over parameters,
updating each via IRLS (Iteratively Reweighted Least Squares) until convergence.

---

## The Benchmark Setup

We ran 81 tests: 9 distributions × 3 data sizes (n=100, 500, 5000) × 3 formulas.
Each R call is a fresh `Rscript` subprocess (no within-process caching).
Python times are **steady-state** (after JAX JIT warm-up).

Hardware: Intel Core i7-12700K, Windows 11, Python 3.12, JAX 0.10.0.

---

## Results

| Distribution | Mean speedup | Range |
|-------------|-------------|-------|
| NO (Normal) | **104×** | 88–122× |
| LOGNO (Log-Normal) | **110×** | 100–148× |
| PO (Poisson) | **62×** | 32–84× |
| GA (Gamma) | **34×** | 29–43× |
| BI (Binomial) | **65×** | 29–119× |
| NBI (Neg. Binomial) | **35×** | 13–55× |
| BE (Beta) | **27×** | 22–33× |
| ZIP (Zero-Inflated Poisson) | **22×** | 12–27× |
| ZAGA (Zero-Adj. Gamma) | **16×** | 14–20× |

**Overall: 30–120× faster than R gamlss on CPU.**

Numerical accuracy: all 81 tests pass with deviance difference < 1×10⁻⁴.

---

## Why Is JAX Faster?

### 1. JIT compilation eliminates Python overhead

R's GAMLSS calls compiled C/Fortran code for the inner IRLS loop.
Python normally can't compete — but JAX's `jit` decorator compiles the
entire computation graph to XLA, which runs at near-native speed.

```python
import jax

@jax.jit
def irls_step(X, y, w, beta):
    # This entire function compiles to a single XLA kernel
    XtWX = X.T @ (w[:, None] * X)
    Xty  = X.T @ (w * y)
    return jnp.linalg.solve(XtWX, Xty)
```

The first call compiles (~0.5–2s). Every subsequent call with the same
structure runs at XLA speed.

### 2. Vectorized operations over the full dataset

R's IRLS loop processes observations sequentially in some inner steps.
JAX operates on the full array at once, leveraging SIMD and cache efficiency.

### 3. No R startup overhead

Each R benchmark call spawns a fresh `Rscript` process (~0.6s startup).
Python's JAX computation is ~0.005–0.03s for the same model.

---

## The Speedup Isn't Uniform — Here's Why

Notice that GA (Gamma) achieves only 34× while NO (Normal) achieves 104×.
The difference is in the number of IRLS iterations:

- **NO**: 1–2 iterations (closed-form solution exists)
- **GA**: 3–5 iterations (requires iterative sigma update)
- **NBI**: 5–10 iterations (overdispersion parameter is harder to estimate)

Each iteration is fast in JAX, but more iterations means less relative advantage
over R's compiled C code.

---

## The Bug We Found: JAX Scatter in a Python Loop

During development, we discovered a critical performance regression in our
smoother implementation. The `pb()` (P-spline) smoother was taking **4.5 seconds**
per fit — slower than R's 0.05 seconds.

The culprit was this code in `penalized_wls_no_jit()`:

```python
# SLOW: O(p²) JAX scatter operations
penalty_total = jnp.zeros((p, p))
for i in range(expected_size):
    for j in range(expected_size):
        penalty_total = penalty_total.at[start + i, start + j].add(
            P_scaled[i, j]
        )
```

Each `.at[].add()` call triggers JAX's tracing machinery. With p=20 basis
functions, this is 400 JAX operations per IRLS iteration, called 10+ times
per model fit = 4,000+ JAX trace calls.

The fix was trivial — use numpy for the block assignment:

```python
# FAST: single numpy block assignment
penalty_total = np.zeros((p, p))
penalty_total[start:end, start:end] += lambda_ * P_np
```

Result: **pb: 4.5s → 0.34s (13× faster)**.

**Lesson**: Never put JAX operations inside Python loops. Vectorize or use numpy.

---

## The JIT Warm-Up Caveat

JAX's JIT compilation is a one-time cost per Python session. The first call
to `gamlss()` takes 0.5–2s to compile. Subsequent calls are fast.

```python
import time
from omnilss import gamlss, NO
import numpy as np

data = {"y": np.random.randn(500), "x": np.random.randn(500)}

# First call: JIT compilation (~1.2s)
t0 = time.perf_counter()
model = gamlss("y ~ x", family=NO(), data=data)
print(f"Cold start: {time.perf_counter() - t0:.3f}s")  # ~1.2s

# Second call: steady state (~0.007s)
t0 = time.perf_counter()
model = gamlss("y ~ x", family=NO(), data=data)
print(f"Warm:       {time.perf_counter() - t0:.3f}s")  # ~0.007s
```

For scripts that call `gamlss()` only once, add a warm-up call first:

```python
# Warm up with a tiny dataset
_ = gamlss("y ~ x", family=NO(), data={"y": y[:10], "x": x[:10]})
# Now time the real call
model = gamlss("y ~ x", family=NO(), data=data)
```

---

## Beyond Speed: What OmniLSS Adds

The speedup is useful, but the real reason to use OmniLSS over R is the
capabilities it unlocks:

### GPU/TPU acceleration

```python
import jax

# Run on GPU — same code, no changes
with jax.default_device(jax.devices('gpu')[0]):
    model = gamlss("y ~ x", family=NO(), data=large_data)
```

For n > 10,000, GPU provides an additional 3–10× speedup over CPU.

### Neural GAMLSS

Distribution parameters output by a neural network:

```python
from omnilss.deep import fit_deep_gamlss

model, params, history = fit_deep_gamlss(
    X, y,
    family=NO(),
    hidden_dims=(64, 32),
    n_epochs=100,
)
```

This is impossible in R gamlss. It's the key differentiator.

### scikit-learn integration

```python
from omnilss.sklearn_compat import GAMLSSRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("gamlss", GAMLSSRegressor(family="NO")),
])
pipe.fit(X_train, y_train)
```

### Probabilistic scoring

```python
from omnilss.scoring import crps, scoring_summary

summary = scoring_summary(model, newdata, y_test)
print(f"CRPS:     {summary['crps']:.4f}")
print(f"Coverage: {summary['coverage_90']:.1%}")
```

---

## Numerical Accuracy

Speed means nothing if the results are wrong. We verified OmniLSS against R
on 33 test cases:

| Category | Tests | Pass rate | Max absolute error |
|----------|-------|-----------|-------------------|
| d/p/q functions | 18 | 100% | < 5×10⁻⁵ (continuous) |
| Model fitting | 12 | 100% | < 5×10⁻⁵ |
| Smoothers | 3 | 100% | < 1 (cs: 0.8, pb: 0.06) |

The smoother errors (cs: 0.8, pb: 0.06) are in fitted values, not parameters.
For point prediction this is acceptable; for research requiring exact R
equivalence, use `cs()` which achieves < 0.01 error.

---

## Try It

```bash
pip install git+https://github.com/dongfangzhizhu/OmniLSS.git#subdirectory=omnilss
```

```python
import numpy as np
from omnilss import gamlss, NO

np.random.seed(42)
n = 1000
x = np.random.randn(n)
y = 2 + 3*x + np.random.randn(n)
data = {"y": y, "x": x}

model = gamlss("y ~ x", sigma_formula="~ x", family=NO(), data=data)
print(f"Deviance: {model.g_dev:.2f}")

# Predict quantiles
newdata = {"x": np.array([0.0, 1.0, 2.0])}
q = model.predict_quantiles(newdata, quantiles=[0.05, 0.5, 0.95])
print(f"Median: {q[0.5]}")
```

Benchmarks and reproducible scripts:
```bash
git clone https://github.com/dongfangzhizhu/OmniLSS.git
python benchmarks/comprehensive_performance_test.py --quick
```

---

## What's Next

- **PyPI release** (v0.3.0) — coming soon
- **JOSS paper** — in preparation
- **Neural GAMLSS paper** — planned
- **More distributions** — currently 80+, targeting 100+

---

*OmniLSS is open source (GPL-3.0). Contributions welcome.*

*GitHub: https://github.com/dongfangzhizhu/OmniLSS*
