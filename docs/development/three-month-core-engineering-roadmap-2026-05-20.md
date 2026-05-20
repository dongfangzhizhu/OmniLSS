# OmniLSS Three-Month Core Engineering Roadmap (Foundation-First Strategy)

## Goal

Focus exclusively on stabilizing and completing `omnilss` itself before prioritizing `omnilss-pro` or `omnilss-server`.

Within three months, transform `omnilss` into a:

- Numerically stable runtime
- API-stable runtime
- Architecturally stable runtime
- Benchmark-reproducible runtime
- Extensible family runtime
- CPU-first high-performance runtime
- Scientifically credible probabilistic/statistical core

---

## Strategic Understanding (Critical Context)

Current architectural findings:

1. GPU provides little to no advantage for the current IRLS/RS architecture, especially with small design matrices (`p` small).
2. `jax.while_loop + jnp.linalg.lstsq` is fundamentally inefficient for this workload on GPU.
3. Warm-starting from NumPy RS weakens the value proposition of a JAX runtime.
4. Immediate priorities are:
   - Numerical stability
   - Runtime correctness
   - Deterministic optimization
   - Extensible family systems
5. Near-term success criteria are not “faster than NumPy”, but “a trustworthy GAMLSS runtime”.

---

## Overall Three-Month Strategy

### Phase 1 (Month 1): Runtime Stability & Numerical Correctness

**Goal:** stabilize the system, eliminate optimizer instability, establish deterministic execution, and build a proper family runtime abstraction.

### Phase 2 (Month 2): Engineering Completeness & Performance Infrastructure

**Goal:** establish benchmark discipline, profiling infrastructure, runtime correctness gates, and CPU-first optimization.

### Phase 3 (Month 3): Scientific Credibility & API Stabilization

**Goal:** statistical validation, reproducibility, stable public API, and extensible distribution runtime.

---

## Month 1 — Runtime Stabilization

### Week 1 — Runtime Architecture Refactor

**Goal:** eliminate runtime chaos and non-deterministic behavior.

#### Task 1 — Remove monkey patching

Current issue:

```python
setattr(family_class, "pdf", _pdf)
```

Risks:

- Class pollution
- Non-determinism
- Threading hazards
- JIT cache corruption/confusion

Refactor target:

```text
FamilyDefinition
    ↓
ADFamily
    ↓
CompiledFamilyRuntime
```

Implementation requirements:

1. No `setattr` monkey patching.
2. Use immutable dataclasses.
3. `score`/`hessian`/`pdf`/`d`/`p`/`q`/`r` must be explicit fields.
4. Runtime functions must be compile-safe.
5. Family objects must be thread-safe.
6. Runtime modification of class state is forbidden.

Deliverables:

- New `ADFamily` dataclass
- `FamilyRuntime` abstraction
- Migration layer

#### Task 2 — Introduce runtime backend interface

Target architecture:

```text
Formula Layer
    ↓
ModelSpec
    ↓
RuntimeBackend
        ↓
    NumPyRuntime
    JAXRuntime
```

Required interface:

```python
class RuntimeBackend(Protocol):
    def fit(...)
    def predict(...)
    def score(...)
```

#### Task 3 — Deterministic execution layer

Must implement:

- Fixed random seed management
- Global dtype policy
- Fixed tolerance policy
- Deterministic convergence ordering

Deliverables:

- `runtime/config.py`
- Deterministic policy manager
- Seed manager

### Week 2 — Numerical Stability (Highest Priority)

**Core goal:** fix IRLS/RS oscillation and Hessian instability.

#### Task 1 — Implement damped IRLS

Replace unsafe update:

```text
beta_new = beta_old + delta
```

with:

```text
beta_new = beta_old + α * delta
```

where `0 < α <= 1`.

Must implement:

- Backtracking line search
- Adaptive damping
- Step clipping

#### Task 2 — Hessian stabilization

Must implement Levenberg-style stabilization:

```text
H + λI
```

Requirements:

1. Singular Hessian detection
2. Adaptive lambda
3. Curvature clipping
4. NaN/Inf guards
5. Condition number monitoring

Deliverables:

- `stabilized_hessian.py`
- Tests for singular distributions

#### Task 3 — Working response stability

Must implement:

- Eta clipping in `[-20, 20]`
- `z` clipping
- Weight clipping

### Week 3 — Optimizer Runtime Refactor

**Core goal:** turn optimizer into a real runtime subsystem.

#### Task 1 — Optimizer abstraction layer

```text
Optimizer
    ↓
RSOptimizer
CGOptimizer
NewtonOptimizer
TrustRegionOptimizer
```

#### Task 2 — Convergence framework

Must support:

- Gradient norm
- Deviance delta
- Parameter delta
- Curvature stability

#### Task 3 — Iteration trace system

Must record each iteration:

- Iteration index
- Deviance
- Gradient norm
- Step size
- Condition number
- Runtime

Deliverables:

- `optimizer_trace.py`
- Convergence monitor
- Replay utility

### Week 4 — Family System Completion

**Core goal:** turn family system into a true extensible DSL/runtime.

Must support explicit family metadata:

- `support`
- `parameter_constraints`
- `default_links`
- `tail_behavior`
- `moments`

Must support:

- Parameter validation (e.g., `sigma > 0`, valid `nu` range)
- Automatic domain checking
- Automatic invalid likelihood detection

Deliverables:

- `family_schema.py`
- Constraint system
- Validation engine

---

## Month 2 — Engineering Completeness

### Week 5 — Benchmark Framework

Must establish:

```text
benchmarks/
    correctness/
    performance/
    convergence/
    stress/
```

Correctness benchmarks must compare against:

- R `gamlss`
- `scipy`
- `statsmodels`

Must validate:

- Coefficient agreement
- Deviance agreement
- Convergence agreement

Deliverables:

- Benchmark runner
- Benchmark report generator
- Benchmark dataset registry

### Week 6 — CPU-First Optimization

Strategic shift: GPU optimization is no longer primary for current workload characteristics (`small p`, `large n`, sequential IRLS).

Optimization priorities:

1. BLAS/LAPACK optimization
2. Batched CPU vectorization
3. Memory locality optimization
4. Sparse design matrix support

Must implement Cholesky-based WLS and replace `jnp.linalg.lstsq` in the critical path.

Deliverables:

- `fast_wls.py`
- Cholesky solver
- Profiling benchmarks

### Week 7 — Memory & Profiling Infrastructure

Must build runtime profiler for:

- Memory allocation
- Kernel time
- Solver time
- Compilation time

Must build flamegraph tooling.

### Week 8 — Error System & Observability

Must build typed error hierarchy:

```text
OmniLSSError
    ↓
NumericalError
ConvergenceError
DistributionError
RuntimeError
```

Must build structured logging with:

- Iteration
- Family
- Condition number
- NaN state

---

## Month 3 — Scientific Credibility

### Week 9 — Statistical Validation

Must implement:

- Synthetic parameter recovery
- Confidence interval validation
- Asymptotic consistency tests
- Calibration diagnostics
- Distribution recovery benchmarks

Deliverables:

- Validation suite
- Statistical diagnostics
- Reproducibility reports

### Week 10 — Reproducibility

Must implement artifact locking for:

- Runtime config hash
- Solver config hash
- Family version hash

Must implement deterministic replay.

### Week 11 — Public API Freeze

Must stabilize:

- `gamlss(...)`
- `predict(...)`
- `fit(...)`
- `family(...)`

Must provide:

- Typed API
- Stable return schemas
- Immutable result objects

### Week 12 — Final Hardening

Must complete stress testing for:

- Extreme tails
- Huge sigma
- Singular matrices
- Collinearity
- Zero inflation

Must complete failure-mode testing to guarantee:

- No silent failures
- No NaN propagation
- No infinite loops

---

## Final Target Architecture (After 3 Months)

```text
omnilss/
├── runtime/
│   ├── backend/
│   ├── optimizer/
│   ├── convergence/
│   ├── profiling/
│   └── determinism/
│
├── families/
│   ├── metadata/
│   ├── constraints/
│   └── validation/
│
├── solvers/
│   ├── wls/
│   ├── cholesky/
│   └── sparse/
│
├── benchmarks/
│   ├── correctness/
│   ├── performance/
│   └── stress/
│
├── validation/
│   ├── statistical/
│   └── reproducibility/
│
└── api/
```

---

## Expected Outcome After 3 Months

OmniLSS should move from “a JAX GAMLSS experiment” to “a trustworthy probabilistic modeling runtime.”

## Most Important Strategic Principles

1. **CPU-first:** current workloads do not justify GPU-first architecture.
2. **Numerical stability over benchmark vanity:** never trade correctness for superficial speed claims.
3. **Runtime correctness over feature count:** unstable optimization and silent NaN propagation are higher risk than missing distribution coverage.

## Final Strategic Recommendation

The long-term moat is not “how many distributions are implemented,” but “how robustly difficult distributions can be fit under adverse numerical conditions.”
