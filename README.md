# OmniLSS

[中文版本](README_cn.md)

**Differentiable Distributional Regression in JAX** — a Python implementation of GAMLSS (Generalized Additive Models for Location, Scale and Shape) built from scratch on JAX / NumPy / SciPy, with live-R parity verification, automatic differentiation, neural network integration, and serving boundaries.

## Key Capabilities & Verified Status

> Status claims below are backed by machine-regenerable artifacts in [`docs/development/`](docs/development/) rather than static estimates. Always inspect the generated JSON artifacts for exact current counts.

- **146 Registered Distribution Families (0 Placeholders)**:
  - **22 production-safe admitted families** (`BE`, `BEINF`, `BI`, `EXP`, `GA`, `GEOM`, `GU`, `IG`, `IGAMMA`, `LO`, `LOGNO`, `LOGNO2`, `NBI`, `NBII`, `NET`, `NO`, `NO2`, `PE`, `PO`, `RG`, `WEI`, `ZAGA`) backed by live-R golden fit + `newdata` prediction + JSON serialization round-trip + RS/CG/MIXED layered parity (`docs/development/m3-admission-evidence-2026-11-15.json`).
  - **104 experimental families** + **20 compatibility aliases** (guarded fail-closed on production routes until promoted through the admission gate).
  - Three-gate full-family parity scanner (`tools/rs_family_parity_scan.py`) covering derivatives, `d/p/q/r`, and RS deviance against live R with **0 unclassified failures** (`docs/development/p2-defect-classification-2026-11-15.json`).
- **100% In-Scope R Surface Coverage (11 R Packages)**:
  - All **1,036 / 1,036** exported symbols across `gamlss`, `gamlss.dist`, `gamlss.cens`, `gamlss.tr`, `gamlss.mx`, `gamlss.add`, `gamlss.spatial`, `gamlss.foreach`, `gamlss.prepdata`, `gamlss.ggplots`, and `gamlss.data` have Python surfaces (`docs/development/r-symbol-coverage.json`).
  - `gamlss2` is explicitly **out of scope** for OmniLSS 1.0 (separate specification and evidence track).
- **Fitting Algorithms**:
  - **RS** (Rigby–Stasinopoulos, default), **CG** (Cole–Green full-Hessian & experimental `irls_cross` backend), **MIXED** (`RS(n1) -> CG(n2)` matching R `mixed()`), **Joint** (Adam/Optax), **L-BFGS**, and experimental **RS_JAX**.
  - R-aligned `gamlss()` controls: `mu_start`/`sigma_start`/`nu_start`/`tau_start`, `mu_fix`/`sigma_fix`/`nu_fix`/`tau_fix` across all routes, `start_from`, `contrasts`, `n_cyc`, `trace`.
- **Full Family/Link & Smoother Suite**:
  - `gamlss.family` / `as.family` / `checklink` / `show.link` / `make.link.gamlss` / `gen.Family` (`logNO`, `logitNO` fittable with live-R parity).
  - Smoothers & random effects: `pb`, `ps`, `cs`, `cy`, `pbm`/`pbo`/`pbp`/`pbz`/`pbc`/`pvc`, `fp`/`bfp`, `pp`, `pcat`, `ri`, `random`, `lo`/`gamlss_lo` (weighted local polynomial LOESS), `re`/`gamlss_re` (exact marginal ML/REML mixed effects), `scs`, plus thin-plate (`tps`) and tensor-product splines with GCV/REML/UBRE selection.
- **Surrounding R Package Capabilities**:
  - Censoring (`gamlss.cens`), truncation (`gamlss.tr`), finite mixtures & NPML (`gamlss.mx`), additive learners (`gamlss.add`), spatial MRF/GMRF (`gamlss.spatial`), bootstrap/rolling/QR (`gamlss.foreach`), data diagnostics (`gamlss.prepdata`), and ggplot-style diagnostic data layers (`gamlss.ggplots`).
- **ML & Production Extensions**:
  - Neural GAMLSS (`omnilss.deep`), scikit-learn `GAMLSSRegressor`, probabilistic scoring (`CRPS`, `log_score`, `DSS`, `interval_score`, PIT), JSON/pickle model persistence, and REST (`omnilss-server/`) + gRPC (`omnilss.api.grpc`) serving boundaries with fail-closed capability admission.

## Installation & Development Environments

OmniLSS uses [`uv`](https://docs.astral.sh/uv/) for virtual environment management:

- **Windows CPU environment**: `.venv/` (managed by `uv`)
- **WSL Debian GPU environment**: `debianjaxgpu/` (managed by `uv`)

```bash
# Editable install from source using uv
uv pip install -e "./omnilss[dev]"

# Or with server/gRPC extras
uv pip install -e "./omnilss[server,dev]"
```

## Quick Start

```python
import numpy as np
from omnilss import gamlss, NO

rng = np.random.default_rng(42)
n = 500
x = rng.normal(size=n)
y = 2.0 + 3.0 * x + rng.normal(size=n)
data = {"y": y, "x": x}

# Fit model (RS default; CG / MIXED / joint / lbfgs also supported)
model = gamlss("y ~ x", family=NO(), data=data)
print(f"Deviance: {model.g_dev:.4f}")

# Predict distribution parameters and quantiles on newdata
newdata = {"x": np.array([0.0, 1.0, 2.0])}
params = model.predict_params(newdata)
q = model.predict_quantiles(newdata, quantiles=[0.05, 0.5, 0.95])
```

## Smoothing & Diagnostics

```python
from omnilss import gamlss, NO, get_rqres, rqres_plot, calibration, fittedPlot

# P-spline with automatic lambda selection (GCV/REML)
model = gamlss("y ~ pb(x)", family=NO(), data=data)

# Randomized quantile residuals, worm-plot bands, and centile calibration
rq = get_rqres(model, howmany=6, order=True, seed=0)
wp = rqres_plot(model, howmany=6, plot_type="all", type="wp", seed=0)
cal = calibration(model, xvar="x")
fp = fittedPlot(model, x="x")
```

## Testing & Verification Gates

OmniLSS enforces fail-closed tiered collection and milestone artifact checks:

```bash
# 1. Fail-closed 4-tier test collection (omnilss, omnilss-server, omnilss-pro, benchmarks)
python tools/tiered_collection.py --mode collect --strict-skips --json

# 2. Run subproject tiers directly
python tools/tiered_collection.py --tier omnilss-server --tier omnilss-pro --mode run

# 3. Milestone, contract, and symbol-coverage gates
python tools/generate_symbol_coverage.py --check
python tools/coverage_diff_report.py
python tools/milestone_manifest.py --check
python tools/highfreq_family_freeze.py --check
python tools/contract_freeze.py --check
python tools/m1_assessment.py --check
python tools/m2_assessment.py --check
python tools/m3_assessment.py --check
python tools/m4_scope_inventory.py --check

# 4. Full-suite 8-shard deterministic execution (matches release.yml)
python tools/pytest_shards.py --shards 8 --check
```

| Tier | Directory | Scope |
|------|-----------|-------|
| `omnilss` | `omnilss/tests` | Main suite (604 test files, 5,800+ cases) |
| `omnilss-server` | `omnilss-server` | FastAPI REST boundary, auth, DoS, capability admission |
| `omnilss-pro` | `omnilss-pro` | Commercial gRPC client & AutoML prototype |
| `benchmarks` | `benchmarks` | Benchmark/diagnostic harnesses |

## Reference Projects Policy (Not Dependencies)

The local directories `R/`, `ondil/`, and `gamlss-python/` are gitignored upstream reference checkouts kept strictly for **design reference and R parity verification**. OmniLSS **never imports, copies, or depends on** `gamlss-python` or `ondil` code; all OmniLSS implementations are self-contained on JAX / NumPy / SciPy and verified against live R (`gamlss` / `gamlss.dist`) or independent analytical/DGP goldens.

## Benchmarks & Performance Policy

Performance claims must be generated from reproducible benchmark runs after passing numerical parity gates, separating cold (JIT compile) and warm (steady-state) timings and never extrapolating CPU measurements to GPU:

```bash
python benchmarks/run_local_validation.py --quick
python benchmarks/comprehensive_performance_test.py --quick --no-r --n-repeats 10
```

See [`benchmarks/README.md`](benchmarks/README.md) for methodology.

## Service APIs

- **REST (`omnilss-server/`)**: FastAPI server sharing the persistent SQLite/artifact registry with gRPC (`/fit`, `/predict`, `/diagnostics/{model_id}`, `/models`, `/distributions/select`), enforcing Bearer token auth (`OMNILSS_API_TOKEN`), bounded sliding-window rate limiting, body size limits, and fail-closed production family admission.
- **gRPC (`omnilss.api.grpc`)**: Protobuf definitions under `omnilss/src/omnilss/api/grpc/proto/`. Regenerate stubs with `cd omnilss && python tools/generate_grpc_stubs.py`.

## Offline Dev + R Validation Container

An offline-capable devcontainer is provided under `.devcontainer/` (with `.codex/setup.sh` for post-create bootstrap):

```bash
devcontainer build --workspace-folder .
devcontainer up --workspace-folder .
bash .codex/setup.sh
```

See [`docs/development/offline-dev-bootstrap-2026-05-17.md`](docs/development/offline-dev-bootstrap-2026-05-17.md) and [`docs/development/r-environment-setup.md`](docs/development/r-environment-setup.md).

## Key Documents

- [Comprehensive Audit (2026-11-21, D2.270+)](docs/development/full-project-audit-2026-11-21.md)
- [Master Development Plan](DEVELOPMENT_PLAN.md)
- [Symbol Coverage Matrix](docs/development/r-symbol-coverage.md)
- [M0 Manifest](docs/development/milestone-manifest-m0.md) | [M1](docs/development/m1-assessment-2026-11-15.md) | [M2](docs/development/m2-assessment-2026-11-15.md) | [M3](docs/development/m3-assessment-2026-11-15.md) | [M4 Scope](docs/development/m4-scope-2026-11-15.md)
- [Tutorials](docs/tutorials/) & [Benchmarks](benchmarks/)
- [Changelog](CHANGELOG.md)

## License

GNU General Public License v3 or later (`GPL-3.0-or-later`).
