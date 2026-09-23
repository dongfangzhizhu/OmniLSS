# OmniLSS Release 1 Environment Baseline (2026-07-23)

## Critical Requirements
- **JAX x64 MUST be enabled** for R parity tests and release validation
- R must be installed from vendored source in `R/gamlss` / `R/gamlss.dist`
- No external R library overrides allowed in CI

## Python Environment

| Package | Version | Notes |
|---------|---------|-------|
| Python | 3.10.16 | .venv |
| JAX | 0.6.2 | x64 required |
| NumPy | 2.2.6 | |
| SciPy | 1.15.3 | |

## R Environment

| Package | Version | Source |
|---------|---------|--------|
| R | 4.6.0 | System |
| gamlss | 5.5-3 | Vendored from `R/gamlss/` |
| gamlss.dist | 6.1-8 | Vendored from `R/gamlss.dist/` |
| jsonlite | 2.0.0 | CRAN |

## How to Enable x64

```powershell
$env:JAX_ENABLE_X64 = 'True'
$env:PYTHONPATH = "$PWD\omnilss\src"
```

Or add to `.venv/Scripts/Activate.ps1`:
```powershell
$env:JAX_ENABLE_X64 = 'True'
```

## CI Baseline File

This file should be referenced by CI jobs. Any deviation from these versions
must be explicitly justified and recorded.

## WSL Debian Environment (GPU / secondary dev)

The primary development environment is Windows `.venv` (above). The WSL Debian
environment (`debianjaxgpu/`) is the secondary GPU/performance environment.

| Package | Version | Notes |
|---------|---------|-------|
| Python | 3.11.15 | debianjaxgpu/bin/python |
| JAX | 0.10.0 | CUDA 12 (RTX 3060); use `JAX_PLATFORMS=cpu` for fast CPU-only runs |
| R | 4.5.0 | System R in WSL |
| gamlss | 5.5.0 | Vendored from `R/gamlss/` |
| gamlss.dist | 6.1.1 | Vendored from `R/gamlss.dist/` |

### Running from source in WSL

`omnilss` is not pip-installed in `debianjaxgpu`.  Run from source by adding the
source tree to `PYTHONPATH`:

```bash
export PYTHONPATH="$PWD/omnilss/src"
./debianjaxgpu/bin/python -c "import omnilss; print(omnilss.__version__)"
```

### Version source contract (A0.1)

`omnilss/_version.py` provides the lightweight runtime version used by both
`omnilss.__version__` and model artifact producers. It is kept in sync with
`pyproject.toml` by a contract test. Installed `dist-info` is diagnostic
metadata only and never overrides the source runtime version, avoiding stale
editable-install metadata and lexicographic version comparisons.
