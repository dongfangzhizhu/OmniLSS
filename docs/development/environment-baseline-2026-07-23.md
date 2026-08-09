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
