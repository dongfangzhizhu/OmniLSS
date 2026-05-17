# Offline-capable development bootstrap (2026-05-17)

[中文版本](offline-dev-bootstrap-2026-05-17_cn.md)

This note documents the `.codex/` and `.devcontainer/` environment bootstrap used for subsequent OmniLSS development passes. It is the checklist for building an offline-capable development image that already contains the Python, R, protobuf, package-build, documentation, benchmark, and service dependencies needed by the repository.

## Non-negotiable requirement

The next offline development image must be built **online once** and must install every required tool during the image build, not during later development sessions. A reused offline container should not block work because `optax`, `build`, `twine`, R, `grpc_tools`, protobuf compiler, MkDocs, or R `gamlss` packages are missing.

If a future task discovers a missing dependency, the fix is to update this bootstrap manifest and `.devcontainer/requirements-dev.txt` / `.devcontainer/Dockerfile` before rebuilding the image. Do not rely on ad-hoc `pip install`, `apt-get`, or `install.packages()` commands inside an already-offline session.

## Goals

- Build a container with Python, R, protobuf tooling, and system libraries required by OmniLSS Core, gRPC, benchmarks, and documentation.
- Pre-download Python wheels into `/opt/omnilss-wheelhouse` during image build so `postCreateCommand` can reinstall dependencies without network access.
- Pre-install R packages required by R-backed validation (`renv`, `gamlss`, `gamlss.dist`, `jsonlite`) and editor support (`languageserver`).
- Install package-build tooling (`build`, `twine`, `wheel`, `setuptools`) at image build time so `omnilss/tools/release_check.py` is not blocked in an offline session.
- Install ML/dev extras (`flax`, `optax`, `scikit-learn`, `scoringrules`) at image build time so architecture-contract and optional-wrapper tests do not fail at collection.
- Install gRPC/protobuf tooling (`protobuf-compiler`, `grpcio`, `protobuf`, `grpcio-tools`) at image build time so committed stubs can be regenerated without downloading packages.
- Install documentation tooling (`mkdocs-material`, `mkdocstrings[python]`, `mkdocs-jupyter`, `sphinx`, `myst-parser`) at image build time so documentation changes can be previewed or validated offline.
- Keep editable installs for `omnilss/` and `omnilss-pro/` in `postCreateCommand`, because source code is mounted after the image is built.

## Files and ownership

- `.devcontainer/requirements-dev.txt` is the canonical Python bootstrap list for the devcontainer/Codex image. It must include all runtime, optional, dev/test, docs, release, benchmark, server, and prototype dependencies that are expected to work offline.
- `.devcontainer/Dockerfile` creates `/opt/omnilss-venv`, fills `/opt/omnilss-wheelhouse`, installs Python dependencies from the wheelhouse, installs system packages, installs R packages, sets CPU JAX defaults, and fails the image build if required Python/R dependencies are missing.
- `.devcontainer/devcontainer.json` points VS Code/Codex to the prebuilt venv and runs `.codex/setup.sh` after source checkout.
- `.devcontainer/devcontainer.gpu.json` is an optional NVIDIA/CUDA variant for future GPU validation.
- `.codex/setup.sh` refreshes the wheelhouse when online, falls back to the cached wheelhouse when offline, installs editable packages, refreshes gRPC stubs, optionally refreshes R reference outputs, and prints an installation health summary that includes release, ML, docs, and gRPC modules.
- `.github/workflows/environment-validation.yml` is the networked follow-up path for validating package build, gRPC generation, R references, and optional self-hosted GPU checks.

## Build-time dependency manifest

The image build must install the following inventory before the workspace is used offline.

### System packages

Required through `.devcontainer/Dockerfile`:

- Shell and source tools: `bash`, `git`, `curl`, `wget`, `ca-certificates`, `less`, `jq`.
- Python build/runtime: `python3`, `python3-dev`, `python3-pip`, `python3-venv`, `build-essential`, `gfortran`, `pkg-config`, `cmake`.
- gRPC/protobuf: `protobuf-compiler`.
- Numerical libraries: `libopenblas-dev`, `liblapack-dev`.
- R/package compilation libraries: `libcurl4-openssl-dev`, `libssl-dev`, `libxml2-dev`, `libicu-dev`, `libgit2-dev`.
- Plotting/docs native libraries: `libfontconfig1-dev`, `libharfbuzz-dev`, `libfribidi-dev`, `libfreetype6-dev`, `libpng-dev`, `libtiff5-dev`, `libjpeg-dev`, `pandoc`.

### Python packages

Required through `.devcontainer/requirements-dev.txt` and installed into `/opt/omnilss-venv` from `/opt/omnilss-wheelhouse`:

- Release/build: `build`, `twine`, `wheel`, `setuptools`.
- Runtime: `jax`, `jaxlib`, `numpy`, `scipy`, `pandas`.
- Optional OmniLSS extras: `cloudpickle`, `flax`, `optax`, `scikit-learn`, `scoringrules`, `grpcio`, `protobuf`, `grpcio-tools`, `matplotlib`.
- Dev/test: `pytest`, `pytest-cov`, `pytest-xdist`, `black`, `ruff`, `mypy`, `pre-commit`.
- Docs: `mkdocs-material`, `mkdocstrings[python]`, `mkdocs-jupyter`, `sphinx`, `sphinx-rtd-theme`, `myst-parser`.
- Server/prototype: `fastapi`, `uvicorn`, `httpx`.

### R runtime and R packages

Required through the `rocker/r-ver:4.4.1` base image plus the Dockerfile R install step:

- R commands: `R`, `Rscript`.
- Validation packages: `gamlss`, `gamlss.dist`, `jsonlite`.
- Environment/editor package: `renv`, `languageserver`.

## Expected online-first workflow

```bash
# Build once with network access. This is the only phase allowed to fetch apt,
# PyPI, and CRAN artifacts for the offline image.
devcontainer build --workspace-folder .

# Reopen/reuse later without network; postCreate can install from wheelhouse and
# use already-installed R packages.
OMNILSS_OFFLINE=1 bash .codex/setup.sh
```

## Mandatory post-build validation before declaring the image reusable offline

The Dockerfile now runs the Python and R inventory checks during image creation, so the build fails early if these dependencies are absent. Run these commands again immediately after the online build, while network access still exists, to verify that later offline development will not be blocked by missing package-build, ML, R, docs, or gRPC tooling.

```bash
# Python release/build tools and common optional dependencies.
python - <<'PY'
import importlib.util
required = [
    "build",
    "twine",
    "jax",
    "flax",
    "optax",
    "sklearn",
    "scoringrules",
    "grpc",
    "grpc_tools",
    "google.protobuf",
    "mkdocs",
]
missing = [name for name in required if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit(f"missing Python modules: {missing}")
print("Python offline dependency inventory: ok")
PY
```

```bash
# R runtime and R-backed validation packages.
Rscript -e "missing <- setdiff(c('renv','gamlss','gamlss.dist','jsonlite','languageserver'), rownames(installed.packages())); if (length(missing)) stop(paste('missing R packages:', paste(missing, collapse=', '))); cat('R offline dependency inventory: ok\n')"
```

```bash
# gRPC generation should work from installed build-time dependencies.
cd /workspace/OmniLSS/omnilss
python tools/generate_grpc_stubs.py
```

```bash
# Release packaging should not be blocked by missing build/twine.
cd /workspace/OmniLSS/omnilss
python tools/release_check.py
```

```bash
# Architecture smoke test should not fail during collection because optax/flax is missing.
cd /workspace/OmniLSS
PYTHONPATH=omnilss/src python -m pytest omnilss/tests/test_core_architecture_contracts.py -q
```

```bash
# R-backed validation gate should run in the fully provisioned image.
cd /workspace/OmniLSS
python benchmarks/run_local_validation.py --quick
```

## Offline reuse validation

After disconnecting the container from the network, run:

```bash
cd /workspace/OmniLSS
OMNILSS_OFFLINE=1 bash .codex/setup.sh
```

The command is allowed to skip network refreshes, but it must not fail because a wheelhouse marker, Python module, R package, protobuf generator, or committed gRPC stub is missing.

## Environment-limited behavior

If network access is blocked during setup, dependency refresh may warn and continue only when `/opt/omnilss-wheelhouse/.complete` exists from a previous online build. R-backed checks still require the R package cache to have been populated during the online build.

If `/opt/omnilss-wheelhouse/.complete` is absent in `OMNILSS_OFFLINE=1` mode, the environment is invalid and must be rebuilt online. Do not proceed by manually installing a partial set of dependencies.

## Maintenance rules

- Whenever `omnilss/pyproject.toml` optional dependencies change, update `.devcontainer/requirements-dev.txt` in the same change.
- Whenever a test imports a new top-level Python package at collection time, add it to `.devcontainer/requirements-dev.txt` unless the test is explicitly and safely skipped when the package is missing.
- Whenever a benchmark or validation script requires a new R package, add it to the Dockerfile R install command and to the R inventory check above.
- Whenever a new system tool is needed by setup, release, docs, tests, or benchmarks, add it to `.devcontainer/Dockerfile` rather than documenting a manual install.
- If an offline session fails because a dependency is missing, treat it as a bootstrap bug and update this document plus the bootstrap files before the next image build.
