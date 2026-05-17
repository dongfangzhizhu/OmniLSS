# Offline-capable development bootstrap (2026-05-17)

This note documents the `.codex/` and `.devcontainer/` environment bootstrap used for subsequent OmniLSS development passes.

## Goals

- Build a container with Python, R, protobuf tooling, and system libraries required by OmniLSS Core, gRPC, benchmarks, and documentation.
- Pre-download Python wheels into `/opt/omnilss-wheelhouse` during image build so `postCreateCommand` can reinstall dependencies without network access.
- Pre-install R packages required by R-backed validation (`gamlss`, `gamlss.dist`, `jsonlite`) and editor support (`languageserver`).
- Keep editable installs for `omnilss/` and `omnilss-pro/` in `postCreateCommand`, because source code is mounted after the image is built.

## Files

- `.devcontainer/requirements-dev.txt` is the canonical Python bootstrap list for the devcontainer/Codex image.
- `.devcontainer/Dockerfile` creates `/opt/omnilss-venv`, fills `/opt/omnilss-wheelhouse`, installs R packages, and sets CPU JAX defaults.
- `.devcontainer/devcontainer.json` points VS Code/Codex to the prebuilt venv and runs `.codex/setup.sh` after source checkout.
- `.codex/setup.sh` refreshes the wheelhouse when online, falls back to the cached wheelhouse when offline, installs editable packages, refreshes gRPC stubs, and prints an installation health summary.

## Expected online-first workflow

```bash
# Build once with network access.
devcontainer build --workspace-folder .

# Reopen/reuse later without network; postCreate can install from wheelhouse.
bash .codex/setup.sh
```

## Environment-limited checks

If network access is blocked, dependency refresh may warn and continue only when `/opt/omnilss-wheelhouse/.complete` exists from a previous online build. R-backed checks still require the R package cache to have been populated during the online build.

## Follow-up hardening (same day)

- `.codex/setup.sh` now supports an explicit `OMNILSS_OFFLINE=1` mode that skips pip/R network refreshes and requires the prebuilt wheelhouse marker before installing dependencies.
- pip bootstrap upgrades are best-effort, so a reused offline container is not blocked by a transient package-index failure.
- gRPC stub generation now falls back to the committed generated files when regeneration fails but the checked-in `fit/predict/sample` stubs are complete; setup still fails if neither regeneration nor committed stubs are available.

- `.codex/setup.sh` now reports `google.protobuf` and `build` availability in the health summary and supports `OMNILSS_REFRESH_R_REFERENCE=1` to regenerate `benchmarks/r_reference_results.json` during a fully provisioned R build.
- `.devcontainer/devcontainer.gpu.json` is an optional NVIDIA/CUDA variant for future GPU validation; it installs `jax[cuda12]`, sets `JAX_PLATFORMS=cuda`, and runs the GPU conditional test configuration when host GPU runtime is available.
- `.github/workflows/environment-validation.yml` provides a networked follow-up path for gRPC/package build, R reference, and optional self-hosted GPU validation after constrained local runs.
