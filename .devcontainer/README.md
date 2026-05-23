# OmniLSS Devcontainer

This directory defines the offline-capable environment used by Codex Cloud and
local devcontainer workflows.

- `Dockerfile` builds the CPU base image with Python, R, cached PyPI wheels,
  cached CRAN source packages, bundled `R/gamlss` sources, docs tooling, gRPC
  tooling, and test dependencies.
- `devcontainer.json` is the default CPU profile. Use this for Codex Cloud.
- `devcontainer.gpu.json` is an optional self-hosted GPU profile. It may need
  network access to install CUDA-enabled JAX wheels.
- `requirements-dev.txt` is the Python dependency manifest used to populate the
  offline wheelhouse.

Build once with network access:

```bash
devcontainer build --workspace-folder .
devcontainer up --workspace-folder .
```

Then work offline inside the container:

```bash
bash .codex/setup.sh
python benchmarks/run_local_validation.py --quick
```

When adding dependencies, update this directory together with
`docs/development/offline-dev-bootstrap-2026-05-17.md`.
