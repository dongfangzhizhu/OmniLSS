# OmniLSS Devcontainer

This directory defines the offline-capable environment used by Codex Cloud and
local devcontainer workflows.

- `Dockerfile` builds the CPU base image with Python, R, cached PyPI wheels,
  cached CRAN source packages, docs tooling, gRPC tooling, and test dependencies.
  The `R/` vendored reference trees are gitignored and absent in a fresh clone,
  so they are NOT copied at image build time; `.codex/setup.sh` bundles them into
  `/opt/omnilss-r-source` at post-create time only when present (best-effort).
  The CRAN tarball cache is the authoritative offline R install source.
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
