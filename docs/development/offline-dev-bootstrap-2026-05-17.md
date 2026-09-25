# Offline Dev Bootstrap for OmniLSS (2026-05-17)

[中文版本](offline-dev-bootstrap-2026-05-17_cn.md)

This runbook describes the Codex Cloud / devcontainer environment used to keep
OmniLSS development testable when the later development session has no network
access. It complements `docs/development/r-environment-setup.md` and the active
R migration plan in `docs/development/r-gamlss-coverage-and-plan-2026-05-22.md`.

## Goal

Build the image once while network access is available, then reuse it offline
for Python development, R-backed parity checks, gRPC stub generation, docs
builds, and release-package smoke checks.

The offline image must contain:

- Python virtualenv: `/opt/omnilss-venv`.
- Python wheelhouse: `/opt/omnilss-wheelhouse`.
- R package source cache: `/opt/omnilss-r-cache`.
- Bundled R sources copied from this repo: `/opt/omnilss-r-source`.
- `R`, `Rscript`, `gamlss`, `gamlss.dist`, `jsonlite`, `renv`, and
  `languageserver`.

## Build Once Online

From the repository root:

```bash
devcontainer build --workspace-folder .
devcontainer up --workspace-folder .
```

The Dockerfile performs all network-dependent work at build time:

- installs apt packages for Python, R, Fortran/C/C++ builds, gRPC, docs, and
  R package compilation;
- downloads every Python dependency in `.devcontainer/requirements-dev.txt`
  into `/opt/omnilss-wheelhouse`;
- installs the same Python dependency set into `/opt/omnilss-venv`;
- downloads recursive CRAN source dependencies (`Depends`, `Imports`, and
  `LinkingTo`) for the R validation stack into `/opt/omnilss-r-cache` and
  writes a local `PACKAGES` index;
- installs the required R packages;
- copies `R/gamlss/` and `R/gamlss.dist/` into `/opt/omnilss-r-source`.

The image build fails if required Python or R modules are missing.

## Reuse Offline

Inside the container:

```bash
bash .codex/setup.sh
python benchmarks/run_local_validation.py --quick
```

The default devcontainer sets `OMNILSS_OFFLINE=1`, so `.codex/setup.sh` reuses
the wheelhouse and local R source repository instead of attempting PyPI or CRAN
refreshes. To refresh caches during a connected maintenance session:

```bash
OMNILSS_OFFLINE=0 bash .codex/setup.sh
```

## Codex Cloud Notes

Codex Cloud should use the checked-in `.devcontainer/devcontainer.json` for the
CPU environment. The `.codex/setup.sh` script is idempotent and is safe to rerun
after checkout changes; it refreshes editable installs, verifies R availability,
regenerates gRPC stubs when possible, and prints the validation-gate command.

If Codex Cloud starts from a non-devcontainer base image, `.codex/setup.sh` can
still bootstrap the Python side online. Full R parity requires either the
devcontainer image or a host image that already contains R plus the required R
packages.

## Validation Commands

Use these from the repository root:

```bash
# Quick release-style gate: R consistency first, performance second.
python benchmarks/run_local_validation.py --quick

# Python-only smoke check for constrained environments. This is not proof of
# equivalence with native R gamlss.
python benchmarks/run_local_validation.py --quick --allow-python-only --no-fit --no-smooth

# Architecture-contract smoke tests.
PYTHONPATH=omnilss/src python -m pytest omnilss/tests/test_core_architecture_contracts.py -q

# gRPC/server boundary smoke.
cd omnilss && python -m pytest tests/test_grpc_server.py -q -rs
```

## Maintenance Checklist

When adding a Python, benchmark, docs, service, or R validation dependency,
update all of the following in the same change:

- `.devcontainer/requirements-dev.txt` for Python dependencies.
- `.devcontainer/Dockerfile` for system packages and required R packages.
- `.codex/setup.sh` if the dependency needs a new offline cache or smoke check.
- `docs/development/r-environment-setup.md` if R package requirements change.
- this runbook.

Do not rely on `apt-get`, `pip install`, or `install.packages()` during an
offline development session. The connected image build is the dependency
capture point; offline sessions should only consume the cached artifacts.
