#!/usr/bin/env bash
# Offline-capable dev bootstrap for OmniLSS.
# Mirrors docs/development/offline-dev-bootstrap-2026-05-17.md.
# Idempotent: safe to re-run after a checkout change.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

offline="${OMNILSS_OFFLINE:-1}"
wheelhouse="${OMNILSS_WHEELHOUSE:-/opt/omnilss-wheelhouse}"
venv_dir="${OMNILSS_VENV:-/opt/omnilss-venv}"

echo "[setup] repo: $repo_root (offline=$offline)"

# 1. Pick a Python: prebuilt /opt venv in the devcontainer, else the local .venv.
if [[ -x "$venv_dir/bin/python" ]]; then
  python="$venv_dir/bin/python"
elif [[ -x "$repo_root/.venv/Scripts/python.exe" ]]; then
  python="$repo_root/.venv/Scripts/python.exe"
elif [[ -x "$repo_root/.venv/bin/python" ]]; then
  python="$repo_root/.venv/bin/python"
else
  python="$(command -v python3 || command -v python)"
  echo "[setup] no prebuilt venv; using $python"
fi
echo "[setup] python: $python ($("$python" --version 2>&1))"

# 2. Install package (editable). Offline uses the cached wheelhouse.
if [[ "$offline" == "1" && -d "$wheelhouse" ]]; then
  "$python" -m pip install --no-index --find-links "$wheelhouse" -e "$repo_root/omnilss"
else
  "$python" -m pip install -e "$repo_root/omnilss"
fi

# 3. Verify R + the packages the live-R parity gates need.
if command -v Rscript >/dev/null 2>&1; then
  Rscript -e 'suppressPackageStartupMessages({library(gamlss); library(gamlss.dist); library(jsonlite)}); cat("[setup] R OK\n")' \
    || echo "[setup] WARNING: R present but gamlss/gamlss.dist/jsonlite missing (R parity will skip)"
else
  echo "[setup] WARNING: Rscript not found; live-R parity tests will skip"
fi

# 4. gRPC tooling (best effort; stubs are generated at build time).
if "$python" -c "import grpc_tools" >/dev/null 2>&1; then
  echo "[setup] grpc_tools available"
fi

echo "[setup] done. Suggested next commands:"
echo "  python benchmarks/run_local_validation.py --quick"
echo "  python -m pytest omnilss/tests -q"
