#!/usr/bin/env bash
set -Eeuo pipefail

log() {
  printf '\n===== %s =====\n' "$*"
}

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
VENV_DIR="${VIRTUAL_ENV:-/opt/omnilss-venv}"
WHEELHOUSE="${OMNILSS_WHEELHOUSE:-/opt/omnilss-wheelhouse}"
REQ_FILE="${REPO_ROOT}/.devcontainer/requirements-dev.txt"
PYTHON_BIN="${PYTHON_BIN:-python3}"
export JAX_PLATFORMS="${JAX_PLATFORMS:-cpu}"
export JAX_ENABLE_X64="${JAX_ENABLE_X64:-true}"
export RENV_PATHS_CACHE="${RENV_PATHS_CACHE:-/opt/renv/cache}"

log "Repository"
echo "${REPO_ROOT}"
cd "${REPO_ROOT}"

log "Python"
if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"
python --version
python -m pip install --upgrade pip setuptools wheel

log "Python dependency wheelhouse"
mkdir -p "${WHEELHOUSE}"
if python -m pip download --dest "${WHEELHOUSE}" -r "${REQ_FILE}"; then
  date -u +%Y-%m-%dT%H:%M:%SZ > "${WHEELHOUSE}/.complete"
  echo "Wheelhouse refreshed at ${WHEELHOUSE}"
else
  echo "WARNING: Could not refresh wheelhouse; will prefer online install before offline fallback." >&2
fi

if [[ -f "${WHEELHOUSE}/.complete" ]]; then
  python -m pip install --no-index --find-links "${WHEELHOUSE}" -r "${REQ_FILE}"
elif ! python -m pip install -r "${REQ_FILE}"; then
  echo "WARNING: Online install failed; attempting offline install from existing wheelhouse." >&2
  python -m pip install --no-index --find-links "${WHEELHOUSE}" -r "${REQ_FILE}"
fi

log "Editable packages"
python -m pip install --no-deps -e "${REPO_ROOT}/omnilss"
if [[ -f "${REPO_ROOT}/omnilss-pro/pyproject.toml" ]]; then
  python -m pip install --no-deps -e "${REPO_ROOT}/omnilss-pro"
fi

log "R"
if command -v R >/dev/null 2>&1; then
  R --version | head -1
  R -q -e "options(repos=c(CRAN='https://cloud.r-project.org')); missing <- setdiff(c('renv','gamlss','gamlss.dist','jsonlite','languageserver'), rownames(installed.packages())); if (length(missing)) install.packages(missing, Ncpus=max(1, parallel::detectCores()-1)); if (file.exists('renv.lock')) renv::restore(prompt=FALSE)"
else
  echo "WARNING: R is not installed; R-backed checks will be skipped." >&2
fi

log "gRPC stubs"
python "${REPO_ROOT}/omnilss/tools/generate_grpc_stubs.py"

log "Sanity checks"
python - <<'PY'
import importlib.util
import omnilss
print("omnilss", omnilss.__version__)
print(omnilss.check_installation())
for mod in ["grpc", "grpc_tools", "jax", "flax", "sklearn", "scoringrules"]:
    print(f"{mod}: {'ok' if importlib.util.find_spec(mod) else 'missing'}")
PY

log "Environment Ready"
