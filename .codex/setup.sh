#!/usr/bin/env bash
set -Eeuo pipefail

log() {
  printf '\n===== %s =====\n' "$*"
}

warn() {
  echo "WARNING: $*" >&2
}

is_truthy() {
  case "${1:-}" in
    1|true|TRUE|yes|YES|on|ON) return 0 ;;
    *) return 1 ;;
  esac
}

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
VENV_DIR="${VIRTUAL_ENV:-/opt/omnilss-venv}"
WHEELHOUSE="${OMNILSS_WHEELHOUSE:-/opt/omnilss-wheelhouse}"
REQ_FILE="${REPO_ROOT}/.devcontainer/requirements-dev.txt"
PYTHON_BIN="${PYTHON_BIN:-python3}"
OFFLINE="${OMNILSS_OFFLINE:-0}"
REFRESH_R_REFERENCE="${OMNILSS_REFRESH_R_REFERENCE:-0}"
export JAX_PLATFORMS="${JAX_PLATFORMS:-cpu}"
export JAX_ENABLE_X64="${JAX_ENABLE_X64:-true}"
export RENV_PATHS_CACHE="${RENV_PATHS_CACHE:-/opt/renv/cache}"

required_grpc_stubs_exist() {
  local generated_dir="${REPO_ROOT}/omnilss/src/omnilss/api/grpc/generated"
  local stem
  for stem in fit predict sample; do
    [[ -f "${generated_dir}/${stem}_pb2.py" ]] || return 1
    [[ -f "${generated_dir}/${stem}_pb2_grpc.py" ]] || return 1
  done
}

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
if is_truthy "${OFFLINE}"; then
  warn "OMNILSS_OFFLINE=${OFFLINE}; skipping pip bootstrap upgrade."
elif ! python -m pip install --upgrade pip setuptools wheel; then
  warn "Could not upgrade pip/setuptools/wheel; continuing with existing bootstrap tools."
fi

log "Python dependency wheelhouse"
mkdir -p "${WHEELHOUSE}"
if is_truthy "${OFFLINE}"; then
  warn "OMNILSS_OFFLINE=${OFFLINE}; skipping wheelhouse refresh."
elif python -m pip download --dest "${WHEELHOUSE}" -r "${REQ_FILE}"; then
  date -u +%Y-%m-%dT%H:%M:%SZ > "${WHEELHOUSE}/.complete"
  echo "Wheelhouse refreshed at ${WHEELHOUSE}"
else
  warn "Could not refresh wheelhouse; will prefer cached wheels before online fallback."
fi

if [[ -f "${WHEELHOUSE}/.complete" ]]; then
  python -m pip install --no-index --find-links "${WHEELHOUSE}" -r "${REQ_FILE}"
elif is_truthy "${OFFLINE}"; then
  echo "ERROR: OMNILSS_OFFLINE=${OFFLINE}, but ${WHEELHOUSE}/.complete is missing." >&2
  echo "Build the devcontainer once online or unset OMNILSS_OFFLINE to allow network install." >&2
  exit 1
elif ! python -m pip install -r "${REQ_FILE}"; then
  warn "Online install failed; attempting offline install from existing wheelhouse."
  python -m pip install --no-index --find-links "${WHEELHOUSE}" -r "${REQ_FILE}"
fi

log "Editable packages"
python -m pip install --no-deps -e "${REPO_ROOT}/omnilss"
if [[ -f "${REPO_ROOT}/omnilss-pro/pyproject.toml" ]]; then
  python -m pip install --no-deps -e "${REPO_ROOT}/omnilss-pro"
fi

log "R"
R_CACHE="${OMNILSS_R_CACHE:-/opt/omnilss-r-cache}"
R_SOURCE="${OMNILSS_R_SOURCE:-/opt/omnilss-r-source}"
if command -v R >/dev/null 2>&1; then
  R --version | head -1
  if is_truthy "${OFFLINE}"; then
    warn "OMNILSS_OFFLINE=${OFFLINE}; skipping CRAN-backed R package refresh."
    if [[ -d "${R_CACHE}" ]]; then
      Rscript -e "missing <- setdiff(c('renv','gamlss','gamlss.dist','jsonlite','languageserver'), rownames(installed.packages())); cache <- Sys.getenv('OMNILSS_R_CACHE','${R_CACHE}'); for (pkg in missing) { tarballs <- list.files(cache, pattern=paste0('^', pkg, '_.*\\.tar\\.gz$'), full.names=TRUE); if (length(tarballs)) install.packages(tarballs[1], repos=NULL, type='source') }"
    fi
    if [[ -d "${R_SOURCE}/gamlss" || -d "${R_SOURCE}/gamlss.dist" ]]; then
      Rscript -e "missing <- setdiff(c('gamlss','gamlss.dist'), rownames(installed.packages())); src <- Sys.getenv('OMNILSS_R_SOURCE','${R_SOURCE}'); for (pkg in missing) { dir <- file.path(src, pkg); if (dir.exists(dir)) install.packages(dir, repos=NULL, type='source') }"
    fi
  else
    R -q -e "options(repos=c(CRAN='https://cloud.r-project.org')); missing <- setdiff(c('renv','gamlss','gamlss.dist','jsonlite','languageserver'), rownames(installed.packages())); if (length(missing)) install.packages(missing, Ncpus=max(1, parallel::detectCores()-1)); if (file.exists('renv.lock')) renv::restore(prompt=FALSE)"
  fi
  Rscript -e "missing <- setdiff(c('renv','gamlss','gamlss.dist','jsonlite','languageserver'), rownames(installed.packages())); if (length(missing)) { warning(paste('R packages still missing:', paste(missing, collapse=', '))); quit(status=0) }; cat('gamlss', as.character(packageVersion('gamlss')), 'gamlss.dist', as.character(packageVersion('gamlss.dist')), '\n')"
  if is_truthy "${REFRESH_R_REFERENCE}"; then
    Rscript "${REPO_ROOT}/benchmarks/generate_r_reference_results.R" "${REPO_ROOT}/benchmarks/r_reference_results.json"
  else
    echo "R reference refresh disabled; set OMNILSS_REFRESH_R_REFERENCE=1 to regenerate benchmarks/r_reference_results.json."
  fi
else
  warn "R is not installed; R-backed checks will be skipped."
fi

log "gRPC stubs"
if ! python "${REPO_ROOT}/omnilss/tools/generate_grpc_stubs.py"; then
  if required_grpc_stubs_exist; then
    warn "Could not regenerate gRPC stubs; continuing with committed generated files."
  else
    echo "ERROR: gRPC stub generation failed and committed generated files are incomplete." >&2
    exit 1
  fi
fi

log "Sanity checks"
python - <<'PY'
import importlib.util
import omnilss
print("omnilss", omnilss.__version__)
print(omnilss.check_installation())
def has_module(name):
    try:
        return importlib.util.find_spec(name) is not None
    except ModuleNotFoundError:
        return False

for mod in ["grpc", "google.protobuf", "grpc_tools", "build", "twine", "jax", "flax", "optax", "sklearn", "scoringrules", "mkdocs"]:
    print(f"{mod}: {'ok' if has_module(mod) else 'missing'}")
PY

log "R bridge smoke check"
if command -v Rscript >/dev/null 2>&1; then
  Rscript -e "suppressMessages(library(gamlss)); suppressMessages(library(gamlss.dist)); suppressMessages(library(jsonlite)); cat('R bridge ready: gamlss', as.character(packageVersion('gamlss')), 'gamlss.dist', as.character(packageVersion('gamlss.dist')), '\n')" \
    || warn "Rscript ran but at least one bridge package failed to load; R-backed tests will skip."
  echo "Validation gate (run from the repo root when ready):"
  echo "  python benchmarks/run_local_validation.py --quick"
else
  warn "Rscript not found; R-backed validation will skip. Use --allow-python-only for smoke runs only."
fi

log "Environment Ready"
