FROM rocker/r-ver:4.4.1

# Offline-capable OmniLSS devcontainer:
# - Python 3 (with venv) + a wheelhouse pre-populated from
#   .devcontainer/requirements-dev.txt.
# - R 4.4.1 with gamlss, gamlss.dist, jsonlite, renv and language server,
#   plus a local CRAN tarball cache and a copy of the bundled R sources so
#   `gamlss` / `gamlss.dist` can be rebuilt without network access.
# - Build/test/release/docs/grpc tooling installed at image build time so a
#   reused offline container is never blocked by missing packages.

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=0 \
    PYTHONUNBUFFERED=1 \
    VIRTUAL_ENV=/opt/omnilss-venv \
    OMNILSS_WHEELHOUSE=/opt/omnilss-wheelhouse \
    OMNILSS_R_CACHE=/opt/omnilss-r-cache \
    OMNILSS_R_SOURCE=/opt/omnilss-r-source \
    RENV_PATHS_CACHE=/opt/renv/cache
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"

# System dependencies for Python build, R compilation, gRPC tooling, and the
# native libraries required by gamlss / nlme / survival.
RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    ca-certificates \
    curl \
    git \
    wget \
    build-essential \
    gfortran \
    pkg-config \
    protobuf-compiler \
    python3 \
    python3-dev \
    python3-pip \
    python3-venv \
    libcurl4-openssl-dev \
    libssl-dev \
    libxml2-dev \
    libfontconfig1-dev \
    libharfbuzz-dev \
    libfribidi-dev \
    libfreetype6-dev \
    libpng-dev \
    libtiff5-dev \
    libjpeg-dev \
    libopenblas-dev \
    liblapack-dev \
    libicu-dev \
    libgit2-dev \
    cmake \
    jq \
    less \
    pandoc \
    r-base-dev \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv "${VIRTUAL_ENV}" && \
    python -m pip install --upgrade pip setuptools wheel

COPY .devcontainer/requirements-dev.txt /tmp/requirements-dev.txt

# Build a wheelhouse during image creation so post-create setup can reinstall
# dependencies without network access when the container is reused offline.
RUN mkdir -p "${OMNILSS_WHEELHOUSE}" && \
    python -m pip download --dest "${OMNILSS_WHEELHOUSE}" -r /tmp/requirements-dev.txt && \
    date -u +%Y-%m-%dT%H:%M:%SZ > "${OMNILSS_WHEELHOUSE}/.complete" && \
    python -m pip install --no-index --find-links "${OMNILSS_WHEELHOUSE}" -r /tmp/requirements-dev.txt

# NOTE: R/gamlss and R/gamlss.dist are vendored reference sources that are
# gitignored (`.gitignore: R/`) and therefore ABSENT in a fresh clone.  They
# must NOT be COPY'd here: a hard COPY of a missing path fails `docker build`
# from a clean checkout (P0-1).  The image gets gamlss/gamlss.dist from the
# online `install.packages` below plus the CRAN source tarball cache; the
# optional vendored source bundle is done at *post-create* time by
# `.codex/setup.sh`, which sees the workspace mount (where R/ may exist) and
# copies it into OMNILSS_R_SOURCE only when present -- never failing otherwise.
RUN mkdir -p "${OMNILSS_R_SOURCE}"

# Cache CRAN tarballs for the R packages required by the validation gate,
# including recursive dependencies. The cache is indexed as a local source
# repository so `.codex/setup.sh` can reinstall from file:// URLs offline.
RUN mkdir -p "${RENV_PATHS_CACHE}" "${OMNILSS_R_CACHE}" && \
    R -e "options(repos=c(CRAN='https://cloud.r-project.org')); \
          required <- c('renv','gamlss','gamlss.dist','jsonlite','languageserver'); \
          available <- available.packages(); \
          dep_types <- c('Depends','Imports','LinkingTo'); \
          deps <- tools::package_dependencies(required, db=available, which=dep_types, recursive=TRUE); \
          pkgs <- unique(c(required, unlist(deps, use.names=FALSE))); \
          download.packages(pkgs, destdir='${OMNILSS_R_CACHE}', type='source')" && \
    R -e "tools::write_PACKAGES('${OMNILSS_R_CACHE}', type='source')" && \
    R -e "options(repos=c(CRAN='https://cloud.r-project.org')); \
          install.packages(c('renv','gamlss','gamlss.dist','jsonlite','languageserver'), \
                           dependencies=c('Depends','Imports','LinkingTo'), \
                           Ncpus=max(1, parallel::detectCores()-1))"

# Fail the image build if a required offline-development dependency was not
# installed into the venv or R library during the online build phase.
RUN python - <<'PY'
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
    raise SystemExit(f"missing Python modules in offline image: {missing}")
print("Python offline dependency inventory: ok")
PY

RUN Rscript -e "missing <- setdiff(c('renv','gamlss','gamlss.dist','jsonlite','languageserver'), rownames(installed.packages())); if (length(missing)) stop(paste('missing R packages in offline image:', paste(missing, collapse=', '))); cat('R offline dependency inventory: ok\n')"

# Confirm the CRAN source tarball cache exists so an offline rebuild is possible
# without re-running the online image build.  The CRAN tarballs -- NOT the
# gitignored R/ vendored trees -- are the authoritative offline install source;
# the vendored R source bundle (OMNILSS_R_SOURCE) is best-effort and filled by
# .codex/setup.sh at post-create time when R/ happens to be present.
RUN test -d "${OMNILSS_R_SOURCE}" \
    && ls "${OMNILSS_R_CACHE}"/gamlss_*.tar.gz >/dev/null 2>&1 \
    && ls "${OMNILSS_R_CACHE}"/gamlss.dist_*.tar.gz >/dev/null 2>&1 \
    && echo "R offline tarball inventory: ok"

WORKDIR /workspace/OmniLSS

CMD ["bash"]
