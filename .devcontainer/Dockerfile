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

# Bundle the local R/gamlss and R/gamlss.dist sources so an offline contributor
# can reinstall the R packages from disk if the renv/CRAN cache is missing.
COPY R/gamlss "${OMNILSS_R_SOURCE}/gamlss"
COPY R/gamlss.dist "${OMNILSS_R_SOURCE}/gamlss.dist"

# Cache CRAN tarballs for the R packages required by the validation gate. The
# download.packages step writes versioned source tarballs into
# /opt/omnilss-r-cache/, which `install.packages(..., repos=NULL, type='source')`
# can reuse without network access.
RUN mkdir -p "${RENV_PATHS_CACHE}" "${OMNILSS_R_CACHE}" && \
    R -e "options(repos=c(CRAN='https://cloud.r-project.org')); \
          download.packages(c('renv','gamlss','gamlss.dist','jsonlite','languageserver'), \
                            destdir='${OMNILSS_R_CACHE}', type='source')" && \
    R -e "options(repos=c(CRAN='https://cloud.r-project.org')); \
          install.packages(c('renv','gamlss','gamlss.dist','jsonlite','languageserver'), \
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

# Confirm the bundled R source trees and CRAN cache exist so an offline rebuild
# is possible without re-running the online image build.
RUN test -f "${OMNILSS_R_SOURCE}/gamlss/DESCRIPTION" \
    && test -f "${OMNILSS_R_SOURCE}/gamlss.dist/DESCRIPTION" \
    && ls "${OMNILSS_R_CACHE}"/gamlss_*.tar.gz >/dev/null 2>&1 \
    && ls "${OMNILSS_R_CACHE}"/gamlss.dist_*.tar.gz >/dev/null 2>&1 \
    && echo "R offline source/tarball inventory: ok"

WORKDIR /workspace/OmniLSS

CMD ["bash"]
