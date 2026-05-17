FROM rocker/r-ver:4.4.1

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=0 \
    PYTHONUNBUFFERED=1 \
    VIRTUAL_ENV=/opt/omnilss-venv \
    OMNILSS_WHEELHOUSE=/opt/omnilss-wheelhouse \
    RENV_PATHS_CACHE=/opt/renv/cache
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"

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

RUN mkdir -p "${RENV_PATHS_CACHE}" && \
    R -e "options(repos=c(CRAN='https://cloud.r-project.org')); install.packages(c('renv','gamlss','gamlss.dist','jsonlite','languageserver'), Ncpus=max(1, parallel::detectCores()-1))"

WORKDIR /workspace/OmniLSS

CMD ["bash"]
