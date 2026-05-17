FROM rocker/r-ver:4.4.1

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    build-essential \
    gfortran \
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
    git \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Python packages
COPY requirements.txt /tmp/requirements.txt

RUN pip3 install --upgrade pip && \
    pip3 install -r /tmp/requirements.txt

# R packages
RUN R -e "install.packages(c( \
    'renv', \
    'gamlss', \
    'gamlss.dist', \
    'languageserver' \
    ), repos='https://cloud.r-project.org')"

WORKDIR /workspace