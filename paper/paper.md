[中文版本](paper_cn.md)
---
title: 'OmniLSS: High-Performance GAMLSS in Python using JAX'
tags:
  - Python
  - JAX
  - statistical modeling
  - GAMLSS
  - distributional regression
authors:
  - name: OmniLSS contributors
    orcid: TBD
    affiliation: 1
affiliations:
  - name: Independent
    index: 1
date: 2026-05-18
bibliography: paper.bib
---

# Summary

OmniLSS is a Python implementation of Generalized Additive Models for Location,
Scale and Shape (GAMLSS) [@rigby2005] built around NumPy and JAX [@jax2018]. It
provides a broad catalogue of distribution families, parameter-specific formulas,
a stable Rigby-Stasinopoulos (RS) fitting path, a JAX-native cold-start RS path
for accelerator-oriented workloads, and optional service interfaces for HTTP and
gRPC deployment.

# Statement of Need

The R `gamlss` ecosystem remains the reference implementation for distributional
regression, but many scientific, machine-learning, and production-serving
workflows are Python-first. Practitioners in those environments often need three
capabilities at the same time: familiar GAMLSS distribution families,
Python-native model artifacts and prediction schemas, and numerical kernels that
can participate in modern autodiff and accelerator workflows. OmniLSS addresses
that gap by offering a Python package that preserves core GAMLSS concepts while
using explicit validation gates and service boundaries suitable for deployment.

# Algorithmic Approach

OmniLSS exposes the RS algorithm as the conservative default fitting route. The
NumPy implementation prioritizes reference stability and R-consistency checks.
The JAX RS implementation keeps the stable `max_inner=1` cadence, uses
data-aware cold-start initialization instead of a NumPy RS warm-start, preserves
float64 computations, and provides batched fitting for independent same-family
models. These batched entry points are intended for workloads such as bootstrap,
cross-validation, and candidate-family evaluation, where accelerator throughput
is more meaningful than a single small model fit.

The package also contains a Cole-Green full-Hessian fitting route and deprecated
compatibility aliases for historical L-BFGS entry points. Distribution-family
metadata is centralized through the registry so that formula fitting, validation,
service APIs, and downstream automation use the same family-resolution contract.

# Validation and Benchmarking

Benchmark claims are generated from scripts under `benchmarks/`. The v1.0
benchmark workflow separates Python-only smoke checks, optional R comparisons,
and optional GPU runs. Reports must distinguish cold JAX compilation time from
warm repeated timings, include confidence intervals for repeated warm timings,
and state whether R or GPU resources were available. The CI smoke job does not
require R or GPU and is only a health check; publication-quality comparisons
must use the optional-R suite and cite the generated artifacts, hardware, JAX
backend, dtype, data size, formulas, and repetitions.

# Limitations

OmniLSS is not a drop-in replacement for every feature in the R ecosystem. Some
families and smoothers have more extensive validation than others, GPU benefits
are workload- and shape-dependent, and JAX compilation overhead can dominate
small one-off fits. Performance conclusions should therefore remain conservative
and should separate cold and warm behavior. The project also intentionally keeps
commercial add-ons outside the GPL core process boundary; Pro-side automation
communicates with Core through gRPC rather than importing `omnilss` directly.

# Acknowledgements

OmniLSS is inspired by the R GAMLSS project and the statistical work by Rigby,
Stasinopoulos, Cole, Green, and the wider distributional-regression community.

# References
