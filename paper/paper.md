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
date: 2026-05-17
bibliography: paper.bib
---

# Summary

OmniLSS is a Python implementation of Generalized Additive Models for Location,
Scale and Shape (GAMLSS) [@rigby2005] built on JAX [@jax2018]. It provides a
broad catalogue of distribution families, automatic differentiation-assisted
score and Hessian computation, and optional integration layers for scikit-learn,
HTTP, and gRPC deployments.

# Statement of Need

The R `gamlss` ecosystem is the reference implementation for distributional
regression, but many modern scientific and machine-learning workflows are
Python-first. OmniLSS targets users who need a Python-native interface,
JAX-compatible numerical kernels, and deployment-friendly service boundaries
while preserving familiar GAMLSS concepts such as parameter-specific formulas
and distribution-family objects.

# Key Design Decisions

## Automatic Differentiation Factory

The `build_ad_family()` workflow allows new distribution families to be added by
providing scalar log-density logic while JAX autodiff supplies score and Hessian
helpers. This reduces boilerplate and provides a path for community-contributed
families.

## Conservative Core/Service Boundary

The core statistical implementation remains in the GPL-licensed `omnilss`
package. Remote callers communicate through an explicit gRPC API. This makes the
interface testable, keeps runtime dependencies optional for local users, and
provides a clean process boundary for downstream service experiments.

## Honest Benchmark Reporting

Performance claims are generated from scripts under `benchmarks/`. Reports must
separate cold-start/JIT latency from warm steady-state timings, use warm-up runs,
and record when R or GPU comparisons were unavailable in the execution
environment.

# Acknowledgements

OmniLSS is inspired by the R GAMLSS project and the statistical work by Rigby,
Stasinopoulos, Cole, Green, and the wider distributional-regression community.

# References
