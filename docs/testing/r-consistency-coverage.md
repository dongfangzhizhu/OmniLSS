# R Consistency Coverage

[中文版本](r-consistency-coverage_cn.md)

This document tracks the R-backed checks that protect compatibility with the
reference `gamlss`/`gamlss.dist` implementation. It is a coverage map, not a
claim that OmniLSS is fully equivalent to R for every family.

## Execution Policy

- General Python CI may run without R and should skip R-backed tests cleanly.
- R-equivalence claims require an R-enabled environment with `jsonlite`,
  `gamlss`, and `gamlss.dist` installed.
- Python-only smoke checks prove importability and code-path execution only.

## Current d/p/q Coverage

`tests/test_dpqr_r_consistency.py` currently compares scalarized R reference
values against OmniLSS d/p/q implementations for:

| Family | d | p | q | Notes |
| --- | --- | --- | --- | --- |
| `NO` | yes | yes | yes | Normal baseline |
| `GA` | yes | yes | yes | Gamma baseline |
| `PO` | yes | yes | yes | Discrete baseline |
| `GU` | yes | yes | yes | Batch 1 AD family |
| `RG` | yes | yes | yes | Batch 1 AD family |
| `TF` | yes | yes | yes | Three-parameter continuous family |
| `BB` | yes | yes | yes | Fixed data parameter `bd` coverage |

## Current Bridge Requirements

The `RTestBus` setup now checks all required R packages up front so a missing
R dependency is reported as a skipped R-backed suite rather than as a late test
failure. The R bridge script loads both `gamlss` and `gamlss.dist`, invokes
family functions element-wise, and returns structured JSON errors from R.

## Remaining Coverage Gaps

- Add d/p/q reference cases for all stable families exposed through
  `resolve_family()` before claiming broad d/p/q equivalence.
- Add R-backed `r` distributional checks where deterministic equality is not
  expected; use support, moments, or seeded distributional diagnostics instead.
- Add fitting-level parity checks for families that have complete d/p/q coverage.
- Record generated R-enabled reports as benchmark artifacts before publishing
  static compatibility or performance summaries.
