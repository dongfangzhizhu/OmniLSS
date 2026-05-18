# OmniLSS Benchmarks

[中文版本](README_cn.md)

Benchmark scripts for validating OmniLSS against native R `gamlss` before making
performance claims.

## Validation Order

Always run the checks in this order:

1. **Numerical consistency** against native R `gamlss`.
2. **Performance comparison** only after consistency passes.
3. **Documentation/report update** using the generated JSON/Markdown artifacts.

The orchestration script enforces that order:

```bash
python benchmarks/run_local_validation.py --quick
```

By default, the validation gate requires `Rscript` plus the R packages `gamlss`
and `jsonlite`. In environments without R, a smoke-only run is available:

```bash
python benchmarks/run_local_validation.py --quick --allow-python-only --no-fit --no-smooth
```

Python-only runs do **not** prove equivalence with native `gamlss`; they only
check that the benchmark scripts and OmniLSS code paths execute.

## Individual Scripts

| Script | Purpose | Primary output |
|--------|---------|----------------|
| `run_local_validation.py` | Runs consistency first, then performance if consistency passes | Delegates to the two scripts below |
| `comprehensive_r_consistency_test.py` | Compares d/p/q, fitting, and smoothing results with R `gamlss` | `benchmarks/results/raw/r_consistency_*.json` and `benchmarks/results/reports/consistency_report_*.md` |
| `comprehensive_performance_test.py` | Measures cold time, warm steady-state time, Python heap peak memory, and setup-excluded in-process R timings | `benchmarks/results/raw/quick_results_*.json` and `benchmarks/results/reports/quick_report_*.md` |
| `three_way_comparison.py` | Optional OmniLSS/R/ondil exploratory comparison | `benchmarks/results/raw/three_way_*.json` and report Markdown |
| `generate_plots.py` | Plot benchmark result artifacts | `benchmarks/results/figures/` |

## Consistency Tolerances

Default tolerances are intentionally explicit:

| Comparison | Absolute tolerance | Relative tolerance |
|------------|-------------------:|-------------------:|
| d/p/q functions | `1e-5` | `1e-5` |
| fitting and smoothing | `1e-3` | `1e-3` |
| performance deviance check | `1e-5` | `1e-5` |

Override them from the command line when a numerical-methodology note justifies a
different threshold, for example:

```bash
python benchmarks/comprehensive_r_consistency_test.py --quick \
  --dpqr-abs-tol 1e-6 --dpqr-rel-tol 1e-6
```

## Performance Reporting Rules

Reports separate:

- cold OmniLSS time: first fit, including JAX compilation;
- warm OmniLSS time: repeated steady-state fit time;
- R time: in-process elapsed timing from a single `Rscript` run after package/CSV setup, with one untimed R warm-up fit;
- Python heap peak memory: `tracemalloc` peak during cold fit.

Do not summarize results with static marketing claims. Cite the generated report,
hardware, backend, dtype, data size, formula, repetitions, and whether R was
available.


## Phase 5 Benchmark Suites

The v1.0 benchmark workflow is split into three suites so publication claims can
be reproduced without hiding optional dependencies:

| Suite | Command | Dependency expectation | Intended use |
|-------|---------|------------------------|--------------|
| No-R smoke | `python benchmarks/jax_rs_benchmark.py --suite no-r --smoke --families NO --n-values 100 --n-reps 2` | Python/JAX only; no R or GPU | CI smoke and quick local health checks |
| Optional-R | `python benchmarks/jax_rs_benchmark.py --suite optional-r` | Uses R when `Rscript` and packages are available; records unavailable R as missing comparison data | Release and paper validation |
| Optional-GPU | `python benchmarks/jax_rs_benchmark.py --suite optional-gpu --smoke` | Requires a JAX GPU device for measurements; otherwise writes a skipped artifact and exits successfully | GPU crossover exploration |

`jax_rs_benchmark.py` now records repeated warm timing samples plus a 95%
confidence interval (`*_ci95`) in the JSON and Markdown artifacts. Cold JAX
compilation time is reported separately and is never combined with warm steady
state timing.

## Requirements

- Python with this repository importable (`PYTHONPATH=omnilss/src` or editable install).
- Optional but required for true validation: `Rscript` on `PATH` with:

```r
install.packages(c("gamlss", "jsonlite"))
```

## Output Locations

Generated benchmark artifacts are written under `benchmarks/results/` and are
not meant to be committed unless a release process explicitly asks for frozen
reference results.
