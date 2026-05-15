# OmniLSS Benchmark and Consistency Guide

## Purpose

The benchmark suite is a validation gate, not a marketing tool. It should answer
two questions in order:

1. Does this OmniLSS version reproduce native R `gamlss` results within declared
   numerical tolerances?
2. After that is true, how do cold runtime, warm runtime, memory, and R runtime
   compare for the same workload?

## Gate Script

```bash
python benchmarks/run_local_validation.py --quick
```

Default behavior requires R. The script exits before performance testing if
consistency fails.

Use this only for environments that cannot install R:

```bash
python benchmarks/run_local_validation.py --quick --allow-python-only --no-fit --no-smooth
```

That command is a smoke check, not an R-equivalence result.

## Consistency Script

```bash
python benchmarks/comprehensive_r_consistency_test.py --quick --require-r
```

Default tolerances:

| Layer | Absolute | Relative |
|-------|---------:|---------:|
| d/p/q | `1e-5` | `1e-5` |
| fitting | `1e-3` | `1e-3` |
| smoothing | `1e-3` | `1e-3` |

The generated JSON records the tolerances used for each run.

## Performance Script

```bash
python benchmarks/comprehensive_performance_test.py --quick --require-r --n-repeats 3
```

The report separates:

- OmniLSS cold time, including first-call JAX compilation;
- OmniLSS warm steady-state time;
- R wall-clock time from a fresh `Rscript` subprocess;
- deviance difference and pass/fail against configured tolerances;
- Python heap peak memory via `tracemalloc`.

## Reporting Policy

When updating documentation from benchmark results, include:

- command line;
- timestamp;
- OS/CPU/GPU/backend if known;
- data sizes, formulas, distributions, repetitions;
- whether R comparison was live or skipped;
- generated artifact paths.

Avoid fixed claims such as “X× faster” outside the context of a specific report.
