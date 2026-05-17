# Benchmarks Quick Start

[中文版本](QUICK_START_cn.md)

## Recommended Local Gate

Run consistency first, then performance:

```bash
python benchmarks/run_local_validation.py --quick
```

This requires native R `gamlss`. If `Rscript` is not available, use the smoke
mode only to check script health:

```bash
python benchmarks/run_local_validation.py --quick --allow-python-only --no-fit --no-smooth
```

Smoke mode skips R comparison and therefore cannot confirm numerical equivalence.

## Run Scripts Separately

```bash
# Numerical consistency with R required
python benchmarks/comprehensive_r_consistency_test.py --quick --require-r

# Performance comparison with R required
python benchmarks/comprehensive_performance_test.py --quick --require-r --n-repeats 3
```

Useful local options:

```bash
# Python-only script smoke check
python benchmarks/comprehensive_r_consistency_test.py --quick --no-r --no-fit --no-smooth
python benchmarks/comprehensive_performance_test.py --quick --no-r --n-repeats 1

# Stricter d/p/q tolerance
python benchmarks/comprehensive_r_consistency_test.py --quick --require-r \
  --dpqr-abs-tol 1e-6 --dpqr-rel-tol 1e-6
```

## Interpreting Results

- Consistency failures stop `run_local_validation.py` before performance runs.
- Performance reports include cold time, warm steady-state time, setup-excluded R elapsed time, deviance difference,
  and Python heap peak memory.
- A speed ratio is only meaningful for the exact generated report context:
  hardware, backend, distribution, data size, formula, and repetitions.

## Outputs

- Raw JSON: `benchmarks/results/raw/`
- Markdown reports: `benchmarks/results/reports/`
