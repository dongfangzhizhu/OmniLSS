# OmniLSS Benchmarks

Performance and consistency benchmarks comparing OmniLSS against R gamlss.

## Quick Start

```bash
# Performance benchmark (OmniLSS vs R, 81 tests, auto-generates report)
python benchmarks/comprehensive_performance_test.py

# Consistency test (numerical agreement with R, auto-generates report)
python benchmarks/comprehensive_r_consistency_test.py

# Quick smoke test (3 distributions, ~2 minutes)
python benchmarks/comprehensive_performance_test.py --quick
python benchmarks/comprehensive_r_consistency_test.py --quick

# Python-only (no R required)
python benchmarks/comprehensive_performance_test.py --no-r
python benchmarks/comprehensive_r_consistency_test.py --no-r
```

Reports are written automatically to `results/reports/` after each run.

---

## Scripts

| Script | Purpose | Output |
|--------|---------|--------|
| `comprehensive_performance_test.py` | Time OmniLSS vs R on 81 test cases | `results/raw/quick_results_*.json` + `results/reports/quick_report_*.md` |
| `comprehensive_r_consistency_test.py` | Numerical agreement with R | `results/raw/r_consistency_*.json` + `results/reports/consistency_report_*.md` |
| `generate_plots.py` | Plot benchmark results | `results/figures/` |

## Performance Results

Latest benchmark (Windows, Intel Core i7, JAX CPU):

| Distribution | Mean speedup | Range |
|-------------|-------------|-------|
| NO (Normal) | ~104× | 88–122× |
| LOGNO | ~110× | 100–148× |
| PO (Poisson) | ~62× | 32–84× |
| GA (Gamma) | ~34× | 29–43× |
| BI (Binomial) | ~65× | 29–119× |
| NBI (Neg. Binomial) | ~35× | 13–55× |
| BE (Beta) | ~27× | 22–33× |
| ZIP | ~22× | 12–27× |
| ZAGA | ~16× | 14–20× |

**Overall: 30–120× faster than R gamlss on CPU.**

## Consistency Results

All 33 tests pass (100%) with R comparison enabled:

| Category | Tests | Pass rate |
|----------|-------|-----------|
| d/p/q functions | 18 | 100% |
| Model fitting (RS/CG/Mixed) | 12 | 100% |
| Smoothers (pb/ps/cs) | 3 | 100% |

Numerical accuracy vs R:
- Continuous distributions (NO, GA, LOGNO, BE): max absolute error < 1×10⁻⁴
- Discrete distributions (PO, NBI): differences due to integer rounding, expected
- Model fitted values: max absolute error < 1×10⁻³

## Directory Structure

```
benchmarks/
├── README.md                              # This file
├── QUICK_START.md                         # Detailed usage guide
├── COMPREHENSIVE_TESTING_GUIDE.md         # Testing methodology
├── comprehensive_performance_test.py      # Performance benchmark
├── comprehensive_r_consistency_test.py    # Consistency test
├── generate_plots.py                      # Plot generator
├── config.py                              # Shared configuration
├── data_generators.py                     # Test data generation
├── base.py                                # Base benchmark classes
├── distributions.py                       # Distribution benchmarks
├── reporters/                             # Report formatters
├── diagnostic/                            # Diagnostic scripts
└── results/
    ├── raw/                               # JSON results (gitignored)
    └── reports/                           # Markdown reports
```

## Requirements

- Python with OmniLSS installed (`uv sync --active --project omnilss`)
- R with gamlss and jsonlite packages for R comparison:
  ```r
  install.packages(c("gamlss", "jsonlite"))
  ```
- Rscript must be on PATH

R is optional — both scripts run in Python-only mode with `--no-r`.
