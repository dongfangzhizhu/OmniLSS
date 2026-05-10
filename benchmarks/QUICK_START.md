# Benchmarks Quick Start

## Run in 30 seconds

```bash
# Quick smoke test — 3 distributions, Python only, ~30 seconds
python benchmarks/comprehensive_performance_test.py --quick --no-r
python benchmarks/comprehensive_r_consistency_test.py --quick --no-r
```

Both scripts **auto-generate a Markdown report** in `benchmarks/results/reports/`.

---

## Full Benchmarks (requires R)

### Prerequisites

```r
# In R
install.packages(c("gamlss", "jsonlite"))
```

Rscript must be on PATH. Verify:
```bash
Rscript --version
```

### Performance Benchmark

Tests OmniLSS speed vs R gamlss across 9 distributions × 3 data sizes × 3 formulas = **81 tests**.

```bash
# Full run (~15 minutes with R)
python benchmarks/comprehensive_performance_test.py

# Quick run — 3 distributions (~5 minutes)
python benchmarks/comprehensive_performance_test.py --quick

# Python only — no R needed
python benchmarks/comprehensive_performance_test.py --no-r

# Fewer repetitions for faster results
python benchmarks/comprehensive_performance_test.py --quick --n-repeats 1
```

**Output:**
- Console: live progress with per-test speedup
- `results/raw/quick_results_TIMESTAMP.json` — raw data
- `results/reports/quick_report_TIMESTAMP.md` — formatted report

**Expected results:**
```
NO:   87–122× speedup
GA:   29–43×  speedup
PO:   32–84×  speedup
Overall mean: ~60×
```

### Consistency Test

Tests numerical agreement between OmniLSS and R gamlss.

```bash
# Full run (~10 minutes with R)
python benchmarks/comprehensive_r_consistency_test.py

# Quick run — 6 distributions
python benchmarks/comprehensive_r_consistency_test.py --quick

# Python only
python benchmarks/comprehensive_r_consistency_test.py --no-r
```

**Output:**
- Console: per-distribution pass/fail with max error
- `results/raw/r_consistency_TIMESTAMP.json` — raw data
- `results/reports/consistency_report_TIMESTAMP.md` — formatted report

**Expected results:**
```
d/p/q functions : 45/45 ✅
Model fitting   : 12/12 ✅
Smoothers       : 3/3   ✅
Overall         : 100%
```

---

## Understanding the Results

### Performance Metrics

| Metric | Meaning |
|--------|---------|
| Speedup | R time / Python time — higher is better |
| Python time | Wall-clock time for OmniLSS fit |
| R time | Wall-clock time for Rscript subprocess (includes R startup ~0.6s) |

> **Note on R timing**: Each R call spawns a fresh Rscript process, so R time
> includes ~0.6s startup overhead. This is the same methodology used in the
> original benchmarks and reflects real-world usage where R is called from a
> script or pipeline.

### Consistency Metrics

| Metric | Meaning |
|--------|---------|
| Max absolute error | `max|python - r|` across all test points |
| Mean absolute error | `mean|python - r|` |
| RMSE | Root mean square error of fitted values |
| Correlation | Pearson r between Python and R outputs (1.0 = perfect) |

For continuous distributions, errors < 1×10⁻⁴ indicate numerical equivalence.
Discrete distributions (PO, NBI, BI) may show errors of 1–2 due to integer
rounding differences — this is expected and correct.

---

## Diagnostic Scripts

The `diagnostic/` directory contains focused scripts for investigating specific issues:

```bash
# Compare fitted parameters for NBI
python benchmarks/diagnostic/compare_fitted_parameters.py

# Check R convergence
Rscript benchmarks/diagnostic/check_r_convergence.R

# Trace RS algorithm execution
python benchmarks/diagnostic/trace_rs_detailed.py
```

---

## Troubleshooting

**R not found:**
```
Checking R availability... ✗ not found
```
→ Install R and add Rscript to PATH, or use `--no-r`

**R gamlss not installed:**
```
Error in library(gamlss) : there is no package called 'gamlss'
```
→ Run `install.packages(c("gamlss", "jsonlite"))` in R

**Slow R times:**
R startup takes ~0.6s per call. This is expected — it's the real cost of
calling R from a script. The speedup numbers reflect this accurately.

**Import errors:**
```bash
# Make sure omnilss is installed
uv sync --active --project omnilss
```
