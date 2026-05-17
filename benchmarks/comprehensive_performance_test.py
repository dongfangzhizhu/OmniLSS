"""OmniLSS vs R GAMLSS comprehensive performance benchmark.

Runs the canonical test matrix (9 distributions × data sizes × 3 formulas)
against a live R process, then auto-generates a Markdown report.

Timing methodology
------------------
- Python: cold-start time is reported separately. Warm timings use multiple
  JAX warm-up runs before timing; `jax.block_until_ready()` ensures async
  computation completes before stopping the clock. Warm performance is reported
  as the median of timed runs rather than the fastest value.
- R: one Rscript subprocess per benchmark case. CSV loading and package startup
  are excluded from reported timings; the R script performs one untimed warm-up
  fit, then reports in-process elapsed time for the requested timed repeats.

Usage
-----
    python benchmarks/comprehensive_performance_test.py
    python benchmarks/comprehensive_performance_test.py --quick      # 3 dists
    python benchmarks/comprehensive_performance_test.py --no-r       # Python only
    python benchmarks/comprehensive_performance_test.py --large      # add n=50000,500000,5000000
    python benchmarks/comprehensive_performance_test.py --n-repeats 5
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
import tracemalloc
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import jax
import numpy as np

# Force UTF-8 output for stdout/stderr on Windows so emoji/special characters
# in console output don't crash. This must run before any print() calls.
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
        sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass

# ── path setup ────────────────────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT / "omnilss" / "src"))

from omnilss import gamlss  # noqa: E402
from omnilss.distributions import resolve_family  # noqa: E402

# ── constants ─────────────────────────────────────────────────────────────────
# All distributions with confirmed working d/p/q/r and fitting support
DISTRIBUTIONS = [
    # Core continuous
    "NO",
    "GA",
    "LOGNO",
    "WEI",
    "EXP",
    "IG",
    "LO",
    "TF",
    # Core discrete
    "PO",
    "BI",
    "NBI",
    "NBII",
    "ZIP",
    # Beta / bounded
    "BE",
    # Zero-altered
    "ZAGA",
    # Batch 1
    "GU",
    "RG",
    "PE",
    # Batch 3 heavy-tail / skewed
    "SHASH",
    "JSU",
    # Box-Cox
    "BCCG",
    "BCT",
    "BCPE",
]
QUICK_DISTRIBUTIONS = ["NO", "GA", "PO"]
DATA_SIZES = [100, 500, 5000]
LARGE_DATA_SIZES = [100, 500, 5000, 50_000, 500_000, 5_000_000]
FORMULAS = ["y ~ 1", "y ~ x1", "y ~ x1 + x2"]
N_REPEATS = 3
WARMUP = 3
DEFAULT_DEVIANCE_ABS_TOL = 1e-5
DEFAULT_DEVIANCE_REL_TOL = 1e-5

# R script — one warm-up call (untimed) then n_repeats timed calls
_R_SCRIPT = r"""
suppressMessages(library(gamlss))
suppressMessages(library(jsonlite))
args <- commandArgs(trailingOnly=TRUE)
data_file <- args[1]
dist      <- args[2]
formula   <- args[3]
n_repeats <- as.integer(args[4])

df <- read.csv(data_file)
mu_formula <- as.formula(formula)

# One warm-up call (not timed)
tryCatch(gamlss(mu_formula, family=dist, data=df, trace=FALSE), error=function(e) NULL)

times <- numeric(n_repeats)
deviance_val <- NA_real_
converged <- FALSE

for (i in seq_len(n_repeats)) {
  t0 <- proc.time()["elapsed"]
  fit <- tryCatch(gamlss(mu_formula, family=dist, data=df, trace=FALSE), error=function(e) NULL)
  times[i] <- proc.time()["elapsed"] - t0
  if (!is.null(fit)) { deviance_val <- fit$G.deviance; converged <- TRUE }
}

cat(toJSON(list(mean_time=mean(times), min_time=min(times),
               deviance=deviance_val, converged=converged), auto_unbox=TRUE), "\n")
"""


# ── data generation ───────────────────────────────────────────────────────────


def _generate_data(dist: str, n: int, seed: int = 42) -> dict[str, np.ndarray]:
    """Generate reproducible test data for a given distribution."""
    rng = np.random.RandomState(seed)
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)

    if dist == "NO":
        y = 10.0 + 2.0 * x1 + rng.normal(0, 2.5, n)
    elif dist in ("LOGNO", "LOGNO2"):
        y = np.exp(rng.normal(2.0 + 0.3 * x1, 0.6, n))
    elif dist == "GA":
        mu = np.exp(2.0 + 0.5 * x1)
        y = rng.gamma(4.0, mu / 4.0)
    elif dist == "PO":
        y = rng.poisson(np.exp(1.0 + 0.3 * x1)).astype(float)
    elif dist == "BI":
        y = rng.binomial(1, 1 / (1 + np.exp(-0.5 * x1))).astype(float)
    elif dist in ("NBI", "PIG", "SICHEL"):
        mu = np.exp(1.5 + 0.3 * x1)
        y = rng.negative_binomial(2.0, 2.0 / (2.0 + mu)).astype(float)
    elif dist == "NBII":
        mu = np.exp(1.5 + 0.3 * x1)
        y = rng.negative_binomial(2.0, 2.0 / (2.0 + mu)).astype(float)
    elif dist == "BE":
        mu = 1 / (1 + np.exp(-0.5 * x1))
        y = np.clip(rng.beta(mu * 4, (1 - mu) * 4), 1e-4, 1 - 1e-4)
    elif dist == "ZIP":
        mu = np.exp(1.0 + 0.3 * x1)
        y = np.where(rng.binomial(1, 0.2, n), 0, rng.poisson(mu)).astype(float)
    elif dist == "ZAGA":
        mu = np.exp(1.5 + 0.2 * x1)
        y = np.where(rng.binomial(1, 0.25, n), 0.0, rng.gamma(2.78, mu / 2.78))
    elif dist == "ZAIG":
        mu = np.exp(1.5 + 0.2 * x1)
        y = np.where(rng.binomial(1, 0.25, n), 0.0, rng.gamma(2.78, mu / 2.78))
    elif dist == "WEI":
        mu = np.exp(1.5 + 0.3 * x1)
        y = rng.weibull(2.0, n) * mu
    elif dist == "EXP":
        mu = np.exp(1.0 + 0.3 * x1)
        y = rng.exponential(mu)
    elif dist == "IG":
        mu = np.exp(1.0 + 0.3 * x1)
        # Approximate inverse Gaussian via normal approximation for speed
        y = np.abs(rng.normal(mu, mu * 0.5))
    elif dist in ("LO",):
        y = 0.5 + 0.3 * x1 + rng.logistic(0, 1, n)
    elif dist == "TF":
        y = 0.5 + 0.3 * x1 + rng.standard_t(10, n)
    elif dist in ("GU",):
        y = 0.5 + 0.3 * x1 + rng.gumbel(0, 1, n)
    elif dist in ("RG",):
        y = 0.5 + 0.3 * x1 - rng.gumbel(0, 1, n)
    elif dist in ("PE", "NO2"):
        y = 0.5 + 0.3 * x1 + rng.normal(0, 1, n)
    elif dist in ("SHASH", "SN1", "SN2", "JSU", "GT"):
        y = 0.5 + 0.3 * x1 + rng.normal(0, 1, n)
    elif dist in ("BCCG", "BCT", "BCPE"):
        y = np.exp(0.5 + 0.3 * x1 + rng.normal(0, 0.3, n))
    elif dist in ("IGAMMA", "PARETO2", "GG", "GB2"):
        mu = np.exp(1.0 + 0.3 * x1)
        y = rng.gamma(2.0, mu / 2.0)
    else:
        y = rng.normal(0, 1, n)

    return {"y": y, "x1": x1, "x2": x2}


# ── Python timing ─────────────────────────────────────────────────────────────


def _time_python(
    dist: str,
    formula: str,
    data: dict,
    n_repeats: int,
    warmup: int,
) -> dict:
    """Time gamlss() with honest JAX synchronization and warm-up.

    The first fit is reported separately as cold-start latency because it may
    include JIT compilation. Subsequent warm-up fits are excluded from reported
    performance. Timed warm runs report median/min/max so benchmark reports do
    not accidentally present first-run or fastest-run numbers as steady-state.
    """
    family = resolve_family(dist)

    def _fit():
        model = gamlss(formula, family=family, data=data)
        # Block until all JAX async computation is complete
        try:
            for fv in model.fitted_values.values():
                jax.block_until_ready(fv)
        except Exception:
            pass
        return model

    # Cold-start: first call includes JIT compilation.  tracemalloc reports
    # Python heap peak memory only; JAX device allocator memory is backend-specific.
    t_cold_start = time.perf_counter()
    tracemalloc.start()
    try:
        model = _fit()
        _, peak_bytes = tracemalloc.get_traced_memory()
        cold_time = time.perf_counter() - t_cold_start
        deviance = float(model.g_dev)
    except Exception as e:
        return {"success": False, "error": str(e)}
    finally:
        tracemalloc.stop()

    # Additional warm-up calls excluded from timed performance.
    for _ in range(max(warmup - 1, 0)):
        try:
            _fit()
        except Exception:
            pass

    # Timed warm runs (steady-state performance)
    times = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        try:
            model = _fit()
            elapsed = time.perf_counter() - t0
            times.append(elapsed)
            deviance = float(model.g_dev)
        except Exception as e:
            return {"success": False, "error": str(e)}

    return {
        "success": True,
        "cold_time": cold_time,
        "mean_time": float(np.mean(times)),
        "median_time": float(np.median(times)),
        "min_time": float(np.min(times)),
        "max_time": float(np.max(times)),
        "deviance": deviance,
        "python_peak_memory_mb": float(peak_bytes / (1024 * 1024)),
    }


def honest_benchmark(
    family_name: str,
    n: int,
    n_warmup: int = 3,
    n_runs: int = 10,
) -> dict[str, object]:
    """Small standalone benchmark helper using the documented JAX method.

    This is intended for README/performance-table refreshes where a compact,
    reproducible measurement is preferable to the full R comparison matrix.
    """
    rng = np.random.default_rng(42)
    x = np.linspace(0, 5, n)
    y = 2 + 0.5 * x + rng.normal(size=n)
    data = {"y": y, "x": x}

    def _fit_once():
        model = gamlss("y ~ x", family=family_name, data=data)
        try:
            jax.block_until_ready(model.fitted_values.get("mu"))
        except Exception:
            pass
        return model

    for _ in range(n_warmup):
        _fit_once()

    times: list[float] = []
    deviance = None
    for _ in range(n_runs):
        t0 = time.perf_counter()
        model = _fit_once()
        times.append(time.perf_counter() - t0)
        deviance = float(model.g_dev)

    return {
        "family": family_name,
        "n": n,
        "median_s": float(np.median(times)),
        "min_s": float(np.min(times)),
        "max_s": float(np.max(times)),
        "runs": n_runs,
        "warmup_runs": n_warmup,
        "deviance": deviance,
        "note": f"JIT warm-up ({n_warmup} runs) excluded",
    }


# ── R timing ──────────────────────────────────────────────────────────────────


def _check_r_available() -> bool:
    try:
        result = subprocess.run(
            [
                "Rscript",
                "-e",
                "suppressMessages({library(gamlss); library(jsonlite)}); cat('ok')",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        return result.returncode == 0 and "ok" in result.stdout
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _time_r(dist: str, formula: str, data: dict, n_repeats: int) -> dict:
    """Time R gamlss() using in-process elapsed time from one Rscript run."""
    import csv

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False, newline=""
    ) as f:
        csv_path = f.name
        writer = csv.writer(f)
        keys = list(data.keys())
        writer.writerow(keys)
        for row in zip(*[data[k] for k in keys]):
            writer.writerow(row)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".R", delete=False) as f:
        r_path = f.name
        f.write(_R_SCRIPT)

    try:
        result = subprocess.run(
            ["Rscript", r_path, csv_path, dist, formula, str(n_repeats)],
            capture_output=True,
            text=True,
            timeout=120,
        )

        if result.returncode != 0:
            return {"success": False, "error": result.stderr.strip()[:200]}

        lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        if not lines:
            return {"success": False, "error": "no output from R"}
        try:
            parsed = json.loads(lines[-1])
        except json.JSONDecodeError:
            return {"success": False, "error": f"bad JSON: {lines[-1][:100]}"}

        if not parsed.get("converged", False):
            return {"success": False, "error": "R did not converge"}

        return {
            "success": True,
            "mean_time": float(parsed["mean_time"]),
            "min_time": float(parsed["min_time"]),
            "deviance": float(parsed["deviance"]),
        }
    except subprocess.TimeoutExpired:
        return {"success": False, "error": "R timeout"}
    except Exception as e:
        return {"success": False, "error": str(e)}
    finally:
        Path(csv_path).unlink(missing_ok=True)
        Path(r_path).unlink(missing_ok=True)


# ── result dataclass ──────────────────────────────────────────────────────────


@dataclass
class ComparisonResult:
    distribution: str
    data_size: int
    model_config: str
    python_success: bool = False
    r_success: bool = False
    python_cold_time: float | None = None  # includes JIT compilation
    python_time: float | None = None  # median steady-state (warm)
    r_time: float | None = None
    speedup: float | None = None
    python_deviance: float | None = None
    r_deviance: float | None = None
    deviance_diff: float | None = None
    deviance_rel_diff: float | None = None
    deviance_consistent: bool | None = None
    deviance_abs_tolerance: float | None = None
    deviance_rel_tolerance: float | None = None
    python_peak_memory_mb: float | None = None
    python_error: str | None = None
    r_error: str | None = None

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


# ── report generation ─────────────────────────────────────────────────────────


def _generate_report(
    results: list[ComparisonResult],
    output_path: Path,
    r_available: bool,
) -> None:
    """Write a Markdown performance report."""
    successful = [r for r in results if r.python_success and r.r_success]
    python_only = [r for r in results if r.python_success and not r.r_success]
    r_only = [r for r in results if not r.python_success and r.r_success]
    both_failed = [r for r in results if not r.python_success and not r.r_success]
    deviance_mismatches = [r for r in successful if r.deviance_consistent is False]

    lines: list[str] = []
    a = lines.append

    a("# OmniLSS vs R GAMLSS Performance Report")
    a("")
    a(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    a(f"**R comparison**: {'live Rscript' if r_available else 'skipped'}")
    a("")
    a("## Timing Methodology")
    a("")
    a(
        "- **Python warm time**: median steady-state runtime after JAX warm-up; "
        "`jax.block_until_ready()` ensures async computation completes."
    )
    a("- **Python cold time**: first call including JIT compilation overhead.")
    a(
        "- **Python memory**: peak Python heap during cold fit via `tracemalloc`; "
        "JAX device allocator memory is not included."
    )
    a(
        "- **R time**: in-process elapsed time reported by one Rscript run after "
        "CSV/package setup, with one untimed warm-up fit before timed repeats."
    )
    a("")
    a("---")
    a("")
    a("## Executive Summary")
    a("")
    a(f"- **Total tests**: {len(results)}")
    a(
        f"- **Both succeeded**: {len(successful)} ({100*len(successful)/max(len(results),1):.1f}%)"
    )
    a(f"- **Python only**: {len(python_only)}")
    a(f"- **R only**: {len(r_only)}")
    a(f"- **Both failed**: {len(both_failed)}")
    if r_available:
        a(f"- **Deviance mismatches**: {len(deviance_mismatches)}")
    a("")

    if successful and r_available:
        speedups = [r.speedup for r in successful if r.speedup is not None]
        if speedups:
            faster = sum(1 for s in speedups if s > 1.0)
            a("### Speedup (Python warm vs R)")
            a("")
            a("| Metric | Value |")
            a("|--------|-------|")
            a(f"| Mean   | **{np.mean(speedups):.1f}×** |")
            a(f"| Median | {np.median(speedups):.1f}× |")
            a(f"| Min    | {np.min(speedups):.1f}× |")
            a(f"| Max    | {np.max(speedups):.1f}× |")
            a(f"| Faster than R | {faster}/{len(speedups)} |")
            a("")

    a("---")
    a("")
    a("## Results by Distribution")
    a("")

    by_dist: dict[str, list[ComparisonResult]] = {}
    for r in results:
        by_dist.setdefault(r.distribution, []).append(r)

    for dist in sorted(by_dist):
        dist_results = by_dist[dist]
        ok = [r for r in dist_results if r.python_success and r.r_success]
        a(f"### {dist}")
        a("")
        if ok and r_available:
            speedups = [r.speedup for r in ok if r.speedup is not None]
            consistent_count = sum(1 for r in ok if r.deviance_consistent is not False)
            a(f"- Passed: {len(ok)}/{len(dist_results)}")
            a(f"- Deviance consistent: {consistent_count}/{len(ok)}")
            if speedups:
                a(
                    f"- Mean speedup: **{np.mean(speedups):.1f}×** "
                    f"(range {np.min(speedups):.1f}–{np.max(speedups):.1f}×)"
                )
            a("")
            a(
                "| n | Formula | Python warm (s) | Python cold (s) | Python peak MB | R (s) | Speedup | Dev diff | Consistent |"
            )
            a(
                "|---|---------|----------------|----------------|----------------|-------|---------|----------|------------|"
            )
            for r in sorted(ok, key=lambda x: (x.data_size, x.model_config)):
                dev = f"{r.deviance_diff:.2e}" if r.deviance_diff is not None else "—"
                sp = f"{r.speedup:.1f}×" if r.speedup else "—"
                cold = f"{r.python_cold_time:.4f}" if r.python_cold_time else "—"
                mem = (
                    f"{r.python_peak_memory_mb:.2f}"
                    if r.python_peak_memory_mb is not None
                    else "—"
                )
                consistent = "✅" if r.deviance_consistent else "❌"
                a(
                    f"| {r.data_size} | `{r.model_config}` | "
                    f"{r.python_time:.4f} | {cold} | {mem} | {r.r_time:.4f} | {sp} | {dev} | {consistent} |"
                )
        else:
            a(
                f"- Passed: {sum(1 for r in dist_results if r.python_success)}/{len(dist_results)}"
            )
            a("")
            a(
                "| n | Formula | Python warm (s) | Python cold (s) | Python peak MB | Status |"
            )
            a(
                "|---|---------|----------------|----------------|----------------|--------|"
            )
            for r in sorted(dist_results, key=lambda x: (x.data_size, x.model_config)):
                status = "✅" if r.python_success else f"❌ {r.python_error or ''}"
                pt = f"{r.python_time:.4f}" if r.python_time else "—"
                cold = f"{r.python_cold_time:.4f}" if r.python_cold_time else "—"
                mem = (
                    f"{r.python_peak_memory_mb:.2f}"
                    if r.python_peak_memory_mb is not None
                    else "—"
                )
                a(
                    f"| {r.data_size} | `{r.model_config}` | {pt} | {cold} | {mem} | {status} |"
                )
        a("")

    if r_only or both_failed:
        a("---")
        a("")
        a("## Failures")
        a("")
        if r_only:
            a("### R succeeded, Python failed")
            a("")
            a("| Distribution | n | Formula | Error |")
            a("|-------------|---|---------|-------|")
            for r in r_only:
                err = (r.python_error or "")[:80]
                a(f"| {r.distribution} | {r.data_size} | `{r.model_config}` | {err} |")
            a("")

    a("---")
    a("")
    a("## Performance by Data Size")
    a("")
    if successful and r_available:
        by_size: dict[int, list] = {}
        for r in successful:
            by_size.setdefault(r.data_size, []).append(r)
        a(
            "| n | Tests | Mean speedup | Mean Python warm (s) | Mean Python cold (s) | Mean R (s) |"
        )
        a(
            "|---|-------|-------------|---------------------|---------------------|-----------|"
        )
        for size in sorted(by_size):
            rs = by_size[size]
            speedups = [r.speedup for r in rs if r.speedup]
            py_times = [r.python_time for r in rs if r.python_time]
            cold_times = [r.python_cold_time for r in rs if r.python_cold_time]
            r_times = [r.r_time for r in rs if r.r_time]
            sp = f"{np.mean(speedups):.1f}×" if speedups else "—"
            pt = f"{np.mean(py_times):.4f}" if py_times else "—"
            ct = f"{np.mean(cold_times):.4f}" if cold_times else "—"
            rt = f"{np.mean(r_times):.4f}" if r_times else "—"
            a(f"| {size:,} | {len(rs)} | {sp} | {pt} | {ct} | {rt} |")
    else:
        a("*(R comparison not available)*")
    a("")
    a("---")
    a("")
    a("*Report generated by OmniLSS benchmark suite*")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Report saved → {output_path}")


# ── main ──────────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(description="OmniLSS vs R GAMLSS benchmark")
    parser.add_argument(
        "--quick", action="store_true", help="Run only 3 distributions (NO, GA, PO)"
    )
    parser.add_argument(
        "--no-r", action="store_true", help="Skip R comparison (Python timing only)"
    )
    parser.add_argument(
        "--large",
        action="store_true",
        help="Include large data sizes (50k, 500k, 5M) — Python only for n>=50k",
    )
    parser.add_argument(
        "--n-repeats",
        type=int,
        default=N_REPEATS,
        help=f"Timing repetitions (default {N_REPEATS})",
    )
    parser.add_argument(
        "--require-r",
        action="store_true",
        help="Fail immediately if Rscript with gamlss/jsonlite is unavailable",
    )
    parser.add_argument(
        "--deviance-abs-tol",
        type=float,
        default=DEFAULT_DEVIANCE_ABS_TOL,
        help=f"Absolute tolerance for OmniLSS/R deviance checks (default {DEFAULT_DEVIANCE_ABS_TOL:g})",
    )
    parser.add_argument(
        "--deviance-rel-tol",
        type=float,
        default=DEFAULT_DEVIANCE_REL_TOL,
        help=f"Relative tolerance for OmniLSS/R deviance checks (default {DEFAULT_DEVIANCE_REL_TOL:g})",
    )
    args = parser.parse_args()

    distributions = QUICK_DISTRIBUTIONS if args.quick else DISTRIBUTIONS
    data_sizes = LARGE_DATA_SIZES if args.large else DATA_SIZES
    n_repeats = args.n_repeats
    if n_repeats < 1:
        print("--n-repeats must be >= 1")
        return 2

    # Large datasets: Python only (R would be too slow)
    if args.large:
        args.no_r = True

    r_available = False
    if not args.no_r:
        print("Checking R availability...", end=" ", flush=True)
        r_available = _check_r_available()
        print("✓ found" if r_available else "✗ not found")
        if args.require_r and not r_available:
            print(
                "R comparison is required but Rscript with gamlss/jsonlite is unavailable."
            )
            return 2
        if not r_available:
            print(
                "Continuing in Python-only mode; performance against R is not measured."
            )
    print()

    total = len(distributions) * len(data_sizes) * len(FORMULAS)
    print(f"{'='*70}")
    print("OmniLSS Performance Benchmark")
    print(f"  Distributions : {', '.join(distributions)}")
    print(f"  Data sizes    : {data_sizes}")
    print(f"  Formulas      : {FORMULAS}")
    print(f"  Total tests   : {total}")
    print(f"  R comparison  : {'yes' if r_available else 'no'}")
    print(
        f"  Deviance tol  : abs={args.deviance_abs_tol:.1e}, rel={args.deviance_rel_tol:.1e}"
    )
    print("  Timing        : warm (steady-state) + cold (first call / JIT)")
    print(f"{'='*70}")
    print()

    results: list[ComparisonResult] = []
    done = 0

    for dist in distributions:
        print(f"[{dist}]")
        for n in data_sizes:
            data = _generate_data(dist, n)
            for formula in FORMULAS:
                done += 1
                label = f"  n={n:7,}  {formula:<20s}"
                print(f"  ({done:2d}/{total}) {label}", end="", flush=True)

                cr = ComparisonResult(
                    distribution=dist,
                    data_size=n,
                    model_config=formula,
                )

                # Python timing
                py = _time_python(dist, formula, data, n_repeats, WARMUP)
                if py["success"]:
                    cr.python_success = True
                    cr.python_time = py.get("median_time", py["mean_time"])
                    cr.python_cold_time = py.get("cold_time")
                    cr.python_deviance = py.get("deviance")
                    cr.python_peak_memory_mb = py.get("python_peak_memory_mb")
                else:
                    cr.python_error = py.get("error", "")

                # R timing
                if r_available:
                    r = _time_r(dist, formula, data, n_repeats)
                    if r["success"]:
                        cr.r_success = True
                        cr.r_time = r["mean_time"]
                        cr.r_deviance = r.get("deviance")
                        if cr.python_success and cr.python_time and cr.r_time:
                            cr.speedup = cr.r_time / cr.python_time
                        if cr.python_deviance is not None and cr.r_deviance is not None:
                            cr.deviance_diff = abs(cr.python_deviance - cr.r_deviance)
                            denom = abs(cr.r_deviance) if cr.r_deviance else 1.0
                            cr.deviance_rel_diff = cr.deviance_diff / denom
                            cr.deviance_abs_tolerance = args.deviance_abs_tol
                            cr.deviance_rel_tolerance = args.deviance_rel_tol
                            cr.deviance_consistent = (
                                cr.deviance_diff <= args.deviance_abs_tol
                                or cr.deviance_rel_diff <= args.deviance_rel_tol
                            )
                    else:
                        cr.r_error = r.get("error", "")

                results.append(cr)

                # One-line summary
                if cr.python_success:
                    cold_str = (
                        f" cold={cr.python_cold_time:.3f}s"
                        if cr.python_cold_time
                        else ""
                    )
                    if r_available and cr.r_success and cr.r_time is not None:
                        sp = f"{cr.speedup:.1f}×" if cr.speedup else "—"
                        print(
                            f"  warm={cr.python_time:.4f}s{cold_str}  R={cr.r_time:.4f}s  → {sp}"
                        )
                    elif r_available and not cr.r_success:
                        print(f"  warm={cr.python_time:.4f}s{cold_str}  R FAIL")
                    else:
                        print(f"  warm={cr.python_time:.4f}s{cold_str}")
                else:
                    print(f"  FAIL: {cr.python_error or '?'}")

        print()

    # Summary
    py_ok = sum(1 for r in results if r.python_success)
    r_only_count = sum(1 for r in results if not r.python_success and r.r_success)

    print(f"{'='*70}")
    print("Summary")
    print(f"{'='*70}")
    print(f"  Python passed : {py_ok}/{total}")
    if r_available:
        both_ok = sum(1 for r in results if r.python_success and r.r_success)
        print(f"  Both passed   : {both_ok}/{total}")
        if r_only_count:
            print(f"  R only (regressions!) : {r_only_count}")
        inconsistent = [r for r in results if r.deviance_consistent is False]
        print(f"  Deviance consistent : {both_ok - len(inconsistent)}/{both_ok}")
        if inconsistent:
            print(f"  Deviance mismatches : {len(inconsistent)}")
        speedups = [r.speedup for r in results if r.speedup is not None]
        if speedups:
            print(f"  Mean speedup (warm) : {np.mean(speedups):.1f}×")
            print(f"  Median speedup      : {np.median(speedups):.1f}×")
            print(
                f"  Min / Max           : {np.min(speedups):.1f}× / {np.max(speedups):.1f}×"
            )
    cold_times = [r.python_cold_time for r in results if r.python_cold_time]
    warm_times = [r.python_time for r in results if r.python_time]
    if cold_times and warm_times:
        print(
            f"  Mean cold-start : {np.mean(cold_times):.3f}s (includes JIT compilation)"
        )
        print(f"  Mean warm time  : {np.mean(warm_times):.4f}s (steady-state)")
    print()

    # Save JSON
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    raw_dir = Path(__file__).parent / "results" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    json_path = raw_dir / f"quick_results_{timestamp}.json"

    payload = {
        "timestamp": timestamp,
        "r_available": r_available,
        "config": {
            "distributions": distributions,
            "data_sizes": data_sizes,
            "formulas": FORMULAS,
            "n_repeats": n_repeats,
            "warmup_runs": WARMUP,
            "timing_method": "Python median warm runtime after JAX warm-up with block_until_ready plus separate cold time; R in-process elapsed timing after setup",
            "deviance_tolerances": {
                "abs": args.deviance_abs_tol,
                "rel": args.deviance_rel_tol,
            },
            "memory_method": "Python tracemalloc peak during cold fit; excludes JAX device allocator",
        },
        "results": [r.to_dict() for r in results],
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Raw results  → {json_path}")

    # Auto-generate Markdown report
    reports_dir = Path(__file__).parent / "results" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / f"quick_report_{timestamp}.md"
    _generate_report(results, report_path, r_available)

    deviance_mismatches = sum(1 for r in results if r.deviance_consistent is False)
    return 0 if r_only_count == 0 and deviance_mismatches == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
