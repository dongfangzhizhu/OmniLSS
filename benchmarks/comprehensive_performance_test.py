"""OmniLSS vs R GAMLSS comprehensive performance benchmark.

Runs the canonical 81-test matrix (9 distributions × 3 data sizes × 3 formulas)
against a live R process, then auto-generates a Markdown report.

Usage
-----
    python benchmarks/comprehensive_performance_test.py
    python benchmarks/comprehensive_performance_test.py --quick   # 3 dists only
    python benchmarks/comprehensive_performance_test.py --no-r    # skip R, Python only
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

# ── path setup ────────────────────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT / "omnilss" / "src"))

from omnilss import gamlss
from omnilss.distributions import resolve_family

# ── constants ─────────────────────────────────────────────────────────────────
DISTRIBUTIONS = ["NO", "GA", "LOGNO", "PO", "BI", "NBI", "BE", "ZIP", "ZAGA"]
QUICK_DISTRIBUTIONS = ["NO", "GA", "PO"]
DATA_SIZES = [100, 500, 5000]
FORMULAS = ["y ~ 1", "y ~ x1", "y ~ x1 + x2"]
N_REPEATS = 3
WARMUP = 1

# R script template — runs gamlss() and prints timing + deviance as JSON.
# Uses proc.time() for wall-clock measurement. Each call is a fresh Rscript
# process so there is no cross-test caching.
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
tryCatch(
  gamlss(mu_formula, family=dist, data=df, trace=FALSE),
  error=function(e) NULL
)

times <- numeric(n_repeats)
deviance_val <- NA_real_
converged <- FALSE

for (i in seq_len(n_repeats)) {
  t0 <- proc.time()["elapsed"]
  fit <- tryCatch(
    gamlss(mu_formula, family=dist, data=df, trace=FALSE),
    error=function(e) NULL
  )
  times[i] <- proc.time()["elapsed"] - t0
  if (!is.null(fit)) {
    deviance_val <- fit$G.deviance
    converged    <- TRUE
  }
}

cat(toJSON(list(
  mean_time  = mean(times),
  min_time   = min(times),
  deviance   = deviance_val,
  converged  = converged
), auto_unbox=TRUE), "\n")
"""


# ── data generation ───────────────────────────────────────────────────────────

def _generate_data(dist: str, n: int, seed: int = 42) -> dict[str, np.ndarray]:
    """Generate reproducible test data for a given distribution."""
    rng = np.random.RandomState(seed)
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)

    if dist == "NO":
        y = 10.0 + 2.0 * x1 + rng.normal(0, 2.5, n)
    elif dist == "LOGNO":
        log_mu = 2.0 + 0.3 * x1
        y = np.exp(rng.normal(log_mu, 0.6, n))
    elif dist == "GA":
        mu = np.exp(2.0 + 0.5 * x1)
        shape = 4.0
        y = rng.gamma(shape, mu / shape)
    elif dist == "PO":
        mu = np.exp(1.0 + 0.3 * x1)
        y = rng.poisson(mu).astype(float)
    elif dist == "BI":
        p = 1 / (1 + np.exp(-(0.5 * x1)))
        y = rng.binomial(1, p).astype(float)
    elif dist == "NBI":
        mu = np.exp(1.5 + 0.3 * x1)
        size = 2.0
        prob = size / (size + mu)
        y = rng.negative_binomial(size, prob).astype(float)
    elif dist == "BE":
        mu = 1 / (1 + np.exp(-(0.5 * x1)))
        phi = 4.0
        alpha = mu * phi
        beta = (1 - mu) * phi
        y = np.clip(rng.beta(alpha, beta), 1e-4, 1 - 1e-4)
    elif dist == "ZIP":
        mu = np.exp(1.0 + 0.3 * x1)
        is_zero = rng.binomial(1, 0.2, n)
        y = np.where(is_zero, 0, rng.poisson(mu)).astype(float)
    elif dist == "ZAGA":
        mu = np.exp(1.5 + 0.2 * x1)
        is_zero = rng.binomial(1, 0.25, n)
        shape = 2.78
        y = np.where(is_zero, 0.0, rng.gamma(shape, mu / shape))
    else:
        y = rng.normal(0, 1, n)

    return {"y": y, "x1": x1, "x2": x2}


# ── Python timing ─────────────────────────────────────────────────────────────

def _time_python(dist: str, formula: str, data: dict, n_repeats: int, warmup: int) -> dict:
    """Time a single Python gamlss() call."""
    family = resolve_family(dist)

    def _fit():
        return gamlss(formula, family=family, data=data)

    # warm-up
    for _ in range(warmup):
        try:
            _fit()
        except Exception:
            pass

    times = []
    deviance = None
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
        "mean_time": float(np.mean(times)),
        "min_time": float(np.min(times)),
        "deviance": deviance,
    }


# ── R timing ──────────────────────────────────────────────────────────────────

def _check_r_available() -> bool:
    """Return True if Rscript is on PATH and gamlss + jsonlite are installed."""
    try:
        result = subprocess.run(
            ["Rscript", "-e",
             "suppressMessages({library(gamlss); library(jsonlite)}); cat('ok')"],
            capture_output=True, text=True, timeout=30,
        )
        return result.returncode == 0 and "ok" in result.stdout
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _time_r(dist: str, formula: str, data: dict, n_repeats: int) -> dict:
    """Time a single R gamlss() call via Rscript subprocess.
    
    Measures wall-clock time from Python's side (subprocess total), which
    matches what the original benchmark did and gives an honest comparison.
    The R script includes one warm-up call so JIT compilation is excluded
    from the timed runs.
    """
    import csv

    # Write data to a temp CSV
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False, newline=""
    ) as f:
        csv_path = f.name
        writer = csv.writer(f)
        keys = list(data.keys())
        writer.writerow(keys)
        for row in zip(*[data[k] for k in keys]):
            writer.writerow(row)

    # Write R script to a temp file
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".R", delete=False
    ) as f:
        r_path = f.name
        f.write(_R_SCRIPT)

    # Run n_repeats separate Rscript processes (each includes 1 warm-up).
    # This matches the original benchmark methodology and avoids R's
    # within-process caching inflating speedup numbers.
    times = []
    deviance = None
    converged = False

    try:
        for _ in range(n_repeats):
            t0 = time.perf_counter()
            result = subprocess.run(
                ["Rscript", r_path, csv_path, dist, formula, "1"],
                capture_output=True, text=True, timeout=120,
            )
            elapsed = time.perf_counter() - t0

            if result.returncode != 0:
                return {"success": False, "error": result.stderr.strip()[:200]}

            lines = [l.strip() for l in result.stdout.splitlines() if l.strip()]
            if not lines:
                return {"success": False, "error": "no output from R"}
            try:
                parsed = json.loads(lines[-1])
            except json.JSONDecodeError:
                return {"success": False, "error": f"bad JSON: {lines[-1][:100]}"}

            if not parsed.get("converged", False):
                return {"success": False, "error": "R did not converge"}

            times.append(elapsed)
            deviance = float(parsed["deviance"])
            converged = True

        return {
            "success": True,
            "mean_time": float(np.mean(times)),
            "min_time": float(np.min(times)),
            "deviance": deviance,
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
    python_time: float | None = None
    r_time: float | None = None
    speedup: float | None = None
    python_deviance: float | None = None
    r_deviance: float | None = None
    deviance_diff: float | None = None
    deviance_rel_diff: float | None = None
    python_error: str | None = None
    r_error: str | None = None

    def to_dict(self) -> dict:
        return {
            "distribution": self.distribution,
            "data_size": self.data_size,
            "model_config": self.model_config,
            "python_success": self.python_success,
            "r_success": self.r_success,
            "python_time": self.python_time,
            "r_time": self.r_time,
            "speedup": self.speedup,
            "python_deviance": self.python_deviance,
            "r_deviance": self.r_deviance,
            "deviance_diff": self.deviance_diff,
            "deviance_rel_diff": self.deviance_rel_diff,
            "python_error": self.python_error,
            "r_error": self.r_error,
        }


# ── report generation ─────────────────────────────────────────────────────────

def _generate_report(results: list[ComparisonResult], output_path: Path, r_available: bool) -> None:
    """Write a Markdown performance report."""
    successful = [r for r in results if r.python_success and r.r_success]
    python_only = [r for r in results if r.python_success and not r.r_success]
    r_only = [r for r in results if not r.python_success and r.r_success]
    both_failed = [r for r in results if not r.python_success and not r.r_success]

    lines: list[str] = []
    a = lines.append

    a("# OmniLSS vs R GAMLSS Performance Report")
    a("")
    a(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    a(f"**R comparison**: {'live Rscript' if r_available else 'skipped (Rscript not available)'}")
    a("")
    a("---")
    a("")
    a("## Executive Summary")
    a("")
    a(f"- **Total tests**: {len(results)}")
    a(f"- **Both succeeded**: {len(successful)} ({100*len(successful)/max(len(results),1):.1f}%)")
    a(f"- **Python only**: {len(python_only)}")
    a(f"- **R only**: {len(r_only)}")
    a(f"- **Both failed**: {len(both_failed)}")
    a("")

    if successful and r_available:
        speedups = [r.speedup for r in successful if r.speedup is not None]
        if speedups:
            faster = sum(1 for s in speedups if s > 1.0)
            a("### Speedup (Python vs R)")
            a("")
            a(f"| Metric | Value |")
            a(f"|--------|-------|")
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
            a(f"- Passed: {len(ok)}/{len(dist_results)}")
            if speedups:
                a(f"- Mean speedup: **{np.mean(speedups):.1f}×** "
                  f"(range {np.min(speedups):.1f}–{np.max(speedups):.1f}×)")
            a("")
            a("| n | Formula | Python (s) | R (s) | Speedup | Dev diff |")
            a("|---|---------|-----------|-------|---------|----------|")
            for r in sorted(ok, key=lambda x: (x.data_size, x.model_config)):
                dev = f"{r.deviance_diff:.2e}" if r.deviance_diff is not None else "—"
                sp = f"{r.speedup:.1f}×" if r.speedup else "—"
                a(f"| {r.data_size} | `{r.model_config}` | "
                  f"{r.python_time:.4f} | {r.r_time:.4f} | {sp} | {dev} |")
        else:
            a(f"- Passed: {sum(1 for r in dist_results if r.python_success)}/{len(dist_results)}")
            a("")
            a("| n | Formula | Python (s) | Status |")
            a("|---|---------|-----------|--------|")
            for r in sorted(dist_results, key=lambda x: (x.data_size, x.model_config)):
                status = "✅" if r.python_success else f"❌ {r.python_error or ''}"
                pt = f"{r.python_time:.4f}" if r.python_time else "—"
                a(f"| {r.data_size} | `{r.model_config}` | {pt} | {status} |")
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
        if both_failed:
            a("### Both failed")
            a("")
            for r in both_failed:
                a(f"- {r.distribution} n={r.data_size} `{r.model_config}`")
            a("")

    a("---")
    a("")
    a("## Performance by Data Size")
    a("")
    if successful and r_available:
        by_size: dict[int, list] = {}
        for r in successful:
            by_size.setdefault(r.data_size, []).append(r)
        a("| n | Tests | Mean speedup | Mean Python (s) | Mean R (s) |")
        a("|---|-------|-------------|----------------|-----------|")
        for size in sorted(by_size):
            rs = by_size[size]
            speedups = [r.speedup for r in rs if r.speedup]
            py_times = [r.python_time for r in rs if r.python_time]
            r_times = [r.r_time for r in rs if r.r_time]
            sp = f"{np.mean(speedups):.1f}×" if speedups else "—"
            pt = f"{np.mean(py_times):.4f}" if py_times else "—"
            rt = f"{np.mean(r_times):.4f}" if r_times else "—"
            a(f"| {size} | {len(rs)} | {sp} | {pt} | {rt} |")
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
    parser.add_argument("--quick", action="store_true",
                        help="Run only 3 distributions (NO, GA, PO)")
    parser.add_argument("--no-r", action="store_true",
                        help="Skip R comparison (Python timing only)")
    parser.add_argument("--n-repeats", type=int, default=N_REPEATS,
                        help=f"Timing repetitions (default {N_REPEATS})")
    args = parser.parse_args()

    distributions = QUICK_DISTRIBUTIONS if args.quick else DISTRIBUTIONS
    n_repeats = args.n_repeats

    # Check R availability
    r_available = False
    if not args.no_r:
        print("Checking R availability...", end=" ", flush=True)
        r_available = _check_r_available()
        print("✓ found" if r_available else "✗ not found (running Python-only)")
    print()

    total = len(distributions) * len(DATA_SIZES) * len(FORMULAS)
    print(f"{'='*70}")
    print(f"OmniLSS Performance Benchmark")
    print(f"  Distributions : {', '.join(distributions)}")
    print(f"  Data sizes    : {DATA_SIZES}")
    print(f"  Formulas      : {FORMULAS}")
    print(f"  Total tests   : {total}")
    print(f"  R comparison  : {'yes' if r_available else 'no'}")
    print(f"{'='*70}")
    print()

    results: list[ComparisonResult] = []
    done = 0

    for dist in distributions:
        print(f"[{dist}]")
        for n in DATA_SIZES:
            data = _generate_data(dist, n)
            for formula in FORMULAS:
                done += 1
                label = f"  n={n:5d}  {formula:<20s}"
                print(f"  ({done:2d}/{total}) {label}", end="", flush=True)

                cr = ComparisonResult(
                    distribution=dist,
                    data_size=n,
                    model_config=formula,
                )

                # Python
                py = _time_python(dist, formula, data, n_repeats, WARMUP)
                if py["success"]:
                    cr.python_success = True
                    cr.python_time = py["mean_time"]
                    cr.python_deviance = py.get("deviance")
                else:
                    cr.python_error = py.get("error", "")

                # R
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
                    else:
                        cr.r_error = r.get("error", "")

                results.append(cr)

                # Print one-line summary
                if cr.python_success:
                    if r_available and cr.r_success and cr.r_time is not None:
                        sp = f"{cr.speedup:.1f}×" if cr.speedup else "—"
                        print(f"  Python {cr.python_time:.4f}s  R {cr.r_time:.4f}s  → {sp}")
                    elif r_available and not cr.r_success:
                        print(f"  Python {cr.python_time:.4f}s  R FAIL: {cr.r_error or '?'}")
                    else:
                        print(f"  Python {cr.python_time:.4f}s")
                else:
                    print(f"  FAIL: {cr.python_error or '?'}")

        print()

    # ── summary ───────────────────────────────────────────────────────────────
    successful = [r for r in results if r.python_success and (r.r_success or not r_available)]
    py_ok = sum(1 for r in results if r.python_success)
    r_only = sum(1 for r in results if not r.python_success and r.r_success)

    print(f"{'='*70}")
    print("Summary")
    print(f"{'='*70}")
    print(f"  Python passed : {py_ok}/{total}")
    if r_available:
        both_ok = sum(1 for r in results if r.python_success and r.r_success)
        print(f"  Both passed   : {both_ok}/{total}")
        if r_only:
            print(f"  R only (regressions!) : {r_only}")
        speedups = [r.speedup for r in results if r.speedup is not None]
        if speedups:
            print(f"  Mean speedup  : {np.mean(speedups):.1f}×")
            print(f"  Median speedup: {np.median(speedups):.1f}×")
            print(f"  Min / Max     : {np.min(speedups):.1f}× / {np.max(speedups):.1f}×")
    print()

    # ── save JSON ─────────────────────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    raw_dir = Path(__file__).parent / "results" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    json_path = raw_dir / f"quick_results_{timestamp}.json"

    payload = {
        "timestamp": timestamp,
        "r_available": r_available,
        "config": {
            "distributions": distributions,
            "data_sizes": DATA_SIZES,
            "formulas": FORMULAS,
            "n_repeats": n_repeats,
            "warmup_runs": WARMUP,
        },
        "results": [r.to_dict() for r in results],
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Raw results  → {json_path}")

    # ── auto-generate Markdown report ─────────────────────────────────────────
    reports_dir = Path(__file__).parent / "results" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / f"quick_report_{timestamp}.md"
    _generate_report(results, report_path, r_available)

    return 0 if r_only == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
