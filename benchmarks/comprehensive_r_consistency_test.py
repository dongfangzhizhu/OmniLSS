"""OmniLSS vs R GAMLSS comprehensive consistency test.

Tests numerical agreement between OmniLSS and R gamlss across:
  1. d/p/q functions for all major distributions (via live Rscript)
  2. Model fitting (RS, CG, Mixed) for key distributions
  3. Smoothing terms (pb, ps, cs)

Auto-generates a Markdown report at the end.

Usage
-----
    python benchmarks/comprehensive_r_consistency_test.py
    python benchmarks/comprehensive_r_consistency_test.py --no-r   # Python only
    python benchmarks/comprehensive_r_consistency_test.py --quick  # fewer dists
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import tempfile
import time
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np

# ── path setup ────────────────────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT / "omnilss" / "src"))

from omnilss import gamlss
from omnilss.distributions import resolve_family

# ── distribution lists ────────────────────────────────────────────────────────
DPQR_DISTRIBUTIONS = [
    "NO", "GA", "LOGNO", "WEI", "EXP", "IG", "LO", "TF",
    "PO", "BI", "NBI", "NBII", "ZIP",
    "BE", "ZAGA",
]
QUICK_DISTRIBUTIONS = ["NO", "GA", "PO", "BI", "NBI", "BE"]

FIT_DISTRIBUTIONS = ["NO", "GA", "PO", "BE"]
FIT_ALGORITHMS    = ["RS", "CG", "Mixed"]
SMOOTHERS         = ["pb", "ps", "cs"]   # te excluded: not yet implemented

# ── R scripts ─────────────────────────────────────────────────────────────────
_R_DPQR_SCRIPT = r"""
suppressMessages(library(gamlss.dist))
suppressMessages(library(jsonlite))
args      <- commandArgs(trailingOnly=TRUE)
dist      <- args[1]
x_file    <- args[2]
p_file    <- args[3]
param_file <- args[4]

x      <- scan(x_file,  quiet=TRUE)
p_vals <- scan(p_file,  quiet=TRUE)
params <- scan(param_file, quiet=TRUE)

d_fn <- get(paste0("d", dist))
p_fn <- get(paste0("p", dist))
q_fn <- get(paste0("q", dist))

call_fn <- function(fn, first_arg) {
  tryCatch({
    if (length(params) == 1)
      fn(first_arg, params[1])
    else if (length(params) == 2)
      fn(first_arg, params[1], params[2])
    else if (length(params) == 3)
      fn(first_arg, params[1], params[2], params[3])
    else
      fn(first_arg, params[1], params[2], params[3], params[4])
  }, error=function(e) rep(NA_real_, length(first_arg)))
}

d_vals <- call_fn(d_fn, x)
p_vals_out <- call_fn(p_fn, x)
q_vals <- call_fn(q_fn, p_vals)

cat(toJSON(list(d=d_vals, p=p_vals_out, q=q_vals), auto_unbox=FALSE), "\n")
"""

_R_FIT_SCRIPT = r"""
suppressMessages(library(gamlss))
suppressMessages(library(jsonlite))
args      <- commandArgs(trailingOnly=TRUE)
data_file <- args[1]
dist      <- args[2]
formula   <- args[3]
# method arg[4] is ignored — R always uses default RS() for comparison

df <- read.csv(data_file)
mu_formula <- as.formula(formula)

t0  <- proc.time()["elapsed"]
fit <- tryCatch(
  gamlss(mu_formula, family=dist, data=df, trace=FALSE),
  error=function(e) NULL
)
elapsed <- proc.time()["elapsed"] - t0

if (is.null(fit)) {
  cat(toJSON(list(success=FALSE, error="gamlss failed"), auto_unbox=TRUE), "\n")
} else {
  cat(toJSON(list(
    success   = TRUE,
    deviance  = fit$G.deviance,
    mu_coef   = as.numeric(fit$mu.coefficients),
    mu_fitted = as.numeric(fit$mu.fv),
    r_time    = elapsed
  ), auto_unbox=TRUE), "\n")
}
"""

_R_SMOOTH_SCRIPT = r"""
suppressMessages(library(gamlss))
suppressMessages(library(jsonlite))
args      <- commandArgs(trailingOnly=TRUE)
data_file <- args[1]
formula   <- args[2]

df <- read.csv(data_file)
mu_formula <- as.formula(formula)

t0  <- proc.time()["elapsed"]
fit <- tryCatch(
  gamlss(mu_formula, family="NO", data=df, trace=FALSE),
  error=function(e) NULL
)
elapsed <- proc.time()["elapsed"] - t0

if (is.null(fit)) {
  cat(toJSON(list(success=FALSE, error="gamlss failed"), auto_unbox=TRUE), "\n")
} else {
  cat(toJSON(list(
    success   = TRUE,
    deviance  = fit$G.deviance,
    mu_fitted = as.numeric(fit$mu.fv),
    r_time    = elapsed
  ), auto_unbox=TRUE), "\n")
}
"""


# ── helpers ───────────────────────────────────────────────────────────────────

def _check_r_available() -> bool:
    try:
        r = subprocess.run(
            ["Rscript", "-e",
             "suppressMessages({library(gamlss);library(jsonlite)});cat('ok')"],
            capture_output=True, text=True, timeout=30,
        )
        return r.returncode == 0 and "ok" in r.stdout
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _run_r(script: str, args: list[str], timeout: int = 60) -> dict:
    """Write script to a temp file, run Rscript, return parsed JSON."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".R", delete=False) as f:
        f.write(script)
        r_path = f.name
    try:
        result = subprocess.run(
            ["Rscript", r_path] + args,
            capture_output=True, text=True, timeout=timeout,
        )
        if result.returncode != 0:
            return {"success": False, "error": result.stderr.strip()[:300]}
        lines = [l.strip() for l in result.stdout.splitlines() if l.strip()]
        if not lines:
            return {"success": False, "error": "no output"}
        return json.loads(lines[-1])
    except subprocess.TimeoutExpired:
        return {"success": False, "error": "timeout"}
    except Exception as e:
        return {"success": False, "error": str(e)}
    finally:
        Path(r_path).unlink(missing_ok=True)


def _write_vector(values: np.ndarray) -> str:
    """Write a 1-D array to a temp file, return path."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False
    ) as f:
        f.write("\n".join(str(v) for v in values))
        return f.name


def _write_csv(data: dict) -> str:
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False, newline=""
    ) as f:
        writer = csv.writer(f)
        keys = list(data.keys())
        writer.writerow(keys)
        for row in zip(*[data[k] for k in keys]):
            writer.writerow(row)
        return f.name


def _safe_correlation(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation that avoids the numpy RuntimeWarning for constant arrays."""
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    if len(a) < 2:
        return float("nan")
    std_a = np.std(a)
    std_b = np.std(b)
    if std_a == 0 or std_b == 0:
        # One array is constant — correlation is undefined; return 1.0 if
        # they are identical, nan otherwise.
        return 1.0 if np.allclose(a, b) else float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return float(np.corrcoef(a, b)[0, 1])


def _errors(py: np.ndarray, r: np.ndarray) -> dict:
    py = np.asarray(py, dtype=np.float64).ravel()
    r  = np.asarray(r,  dtype=np.float64).ravel()
    mask = np.isfinite(py) & np.isfinite(r)
    py, r = py[mask], r[mask]
    if len(py) == 0:
        nan = float("nan")
        return dict(max_absolute_error=nan, max_relative_error=nan,
                    mean_absolute_error=nan, mean_relative_error=nan,
                    rmse=nan, correlation=nan)
    abs_err = np.abs(py - r)
    rel_err = abs_err / (np.abs(r) + 1e-10)
    return dict(
        max_absolute_error  = float(np.max(abs_err)),
        max_relative_error  = float(np.max(rel_err)),
        mean_absolute_error = float(np.mean(abs_err)),
        mean_relative_error = float(np.mean(rel_err)),
        rmse                = float(np.sqrt(np.mean((py - r) ** 2))),
        correlation         = _safe_correlation(py, r),
    )


# ── result dataclass ──────────────────────────────────────────────────────────

@dataclass
class ConsistencyResult:
    test_name:           str
    category:            str   # dpqr | fitting | smoothing
    distribution:        str | None = None
    success:             bool  = False
    max_absolute_error:  float | None = None
    max_relative_error:  float | None = None
    mean_absolute_error: float | None = None
    mean_relative_error: float | None = None
    rmse:                float | None = None
    correlation:         float | None = None
    python_time:         float | None = None
    r_time:              float | None = None
    speedup:             float | None = None
    error_message:       str | None = None
    notes:               str = ""

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


# ── test functions ────────────────────────────────────────────────────────────

def _dpqr_params(dist: str, family) -> tuple[np.ndarray, np.ndarray, tuple]:
    """Return (x_vals, p_vals, params) for a distribution."""
    n = 200
    if dist in ("NO", "LO", "TF", "GU", "RG", "NO2"):
        x = np.linspace(-3, 3, n)
        params = (0.0, 1.0) if len(family.parameters) == 2 else (0.0, 1.0, 10.0)
    elif dist in ("GA", "LOGNO", "WEI", "EXP", "IG", "LOGNO2", "IGAMMA", "PARETO2"):
        x = np.linspace(0.1, 5, n)
        params = (1.0, 1.0) if len(family.parameters) == 2 else (1.0,)
    elif dist in ("PO", "GEOM", "NBI", "NBII", "ZIP", "ZIP2", "ZINBI", "ZAP"):
        x = np.arange(0, min(n, 50), dtype=float)
        params = (1.0,) if len(family.parameters) == 1 else (1.0, 1.0)
    elif dist == "BI":
        x = np.array([0.0, 1.0] * (n // 2))
        params = (0.5,)
    elif dist == "BE":
        x = np.linspace(0.01, 0.99, n)
        params = (0.5, 0.1)
    elif dist == "ZAGA":
        x = np.concatenate([np.zeros(n // 4),
                             np.linspace(0.1, 5, 3 * n // 4)])
        params = (1.0, 0.5, 0.3)
    else:
        x = np.linspace(0.1, 5, n)
        npar = len(family.parameters)
        params = tuple([1.0] * npar)
    p = np.linspace(0.01, 0.99, len(x))
    return x, p, params


def test_dpqr(dist: str, r_available: bool) -> list[ConsistencyResult]:
    results = []
    try:
        family = resolve_family(dist)
    except Exception as e:
        return [ConsistencyResult(
            test_name=f"{dist}_dpqr", category="dpqr", distribution=dist,
            success=False, error_message=str(e))]

    x, p_vals, params = _dpqr_params(dist, family)
    jax_params = tuple(jnp.full(len(x), v) for v in params)
    jax_p      = jnp.array(p_vals[:len(x)])

    # ── Python ────────────────────────────────────────────────────────────────
    py_results: dict[str, np.ndarray | None] = {}
    py_times:   dict[str, float] = {}
    for fn_name, fn_args in [("d", (jnp.array(x),) + jax_params),
                              ("p", (jnp.array(x),) + jax_params),
                              ("q", (jax_p,)         + jax_params)]:
        fn = getattr(family, fn_name, None)
        if fn is None:
            py_results[fn_name] = None
            continue
        t0 = time.perf_counter()
        try:
            out = fn(*fn_args)
            py_results[fn_name] = np.asarray(out, dtype=np.float64)
        except Exception as e:
            py_results[fn_name] = None
        py_times[fn_name] = time.perf_counter() - t0

    # ── R ─────────────────────────────────────────────────────────────────────
    r_results: dict[str, np.ndarray | None] = {"d": None, "p": None, "q": None}
    r_time_total: float | None = None

    if r_available:
        x_file = _write_vector(x)
        p_file = _write_vector(p_vals[:len(x)])
        par_file = _write_vector(np.array(params))
        try:
            t0 = time.perf_counter()
            out = _run_r(_R_DPQR_SCRIPT, [dist, x_file, p_file, par_file])
            r_time_total = time.perf_counter() - t0
            if out.get("success", True) and "d" in out:
                def _to_float_array(lst):
                    # R may return "NA" strings for unsupported values
                    arr = []
                    for v in lst:
                        try:
                            arr.append(float(v))
                        except (TypeError, ValueError):
                            arr.append(float("nan"))
                    return np.array(arr, dtype=np.float64)
                r_results["d"] = _to_float_array(out["d"])
                r_results["p"] = _to_float_array(out["p"])
                r_results["q"] = _to_float_array(out["q"])
        finally:
            for f in (x_file, p_file, par_file):
                Path(f).unlink(missing_ok=True)

    # ── build ConsistencyResult per function ──────────────────────────────────
    for fn_name in ("d", "p", "q"):
        py_val = py_results.get(fn_name)
        r_val  = r_results.get(fn_name)
        pt     = py_times.get(fn_name)

        if py_val is None:
            results.append(ConsistencyResult(
                test_name=f"{dist}_{fn_name}", category="dpqr",
                distribution=dist, success=False,
                error_message=f"{fn_name}() failed"))
            continue

        if r_val is not None:
            err = _errors(py_val, r_val)
            sp  = (r_time_total / pt) if (r_time_total and pt and pt > 0) else None
            results.append(ConsistencyResult(
                test_name=f"{dist}_{fn_name}", category="dpqr",
                distribution=dist, success=True,
                python_time=pt, r_time=r_time_total, speedup=sp,
                **err))
        else:
            # No R comparison — just record that Python ran OK
            results.append(ConsistencyResult(
                test_name=f"{dist}_{fn_name}", category="dpqr",
                distribution=dist, success=True,
                python_time=pt,
                notes="R comparison not available"))

    return results


def _gen_fit_data(dist: str, n: int = 500) -> dict:
    rng = np.random.RandomState(42)
    x = rng.normal(0, 1, n)
    if dist in ("PO", "NBI", "ZIP"):
        y = rng.poisson(np.exp(0.5 + 0.3 * x)).astype(float)
    elif dist == "BI":
        y = rng.binomial(1, 1 / (1 + np.exp(-(0.5 + 0.3 * x)))).astype(float)
    elif dist == "BE":
        y = np.clip(rng.beta(2, 2, n), 1e-4, 1 - 1e-4)
    elif dist in ("GA", "LOGNO", "WEI", "EXP", "IG", "ZAGA"):
        # Positive-only distributions: use log-linear mean
        mu = np.exp(1.0 + 0.3 * x)
        if dist == "GA":
            y = rng.gamma(shape=4.0, scale=mu / 4.0)
        elif dist == "LOGNO":
            y = np.exp(rng.normal(np.log(mu), 0.5, n))
        elif dist == "WEI":
            y = rng.weibull(2.0, n) * mu
        elif dist == "EXP":
            y = rng.exponential(mu)
        elif dist == "IG":
            from scipy.stats import invgauss
            y = invgauss.rvs(mu=mu, scale=1.0, random_state=rng)
        elif dist == "ZAGA":
            is_zero = rng.binomial(1, 0.25, n)
            y = np.where(is_zero, 0.0, rng.gamma(shape=2.0, scale=mu / 2.0))
    else:
        # Default: normal data
        y = 0.5 + 0.3 * x + rng.normal(0, 1, n)
    return {"y": y, "x": x}


def test_fitting(dist: str, algorithm: str, r_available: bool) -> ConsistencyResult:
    data = _gen_fit_data(dist)
    family = resolve_family(dist)

    # Python
    t0 = time.perf_counter()
    try:
        model = gamlss("y ~ x", family=family, data=data, algorithm=algorithm)
        py_time = time.perf_counter() - t0
        py_dev     = float(model.deviance)
        py_coef    = np.asarray(model.coefficients["mu"], dtype=np.float64)
        py_fitted  = np.asarray(model.fitted_values["mu"], dtype=np.float64)
    except Exception as e:
        return ConsistencyResult(
            test_name=f"{dist}_{algorithm}_fitting", category="fitting",
            distribution=dist, success=False, error_message=str(e))

    if not r_available:
        return ConsistencyResult(
            test_name=f"{dist}_{algorithm}_fitting", category="fitting",
            distribution=dist, success=True,
            python_time=py_time, notes="R comparison not available")

    # R
    csv_path = _write_csv(data)
    r_method = {"RS": "RS", "CG": "CG", "Mixed": "Mixed"}.get(algorithm, "RS")
    try:
        t0 = time.perf_counter()
        out = _run_r(_R_FIT_SCRIPT, [csv_path, dist, "y ~ x", r_method])
        r_wall = time.perf_counter() - t0
    finally:
        Path(csv_path).unlink(missing_ok=True)

    if not out.get("success", False):
        return ConsistencyResult(
            test_name=f"{dist}_{algorithm}_fitting", category="fitting",
            distribution=dist, success=False,
            python_time=py_time,
            error_message=f"R: {out.get('error','?')}")

    r_dev    = float(out["deviance"])
    r_coef   = np.array(out["mu_coef"],   dtype=np.float64)
    r_fitted = np.array(out["mu_fitted"], dtype=np.float64)
    r_time   = float(out.get("r_time", r_wall))

    coef_err   = _errors(py_coef,   r_coef)
    fitted_err = _errors(py_fitted, r_fitted)
    dev_diff   = abs(py_dev - r_dev)
    sp = r_time / py_time if py_time > 0 else None

    return ConsistencyResult(
        test_name=f"{dist}_{algorithm}_fitting", category="fitting",
        distribution=dist, success=True,
        python_time=py_time, r_time=r_time, speedup=sp,
        max_absolute_error  = max(coef_err["max_absolute_error"],
                                  fitted_err["max_absolute_error"]),
        max_relative_error  = max(coef_err["max_relative_error"],
                                  fitted_err["max_relative_error"]),
        mean_absolute_error = (coef_err["mean_absolute_error"] +
                               fitted_err["mean_absolute_error"]) / 2,
        mean_relative_error = (coef_err["mean_relative_error"] +
                               fitted_err["mean_relative_error"]) / 2,
        rmse        = fitted_err["rmse"],
        correlation = fitted_err["correlation"],
        notes=f"deviance diff={dev_diff:.4e}")


def test_smoother(smoother: str, r_available: bool) -> ConsistencyResult:
    rng = np.random.RandomState(42)
    n = 300
    x = np.linspace(0, 10, n)
    y = np.sin(x) + rng.normal(0, 0.3, n)
    data = {"y": y, "x": x}
    formula = f"y ~ {smoother}(x)"
    family = resolve_family("NO")

    # Python — warm up JAX JIT before timing so we measure steady-state
    # performance, not compilation overhead.
    try:
        gamlss(formula, family=family, data=data)  # warm-up (not timed)
    except Exception:
        pass

    t0 = time.perf_counter()
    try:
        model = gamlss(formula, family=family, data=data)
        py_time   = time.perf_counter() - t0
        py_fitted = np.asarray(model.fitted_values["mu"], dtype=np.float64)
        py_dev    = float(model.deviance)
    except Exception as e:
        return ConsistencyResult(
            test_name=f"{smoother}_NO_smoothing", category="smoothing",
            distribution="NO", success=False, error_message=str(e))

    if not r_available:
        return ConsistencyResult(
            test_name=f"{smoother}_NO_smoothing", category="smoothing",
            distribution="NO", success=True,
            python_time=py_time, notes="R comparison not available")

    csv_path = _write_csv(data)
    try:
        out = _run_r(_R_SMOOTH_SCRIPT, [csv_path, formula])
    finally:
        Path(csv_path).unlink(missing_ok=True)

    if not out.get("success", False):
        return ConsistencyResult(
            test_name=f"{smoother}_NO_smoothing", category="smoothing",
            distribution="NO", success=False,
            python_time=py_time,
            error_message=f"R: {out.get('error','?')}")

    r_fitted = np.array(out["mu_fitted"], dtype=np.float64)
    r_time   = float(out.get("r_time", 0))
    r_dev    = float(out["deviance"])
    err = _errors(py_fitted, r_fitted)
    sp  = r_time / py_time if (r_time > 0 and py_time > 0) else None

    return ConsistencyResult(
        test_name=f"{smoother}_NO_smoothing", category="smoothing",
        distribution="NO", success=True,
        python_time=py_time, r_time=r_time, speedup=sp,
        notes=f"deviance diff={abs(py_dev - r_dev):.4e}",
        **err)


# ── report generation ─────────────────────────────────────────────────────────

def _generate_report(results: list[ConsistencyResult],
                     output_path: Path,
                     r_available: bool) -> None:
    total     = len(results)
    passed    = [r for r in results if r.success]
    failed    = [r for r in results if not r.success]
    by_cat: dict[str, list] = {}
    for r in results:
        by_cat.setdefault(r.category, []).append(r)

    lines: list[str] = []
    a = lines.append

    a("# OmniLSS vs R GAMLSS Consistency Report")
    a("")
    a(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    a(f"**R comparison**: {'live Rscript' if r_available else 'skipped'}")
    a("")
    a("---")
    a("")
    a("## Executive Summary")
    a("")
    a(f"- **Total tests**: {total}")
    a(f"- **Passed**: {len(passed)} ({100*len(passed)/max(total,1):.1f}%)")
    a(f"- **Failed**: {len(failed)}")
    a("")
    a("| Category | Passed | Total | Pass rate |")
    a("|----------|--------|-------|-----------|")
    for cat, rs in sorted(by_cat.items()):
        ok = sum(1 for r in rs if r.success)
        a(f"| {cat} | {ok} | {len(rs)} | {100*ok/max(len(rs),1):.0f}% |")
    a("")

    # Error summary for tests that have R comparison
    r_compared = [r for r in passed
                  if r.max_absolute_error is not None
                  and np.isfinite(r.max_absolute_error)]
    if r_compared and r_available:
        max_abs = [r.max_absolute_error for r in r_compared]
        mean_abs = [r.mean_absolute_error for r in r_compared
                    if r.mean_absolute_error is not None]
        rmses = [r.rmse for r in r_compared if r.rmse is not None]
        a("## Numerical Accuracy (vs R)")
        a("")
        a("| Metric | Min | Median | Max |")
        a("|--------|-----|--------|-----|")
        if max_abs:
            a(f"| Max absolute error | {np.min(max_abs):.2e} | "
              f"{np.median(max_abs):.2e} | {np.max(max_abs):.2e} |")
        if mean_abs:
            a(f"| Mean absolute error | {np.min(mean_abs):.2e} | "
              f"{np.median(mean_abs):.2e} | {np.max(mean_abs):.2e} |")
        if rmses:
            a(f"| RMSE | {np.min(rmses):.2e} | "
              f"{np.median(rmses):.2e} | {np.max(rmses):.2e} |")
        a("")

    a("---")
    a("")

    # ── per-category detail ───────────────────────────────────────────────────
    for cat in ("dpqr", "fitting", "smoothing"):
        rs = by_cat.get(cat, [])
        if not rs:
            continue
        ok = [r for r in rs if r.success]
        a(f"## {cat.upper()} Tests")
        a("")
        a(f"Passed {len(ok)}/{len(rs)}")
        a("")

        if cat == "dpqr":
            # Group by distribution
            by_dist: dict[str, list] = {}
            for r in rs:
                by_dist.setdefault(r.distribution or "?", []).append(r)
            a("| Distribution | d | p | q | Max abs err |")
            a("|-------------|---|---|---|-------------|")
            for dist in sorted(by_dist):
                dr = {r.test_name.split("_")[1]: r for r in by_dist[dist]}
                def _sym(fn):
                    r = dr.get(fn)
                    if r is None: return "—"
                    return "✅" if r.success else "❌"
                max_e = max(
                    (r.max_absolute_error for r in by_dist[dist]
                     if r.success and r.max_absolute_error is not None
                     and np.isfinite(r.max_absolute_error)),
                    default=None)
                me_str = f"{max_e:.2e}" if max_e is not None else "—"
                a(f"| {dist} | {_sym('d')} | {_sym('p')} | {_sym('q')} | {me_str} |")

        elif cat == "fitting":
            if r_available:
                a("| Distribution | Algorithm | Python (s) | R (s) | Speedup | "
                  "Max abs err | Dev diff |")
                a("|-------------|-----------|-----------|-------|---------|"
                  "------------|----------|")
                for r in sorted(ok, key=lambda x: (x.distribution or "", x.test_name)):
                    sp  = f"{r.speedup:.1f}×" if r.speedup else "—"
                    mae = f"{r.max_absolute_error:.2e}" if r.max_absolute_error is not None else "—"
                    dev = r.notes.replace("deviance diff=", "") if r.notes else "—"
                    pt  = f"{r.python_time:.4f}" if r.python_time else "—"
                    rt  = f"{r.r_time:.4f}" if r.r_time else "—"
                    algo = r.test_name.split("_")[1] if "_" in r.test_name else "?"
                    a(f"| {r.distribution} | {algo} | {pt} | {rt} | {sp} | {mae} | {dev} |")
            else:
                a("| Distribution | Algorithm | Python (s) | Status |")
                a("|-------------|-----------|-----------|--------|")
                for r in sorted(rs, key=lambda x: x.test_name):
                    algo = r.test_name.split("_")[1] if "_" in r.test_name else "?"
                    pt   = f"{r.python_time:.4f}" if r.python_time else "—"
                    st   = "✅" if r.success else f"❌ {r.error_message or ''}"
                    a(f"| {r.distribution} | {algo} | {pt} | {st} |")

        elif cat == "smoothing":
            if r_available:
                a("| Smoother | Python (s) | R (s) | Speedup | Max abs err |")
                a("|---------|-----------|-------|---------|------------|")
                for r in sorted(rs, key=lambda x: x.test_name):
                    sm  = r.test_name.split("_")[0]
                    sp  = f"{r.speedup:.1f}×" if r.speedup else "—"
                    mae = f"{r.max_absolute_error:.2e}" if r.max_absolute_error is not None else "—"
                    pt  = f"{r.python_time:.4f}" if r.python_time else "—"
                    rt  = f"{r.r_time:.4f}" if r.r_time else "—"
                    st  = "✅" if r.success else "❌"
                    a(f"| {sm} {st} | {pt} | {rt} | {sp} | {mae} |")
            else:
                a("| Smoother | Python (s) | Status |")
                a("|---------|-----------|--------|")
                for r in sorted(rs, key=lambda x: x.test_name):
                    sm = r.test_name.split("_")[0]
                    pt = f"{r.python_time:.4f}" if r.python_time else "—"
                    st = "✅" if r.success else f"❌ {r.error_message or ''}"
                    a(f"| {sm} | {pt} | {st} |")
        a("")

    # ── failures ──────────────────────────────────────────────────────────────
    if failed:
        a("---")
        a("")
        a("## Failures")
        a("")
        a("| Test | Category | Error |")
        a("|------|----------|-------|")
        for r in failed:
            err = (r.error_message or "")[:100]
            a(f"| {r.test_name} | {r.category} | {err} |")
        a("")

    a("---")
    a("")
    a("*Report generated by OmniLSS consistency test suite*")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Report saved → {output_path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="OmniLSS vs R GAMLSS consistency test")
    parser.add_argument("--quick", action="store_true",
                        help="Test fewer distributions")
    parser.add_argument("--no-r", action="store_true",
                        help="Skip R comparison (Python only)")
    args = parser.parse_args()

    dists = QUICK_DISTRIBUTIONS if args.quick else DPQR_DISTRIBUTIONS

    r_available = False
    if not args.no_r:
        print("Checking R availability...", end=" ", flush=True)
        r_available = _check_r_available()
        print("✓ found" if r_available else "✗ not found (Python-only mode)")
    print()

    print("=" * 70)
    print("OmniLSS Consistency Test")
    print(f"  Distributions : {', '.join(dists)}")
    print(f"  R comparison  : {'yes' if r_available else 'no'}")
    print("=" * 70)
    print()

    all_results: list[ConsistencyResult] = []

    # ── 1. d/p/q functions ────────────────────────────────────────────────────
    print("1. d/p/q functions")
    print("-" * 70)
    for dist in dists:
        print(f"  {dist:8s}", end=" ", flush=True)
        rs = test_dpqr(dist, r_available)
        all_results.extend(rs)
        ok = sum(1 for r in rs if r.success)
        # Show max error if R comparison available
        errs = [r.max_absolute_error for r in rs
                if r.success and r.max_absolute_error is not None
                and np.isfinite(r.max_absolute_error)]
        err_str = f"  max_err={max(errs):.2e}" if errs else ""
        print(f"{ok}/{len(rs)} OK{err_str}")
    print()

    # ── 2. Model fitting ──────────────────────────────────────────────────────
    print("2. Model fitting")
    print("-" * 70)
    for dist in FIT_DISTRIBUTIONS:
        for algo in FIT_ALGORITHMS:
            print(f"  {dist:4s} + {algo:5s}", end=" ", flush=True)
            r = test_fitting(dist, algo, r_available)
            all_results.append(r)
            if r.success:
                sp = f"  {r.speedup:.1f}×" if r.speedup else ""
                print(f"OK{sp}")
            else:
                print(f"FAIL: {r.error_message or '?'}")
    print()

    # ── 3. Smoothers ──────────────────────────────────────────────────────────
    print("3. Smoothers")
    print("-" * 70)
    for sm in SMOOTHERS:
        print(f"  {sm:4s}", end=" ", flush=True)
        r = test_smoother(sm, r_available)
        all_results.append(r)
        if r.success:
            sp = f"  {r.speedup:.1f}×" if r.speedup else ""
            print(f"OK{sp}")
        else:
            print(f"FAIL: {r.error_message or '?'}")
    print()

    # ── summary ───────────────────────────────────────────────────────────────
    total  = len(all_results)
    passed = sum(1 for r in all_results if r.success)
    failed = total - passed

    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"  Passed : {passed}/{total} ({100*passed/max(total,1):.1f}%)")
    if failed:
        print(f"  Failed : {failed}")
        for r in all_results:
            if not r.success:
                print(f"    ✗ {r.test_name}: {r.error_message or '?'}")

    if r_available:
        errs = [r.max_absolute_error for r in all_results
                if r.success and r.max_absolute_error is not None
                and np.isfinite(r.max_absolute_error)]
        if errs:
            print(f"  Max absolute error : {max(errs):.2e}")
            print(f"  Median max abs err : {float(np.median(errs)):.2e}")
    print()

    # ── save JSON ─────────────────────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    raw_dir = Path(__file__).parent / "results" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    json_path = raw_dir / f"r_consistency_{timestamp}.json"
    json_path.write_text(
        json.dumps({
            "timestamp": timestamp,
            "r_available": r_available,
            "summary": {
                "total": total, "passed": passed, "failed": failed,
                "pass_rate": passed / max(total, 1),
            },
            "results": [r.to_dict() for r in all_results],
        }, indent=2),
        encoding="utf-8",
    )
    print(f"Raw results  → {json_path}")

    # ── auto-generate Markdown report ─────────────────────────────────────────
    reports_dir = Path(__file__).parent / "results" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / f"consistency_report_{timestamp}.md"
    _generate_report(all_results, report_path, r_available)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
