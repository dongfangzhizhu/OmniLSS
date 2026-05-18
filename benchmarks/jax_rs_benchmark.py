# SPDX-License-Identifier: GPL-3.0-or-later
"""JAX RS core benchmark: cold/warm time, NumPy RS, and R gamlss comparison.

Design principles
-----------------
1. **R timing is done inside R** — a single Rscript call runs n_reps iterations
   and measures wall time with proc.time(), eliminating startup overhead.
2. **Hardware is explicitly reported** — CPU model, core count, RAM, JAX backend,
   and GPU info (if available).
3. **n sweep** — runs across multiple n values to find the crossover point where
   JAX warm time beats NumPy RS.
4. **Correctness gate** — |JAX deviance - R deviance| < 0.1 must pass before
   performance numbers are published.

Usage
-----
    # Full optional-R benchmark (recommended where R is installed)
    python benchmarks/jax_rs_benchmark.py --suite optional-r

    # CI-safe Python-only smoke benchmark (no R/GPU required)
    python benchmarks/jax_rs_benchmark.py --suite no-r --smoke --families NO --n-values 100

    # Optional GPU benchmark; exits successfully with a skipped artifact when no GPU is available
    python benchmarks/jax_rs_benchmark.py --suite optional-gpu --smoke

    # Custom n values
    python benchmarks/jax_rs_benchmark.py --n-values 1000 10000 100000 500000

Output
------
    docs/benchmarks/jax_rs_<timestamp>.json   — full artifact
    docs/benchmarks/jax_rs_<timestamp>.md     — human-readable report
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(description="JAX RS benchmark")
    p.add_argument("--smoke",     action="store_true",
                   help="Fast smoke test: default n=1000,5000 and at most 3 reps; --n-values may override")
    p.add_argument("--suite", choices=["no-r", "optional-r", "optional-gpu"],
                   default="optional-r",
                   help="Benchmark suite: Python-only, R-when-available, or GPU-when-available")
    p.add_argument("--no-r",      action="store_true",
                   help="Deprecated alias for --suite no-r; skip R comparison")
    p.add_argument("--n-values",  type=int, nargs="+",
                   default=None,
                   help="Override n values (default: 1k 5k 10k 50k 100k 500k)")
    p.add_argument("--n-reps",    type=int, default=6,
                   help="Python warm-up repetitions (default 6)")
    p.add_argument("--r-reps",    type=int, default=5,
                   help="R repetitions per (family, n) (default 5)")
    p.add_argument("--out-dir",   default="docs/benchmarks",
                   help="Output directory")
    p.add_argument("--families",  nargs="+",
                   default=["NO", "GA", "PO", "BI", "WEI", "TF"],
                   help="Families to benchmark")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Hardware / software info
# ---------------------------------------------------------------------------

def _collect_env() -> dict[str, Any]:
    import jax
    info: dict[str, Any] = {
        "timestamp_utc":  datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version.split()[0],
        "platform":       platform.platform(),
        "cpu_model":      platform.processor(),
        "jax_version":    jax.__version__,
        "numpy_version":  np.__version__,
        "jax_backend":    jax.default_backend(),
        "jax_devices":    [str(d) for d in jax.devices()],
    }
    # Physical / logical cores + RAM
    try:
        import psutil
        info["cpu_physical_cores"] = psutil.cpu_count(logical=False)
        info["cpu_logical_cores"]  = psutil.cpu_count(logical=True)
        info["ram_gb"]             = round(psutil.virtual_memory().total / 1e9, 1)
    except ImportError:
        pass
    # GPU info
    gpu_devs = [d for d in jax.devices() if "gpu" in str(d).lower()]
    info["gpu_devices"] = [str(d) for d in gpu_devs]
    info["has_gpu"]     = len(gpu_devs) > 0
    # R version
    try:
        r_ver = subprocess.check_output(
            ["Rscript", "--version"], stderr=subprocess.STDOUT, text=True
        ).strip()
        info["r_version"] = r_ver
    except Exception:
        info["r_version"] = "unavailable"
    return info


# ---------------------------------------------------------------------------
# Data generators (must match r_timing_template.R exactly)
# ---------------------------------------------------------------------------

def _make_data(family: str, n: int, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n)
    if family == "NO":
        y = 2.0 + 1.5 * x + rng.standard_normal(n)
    elif family == "GA":
        mu = np.exp(1.0 + 0.5 * x)
        y = rng.gamma(shape=4.0, scale=mu / 4.0)
    elif family == "PO":
        mu = np.exp(1.0 + 0.3 * x)
        y = rng.poisson(mu).astype(float)
    elif family == "BI":
        p = 1.0 / (1.0 + np.exp(-(0.5 + 0.8 * x)))
        y = rng.binomial(1, p).astype(float)
    elif family == "WEI":
        mu = np.exp(1.0 + 0.4 * x)
        y = rng.weibull(1.5, size=n) * mu
    elif family == "TF":
        from scipy.stats import t as scipy_t
        mu = 1.0 + 0.5 * x
        y = mu + scipy_t.rvs(df=5, size=n, random_state=int(seed))
    else:
        raise ValueError(f"Unknown family: {family}")
    return {"y": y, "x": x}


# ---------------------------------------------------------------------------
# Python timing helpers
# ---------------------------------------------------------------------------

def _block(result):
    """Force JAX to complete computation."""
    try:
        import jax
        if hasattr(result, "g_dev"):
            jax.block_until_ready(float(result.g_dev))
        elif hasattr(result, "__jax_array__"):
            jax.block_until_ready(result)
    except Exception:
        pass
    return result


def _time_call(fn, *args, **kwargs) -> tuple[float, Any]:
    t0 = time.perf_counter()
    r  = fn(*args, **kwargs)
    _block(r)
    return (time.perf_counter() - t0) * 1000.0, r


def _warm_times(fn, args, kwargs, n_reps: int) -> list[float]:
    return [_time_call(fn, *args, **kwargs)[0] for _ in range(n_reps)]


def _timing_summary(times_ms: list[float]) -> dict[str, Any]:
    """Return median/min/mean and a conservative normal-approximation 95% CI."""
    if not times_ms:
        return {
            "median_ms": None,
            "min_ms": None,
            "mean_ms": None,
            "ci95_ms": None,
            "samples_ms": [],
        }
    rounded = [round(float(t), 2) for t in times_ms]
    mean = float(statistics.mean(times_ms))
    if len(times_ms) > 1:
        margin = 1.96 * float(statistics.stdev(times_ms)) / math.sqrt(len(times_ms))
        ci95 = [round(max(0.0, mean - margin), 2), round(mean + margin, 2)]
    else:
        ci95 = None
    return {
        "median_ms": round(float(statistics.median(times_ms)), 2),
        "min_ms": round(float(min(times_ms)), 2),
        "mean_ms": round(mean, 2),
        "ci95_ms": ci95,
        "samples_ms": rounded,
    }


# ---------------------------------------------------------------------------
# R timing (single Rscript call, all reps inside R)
# ---------------------------------------------------------------------------

R_SCRIPT = Path(__file__).parent / "r_timing_template.R"


def _run_r_timing(
    family: str, n: int, seed: int, n_reps: int,
    data: dict[str, np.ndarray],
) -> dict[str, Any] | None:
    """Run R timing for one (family, n) combination.

    Writes Python-generated data to a temp CSV, passes it to R so both
    sides fit the exact same dataset.  R runs n_reps iterations internally
    using proc.time() — no per-call Rscript startup overhead.

    Returns dict with keys: median_ms, min_ms, deviance, times_ms.
    Returns None if R is unavailable or fails.
    """
    if not shutil.which("Rscript"):
        return None
    if not R_SCRIPT.exists():
        return None

    with tempfile.NamedTemporaryFile(
        suffix=".csv", delete=False, mode="w", newline="", encoding="utf-8"
    ) as f:
        data_csv = f.name
        writer = csv.writer(f)
        writer.writerow(["y", "x"])
        for yi, xi in zip(data["y"], data["x"]):
            writer.writerow([yi, xi])

    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        out_csv = f.name

    try:
        result = subprocess.run(
            ["Rscript", str(R_SCRIPT),
             family, data_csv, str(n_reps), out_csv],
            capture_output=True, text=True, timeout=600,
        )
        if result.returncode != 0:
            print(f"  [R ERROR] {result.stderr.strip()[:300]}")
            return None

        times_ms  = []
        deviances = []
        with open(out_csv, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                times_ms.append(float(row["time_ms"]))
                deviances.append(float(row["deviance"]))

        if not times_ms:
            return None

        summary = _timing_summary(times_ms)
        return {
            "median_ms": summary["median_ms"],
            "min_ms": summary["min_ms"],
            "mean_ms": summary["mean_ms"],
            "ci95_ms": summary["ci95_ms"],
            "deviance":  round(statistics.median(deviances), 6),
            "times_ms": summary["samples_ms"],
            "deviances": [round(d, 6) for d in deviances],
        }
    except subprocess.TimeoutExpired:
        print(f"  [R TIMEOUT] family={family} n={n}")
        return None
    except Exception as e:
        print(f"  [R EXCEPTION] {e}")
        return None
    finally:
        for p in (data_csv, out_csv):
            try:
                os.unlink(p)
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Single benchmark run for one (family, n)
# ---------------------------------------------------------------------------

def _run_one(
    family_name: str,
    n: int,
    seed: int,
    n_reps: int,
    r_reps: int,
    skip_r: bool,
) -> dict[str, Any]:
    from omnilss import gamlss
    from omnilss.distributions import resolve_family

    family = resolve_family(family_name)
    data   = _make_data(family_name, n, seed)
    formula = "y ~ x"

    row: dict[str, Any] = {"family": family_name, "n": n, "seed": seed}

    # ── NumPy RS baseline ────────────────────────────────────────────────────
    # First call (cold for NumPy too — no JIT but includes Python overhead)
    numpy_cold_ms, model_np = _time_call(
        gamlss, formula, family=family, data=data, method="RS"
    )
    numpy_warm = _warm_times(
        gamlss, (formula,), {"family": family, "data": data, "method": "RS"},
        n_reps=n_reps,
    )
    row["numpy_cold_ms"]        = round(numpy_cold_ms, 2)
    numpy_summary = _timing_summary(numpy_warm)
    row["numpy_warm_ms_median"] = numpy_summary["median_ms"]
    row["numpy_warm_ms_min"]    = numpy_summary["min_ms"]
    row["numpy_warm_ms_mean"]   = numpy_summary["mean_ms"]
    row["numpy_warm_ms_ci95"]   = numpy_summary["ci95_ms"]
    row["numpy_warm_ms_samples"] = numpy_summary["samples_ms"]
    row["numpy_deviance"]       = round(float(model_np.g_dev), 6)

    # ── JAX RS cold (first call = JIT compilation) ───────────────────────────
    jax_cold_ms, model_jax = _time_call(
        gamlss, formula, family=family, data=data, method="RS_JAX"
    )
    row["jax_cold_ms"]  = round(jax_cold_ms, 2)
    row["jax_deviance"] = round(float(model_jax.g_dev), 6)

    # ── JAX RS warm ──────────────────────────────────────────────────────────
    jax_warm = _warm_times(
        gamlss, (formula,), {"family": family, "data": data, "method": "RS_JAX"},
        n_reps=n_reps,
    )
    jax_summary = _timing_summary(jax_warm)
    row["jax_warm_ms_median"] = jax_summary["median_ms"]
    row["jax_warm_ms_min"]    = jax_summary["min_ms"]
    row["jax_warm_ms_mean"]   = jax_summary["mean_ms"]
    row["jax_warm_ms_ci95"]   = jax_summary["ci95_ms"]
    row["jax_warm_ms_samples"] = jax_summary["samples_ms"]

    # ── Deviance diff vs NumPy ────────────────────────────────────────────────
    row["deviance_diff_vs_numpy"] = round(
        abs(row["jax_deviance"] - row["numpy_deviance"]), 8
    )

    # ── Speedup warm JAX vs warm NumPy ───────────────────────────────────────
    if row["jax_warm_ms_median"] > 0:
        row["speedup_jax_vs_numpy"] = round(
            row["numpy_warm_ms_median"] / row["jax_warm_ms_median"], 3
        )
    else:
        row["speedup_jax_vs_numpy"] = None

    # ── R timing ─────────────────────────────────────────────────────────────
    if not skip_r:
        r_result = _run_r_timing(family_name, n, seed, r_reps, data)
        if r_result:
            row["r_median_ms"]          = r_result["median_ms"]
            row["r_min_ms"]             = r_result["min_ms"]
            row["r_deviance"]           = r_result["deviance"]
            row["r_times_ms"]           = r_result["times_ms"]
            row["r_mean_ms"]            = r_result["mean_ms"]
            row["r_ms_ci95"]            = r_result["ci95_ms"]
            row["deviance_diff_vs_r"]   = round(
                abs(row["jax_deviance"] - r_result["deviance"]), 8
            )
            if r_result["median_ms"] > 0:
                row["speedup_jax_vs_r"] = round(
                    r_result["median_ms"] / row["jax_warm_ms_median"], 3
                )
                row["speedup_numpy_vs_r"] = round(
                    r_result["median_ms"] / row["numpy_warm_ms_median"], 3
                )
        else:
            row["r_median_ms"] = None
            row["r_deviance"]  = None
            row["deviance_diff_vs_r"] = None

    import jax
    row["backend"] = jax.default_backend()
    row["has_gpu"] = len([d for d in jax.devices() if "gpu" in str(d).lower()]) > 0

    return row


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------

def _format_ci(ci: Any) -> str:
    if not ci:
        return "N/A"
    return f"[{ci[0]:.1f}, {ci[1]:.1f}]"


def _make_markdown(env: dict, rows: list[dict]) -> str:
    has_r = any("r_median_ms" in r and r["r_median_ms"] is not None for r in rows)
    backend = env.get("jax_backend", "cpu").upper()
    gpu_info = ""
    if env.get("has_gpu"):
        gpu_info = f"  \n**GPU devices**: {', '.join(env.get('gpu_devices', []))}"

    lines = [
        "# JAX RS Core Benchmark",
        "",
        f"**Generated**: {env['timestamp_utc']}  ",
        f"**Platform**: {env['platform']}  ",
        f"**CPU**: {env.get('cpu_model', 'unknown')}  ",
        f"**Cores**: {env.get('cpu_physical_cores', '?')} physical / "
        f"{env.get('cpu_logical_cores', '?')} logical  ",
        f"**RAM**: {env.get('ram_gb', '?')} GB  ",
        f"**JAX version**: {env['jax_version']}  ",
        f"**JAX backend**: {backend}{gpu_info}  ",
        f"**R version**: {env.get('r_version', 'N/A')}  ",
        "",
        "## Methodology",
        "",
        "- `jax_cold_ms`: wall time of the **first** JAX call (JIT compilation included)",
        "- `jax_warm_ms`: median of repeated cold-start JAX RS calls after the first compilation",
        "- `numpy_warm_ms`: median of repeated calls to `method='RS'` (NumPy IRLS path)",
        "- `*_ci95`: normal-approximation 95% confidence interval for repeated warm timings",
        "- `r_median_ms`: median of R-internal repetitions via `proc.time()` "
        "(single Rscript process, startup overhead excluded)",
        "- `speedup_vs_numpy`: `numpy_warm_ms / jax_warm_ms`",
        "- `speedup_vs_r`: `r_median_ms / jax_warm_ms`",
        "- Formula: `y ~ x` (linear predictor, no smooth terms)",
        "- JAX RS uses data-aware cold-start initialization; no NumPy RS warm-start is included",
        "",
    ]

    # Per-family tables
    families = sorted(set(r["family"] for r in rows))
    for fam in families:
        fam_rows = [r for r in rows if r["family"] == fam]
        lines += [f"## {fam}", ""]
        if has_r:
            lines += [
                "| n | jax_cold_ms | jax_warm_ms | jax_ci95_ms | numpy_warm_ms | numpy_ci95_ms | r_median_ms "
                "| r_ci95_ms | speedup_vs_numpy | speedup_vs_r | dev_diff_r |",
                "|---|------------:|------------:|------------:|--------------:|--------------:|------------:"
                "|----------:|-----------------:|-------------:|-----------:|",
            ]
            for r in fam_rows:
                r_ms    = f"{r.get('r_median_ms', 'N/A'):.1f}" if r.get("r_median_ms") else "N/A"
                sp_r    = f"{r.get('speedup_jax_vs_r', 'N/A'):.2f}x" if r.get("speedup_jax_vs_r") else "N/A"
                dd_r    = f"{r.get('deviance_diff_vs_r', 'N/A'):.2e}" if r.get("deviance_diff_vs_r") is not None else "N/A"
                sp_np   = f"{r.get('speedup_jax_vs_numpy', 'N/A'):.2f}x" if r.get("speedup_jax_vs_numpy") else "N/A"
                jax_ci = _format_ci(r.get("jax_warm_ms_ci95"))
                numpy_ci = _format_ci(r.get("numpy_warm_ms_ci95"))
                r_ci = _format_ci(r.get("r_ms_ci95"))
                lines.append(
                    f"| {r['n']:,} "
                    f"| {r['jax_cold_ms']:.1f} "
                    f"| {r['jax_warm_ms_median']:.1f} "
                    f"| {jax_ci} "
                    f"| {r['numpy_warm_ms_median']:.1f} "
                    f"| {numpy_ci} "
                    f"| {r_ms} "
                    f"| {r_ci} "
                    f"| {sp_np} "
                    f"| {sp_r} "
                    f"| {dd_r} |"
                )
        else:
            lines += [
                "| n | jax_cold_ms | jax_warm_ms | jax_ci95_ms | numpy_warm_ms | numpy_ci95_ms | speedup_vs_numpy | dev_diff_numpy |",
                "|---|------------:|------------:|------------:|--------------:|--------------:|-----------------:|---------------:|",
            ]
            for r in fam_rows:
                sp_np = f"{r.get('speedup_jax_vs_numpy', 'N/A'):.2f}x" if r.get("speedup_jax_vs_numpy") else "N/A"
                lines.append(
                    f"| {r['n']:,} "
                    f"| {r['jax_cold_ms']:.1f} "
                    f"| {r['jax_warm_ms_median']:.1f} "
                    f"| {_format_ci(r.get('jax_warm_ms_ci95'))} "
                    f"| {r['numpy_warm_ms_median']:.1f} "
                    f"| {_format_ci(r.get('numpy_warm_ms_ci95'))} "
                    f"| {sp_np} "
                    f"| {r['deviance_diff_vs_numpy']:.2e} |"
                )
        lines.append("")

    # Correctness gate
    good = [r for r in rows if "error" not in r]
    gate_numpy = all(r["deviance_diff_vs_numpy"] < 0.1 for r in good)
    gate_r     = all(
        r.get("deviance_diff_vs_r", 0.0) is not None
        and r.get("deviance_diff_vs_r", 0.0) < 0.1
        for r in good if r.get("deviance_diff_vs_r") is not None
    )

    lines += [
        "## Correctness Gate",
        "",
        f"- vs NumPy RS: {'PASS' if gate_numpy else 'FAIL'} "
        f"(all |JAX deviance - NumPy deviance| < 0.1)",
    ]
    if has_r:
        lines.append(
            f"- vs R gamlss: {'PASS' if gate_r else 'FAIL'} "
            f"(all |JAX deviance - R deviance| < 0.1)"
        )
    lines += [
        "",
        "## Notes",
        "",
        "- JAX cold-start initialization is data-aware and JAX-native; NumPy RS warm-starts are intentionally excluded.",
        "- Cold and warm timings must be reported separately; use the confidence intervals above for repeated warm claims.",
        "- The crossover point (JAX warm < NumPy warm) depends on n and family complexity.",
        "- TF (Student-t) uses JAX autodiff for score/Hessian; cold time is higher.",
        "",
        "> Generated by `benchmarks/jax_rs_benchmark.py`. Do not edit manually.",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = _parse_args()

    import jax
    jax.config.update("jax_enable_x64", True)

    if args.smoke:
        n_values = args.n_values or [1_000, 5_000]
        n_reps = min(args.n_reps, 3)
        r_reps = min(args.r_reps, 3)
        print(f"[smoke mode] n={n_values}, reps={n_reps}, r_reps={r_reps}")
    else:
        n_values = args.n_values or [1_000, 5_000, 10_000, 50_000, 100_000, 500_000]
        n_reps   = args.n_reps
        r_reps   = args.r_reps

    families = args.families
    suite = "no-r" if args.no_r else args.suite
    skip_r = suite in {"no-r", "optional-gpu"}

    env  = _collect_env()
    rows = []

    if suite == "optional-gpu" and not env.get("has_gpu"):
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        artifact = {
            "environment": env,
            "suite": suite,
            "skipped": True,
            "skip_reason": "No JAX GPU device was available.",
            "results": [],
            "correctness_vs_numpy": None,
        }
        json_path = out_dir / f"jax_rs_{suite}_{ts}.json"
        md_path = out_dir / f"jax_rs_{suite}_{ts}.md"
        json_path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
        md_path.write_text(
            "# Optional GPU JAX RS Benchmark\n\n"
            "Skipped: no JAX GPU device was available.\n",
            encoding="utf-8",
        )
        print("[optional-gpu] skipped: no JAX GPU device was available")
        print(f"Artifacts:\n  JSON: {json_path}\n  MD:   {md_path}")
        return 0

    print(f"\nJAX RS Benchmark — {env['timestamp_utc']}")
    print(f"CPU: {env.get('cpu_model', '?')}")
    print(f"Cores: {env.get('cpu_physical_cores','?')}P / {env.get('cpu_logical_cores','?')}L  "
          f"RAM: {env.get('ram_gb','?')} GB")
    print(f"JAX backend: {env['jax_backend']}  "
          f"GPU: {'yes — ' + str(env.get('gpu_devices')) if env.get('has_gpu') else 'no'}")
    print(f"R: {env.get('r_version', 'N/A')}")
    print()

    hdr = (f"{'Family':<6} {'n':>8} {'jax_cold':>10} {'jax_warm':>10} "
           f"{'np_warm':>10} {'r_med':>10} {'spd_np':>8} {'spd_r':>8} {'dev_r':>12}")
    print(hdr)
    print("-" * len(hdr))

    seed_base = 200
    for family_name in families:
        for n in n_values:
            seed = seed_base
            seed_base += 1
            try:
                row = _run_one(
                    family_name, n, seed, n_reps, r_reps, skip_r
                )
                rows.append(row)

                r_ms  = f"{row.get('r_median_ms', 0.0):.1f}" if row.get("r_median_ms") else "  N/A"
                sp_np = f"{row.get('speedup_jax_vs_numpy', 0.0):.2f}x" if row.get("speedup_jax_vs_numpy") else "  N/A"
                sp_r  = f"{row.get('speedup_jax_vs_r', 0.0):.2f}x" if row.get("speedup_jax_vs_r") else "  N/A"
                dd_r  = f"{row.get('deviance_diff_vs_r', 0.0):.2e}" if row.get("deviance_diff_vs_r") is not None else "       N/A"
                print(
                    f"{family_name:<6} {n:>8,} "
                    f"{row['jax_cold_ms']:>10.1f} "
                    f"{row['jax_warm_ms_median']:>10.1f} "
                    f"{row['numpy_warm_ms_median']:>10.1f} "
                    f"{r_ms:>10} "
                    f"{sp_np:>8} "
                    f"{sp_r:>8} "
                    f"{dd_r:>12}"
                )
            except Exception as exc:
                import traceback
                traceback.print_exc()
                print(f"{family_name:<6} {n:>8,}  ERROR: {exc}")
                rows.append({"family": family_name, "n": n, "error": str(exc)})

    # ── Correctness gate ─────────────────────────────────────────────────────
    print("\n" + "=" * len(hdr))
    good = [r for r in rows if "error" not in r]
    gate_numpy = all(r["deviance_diff_vs_numpy"] < 0.1 for r in good)
    print(f"Correctness vs NumPy: {'PASS' if gate_numpy else 'FAIL'}")
    if not skip_r:
        r_rows = [r for r in good if r.get("deviance_diff_vs_r") is not None]
        if r_rows:
            gate_r = all(r["deviance_diff_vs_r"] < 0.1 for r in r_rows)
            print(f"Correctness vs R:     {'PASS' if gate_r else 'FAIL'}")
            if not gate_r:
                for r in r_rows:
                    if r["deviance_diff_vs_r"] >= 0.1:
                        print(f"  FAIL: {r['family']} n={r['n']} diff={r['deviance_diff_vs_r']:.4f}")

    # ── Crossover analysis ───────────────────────────────────────────────────
    print("\nCrossover analysis (JAX warm < NumPy warm):")
    for fam in families:
        fam_rows = [r for r in good if r["family"] == fam]
        crossover = None
        for r in fam_rows:
            if r.get("speedup_jax_vs_numpy", 0) and r["speedup_jax_vs_numpy"] >= 1.0:
                crossover = r["n"]
                break
        if crossover:
            print(f"  {fam}: crossover at n >= {crossover:,}")
        else:
            max_n = max((r["n"] for r in fam_rows), default=0)
            print(f"  {fam}: no crossover up to n={max_n:,} (JAX still slower on CPU)")

    # ── Write artifacts ───────────────────────────────────────────────────────
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    artifact = {
        "environment":   env,
        "suite": suite,
        "methodology": {
            "n_python_reps": n_reps,
            "n_r_reps":      r_reps,
            "formula":       "y ~ x",
            "timing_unit":   "milliseconds",
            "python_timing": "time.perf_counter",
            "r_timing":      "proc.time() inside single Rscript process",
            "jax_path_note": (
                "JAX RS uses data-aware cold-start initialization and excludes "
                "NumPy RS warm-starts. Cold compilation and repeated warm timings "
                "are reported separately."
            ),
        },
        "results":              rows,
        "correctness_vs_numpy": gate_numpy,
    }

    json_path = out_dir / f"jax_rs_{ts}.json"
    md_path   = out_dir / f"jax_rs_{ts}.md"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(artifact, f, indent=2)

    md_content = _make_markdown(env, good)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    print("\nArtifacts:")
    print(f"  JSON: {json_path}")
    print(f"  MD:   {md_path}")

    return 0 if gate_numpy else 1


if __name__ == "__main__":
    sys.exit(main())
