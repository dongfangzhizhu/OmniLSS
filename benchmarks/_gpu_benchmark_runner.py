# SPDX-License-Identifier: GPL-3.0-or-later
"""GPU vs NumPy RS benchmark runner.

Measures JAX warm time vs NumPy RS warm time across n values to find the
crossover point where JAX (GPU) becomes faster than NumPy (CPU).

Run via:
    source /mnt/d/AI/my/githubprojs/OmniLSS/debianjaxgpu/bin/activate
    PYTHONPATH=omnilss/src python benchmarks/_gpu_benchmark_runner.py

Output:
    docs/benchmarks/gpu_crossover_<timestamp>.json
    docs/benchmarks/gpu_crossover_<timestamp>.md
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n-values", type=int, nargs="+",
                   default=[1_000, 5_000, 10_000, 50_000, 100_000, 200_000, 500_000])
    p.add_argument("--n-reps",   type=int, default=6)
    p.add_argument("--families", nargs="+",
                   default=["NO", "GA", "PO", "BI", "WEI", "TF"])
    p.add_argument("--out-dir",  default="docs/benchmarks")
    return p.parse_args()


def _collect_env() -> dict[str, Any]:
    import jax
    info: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python":        sys.version.split()[0],
        "platform":      platform.platform(),
        "jax_version":   jax.__version__,
        "jax_backend":   jax.default_backend(),
        "jax_devices":   [str(d) for d in jax.devices()],
    }
    try:
        import psutil
        info["cpu_cores_physical"] = psutil.cpu_count(logical=False)
        info["cpu_cores_logical"]  = psutil.cpu_count(logical=True)
        info["ram_gb"]             = round(psutil.virtual_memory().total / 1e9, 1)
    except ImportError:
        pass
    # GPU name via nvidia-smi if available
    try:
        import subprocess
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total",
             "--format=csv,noheader"],
            text=True, timeout=10
        ).strip()
        info["gpu_info"] = out
    except Exception:
        info["gpu_info"] = "unavailable"
    return info


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


def _block(result):
    try:
        import jax
        if hasattr(result, "g_dev"):
            jax.block_until_ready(float(result.g_dev))
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


def _run_one(family_name: str, n: int, seed: int, n_reps: int) -> dict[str, Any]:
    from omnilss import gamlss
    from omnilss.distributions import resolve_family

    family  = resolve_family(family_name)
    data    = _make_data(family_name, n, seed)
    formula = "y ~ x"

    row: dict[str, Any] = {"family": family_name, "n": n, "seed": seed}

    # NumPy RS
    numpy_cold_ms, model_np = _time_call(
        gamlss, formula, family=family, data=data, method="RS"
    )
    numpy_warm = _warm_times(
        gamlss, (formula,), {"family": family, "data": data, "method": "RS"},
        n_reps=n_reps,
    )
    row["numpy_cold_ms"]        = round(numpy_cold_ms, 2)
    row["numpy_warm_ms_median"] = round(statistics.median(numpy_warm), 2)
    row["numpy_warm_ms_min"]    = round(min(numpy_warm), 2)
    row["numpy_deviance"]       = round(float(model_np.g_dev), 6)

    # JAX RS (GPU)
    jax_cold_ms, model_jax = _time_call(
        gamlss, formula, family=family, data=data, method="RS_JAX"
    )
    row["jax_cold_ms"]  = round(jax_cold_ms, 2)
    row["jax_deviance"] = round(float(model_jax.g_dev), 6)

    jax_warm = _warm_times(
        gamlss, (formula,), {"family": family, "data": data, "method": "RS_JAX"},
        n_reps=n_reps,
    )
    row["jax_warm_ms_median"] = round(statistics.median(jax_warm), 2)
    row["jax_warm_ms_min"]    = round(min(jax_warm), 2)

    row["deviance_diff"] = round(abs(row["jax_deviance"] - row["numpy_deviance"]), 8)

    if row["jax_warm_ms_median"] > 0:
        row["speedup"] = round(
            row["numpy_warm_ms_median"] / row["jax_warm_ms_median"], 3
        )
        row["jax_faster"] = row["speedup"] >= 1.0
    else:
        row["speedup"]    = None
        row["jax_faster"] = False

    import jax
    row["backend"] = jax.default_backend()
    return row


def _find_crossover(rows: list[dict], family: str) -> int | None:
    """Return the smallest n where JAX warm < NumPy warm for this family."""
    fam_rows = sorted(
        [r for r in rows if r["family"] == family and "error" not in r],
        key=lambda r: r["n"],
    )
    for r in fam_rows:
        if r.get("jax_faster"):
            return r["n"]
    return None


def _make_markdown(env: dict, rows: list[dict], crossovers: dict) -> str:
    backend = env.get("jax_backend", "?").upper()
    gpu_info = env.get("gpu_info", "N/A")

    lines = [
        "# GPU vs NumPy RS Crossover Benchmark",
        "",
        f"**Generated**: {env['timestamp_utc']}  ",
        f"**Platform**: {env['platform']}  ",
        f"**JAX version**: {env['jax_version']}  ",
        f"**JAX backend**: {backend}  ",
        f"**GPU**: {gpu_info}  ",
        f"**JAX devices**: {', '.join(env['jax_devices'])}  ",
        "",
        "## Crossover Summary",
        "",
        "The crossover point is the smallest `n` where JAX (GPU) warm time < NumPy warm time.",
        "Values below the crossover use NumPy RS automatically (`method='RS'`).",
        "Values at or above use JAX RS (`method='RS_JAX'`).",
        "",
        "| Family | Crossover n | Notes |",
        "|--------|------------:|-------|",
    ]
    families = sorted(set(r["family"] for r in rows if "error" not in r))
    for fam in families:
        co = crossovers.get(fam)
        if co:
            lines.append(f"| {fam} | {co:,} | JAX faster at n >= {co:,} |")
        else:
            max_n = max((r["n"] for r in rows if r["family"] == fam and "error" not in r), default=0)
            lines.append(f"| {fam} | > {max_n:,} | No crossover found up to n={max_n:,} |")

    lines += ["", "## Detailed Results", ""]
    for fam in families:
        fam_rows = sorted(
            [r for r in rows if r["family"] == fam and "error" not in r],
            key=lambda r: r["n"],
        )
        lines += [
            f"### {fam}",
            "",
            "| n | jax_warm_ms | numpy_warm_ms | speedup | jax_faster |",
            "|---|------------:|--------------:|--------:|:----------:|",
        ]
        for r in fam_rows:
            sp = f"{r.get('speedup', 0):.2f}x" if r.get("speedup") else "N/A"
            faster = "YES" if r.get("jax_faster") else "no"
            lines.append(
                f"| {r['n']:,} "
                f"| {r['jax_warm_ms_median']:.1f} "
                f"| {r['numpy_warm_ms_median']:.1f} "
                f"| {sp} "
                f"| {faster} |"
            )
        lines.append("")

    lines += [
        "## Correctness",
        "",
        "All runs: |JAX deviance - NumPy deviance| < 0.1",
        "",
    ]
    good = [r for r in rows if "error" not in r]
    all_pass = all(r["deviance_diff"] < 0.1 for r in good)
    lines.append(f"**Gate: {'PASS' if all_pass else 'FAIL'}**")
    lines += [
        "",
        "> Generated by `benchmarks/_gpu_benchmark_runner.py`.",
    ]
    return "\n".join(lines)


def main():
    args = _parse_args()

    import jax
    jax.config.update("jax_enable_x64", True)

    env  = _collect_env()
    rows = []

    print(f"\nGPU Crossover Benchmark — {env['timestamp_utc']}")
    print(f"Backend: {env['jax_backend']}  GPU: {env.get('gpu_info', 'N/A')}")
    print(f"Devices: {env['jax_devices']}")
    print()

    hdr = f"{'Family':<6} {'n':>8} {'jax_warm':>10} {'np_warm':>10} {'speedup':>9} {'faster':>7}"
    print(hdr)
    print("-" * len(hdr))

    seed_base = 300
    for fam in args.families:
        for n in args.n_values:
            seed = seed_base
            seed_base += 1
            try:
                row = _run_one(fam, n, seed, args.n_reps)
                rows.append(row)
                sp = f"{row.get('speedup', 0):.2f}x" if row.get("speedup") else "  N/A"
                faster = "YES" if row.get("jax_faster") else " no"
                print(
                    f"{fam:<6} {n:>8,} "
                    f"{row['jax_warm_ms_median']:>10.1f} "
                    f"{row['numpy_warm_ms_median']:>10.1f} "
                    f"{sp:>9} "
                    f"{faster:>7}"
                )
            except Exception as exc:
                import traceback
                traceback.print_exc()
                print(f"{fam:<6} {n:>8,}  ERROR: {exc}")
                rows.append({"family": fam, "n": n, "error": str(exc)})

    # Crossover analysis
    crossovers: dict[str, int | None] = {}
    print("\nCrossover points (JAX warm < NumPy warm):")
    for fam in args.families:
        co = _find_crossover(rows, fam)
        crossovers[fam] = co
        if co:
            print(f"  {fam}: n >= {co:,}")
        else:
            max_n = max((r["n"] for r in rows if r["family"] == fam and "error" not in r), default=0)
            print(f"  {fam}: no crossover up to n={max_n:,}")

    # Correctness
    good = [r for r in rows if "error" not in r]
    gate = all(r["deviance_diff"] < 0.1 for r in good)
    print(f"\nCorrectness gate: {'PASS' if gate else 'FAIL'}")

    # Write artifacts
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    artifact = {
        "environment": env,
        "crossovers":  crossovers,
        "results":     rows,
        "correctness_gate_pass": gate,
    }
    json_path = out_dir / f"gpu_crossover_{ts}.json"
    md_path   = out_dir / f"gpu_crossover_{ts}.md"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(artifact, f, indent=2)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(_make_markdown(env, rows, crossovers))

    print(f"\nArtifacts: {json_path}")
    print(f"           {md_path}")
    return 0 if gate else 1


if __name__ == "__main__":
    sys.exit(main())
