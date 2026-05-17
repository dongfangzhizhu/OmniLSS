#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""GPU crossover sweep for OmniLSS RS vs RS_JAX routing.

This benchmark measures the current JAX-supported families across an
``(n, p, family)`` matrix and writes both raw JSON and a Markdown report.
It intentionally keeps the analysis conservative: thresholds are proposed
only when the post-warm-up median for ``RS_JAX`` is strictly below ``RS``.

Example
-------
Run the full task matrix on a GPU host::

    PYTHONPATH=omnilss/src python benchmarks/gpu_crossover_sweep.py

Run a smoke benchmark while developing the script::

    PYTHONPATH=omnilss/src python benchmarks/gpu_crossover_sweep.py \
        --n-values 100 --p-values 2 --families NO --repeats 1 --warmups 1
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
from typing import Any

import numpy as np

try:
    import jax
except ImportError:  # pragma: no cover - benchmark host validation
    jax = None  # type: ignore[assignment]

try:
    from omnilss import BI, GA, NO, PO, TF, WEI, gamlss
    from omnilss.controls import gamlss_control
except ImportError as exc:  # pragma: no cover - CLI environment validation
    raise SystemExit(
        "Could not import omnilss. Run with PYTHONPATH=omnilss/src or install the package."
    ) from exc

DEFAULT_N_VALUES = [100, 500, 1_000, 5_000, 10_000, 50_000, 100_000, 500_000]
DEFAULT_P_VALUES = [2, 5, 10, 20, 50]
DEFAULT_FAMILIES = ["NO", "GA", "PO", "BI", "WEI", "TF"]
FAMILY_FACTORIES = {"NO": NO, "GA": GA, "PO": PO, "BI": BI, "WEI": WEI, "TF": TF}


@dataclass(frozen=True)
class TimingResult:
    values_ms: list[float]
    p50_ms: float
    p95_ms: float


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def make_dataset(family: str, n: int, p: int, seed: int) -> tuple[str, dict[str, np.ndarray]]:
    """Create deterministic synthetic data with a design matrix containing p columns.

    ``p`` includes the intercept column used by OmniLSS, so the generated
    formula contains ``p - 1`` explicit predictors.
    """
    if p < 1:
        raise ValueError("p must be at least 1 because the intercept is a design column")

    rng = np.random.default_rng(seed)
    n_predictors = max(p - 1, 0)
    x = rng.normal(size=(n, n_predictors)) if n_predictors else np.empty((n, 0))
    beta = np.linspace(0.05, 0.35, n_predictors) if n_predictors else np.empty(0)
    eta = 0.5 + x @ beta if n_predictors else np.full(n, 0.5)

    if family == "NO":
        y = eta + rng.normal(scale=1.0, size=n)
    elif family == "GA":
        mu = np.exp(np.clip(eta, -3.0, 5.0))
        shape = 4.0
        y = rng.gamma(shape=shape, scale=mu / shape)
    elif family == "PO":
        mu = np.exp(np.clip(eta, -3.0, 4.0))
        y = rng.poisson(mu).astype(float)
    elif family == "BI":
        y = rng.binomial(1, _sigmoid(eta)).astype(float)
    elif family == "WEI":
        scale = np.exp(np.clip(eta, -3.0, 5.0))
        shape = 1.5
        y = scale * rng.weibull(shape, size=n)
    elif family == "TF":
        y = eta + rng.standard_t(df=5.0, size=n)
    else:
        raise ValueError(f"unsupported benchmark family: {family}")

    data: dict[str, np.ndarray] = {"y": np.asarray(y, dtype=np.float64)}
    for idx in range(n_predictors):
        data[f"x{idx + 1}"] = np.asarray(x[:, idx], dtype=np.float64)
    if family == "BI":
        data["bd"] = np.ones(n, dtype=np.float64)

    rhs = " + ".join(f"x{idx + 1}" for idx in range(n_predictors)) or "1"
    return f"y ~ {rhs}", data


def _block_until_ready(value: Any) -> None:
    if jax is None:
        return
    try:
        jax.block_until_ready(value)
    except Exception:
        pass


def _fit_once(family: str, formula: str, data: dict[str, np.ndarray], method: str, max_iter: int) -> Any:
    control = gamlss_control(n_cyc=max_iter, c_crit=1e-4, trace=False)
    model = gamlss(
        formula,
        family=FAMILY_FACTORIES[family](),
        data=data,
        method=method,
        control=control,
        verbose=False,
    )
    _block_until_ready(getattr(model, "g_dev", None))
    return model


def time_method(
    family: str,
    formula: str,
    data: dict[str, np.ndarray],
    method: str,
    *,
    repeats: int,
    warmups: int,
    max_iter: int,
) -> TimingResult:
    for _ in range(warmups):
        _fit_once(family, formula, data, method, max_iter)

    values_ms: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        _fit_once(family, formula, data, method, max_iter)
        values_ms.append((time.perf_counter() - start) * 1_000.0)

    if len(values_ms) == 1:
        p95 = values_ms[0]
    else:
        p95 = statistics.quantiles(values_ms, n=20, method="inclusive")[18]
    return TimingResult(values_ms=values_ms, p50_ms=statistics.median(values_ms), p95_ms=p95)


def nvidia_smi_snapshot() -> dict[str, Any]:
    cmd = [
        "nvidia-smi",
        "--query-gpu=name,memory.used,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=10)
    except Exception as exc:
        return {"available": False, "error": str(exc)}

    gpus = []
    for line in completed.stdout.strip().splitlines():
        name, memory_used, memory_total, utilization = [part.strip() for part in line.split(",")]
        gpus.append(
            {
                "name": name,
                "memory_used_mib": float(memory_used),
                "memory_total_mib": float(memory_total),
                "utilization_gpu_percent": float(utilization),
            }
        )
    return {"available": True, "gpus": gpus}


def infer_thresholds(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Infer conservative crossover thresholds from raw benchmark rows."""
    by_family_p: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("status") != "ok":
            continue
        by_family_p.setdefault((row["family"], row["p"]), []).append(row)

    thresholds_by_family_p: dict[str, dict[str, float | str]] = {}
    recommended: dict[str, float | str] = {}
    p_sensitivity: dict[str, dict[str, Any]] = {}

    for family in sorted({row["family"] for row in rows}):
        family_thresholds: dict[int, float] = {}
        for (fam, p), group in by_family_p.items():
            if fam != family:
                continue
            sorted_group = sorted(group, key=lambda item: item["n"])
            threshold = math.inf
            for row in sorted_group:
                if row["rs_jax_p50_ms"] < row["rs_p50_ms"]:
                    threshold = float(row["n"])
                    break
            family_thresholds[p] = threshold

        thresholds_by_family_p[family] = {
            str(p): ("inf" if value == math.inf else value)
            for p, value in sorted(family_thresholds.items())
        }
        finite = [value for value in family_thresholds.values() if value != math.inf]
        if finite and len(finite) == len(family_thresholds):
            recommended[family] = max(finite)
            ratio = max(finite) / max(min(finite), 1.0)
            p_sensitivity[family] = {"ratio": ratio, "requires_p_dimension": ratio > 2.0}
        else:
            recommended[family] = "inf"
            p_sensitivity[family] = {"ratio": "inf", "requires_p_dimension": False}

    return {
        "thresholds_by_family_p": thresholds_by_family_p,
        "recommended_gpu_crossover_n": recommended,
        "p_sensitivity": p_sensitivity,
    }


def write_report(payload: dict[str, Any], report_path: Path) -> None:
    rows = payload["rows"]
    analysis = payload["analysis"]
    lines = [
        f"# GPU Crossover Sweep — {payload['timestamp']}",
        "",
        "## Environment",
        "",
        f"- Python: `{payload['environment']['python']}`",
        f"- Platform: `{payload['environment']['platform']}`",
        f"- JAX backend: `{payload['environment']['jax_backend']}`",
        f"- JAX version: `{payload['environment']['jax_version']}`",
        f"- Devices: `{payload['environment']['jax_devices']}`",
        f"- NVIDIA SMI: `{payload['environment']['nvidia_smi']}`",
        "",
        "## Crossover analysis",
        "",
        "Recommended `GPU_CROSSOVER_N` values use a conservative max across p when every p has a finite crossover; otherwise the family remains `math.inf`.",
        "",
        "| Family | Recommended threshold | p-sensitivity |",
        "|---|---:|---|",
    ]
    for family, threshold in analysis["recommended_gpu_crossover_n"].items():
        sensitivity = analysis["p_sensitivity"][family]
        lines.append(f"| {family} | {threshold} | {sensitivity} |")

    lines.extend([
        "",
        "## Raw timing summary",
        "",
        "| Family | n | p | RS p50 ms | RS p95 ms | RS_JAX p50 ms | RS_JAX p95 ms | GPU memory MiB | Status |",
        "|---|---:|---:|---:|---:|---:|---:|---|---|",
    ])
    for row in rows:
        if row.get("status") == "ok":
            lines.append(
                f"| {row['family']} | {row['n']} | {row['p']} | "
                f"{row['rs_p50_ms']:.3f} | {row['rs_p95_ms']:.3f} | "
                f"{row['rs_jax_p50_ms']:.3f} | {row['rs_jax_p95_ms']:.3f} | "
                f"{row.get('gpu_memory', 'n/a')} | ok |"
            )
        else:
            lines.append(
                f"| {row['family']} | {row['n']} | {row['p']} | n/a | n/a | n/a | n/a | "
                f"{row.get('gpu_memory', 'n/a')} | {row.get('error', 'failed')} |"
            )

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-values", nargs="+", type=int, default=DEFAULT_N_VALUES)
    parser.add_argument("--p-values", nargs="+", type=int, default=DEFAULT_P_VALUES)
    parser.add_argument("--families", nargs="+", choices=DEFAULT_FAMILIES, default=DEFAULT_FAMILIES)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--max-iter", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260517)
    parser.add_argument("--output-dir", type=Path, default=Path("benchmarks/results"))
    parser.add_argument("--report-dir", type=Path, default=Path("docs/benchmarks"))
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    environment = {
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "jax_version": getattr(jax, "__version__", "not-installed"),
        "jax_backend": jax.default_backend() if jax is not None else "not-installed",
        "jax_devices": [str(device) for device in jax.devices()] if jax is not None else [],
        "nvidia_smi": nvidia_smi_snapshot(),
    }

    rows: list[dict[str, Any]] = []
    for family in args.families:
        for p in args.p_values:
            for n in args.n_values:
                stable_family = sum((idx + 1) * ord(char) for idx, char in enumerate(family))
                seed = args.seed + stable_family * 1_000_000 + p * 10_000 + n
                formula, data = make_dataset(family, n, p, seed)
                gpu_before = nvidia_smi_snapshot()
                row: dict[str, Any] = {
                    "family": family,
                    "n": n,
                    "p": p,
                    "seed": seed,
                    "formula": formula,
                    "gpu_memory": gpu_before,
                }
                print(f"[gpu-crossover] family={family} n={n} p={p}", flush=True)
                try:
                    rs = time_method(
                        family,
                        formula,
                        data,
                        "RS",
                        repeats=args.repeats,
                        warmups=0,
                        max_iter=args.max_iter,
                    )
                    rs_jax = time_method(
                        family,
                        formula,
                        data,
                        "RS_JAX",
                        repeats=args.repeats,
                        warmups=args.warmups,
                        max_iter=args.max_iter,
                    )
                    row.update(
                        {
                            "status": "ok",
                            "rs_values_ms": rs.values_ms,
                            "rs_p50_ms": rs.p50_ms,
                            "rs_p95_ms": rs.p95_ms,
                            "rs_jax_values_ms": rs_jax.values_ms,
                            "rs_jax_p50_ms": rs_jax.p50_ms,
                            "rs_jax_p95_ms": rs_jax.p95_ms,
                            "gpu_memory_after": nvidia_smi_snapshot(),
                        }
                    )
                except Exception as exc:  # pragma: no cover - benchmark resilience
                    row.update({"status": "error", "error": repr(exc)})
                    if args.fail_fast:
                        raise
                rows.append(row)

    payload = {
        "timestamp": timestamp,
        "config": {
            "n_values": args.n_values,
            "p_values": args.p_values,
            "families": args.families,
            "repeats": args.repeats,
            "warmups": args.warmups,
            "max_iter": args.max_iter,
            "seed": args.seed,
        },
        "environment": environment,
        "rows": rows,
        "analysis": infer_thresholds(rows),
    }

    json_path = args.output_dir / f"gpu_crossover_{timestamp}.json"
    report_path = args.report_dir / f"gpu_crossover_{timestamp}.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    write_report(payload, report_path)
    print(f"Wrote {json_path}")
    print(f"Wrote {report_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
