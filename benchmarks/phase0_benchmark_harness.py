"""Phase 0 benchmark harness for reproducible runtime/convergence/memory/JIT metrics."""

from __future__ import annotations

import argparse
import json
import time
import tracemalloc
from pathlib import Path

import numpy as np

from omnilss.fitting import gamlss


def _dataset(n: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    y = 2.0 + 0.7 * x + rng.normal(scale=1.0, size=n)
    return {"x": x, "y": y}


def run_once(n: int, method: str, family: str) -> dict:
    data = _dataset(n)

    tracemalloc.start()
    t0 = time.perf_counter()
    model = gamlss("y ~ x", family=family, data=data, method=method)
    runtime_s = time.perf_counter() - t0
    _cur, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    slots = model.additional_slots
    out = {
        "n": n,
        "method": method,
        "family": family,
        "runtime_s": runtime_s,
        "converged": bool(slots.get("converged", slots.get("rs_converged", False))),
        "iterations": int(slots.get("cycles", slots.get("rs_iterations", model.iter))),
        "g_dev": float(model.g_dev),
        "memory_peak_bytes": int(peak),
        # For RS this is not JIT; for joint/lbfgs these keys are placeholders for consistency.
        "jit_compile_time_s": float(slots.get("jit_compile_time_s", 0.0)),
    }
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, nargs="+", default=[1000, 10000])
    parser.add_argument("--method", type=str, default="RS")
    parser.add_argument("--family", type=str, default="NO")
    parser.add_argument("--output", type=str, default="docs/benchmarks/phase0-benchmark-results.json")
    args = parser.parse_args()

    rows = [run_once(n=v, method=args.method, family=args.family) for v in args.n]
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False))

    print(json.dumps(rows, indent=2, ensure_ascii=False))
    print(f"\nSaved to: {out_path}")


if __name__ == "__main__":
    main()
