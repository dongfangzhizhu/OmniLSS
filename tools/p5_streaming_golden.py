# SPDX-License-Identifier: GPL-3.0-or-later
"""P5-1c independent streaming golden verification tool (D2.284/C1).

Implements all 7 reproducible streaming golden criteria specified in
``docs/development/p5-ondil-distribution-contract-2026-11-15.md`` §12 using
synthetic fixed-seed DGPs and OmniLSS's self-contained :mod:`omnilss.online`
estimator (zero external imports from reference repositories):

  1. partial_fit_contract     -- first partial_fit == fit; subsequent == update.
  2. forgetting_factor_decay  -- exact exponential weights (1 - forget)^(N - n).
  3. batch_invariance         -- update(A); update(B) == update(concat(A, B)) <= 1e-10.
  4. drift_order_sensitivity  -- forget > 0 adapts to regime jump with short half-life.
  5. warm_start_equivalence   -- partial_fit(max_iter=k) == fit(max_iter=k) <= 1e-8.
  6. missingness_robustness   -- bounded bias under 0% / 5% / 20% missing rows.
  7. memory_and_scaling       -- constant O(p^2) memory & flat per-obs update time.

Usage::

    python tools/p5_streaming_golden.py --write docs/development/p5-streaming-golden-2026-11-21.json
    python tools/p5_streaming_golden.py --check
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "omnilss" / "src"))

from omnilss.online import OnlineGAMLSS, OnlineGram  # noqa: E402

DEFAULT_OUT = ROOT / "docs" / "development" / "p5-streaming-golden-2026-11-21.json"
SCHEMA_VERSION = 1

BATCH_INVARIANCE_TOL = 1e-10
WARM_START_TOL = 1e-8
DRIFT_MAX_HALFLIFE_BATCHES = 15
MISSING_MAX_BIAS = 0.20


def _criterion_1_partial_fit_contract() -> dict:
    rng = np.random.default_rng(101)
    X = rng.normal(size=(180, 3))
    y = 1.5 + 0.8 * X[:, 0] - 0.5 * X[:, 1] + 0.3 * X[:, 2] + rng.normal(scale=0.6, size=180)
    m_fit = OnlineGAMLSS("NO", forget=0.02).fit(X[:100], y[:100]).update(X[100:], y[100:])
    m_pf = OnlineGAMLSS("NO", forget=0.02).partial_fit(X[:100], y[:100]).partial_fit(X[100:], y[100:])
    mu_diff = float(np.max(np.abs(m_fit.coefficients_["mu"] - m_pf.coefficients_["mu"])))
    sig_diff = float(np.max(np.abs(m_fit.coefficients_["sigma"] - m_pf.coefficients_["sigma"])))
    max_diff = max(mu_diff, sig_diff)
    return {
        "met": max_diff <= 1e-12,
        "max_coef_diff": max_diff,
        "tolerance": 1e-12,
    }


def _criterion_2_forgetting_factor_decay() -> dict:
    rng = np.random.default_rng(202)
    X = np.column_stack([np.ones(40), rng.normal(size=(40, 2))])
    y = rng.normal(size=40)
    forget = 0.05
    decay = 1.0 - forget
    gram = OnlineGram(3, forget=forget)
    gram.update(X[:15], y[:15])
    gram.update(X[15:], y[15:])
    expected_w = decay ** np.arange(39, -1, -1, dtype=np.float64)
    expected_G = (X * expected_w[:, None]).T @ X
    expected_h = (X * expected_w[:, None]).T @ y
    g_diff = float(np.max(np.abs(gram.gram - expected_G)))
    h_diff = float(np.max(np.abs(gram.rhs - expected_h)))
    w_diff = abs(gram.sum_w - float(np.sum(expected_w)))
    max_diff = max(g_diff, h_diff, w_diff)
    return {
        "met": max_diff <= 1e-12,
        "forget": forget,
        "max_stat_diff": max_diff,
        "tolerance": 1e-12,
    }


def _criterion_3_batch_invariance() -> dict:
    rng = np.random.default_rng(303)
    X = rng.normal(size=(240, 3))
    y = 2.0 + 1.2 * X[:, 0] - 0.7 * X[:, 1] + 0.4 * X[:, 2] + rng.normal(scale=0.5, size=240)
    results = {}
    worst = 0.0
    for forget in (0.0, 0.02):
        m_split = OnlineGAMLSS("NO", forget=forget)
        m_split.partial_fit(X[:80], y[:80])
        m_split.update(X[80:150], y[80:150])
        m_split.update(X[150:], y[150:])

        m_concat = OnlineGAMLSS("NO", forget=forget)
        m_concat.partial_fit(X[:80], y[:80])
        m_concat.update(X[80:], y[80:])

        diff_mu = float(np.max(np.abs(m_split.coefficients_["mu"] - m_concat.coefficients_["mu"])))
        diff_sig = float(np.max(np.abs(m_split.coefficients_["sigma"] - m_concat.coefficients_["sigma"])))
        gap = max(diff_mu, diff_sig)
        worst = max(worst, gap)
        results[f"forget={forget}"] = {"mu_diff": diff_mu, "sigma_diff": diff_sig}
    return {
        "met": worst <= BATCH_INVARIANCE_TOL,
        "max_coef_diff": worst,
        "tolerance": BATCH_INVARIANCE_TOL,
        "cases": results,
    }


def _criterion_4_drift_order_sensitivity() -> dict:
    rng = np.random.default_rng(404)
    # Regime 1: intercept = 1.0, slope = 0.5; Regime 2: intercept = 6.0, slope = 0.5
    X1 = rng.normal(size=(200, 1))
    y1 = 1.0 + 0.5 * X1[:, 0] + rng.normal(scale=0.3, size=200)
    m_forget = OnlineGAMLSS("NO", forget=0.05).fit(X1, y1)
    m_static = OnlineGAMLSS("NO", forget=0.0).fit(X1, y1)

    target_midpoint = 3.5  # halfway from 1.0 to 6.0
    halflife_batches = None
    batch_size = 10
    for step in range(1, 31):
        Xb = rng.normal(size=(batch_size, 1))
        yb = 6.0 + 0.5 * Xb[:, 0] + rng.normal(scale=0.3, size=batch_size)
        m_forget.update(Xb, yb)
        m_static.update(Xb, yb)
        if halflife_batches is None and float(m_forget.coefficients_["mu"][0]) >= target_midpoint:
            halflife_batches = step

    final_forget = float(m_forget.coefficients_["mu"][0])
    final_static = float(m_static.coefficients_["mu"][0])
    # Order sensitivity check: newer regime gets strictly higher weight than static forget=0
    direction_ok = final_forget > final_static + 1.0 and abs(final_forget - 6.0) < 0.25
    halflife_ok = halflife_batches is not None and halflife_batches <= DRIFT_MAX_HALFLIFE_BATCHES
    return {
        "met": bool(direction_ok and halflife_ok),
        "halflife_batches": halflife_batches,
        "max_halflife_batches": DRIFT_MAX_HALFLIFE_BATCHES,
        "final_intercept_forget": round(final_forget, 6),
        "final_intercept_static": round(final_static, 6),
    }


def _criterion_5_warm_start_equivalence() -> dict:
    rng = np.random.default_rng(505)
    X = rng.normal(size=(120, 2))
    cases = {}
    worst = 0.0
    for family_name, y in (
        ("NO", 1.0 + 0.6 * X[:, 0] - 0.3 * X[:, 1] + rng.normal(scale=0.4, size=120)),
        ("GA", rng.gamma(shape=9.0, scale=np.exp(0.5 + 0.3 * X[:, 0]) / 9.0)),
        ("PO", rng.poisson(lam=np.exp(0.8 + 0.4 * X[:, 0])).astype(float)),
    ):
        for k in (1, 5):
            m_batch = OnlineGAMLSS(family_name, forget=0.0).fit(X, y, max_iter=k)
            m_pf = OnlineGAMLSS(family_name, forget=0.0).partial_fit(X, y, max_iter=k)
            gap = max(
                float(np.max(np.abs(m_batch.coefficients_[p] - m_pf.coefficients_[p])))
                for p in m_batch.coefficients_
            )
            worst = max(worst, gap)
            cases[f"{family_name}_k{k}"] = gap
    return {
        "met": worst <= WARM_START_TOL,
        "max_coef_diff": worst,
        "tolerance": WARM_START_TOL,
        "cases": cases,
    }


def _criterion_6_missingness_robustness() -> dict:
    rng = np.random.default_rng(606)
    n = 1200
    true_beta = np.array([2.0, 1.0, -0.8])
    X_base = rng.normal(size=(n, 2))
    y_base = true_beta[0] + true_beta[1] * X_base[:, 0] + true_beta[2] * X_base[:, 1] + rng.normal(scale=0.4, size=n)
    rates = {}
    worst_bias = 0.0
    for rate in (0.0, 0.05, 0.20):
        X_m = X_base.copy()
        y_m = y_base.copy()
        if rate > 0.0:
            drop_y = rng.random(n) < (rate * 0.5)
            drop_x = rng.random(n) < (rate * 0.5)
            y_m[drop_y] = np.nan
            X_m[drop_x, 0] = np.nan
        est = OnlineGAMLSS("NO", forget=0.0)
        for start in range(0, n, 120):
            est.partial_fit(X_m[start : start + 120], y_m[start : start + 120])
        bias = float(np.max(np.abs(est.coefficients_["mu"] - true_beta)))
        worst_bias = max(worst_bias, bias)
        rates[f"{int(rate * 100)}pct"] = {"max_abs_bias": round(bias, 6), "valid_obs": est.n_obs_}
    return {
        "met": worst_bias <= MISSING_MAX_BIAS,
        "max_abs_bias": round(worst_bias, 6),
        "tolerance": MISSING_MAX_BIAS,
        "rates": rates,
    }


def _criterion_7_memory_and_scaling() -> dict:
    rng = np.random.default_rng(707)
    p = 8
    est = OnlineGAMLSS("NO", forget=0.01)
    # Cold start batch
    X0 = rng.normal(size=(100, p))
    y0 = 1.0 + X0[:, 0] + rng.normal(scale=0.5, size=100)
    t0 = time.perf_counter()
    est.partial_fit(X0, y0)
    cold_ms = (time.perf_counter() - t0) * 1000.0
    mem_initial = est.state_memory_bytes()

    # Warm streaming chunks across growing cumulative N (up to 10,100)
    chunk_size = 200
    chunk_times_us_per_obs: list[float] = []
    mem_samples: list[int] = [mem_initial]
    for _ in range(50):
        Xb = rng.normal(size=(chunk_size, p))
        yb = 1.0 + Xb[:, 0] + rng.normal(scale=0.5, size=chunk_size)
        ts = time.perf_counter()
        est.update(Xb, yb)
        elapsed_us = (time.perf_counter() - ts) * 1e6 / chunk_size
        chunk_times_us_per_obs.append(elapsed_us)
        mem_samples.append(est.state_memory_bytes())

    mem_constant = len(set(mem_samples)) == 1
    # Compare early warm median vs late warm median (should not grow linearly with N)
    early_median = float(np.median(chunk_times_us_per_obs[:10]))
    late_median = float(np.median(chunk_times_us_per_obs[-10:]))
    ratio = late_median / max(early_median, 1e-6)
    return {
        "met": bool(mem_constant and ratio < 3.0),
        "n_features": p,
        "total_obs": est.n_obs_,
        "state_memory_bytes": mem_initial,
        "memory_constant_across_N": mem_constant,
        "cold_ms": round(cold_ms, 3),
        "warm_early_us_per_obs": round(early_median, 3),
        "warm_late_us_per_obs": round(late_median, 3),
        "late_to_early_ratio": round(ratio, 3),
    }


def build() -> dict:
    criteria = {
        "partial_fit_contract": _criterion_1_partial_fit_contract(),
        "forgetting_factor_decay": _criterion_2_forgetting_factor_decay(),
        "batch_invariance": _criterion_3_batch_invariance(),
        "drift_order_sensitivity": _criterion_4_drift_order_sensitivity(),
        "warm_start_equivalence": _criterion_5_warm_start_equivalence(),
        "missingness_robustness": _criterion_6_missingness_robustness(),
        "memory_and_scaling": _criterion_7_memory_and_scaling(),
    }
    met = sum(1 for c in criteria.values() if c["met"])
    return {
        "schema_version": SCHEMA_VERSION,
        "task": "P5-1c",
        "decision": "D2.284",
        "contract": "docs/development/p5-ondil-distribution-contract-2026-11-15.md#section-12",
        "criteria": criteria,
        "counts": {"criteria": len(criteria), "met": met},
        "verdict": {
            "streaming_golden_met": met == len(criteria),
            "notes": (
                "All 7 streaming golden criteria from P5-1a section 12 evaluated on "
                "OmniLSS's self-contained omnilss.online estimator with zero external "
                "reference-repo imports."
            ),
        },
    }


def _rounded(value):
    """Round deterministic floats so cross-platform BLAS/JAX jitter (1e-15)
    does not read as artifact drift, while real numeric changes still do."""
    if isinstance(value, float):
        return round(value, 8)
    if isinstance(value, dict):
        return {key: _rounded(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_rounded(item) for item in value]
    return value


def _stable(artifact: dict) -> dict:
    # Exclude wall-clock microsecond timings from determinism diff while keeping
    # every pass/fail verdict and deterministic numeric metric checked.
    stable_criteria = {}
    for name, item in artifact["criteria"].items():
        if name == "memory_and_scaling":
            stable_criteria[name] = {
                "met": item["met"],
                "n_features": item["n_features"],
                "total_obs": item["total_obs"],
                "state_memory_bytes": item["state_memory_bytes"],
                "memory_constant_across_N": item["memory_constant_across_N"],
            }
        else:
            stable_criteria[name] = _rounded(item)
    return {
        "schema_version": artifact["schema_version"],
        "task": artifact["task"],
        "criteria": stable_criteria,
        "counts": artifact["counts"],
        "verdict": artifact["verdict"],
    }


def _render_markdown(a: dict) -> str:
    c = a["counts"]
    lines = [
        "# P5-1c Streaming Golden Verification",
        "",
        f"- streaming_golden_met: **{a['verdict']['streaming_golden_met']}** ({c['met']}/{c['criteria']})",
        "",
        "| criterion | met | key metric |",
        "|---|---|---|",
    ]
    for name, item in a["criteria"].items():
        summary = ", ".join(
            f"{k}={v}"
            for k, v in item.items()
            if k not in {"met", "cases", "rates"}
        )
        lines.append(f"| {name} | {item['met']} | {summary} |")
    lines += ["", a["verdict"]["notes"], ""]
    return "\n".join(lines)


def write_artifact(artifact: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(artifact, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    path.with_suffix(".md").write_text(_render_markdown(artifact), encoding="utf-8")


def check_artifact(path: Path) -> tuple[bool, list[str]]:
    if not path.is_file():
        return False, [f"artifact not found: {path}"]
    artifact = json.loads(path.read_text(encoding="utf-8"))
    current = build()
    problems: list[str] = []
    if not current["verdict"]["streaming_golden_met"]:
        failed = [k for k, v in current["criteria"].items() if not v["met"]]
        problems.append(f"streaming golden criteria failed: {', '.join(failed)}")
    if _stable(artifact) != _stable(current):
        problems.append("streaming golden artifact drifted vs current tree")
    return (not problems), problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", type=Path, default=None)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--path", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    if args.check:
        ok, problems = check_artifact(args.path)
        print(f"P5-1c streaming golden check: {args.path}")
        if ok:
            print("result: OK")
            return 0
        for problem in problems:
            print(f"  - {problem}")
        print("result: FAIL")
        return 1

    artifact = build()
    if args.write is not None:
        write_artifact(artifact, args.write)
        print(f"wrote {args.write}")
        print(f"wrote {args.write.with_suffix('.md')}")
    if args.json or args.write is None:
        print(json.dumps(artifact, indent=2, sort_keys=True, ensure_ascii=False))
    c = artifact["counts"]
    print(f"streaming_golden_met={artifact['verdict']['streaming_golden_met']} ({c['met']}/{c['criteria']})")
    return 0 if artifact["verdict"]["streaming_golden_met"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
