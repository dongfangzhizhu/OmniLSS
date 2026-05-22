import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from omnilss import gamlss


def _build_data(case_name: str, n: int) -> pd.DataFrame:
    rng = np.random.default_rng(20260522)
    x1 = rng.normal(size=n)
    if case_name.startswith("NO"):
        y = 2.0 + 3.0 * x1 + rng.normal(size=n)
    elif case_name.startswith("PO"):
        mu = np.exp(1.0 + 0.5 * x1)
        y = rng.poisson(mu)
    else:
        raise ValueError(f"Unsupported case: {case_name}")
    return pd.DataFrame({"y": y, "x1": x1})


def test_performance_snapshot_v0_3_0_freeze_guard():
    snap_path = Path(__file__).parent / "fixtures" / "performance_snapshot_v0_3_0.json"
    snapshot = json.loads(snap_path.read_text())

    for case_name, case in snapshot["cases"].items():
        data = _build_data(case_name, int(case["n"]))

        _ = gamlss(
            formula=case["formula"],
            sigma_formula=case["sigma_formula"],
            family=case["family"],
            data=data,
        )

        started = time.perf_counter()
        model = gamlss(
            formula=case["formula"],
            sigma_formula=case["sigma_formula"],
            family=case["family"],
            data=data,
        )
        elapsed = time.perf_counter() - started

        assert model.additional_slots.get("converged", False), f"{case_name} did not converge"
        assert elapsed <= float(case["max_seconds"]), (
            f"{case_name} took {elapsed:.4f}s > snapshot max {case['max_seconds']:.4f}s"
        )
