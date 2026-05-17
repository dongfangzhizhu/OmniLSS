"""Validate OmniLSS results against stored R reference values.

The JSON fixture may intentionally contain null reference values when R was not
available during development. Such cases are reported as skipped instead of
being treated as passing benchmark evidence.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from omnilss import gamlss


def _case_data(case: dict) -> dict[str, np.ndarray]:
    """Return deterministic data matching the R reference generator."""
    n = case["n"]
    x = np.linspace(0, 5, n)
    if case["id"] == "normal_linear":
        y = 2 + 3 * x + np.sin(x)
        return {"y": y, "x": x}
    raise ValueError(f"Unknown R reference case id: {case['id']}")


def test_r_consistency(
    r_results_path: str = "benchmarks/r_reference_results.json",
) -> bool:
    """Return True when all populated R reference cases match OmniLSS."""
    path = Path(r_results_path)
    with path.open(encoding="utf-8") as f:
        r_data = json.load(f)

    failures: list[str] = []
    skipped: list[str] = []
    for case in r_data["test_cases"]:
        if case.get("r_deviance") is None:
            skipped.append(f"{case['id']}: missing R reference deviance")
            continue

        data = _case_data(case)
        model = gamlss(case["formula"], family=case["family"], data=data)
        denominator = max(abs(float(case["r_deviance"])), 1e-12)
        deviance_diff = abs(float(model.g_dev) - float(case["r_deviance"])) / denominator
        if deviance_diff > 0.01:
            failures.append(f"{case['id']}: deviance diff {deviance_diff:.2%}")

    if skipped:
        print("SKIPPED R reference cases:")
        for item in skipped:
            print(f"  {item}")
    if failures:
        print("FAILURES:")
        for item in failures:
            print(f"  {item}")
        return False
    print(
        f"All populated R consistency checks passed ({len(r_data['test_cases']) - len(skipped)} run, {len(skipped)} skipped)."
    )
    return True


if __name__ == "__main__":
    raise SystemExit(0 if test_r_consistency() else 1)
