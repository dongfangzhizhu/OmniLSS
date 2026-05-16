"""Phase 1 autodiff validation pipeline runner."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from omnilss.ad_validation import finite_difference_vs_autodiff_no_mu


def run(output: str = "docs/benchmarks/phase1-ad-validation.json") -> dict:
    y = np.array([0.2, -0.1, 1.5, 0.0], dtype=np.float64)
    mu = np.array([0.0, 0.1, 1.0, -0.2], dtype=np.float64)
    sigma = np.array([1.2, 0.8, 1.5, 2.0], dtype=np.float64)
    metrics = finite_difference_vs_autodiff_no_mu(y, mu, sigma)
    out = {"pipeline": "finite_difference_vs_autodiff_no_mu", **metrics}
    p = Path(output)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    return out


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, ensure_ascii=False))
