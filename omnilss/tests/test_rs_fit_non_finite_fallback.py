"""The non-finite fallback must stay out of a normal fit (D2.92).

rs_fit now keeps the best finite cycle and, if the last cycle ends with a
non-finite deviance, returns that cycle instead (termination_reason
"reverted_to_best_finite").  That is a rescue for families whose likelihood
is unbounded in some direction -- SEP3 dived to inf at cycle 13 and now
returns 123.46.  It must NOT fire for a family that converges normally, or
every fit would silently report a stale cycle.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from omnilss import gamlss  # noqa: E402


def test_a_converging_fit_is_not_reverted():
    rng = np.random.default_rng(7)
    x = rng.normal(size=60)
    y = 1.0 + 0.5 * x + rng.normal(size=60) * 0.3
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = gamlss(
            "y ~ x",
            family="NO",
            data={"y": y, "x": x},
            sigma_formula="~1",
            method="RS",
            max_iter=20,
            allow_compat_alias=True,
        )
    assert np.isfinite(float(model.deviance))
    reason = model.additional_slots.get("termination_reason")
    assert reason != "reverted_to_best_finite", reason