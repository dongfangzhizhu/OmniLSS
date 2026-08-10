"""Fixed R golden artifact replay: BI (Binomial/Bernoulli) core family (Phase 2.1)."""
import json
from pathlib import Path

import numpy as np

from omnilss.distributions import BI
from omnilss.fitting import gamlss
from omnilss.operations import deviance


def test_bi_golden_fixture_replay_matches_expected_metrics():
    fixture_path = Path(__file__).parent / "fixtures" / "r_golden" / "bi_y_x_seed_20260809.json"
    payload = json.loads(fixture_path.read_text())

    data = {
        "y": np.asarray(payload["data"]["y"], dtype=float),
        "x": np.asarray(payload["data"]["x"], dtype=float),
    }
    # BI uses Bernoulli semantics (bd=1), matching the R fixture generator.
    model = gamlss(payload["formula"], family=BI(), data=data)

    np.testing.assert_allclose(
        model.coefficients["mu"],
        np.asarray(payload["expected"]["mu_coefficients"], dtype=float),
        rtol=2e-4,
        atol=2e-4,
    )
    np.testing.assert_allclose(
        deviance(model),
        float(payload["expected"]["deviance"]),
        rtol=2e-4,
        atol=2e-4,
    )
