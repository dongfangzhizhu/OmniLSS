import json
from pathlib import Path

import numpy as np

from omnilss.distributions import GA
from omnilss.fitting import gamlss
from omnilss.operations import deviance


def test_ga_golden_fixture_replay_matches_expected_metrics():
    fixture_path = Path(__file__).parent / "fixtures" / "r_golden" / "ga_y_x_seed_20260525.json"
    payload = json.loads(fixture_path.read_text())

    data = {
        "y": np.asarray(payload["data"]["y"], dtype=float),
        "x": np.asarray(payload["data"]["x"], dtype=float),
    }
    model = gamlss(payload["formula"], family=GA(), data=data)

    np.testing.assert_allclose(
        model.coefficients["mu"],
        np.asarray(payload["expected"]["mu_coefficients"], dtype=float),
        rtol=2e-4,
        atol=2e-4,
    )
    np.testing.assert_allclose(
        model.coefficients["sigma"],
        np.asarray(payload["expected"]["sigma_coefficients"], dtype=float),
        rtol=2e-4,
        atol=2e-4,
    )
    np.testing.assert_allclose(
        deviance(model),
        float(payload["expected"]["deviance"]),
        rtol=2e-4,
        atol=2e-4,
    )
