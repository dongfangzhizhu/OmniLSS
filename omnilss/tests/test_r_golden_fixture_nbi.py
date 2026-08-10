"""Fixed R golden artifact replay: NBI (Negative Binomial I) core family (Phase 2.1)."""
import json
from pathlib import Path

import numpy as np

from omnilss.distributions import NBI
from omnilss.fitting import gamlss
from omnilss.operations import deviance


def test_nbi_golden_fixture_replay_matches_expected_metrics():
    fixture_path = Path(__file__).parent / "fixtures" / "r_golden" / "nbi_y_x_seed_20260810.json"
    payload = json.loads(fixture_path.read_text())

    data = {
        "y": np.asarray(payload["data"]["y"], dtype=float),
        "x": np.asarray(payload["data"]["x"], dtype=float),
    }
    model = gamlss(payload["formula"], family=NBI(), data=data)

    for p in ("mu", "sigma"):
        np.testing.assert_allclose(
            model.coefficients[p],
            np.asarray(payload["expected"][f"{p}_coefficients"], dtype=float),
            rtol=2e-4,
            atol=2e-4,
        )
    np.testing.assert_allclose(
        deviance(model),
        float(payload["expected"]["deviance"]),
        rtol=2e-4,
        atol=2e-4,
    )
