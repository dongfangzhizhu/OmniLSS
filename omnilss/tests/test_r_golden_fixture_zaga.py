"""Fixed R golden artifact replay: ZAGA (Zero-Altered Gamma) core family (Phase 2.1)."""
import json
from pathlib import Path

import numpy as np

from omnilss.distributions_b5 import ZAGA
from omnilss.fitting import gamlss
from omnilss.operations import deviance


def test_zaga_golden_fixture_replay_matches_expected_metrics():
    fixture_path = Path(__file__).parent / "fixtures" / "r_golden" / "zaga_y_x_seed_20260812.json"
    payload = json.loads(fixture_path.read_text())

    data = {
        "y": np.asarray(payload["data"]["y"], dtype=float),
        "x": np.asarray(payload["data"]["x"], dtype=float),
    }
    model = gamlss(payload["formula"], family=ZAGA(), data=data)

    for p in ("mu", "sigma", "nu"):
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
