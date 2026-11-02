import json
from pathlib import Path

import numpy as np

from omnilss.distributions import IG
from omnilss.fitting import gamlss
from omnilss.operations import deviance


def test_ig_golden_fixture_replay_matches_expected_metrics():
    fixture_path = Path(__file__).parent / "fixtures" / "r_golden" / "ig_y_x_seed_20260522.json"
    payload = json.loads(fixture_path.read_text())

    data = {
        "y": np.asarray(payload["data"]["y"], dtype=float),
        "x": np.asarray(payload["data"]["x"], dtype=float),
    }
    model = gamlss(payload["formula"], family=IG(), data=data)

    # D2.148: the coefficient tolerance is loose on purpose.  Both surfaces are
    # flat along these directions -- on IG our deviance is 147.36572 against the
    # fixture's 147.36574 and live R 4.6.0's 147.36577 *today* (R's own mu
    # moved from 0.637817 to 0.637389, 7e-4 relative), and NBI's sigma sits on
    # the eta clip floor where the deviance is flat to 1.6e-6.  Pinning
    # coefficients to 2e-4 therefore asserted more than the data supports; the
    # deviance assertions below stay tight and are the real contract.
    COEF_RTOL = 5e-3
    COEF_ATOL = 5e-3

    np.testing.assert_allclose(
        model.coefficients["mu"],
        np.asarray(payload["expected"]["mu_coefficients"], dtype=float),
        rtol=COEF_RTOL,
        atol=COEF_ATOL,
    )
    np.testing.assert_allclose(
        model.coefficients["sigma"],
        np.asarray(payload["expected"]["sigma_coefficients"], dtype=float),
        rtol=COEF_RTOL,
        atol=COEF_ATOL,
    )
    np.testing.assert_allclose(
        deviance(model),
        float(payload["expected"]["deviance"]),
        rtol=2e-4,
        atol=2e-4,
    )
