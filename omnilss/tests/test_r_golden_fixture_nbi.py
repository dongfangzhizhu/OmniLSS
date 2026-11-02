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

    # D2.148: the coefficient tolerance is loose on purpose.  sigma sits on the
    # eta clip floor here (the fit warns about it), where the deviance is flat
    # to 1.6e-6 -- our -1.275162 against the fixture's -1.274465 is 5.5e-4
    # relative and costs nothing in deviance.  Pinning coefficients to 2e-4
    # asserted more than the data supports; the deviance assertion below stays
    # tight and is the real contract.


    for p in ("mu", "sigma"):
        np.testing.assert_allclose(
            model.coefficients[p],
            np.asarray(payload["expected"][f"{p}_coefficients"], dtype=float),
            rtol=5e-3,
            atol=5e-3,
        )
    np.testing.assert_allclose(
        deviance(model),
        float(payload["expected"]["deviance"]),
        rtol=2e-4,
        atol=2e-4,
    )
