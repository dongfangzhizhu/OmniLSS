import json
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "fixture_name, expected_has_sigma",
    [
        ("ig_y_x_seed_20260522.json", True),
        ("po_y_x_seed_20260523.json", False),
        ("no_y_x_seed_20260524.json", True),
        ("ga_y_x_seed_20260525.json", True),
    ],
)
def test_r_golden_fixture_schema_is_complete(fixture_name, expected_has_sigma):
    fixture_path = Path(__file__).parent / "fixtures" / "r_golden" / fixture_name
    payload = json.loads(fixture_path.read_text())

    assert isinstance(payload.get("family"), str) and payload["family"]
    assert isinstance(payload.get("formula"), str) and payload["formula"]

    data = payload.get("data")
    assert isinstance(data, dict)
    assert set(data.keys()) == {"x", "y"}
    assert len(data["x"]) == len(data["y"]) > 0

    expected = payload.get("expected")
    assert isinstance(expected, dict)
    assert isinstance(expected.get("mu_coefficients"), list)
    assert len(expected["mu_coefficients"]) >= 1
    assert isinstance(expected.get("deviance"), (int, float))

    has_sigma_coeffs = "sigma_coefficients" in expected
    assert has_sigma_coeffs is expected_has_sigma
    if expected_has_sigma:
        assert isinstance(expected["sigma_coefficients"], list)
        assert len(expected["sigma_coefficients"]) >= 1
