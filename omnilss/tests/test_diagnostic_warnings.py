from omnilss.diagnostic_warnings import evaluate_numerical_warnings


def test_evaluate_numerical_warnings_flags_expected_conditions():
    slots = {
        "gradient_norm": 2e5,
        "condition_number": 1e11,
        "step_size_by_param": {"mu": 1e-6},
        "lambda_update_failed_params": ("mu",),
    }
    events = evaluate_numerical_warnings(slots)
    codes = {e.code for e in events}
    assert {"EXPLODING_GRADIENT", "BAD_CONDITIONING", "TINY_STEPS", "UNSTABLE_SPLINE"}.issubset(codes)
