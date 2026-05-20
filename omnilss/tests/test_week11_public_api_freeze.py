from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from omnilss.api.public import (
    FitResult,
    PredictResult,
    ScoreResult,
    freeze_fit_result,
    freeze_predict_result,
    freeze_score_result,
)


def test_public_result_objects_are_immutable():
    fit = freeze_fit_result(
        family="NO",
        method="RS",
        converged=True,
        iterations=4,
        deviance=12.5,
        coefficients={"mu": [1.0]},
        diagnostics={"condition_number": 100.0},
    )
    with pytest.raises(FrozenInstanceError):
        fit.deviance = 1.0  # type: ignore[misc]


def test_public_result_types_are_stable():
    fit = freeze_fit_result(
        family="NO",
        method="RS",
        converged=True,
        iterations=4,
        deviance=12.5,
        coefficients={"mu": [1.0]},
        diagnostics={"condition_number": 100.0},
    )
    pred = freeze_predict_result(family="NO", parameter_predictions={"mu": [1.1, 1.2]})
    score = freeze_score_result(family="NO", metric="deviance", value=12.5)

    assert isinstance(fit, FitResult)
    assert isinstance(pred, PredictResult)
    assert isinstance(score, ScoreResult)
    assert score.metric == "deviance"
