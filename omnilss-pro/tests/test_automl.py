from __future__ import annotations

import numpy as np

from omnilss_pro.automl import (
    bootstrap_deviance_intervals,
    rank_candidate_families,
    select_family_by_deviance,
)


class FakeClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []
        self.batch_sizes: list[int] = []

    def fit(self, formula, family, data, sigma_formula="~ 1"):
        raise AssertionError("Pro AutoML must use Core batch_fit, not per-model fit")

    def batch_fit(self, requests):
        self.batch_sizes.append(len(requests))
        results = []
        for request in requests:
            family = request["family"]
            data = request["data"]
            self.calls.append((family, len(data["y"])))
            deviance = {"NO": 10.0, "GA": 8.0, "LOGNO": 9.0}.get(family, 99.0)
            results.append(
                {
                    "model_id": f"model-{family}-{len(self.calls)}",
                    "success": True,
                    "error": "",
                    "deviance": deviance + 0.01 * len(self.calls),
                    "iterations": 3,
                    "converged": True,
                }
            )
        return results


def test_rank_candidate_families_reports_information_criteria() -> None:
    client = FakeClient()
    data = {"y": np.arange(8.0), "x": np.arange(8.0)}

    result = rank_candidate_families(
        client,
        "y ~ x",
        data,
        families=("NO", "GA", "LOGNO"),
        rank_by="deviance",
    )

    assert result["best_family"] == "GA"
    assert result["ranked"][0]["aic"] > result["ranked"][0]["deviance"]
    assert result["ranked"][0]["bic"] > result["ranked"][0]["deviance"]
    assert result["ranked"][0]["gaic"] > result["ranked"][0]["deviance"]
    assert len(client.calls) == 3
    assert client.batch_sizes == [3]


def test_select_family_by_deviance_keeps_compatibility() -> None:
    client = FakeClient()
    data = {"y": np.arange(5.0), "x": np.arange(5.0)}

    result = select_family_by_deviance(client, "y ~ x", data, families=("NO", "GA"))

    assert result["rank_by"] == "deviance"
    assert result["best_family"] == "GA"


def test_bootstrap_deviance_intervals_use_resampled_core_fits() -> None:
    client = FakeClient()
    data = {"y": np.arange(10.0), "x": np.arange(10.0)}

    result = bootstrap_deviance_intervals(
        client,
        "y ~ x",
        data,
        families=("NO",),
        n_bootstrap=5,
        random_state=1,
    )

    no_result = result["families"]["NO"]
    assert no_result["successful_resamples"] == 5
    assert no_result["failed_resamples"] == 0
    assert len(no_result["deviance_ci"]) == 2
    assert [call[1] for call in client.calls] == [10] * 5
    assert client.batch_sizes == [5]
