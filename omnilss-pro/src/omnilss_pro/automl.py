"""Pro-side AutoML helpers built only on the Core gRPC client boundary."""

from __future__ import annotations

import math
from typing import Any, Iterable, Literal

import numpy as np

from .client import OmniLSSCoreClient

RankMetric = Literal["deviance", "aic", "bic", "gaic"]

_DEFAULT_PARAMETER_COUNTS = {
    "NO": 2,
    "GA": 2,
    "LOGNO": 2,
    "PO": 1,
    "BI": 2,
    "WEI": 2,
    "TF": 3,
}


def _response_length(data: dict[str, Any]) -> int:
    for value in data.values():
        if hasattr(value, "__len__"):
            return len(value)
    return 1


def _take_rows(data: dict[str, Any], indices: np.ndarray) -> dict[str, Any]:
    sampled: dict[str, Any] = {}
    for key, value in data.items():
        arr = np.asarray(value)
        sampled[key] = arr[indices]
    return sampled


def _score_candidate(
    result: dict[str, Any],
    *,
    family: str,
    n_obs: int,
    parameter_counts: dict[str, int],
    gaic_k: float,
) -> dict[str, Any]:
    deviance = float(result["deviance"])
    k_params = int(
        parameter_counts.get(
            family.upper(), _DEFAULT_PARAMETER_COUNTS.get(family.upper(), 2)
        )
    )
    return {
        "family": family,
        **result,
        "parameter_count": k_params,
        "aic": deviance + 2.0 * k_params,
        "bic": deviance + math.log(max(n_obs, 2)) * k_params,
        "gaic": deviance + float(gaic_k) * k_params,
    }


def rank_candidate_families(
    client: OmniLSSCoreClient,
    formula: str,
    data: dict[str, Any],
    families: Iterable[str] = ("NO", "GA", "LOGNO"),
    sigma_formula: str = "~ 1",
    *,
    rank_by: RankMetric = "gaic",
    gaic_k: float = 2.0,
    parameter_counts: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Fit candidate families through Core and rank them by model criteria.

    The Pro package never imports OmniLSS Core. Candidate fits are executed
    through the supplied :class:`OmniLSSCoreClient` batch-fit boundary,
    preserving the commercial/GPL process boundary while returning deviance,
    AIC, BIC, and GAIC rankings.
    """

    counts = {**_DEFAULT_PARAMETER_COUNTS, **(parameter_counts or {})}
    n_obs = _response_length(data)
    family_list = list(families)
    batch_requests = [
        {
            "formula": formula,
            "family": family,
            "data": data,
            "sigma_formula": sigma_formula,
        }
        for family in family_list
    ]
    batch_results = client.batch_fit(batch_requests)
    results: list[dict[str, Any]] = []
    for family, result in zip(family_list, batch_results, strict=True):
        if result.get("success", True):
            results.append(
                _score_candidate(
                    result,
                    family=family,
                    n_obs=n_obs,
                    parameter_counts=counts,
                    gaic_k=gaic_k,
                )
            )
        else:
            results.append(
                {"family": family, "error": str(result.get("error", "fit failed"))}
            )

    successful = [r for r in results if rank_by in r]
    if not successful:
        raise RuntimeError(f"No candidate family fit succeeded: {results}")
    ranked = sorted(successful, key=lambda r: r[rank_by])
    return {
        "best_family": ranked[0]["family"],
        "best": ranked[0],
        "rank_by": rank_by,
        "ranked": ranked,
        "candidates": results,
    }


def select_family_by_deviance(
    client: OmniLSSCoreClient,
    formula: str,
    data: dict[str, Any],
    families: Iterable[str] = ("NO", "GA", "LOGNO"),
    sigma_formula: str = "~ 1",
) -> dict[str, Any]:
    """Backward-compatible AutoML helper that ranks by deviance."""

    return rank_candidate_families(
        client,
        formula,
        data,
        families,
        sigma_formula,
        rank_by="deviance",
    )


def bootstrap_deviance_intervals(
    client: OmniLSSCoreClient,
    formula: str,
    data: dict[str, Any],
    families: Iterable[str] = ("NO",),
    sigma_formula: str = "~ 1",
    *,
    n_bootstrap: int = 100,
    confidence: float = 0.95,
    random_state: int | None = None,
) -> dict[str, Any]:
    """Estimate bootstrap deviance intervals through repeated Core fits.

    Bootstrap samples are sent to Core via the batch-fit gRPC boundary. The
    returned intervals are intentionally based on deviance because the public
    Core fit response already exposes that stable model-quality metric across
    all supported families.
    """

    if n_bootstrap <= 0:
        raise ValueError("n_bootstrap must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between 0 and 1")

    n_obs = _response_length(data)
    rng = np.random.default_rng(random_state)
    alpha = 1.0 - confidence
    family_results: dict[str, Any] = {}

    for family in families:
        deviances: list[float] = []
        errors: list[str] = []
        batch_requests = []
        for _ in range(n_bootstrap):
            indices = rng.integers(0, n_obs, size=n_obs)
            sample = _take_rows(data, indices)
            batch_requests.append(
                {
                    "formula": formula,
                    "family": family,
                    "data": sample,
                    "sigma_formula": sigma_formula,
                }
            )
        for result in client.batch_fit(batch_requests):
            if result.get("success", True):
                deviances.append(float(result["deviance"]))
            else:
                errors.append(str(result.get("error", "fit failed")))

        if deviances:
            lo, hi = np.quantile(deviances, [alpha / 2.0, 1.0 - alpha / 2.0])
            family_results[family] = {
                "deviance_mean": float(np.mean(deviances)),
                "deviance_ci": [float(lo), float(hi)],
                "successful_resamples": len(deviances),
                "failed_resamples": len(errors),
                "errors": errors[:5],
            }
        else:
            family_results[family] = {
                "error": "all bootstrap fits failed",
                "failed_resamples": len(errors),
                "errors": errors[:5],
            }

    return {
        "confidence": confidence,
        "n_bootstrap": n_bootstrap,
        "families": family_results,
    }
