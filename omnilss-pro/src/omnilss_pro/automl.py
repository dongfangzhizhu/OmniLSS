"""Prototype Pro-side automation built only on the gRPC client boundary."""

from __future__ import annotations

from typing import Any, Iterable

from .client import OmniLSSCoreClient


def select_family_by_deviance(
    client: OmniLSSCoreClient,
    formula: str,
    data: dict[str, Any],
    families: Iterable[str] = ("NO", "GA", "LOGNO"),
    sigma_formula: str = "~ 1",
) -> dict[str, Any]:
    """Fit candidate families through Core and return the lowest deviance result."""
    results: list[dict[str, Any]] = []
    for family in families:
        try:
            result = client.fit(
                formula=formula,
                family=family,
                data=data,
                sigma_formula=sigma_formula,
            )
            results.append({"family": family, **result})
        except Exception as exc:  # keep searching other candidate families
            results.append({"family": family, "error": str(exc)})

    successful = [r for r in results if "deviance" in r]
    if not successful:
        raise RuntimeError(f"No candidate family fit succeeded: {results}")
    best = min(successful, key=lambda r: r["deviance"])
    return {"best_family": best["family"], "best": best, "candidates": results}
