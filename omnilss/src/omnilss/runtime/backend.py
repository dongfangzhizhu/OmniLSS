"""Runtime backend protocol for formula/model execution layers."""

from __future__ import annotations

from typing import Any, Protocol


class RuntimeBackend(Protocol):
    """Interface for execution backends.

    This is intentionally minimal while the runtime migration is in progress.
    """

    def fit(self, model_spec: Any, data: Any, **kwargs: Any) -> Any:
        """Fit a model specification against input data."""

    def predict(self, fitted_model: Any, data: Any, **kwargs: Any) -> Any:
        """Run prediction against an already-fitted model."""

    def score(self, fitted_model: Any, data: Any, **kwargs: Any) -> float:
        """Return a scalar score for validation/comparison workflows."""
