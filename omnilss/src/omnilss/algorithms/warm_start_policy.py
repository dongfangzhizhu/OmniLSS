"""Warm-start policies for RS-style fitting.

Policies are intentionally separated from solver backends so the project can
measure warm-start cost independently from core iteration cost.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class WarmStartResult:
    """Warm-start output used by outer RS loop initialization."""

    initial_eta: dict[str, np.ndarray]
    initial_fitted: dict[str, np.ndarray]
    metadata: dict[str, float | str]


class WarmStartPolicy(Protocol):
    """Policy interface for initializing model parameters before RS iterations."""

    name: str

    def initialize(
        self,
        *,
        y: np.ndarray,
        parameter_names: tuple[str, ...],
        family_name: str | None = None,
    ) -> WarmStartResult:
        """Return initial eta/fitted vectors for each estimable parameter."""


class NoWarmStart:
    """Cheap constant initialization used as a baseline policy."""

    name = "none"

    def initialize(
        self,
        *,
        y: np.ndarray,
        parameter_names: tuple[str, ...],
    ) -> WarmStartResult:
        n = y.shape[0]
        eta = {p: np.zeros(n, dtype=np.float64) for p in parameter_names}
        fitted = {p: np.ones(n, dtype=np.float64) for p in parameter_names}
        return WarmStartResult(
            initial_eta=eta,
            initial_fitted=fitted,
            metadata={"policy": self.name, "n_obs": float(n)},
        )


class FamilyHeuristicInit:
    """Heuristic initialization with stable defaults for common parameters.

    For a small set of families the generic ``mu = mean(y)`` heuristic is
    structurally wrong because their ``mu`` is not a location/mean parameter
    (e.g. WEI2 parameterizes the Weibull with a *rate* ``mu``, where
    ``E[Y] = Gamma(1 + 1/sigma) / mu``).  For those families ``fitted["mu"]``
    is intentionally omitted so that ``rs_fit`` falls through to the
    family-aware ``_initial_mu_beta()`` initialiser, which ports R's
    ``gamlss.family`` ``mu.initial`` expressions (the same policy the outer
    loop already applies to non-mu parameters, see the Phase 2.1 note in
    ``rs_algorithm.rs_fit``).
    """

    name = "family_heuristic"

    # Families whose mu must not be initialized at mean(y); each of them has
    # an R-faithful branch in omnilss._fitting_init._initial_mu_beta.
    _NON_MEAN_MU_FAMILIES = frozenset(
        {
            "WEI2",
            "RGE",
            "PE2",
            # BI / DBI: ``mu`` is a binomial *proportion*, but ``y`` is a
            # success *count*.  ``mean(y) = 5.78`` for bd = 10 is not a
            # proportion at all -- logit(5.78) is NaN, the logit link clips it
            # to +15, and the fit starts at mu = 1.0 with deviance 21360.6
            # before the first IRLS step.  DBI is the double-binomial built on
            # the same response convention.  BB never reached this path
            # because it also estimates sigma, but its mu has the same
            # response convention; it is listed for the same reason.
            "BI",
            "DBI",
            "BB",
        }
    )

    def initialize(
        self,
        *,
        y: np.ndarray,
        parameter_names: tuple[str, ...],
        family_name: str | None = None,
    ) -> WarmStartResult:
        n = y.shape[0]
        y_mean = float(np.mean(y)) if n else 0.0
        y_sd = float(np.std(y)) if n else 1.0
        if family_name is not None and str(family_name).upper() == "GP" and n:
            # GP is heavy-tailed; robust starts avoid an extreme observation
            # dominating both positive log-link parameters.
            positive = y[np.isfinite(y) & (y > 0)]
            if positive.size:
                y_mean = float(np.median(positive))
                y_sd = float(np.subtract(*np.percentile(positive, (75, 25))) / 1.349)
        y_sd = max(y_sd, 1e-6)

        skip_mu = (
            family_name is not None
            and str(family_name).upper() in self._NON_MEAN_MU_FAMILIES
        )

        eta: dict[str, np.ndarray] = {}
        fitted: dict[str, np.ndarray] = {}
        for p in parameter_names:
            if p == "mu":
                if skip_mu:
                    # Omit entirely: rs_fit falls back to _initial_mu_beta.
                    continue
                fitted[p] = np.full(n, y_mean, dtype=np.float64)
            elif p == "sigma":
                fitted[p] = np.full(n, y_sd, dtype=np.float64)
            else:
                fitted[p] = np.ones(n, dtype=np.float64)
            eta[p] = np.zeros(n, dtype=np.float64)

        return WarmStartResult(
            initial_eta=eta,
            initial_fitted=fitted,
            metadata={"policy": self.name, "y_mean": y_mean, "y_sd": y_sd},
        )
