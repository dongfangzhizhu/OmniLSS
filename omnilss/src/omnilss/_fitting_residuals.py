"""Residual and randomized quantile residual helpers extracted from fitting.py."""

from __future__ import annotations

import math
import warnings
from statistics import NormalDist

import jax.numpy as jnp
import numpy as np

from .families import FamilyDefinition

_STANDARD_NORMAL = NormalDist()


def _normal_quantile(probabilities: np.ndarray) -> np.ndarray:
    eps = np.finfo(np.float64).eps
    probs = np.clip(np.asarray(probabilities, dtype=np.float64), eps, 1.0 - eps)
    flattened = probs.ravel()
    quantiles = np.array(
        [_STANDARD_NORMAL.inv_cdf(float(value)) for value in flattened],
        dtype=np.float64,
    )
    return quantiles.reshape(probs.shape)




def _poisson_cdf_scalar(k: int, mu: float) -> float:
    if k < 0:
        return 0.0
    mu = max(float(mu), np.finfo(np.float64).eps)
    pmf = math.exp(-mu)
    total = pmf
    for index in range(1, k + 1):
        pmf *= mu / index
        total += pmf
    return float(min(max(total, 0.0), 1.0))




def _negative_binomial_cdf_scalar(k: int, mu: float, sigma: float) -> float:
    if k < 0:
        return 0.0
    eps = np.finfo(np.float64).eps
    mu = max(float(mu), eps)
    sigma = max(float(sigma), eps)
    size = 1.0 / sigma
    pmf = math.exp(size * math.log(size / (size + mu)))
    total = pmf
    for index in range(1, k + 1):
        pmf *= ((index - 1 + size) / index) * (mu / (size + mu))
        total += pmf
    return float(min(max(total, 0.0), 1.0))




def _geometric_cdf_scalar(k: int, mu: float) -> float:
    if k < 0:
        return 0.0
    mu = max(float(mu), np.finfo(np.float64).eps)
    ratio = mu / (1.0 + mu)
    return float(min(max(1.0 - ratio ** (k + 1), 0.0), 1.0))




def _zip_cdf_scalar(k: int, mu: float, sigma: float) -> float:
    if k < 0:
        return 0.0
    eps = np.finfo(np.float64).eps
    sigma = min(max(float(sigma), eps), 1.0 - eps)
    poisson_cdf = _poisson_cdf_scalar(k, mu)
    return float(min(max(sigma + (1.0 - sigma) * poisson_cdf, 0.0), 1.0))




def _discrete_midpoint_rqres(
    family: FamilyDefinition,
    y: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray | None = None,
) -> np.ndarray:
    y_arr = np.asarray(y, dtype=np.float64)
    mu_arr = np.asarray(mu, dtype=np.float64)
    sigma_arr = None if sigma is None else np.asarray(sigma, dtype=np.float64)
    midpoint = np.zeros_like(y_arr, dtype=np.float64)

    for index, value in enumerate(y_arr):
        observed = int(np.floor(value))
        mu_value = float(mu_arr[index])
        sigma_value = None if sigma_arr is None else float(sigma_arr[index])

        if family.name == "PO":
            lower = _poisson_cdf_scalar(observed - 1, mu_value)
            upper = _poisson_cdf_scalar(observed, mu_value)
        elif family.name == "BI":
            prob = min(max(mu_value, np.finfo(np.float64).eps), 1.0 - np.finfo(np.float64).eps)
            lower = 0.0 if observed <= 0 else 1.0 - prob
            upper = 1.0 - prob if observed <= 0 else 1.0
        elif family.name == "GEOM":
            lower = _geometric_cdf_scalar(observed - 1, mu_value)
            upper = _geometric_cdf_scalar(observed, mu_value)
        elif family.name == "NBI":
            lower = _negative_binomial_cdf_scalar(observed - 1, mu_value, float(sigma_value))
            upper = _negative_binomial_cdf_scalar(observed, mu_value, float(sigma_value))
        elif family.name == "ZIP":
            lower = _zip_cdf_scalar(observed - 1, mu_value, float(sigma_value))
            upper = _zip_cdf_scalar(observed, mu_value, float(sigma_value))
        else:
            raise NotImplementedError(f"rqres is not implemented for family {family.name!r}")

        midpoint[index] = 0.5 * (lower + upper)

    return _normal_quantile(midpoint)




#: Families with a hand-written residual kernel.  They predate the generic
#: path below and their values are pinned by tests, so they keep their own
#: implementation; the generic path must reproduce them (see
#: omnilss/tests/test_generic_rqres.py).
_HAND_CODED_RQRES = frozenset({"PO", "BI", "GEOM", "NBI", "ZIP"})


def _cdf_is_placeholder(family: FamilyDefinition) -> bool:
    """True when the family answers its CDF with the build_ad_family stub."""
    cdf = getattr(family, "p", None)
    if cdf is None:
        return True
    return "placeholder" in getattr(cdf, "__name__", "")


def _cdf_call_args(family: FamilyDefinition, values: dict) -> list:
    """Positional CDF arguments, in the family's own parameters order.

    Same shape as the parity scanner's call assembly: the tuple order is what
    the family declares, and fixed parameters kept *out* of the tuple (BI's /
    DBI's / the zero-variant families' bd) are appended where their signatures
    put them.
    """
    declared = tuple(getattr(family, "parameters", ()) or ())
    args = [
        np.asarray(values[name], dtype=np.float64)
        for name in declared
        if name in values
    ]
    args += [
        np.asarray(value, dtype=np.float64)
        for name, value in values.items()
        if name not in declared
    ]
    return args


def _cdf_values(
    family: FamilyDefinition, q: Any, args: list, params: dict
) -> np.ndarray:
    """Evaluate the family CDF, by keyword for truncated families.

    A truncated family's CDF is R's trun.p closure, whose signature is
    fun(q, lower_tail = TRUE, log_p = FALSE, **params): passing the distribution
    parameters positionally binds them to lower_tail / log_p and the parameter
    lookup then fails ("missing family parameter 'mu'"), which is what made every
    truncated fit report NaN residuals.
    """

    if hasattr(family, "truncation_type"):
        ordered = {name: params[name] for name in family.parameters if name in params}
        if len(ordered) == len(family.parameters):
            return np.asarray(family.p(q, **ordered), dtype=np.float64)
    return np.asarray(family.p(q, *args), dtype=np.float64)


def _generic_rqres(family: FamilyDefinition, values: dict) -> np.ndarray:
    """Randomized-quantile residuals from the family's own CDF.

    R's rqres (gamlss/R/gamlss-5.R) has three branches:

    * Continuous -- u <- cdf(y);
    * Discrete   -- u ~ U(F(y-1), F(y)); R draws, and a deterministic
      implementation cannot, so this uses the midpoint (F(y-1)+F(y))/2, the
      convention the five hand-written kernels already use;
    * Mixed      -- R needs mass.p / prob.mp and draws uniformly inside the
      atom mass.  The midpoint of F(y-1) and F(y) is the same quantity in
      expectation and needs no extra bookkeeping: below the support the CDF is
      0, so an atom at ymin lands in the lower half of its mass, exactly as
      R's draw does on average.

    Returns NaN where the CDF answers NaN (outside the support, or a family
    whose CDF is only valid on part of the line) -- a visibly unusable
    residual rather than a silently invented one.
    """
    y = np.asarray(values["y"], dtype=np.float64)
    params = {k: v for k, v in values.items() if k != "y"}
    args = _cdf_call_args(family, params)
    kind = getattr(family, "type", "Continuous")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if kind == "Continuous":
            probabilities = _cdf_values(family, y, args, params)
        else:
            lower = _cdf_values(family, y - 1.0, args, params)
            upper = _cdf_values(family, y, args, params)
            probabilities = 0.5 * (lower + upper)
    return _normal_quantile(probabilities)


def _build_rqres_callable(family: FamilyDefinition) -> Any | None:
    """Return the residual kernel for family, or None when it has no CDF.

    Before D2.74 only PO/BI/GEOM/NBI/ZIP had a kernel, so residuals came back
    as raw y - mu / Pearson values for the other 141 families -- not comparable
    across families, which is the whole point of a quantile residual (worm
    plots, centile diagnostics).  Every family with a real CDF now gets the
    generic path; the four with the build_ad_family placeholder CDF (DPO1,
    GENERALISEDPOISSON, IGA, LOGSHASHO) still return None and fall back to the
    raw residuals.
    """
    if family.name in _HAND_CODED_RQRES:

        def rqres(**kwargs: Any) -> jnp.ndarray:
            y = np.asarray(kwargs["y"], dtype=np.float64)
            mu = np.asarray(kwargs["mu"], dtype=np.float64)
            sigma_value = kwargs.get("sigma")
            sigma = (
                None
                if sigma_value is None
                else np.asarray(sigma_value, dtype=np.float64)
            )
            values = _discrete_midpoint_rqres(family, y=y, mu=mu, sigma=sigma)
            return jnp.asarray(values, dtype=jnp.float64)

        return rqres

    if _cdf_is_placeholder(family):
        return None

    def generic_rqres(**kwargs: Any) -> jnp.ndarray:
        try:
            values = _generic_rqres(family, kwargs)
        except Exception as exc:  # pragma: no cover - defensive
            warnings.warn(
                f"rqres failed for family {family.name!r}: "
                f"{type(exc).__name__}: {exc}. Residuals are NaN.",
                RuntimeWarning,
                stacklevel=3,
            )
            values = np.full(len(np.asarray(kwargs.get("y", [0.0]))), np.nan)
        return jnp.asarray(values, dtype=jnp.float64)

    return generic_rqres




def _compute_residuals(
    family: FamilyDefinition,
    y: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray | None,
) -> jnp.ndarray:
    if family.name == "LOGNO" and sigma is not None:
        log_y = np.log(np.maximum(y, np.finfo(np.float64).eps))
        return jnp.asarray((log_y - mu) / sigma, dtype=jnp.float64)
    if family.name == "GA" and sigma is not None and family.p is not None:
        probabilities = np.asarray(family.p(y, mu, sigma), dtype=np.float64)
        return jnp.asarray(_normal_quantile(probabilities), dtype=jnp.float64)
    if sigma is not None:
        return jnp.asarray((y - mu) / sigma, dtype=jnp.float64)
    return jnp.asarray((y - mu) / np.sqrt(np.maximum(mu, 1e-12)), dtype=jnp.float64)




__all__ = ["_build_rqres_callable", "_compute_residuals", "_discrete_midpoint_rqres", "_geometric_cdf_scalar", "_negative_binomial_cdf_scalar", "_normal_quantile", "_poisson_cdf_scalar", "_zip_cdf_scalar"]
