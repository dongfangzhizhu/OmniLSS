"""OmniLSS -> ondil distribution contract view (P5-1b).

The written contract lives in
docs/development/p5-ondil-distribution-contract-2026-11-15.md. This module is
the executable side of it: it exposes an OmniLSS FamilyDefinition through the
ondil Distribution surface (theta matrices, dl1_dp1/dl2_dp2/dl2_dpp, supports,
links), in plain numpy, **without importing ondil**.

Why not import ondil: ondil/pyproject.toml requires scikit-learn and numba,
neither of which this project's venv has (contract doc, defect D7). ondil stays
a source reference; the numeric baselines are scipy plus analytic derivatives.

Conventions (contract doc sections 4 and 6):

* theta[:, j] is the j-th *estimable* parameter in FamilyDefinition.parameters order.
* dl1_dp1 / dl2_dp2 / dl2_dpp are per observation, of the **log** likelihood,
  and never summed.
 * ondil's link_derivative is d eta / d mu, while OmniLSS Link.derivative is
  d mu / d eta: inverse_derivative is the matching counterpart.
"""
from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any

import numpy as np

import jax
import jax.numpy as jnp

from .core.distributions import as_distribution_protocol
from .core.links import Link
from .families import FamilyDefinition

__all__ = [
    "FAMILY_SUPPORT",
    "KNOWN_BOUNDARY_VIOLATIONS",
    "ONDIL_PARAMETER_ORDER",
    "OndilDistributionView",
    "OndilLink",
    "Support",
    "as_ondil",
    "contract_report",
    "params_from_theta",
    "probe_support",
    "theta_from_params",
]

_LINK_NAMES = {
    "IdentityLink": "Identity",
    "LogLink": "Log",
    "LogitLink": "Logit",
}


@dataclass(frozen=True)
class Support:
    """Normalized response support (contract doc section 9).

    None means unbounded; closed bounds carry atoms (BEINF at 0/1, PO at 0).
    """

    lower: float | None = None
    upper: float | None = None
    lower_closed: bool = False
    upper_closed: bool = False

    def normalized(self) -> dict[str, Any]:
        return {
            "lower": self.lower,
            "upper": self.upper,
            "lower_closed": self.lower_closed,
            "upper_closed": self.upper_closed,
        }

    def contains(self, value: float) -> bool:
        value = float(value)
        if self.lower is not None:
            if value < self.lower or (value == self.lower and not self.lower_closed):
                return False
        if self.upper is not None:
            if value > self.upper or (value == self.upper and not self.upper_closed):
                return False
        return True

    def probe_points(self) -> tuple[float, float, float]:
        """Return (inside, at_lower, below_lower) probe values."""
        if self.lower is None:
            return 0.5, -10.0, -1e3
        step = max(abs(self.lower), 1.0) * 1e-3
        return self.lower + step, self.lower, self.lower - step


#: Response support for the 22 families ondil implements, copied from
#: ondil/src/ondil/distributions/*.py and normalized to open/closed flags.
FAMILY_SUPPORT: dict[str, Support] = {
    "NO": Support(),
    "NO2": Support(),
    "GU": Support(),
    "RG": Support(),
    "LO": Support(),
    "PE": Support(),
    "JSUo": Support(),
    "ST3": Support(),
    "SST": Support(),
    "TF": Support(),
    "GA": Support(0.0, None),
    "IG": Support(0.0, None),
    "IGAMMA": Support(0.0, None),
    "LOGNO": Support(0.0, None),
    "LOGNO2": Support(0.0, None),
    "WEI": Support(0.0, None),
    # R: dexp(0) = rate > 0, so 0 is in the (closed) support; ondil declares
    # (nextafter(0, 1), inf) and is one sided here -- R semantics win.
    "EXP": Support(0.0, None, lower_closed=True),
    "BE": Support(0.0, 1.0),
    "BEINF": Support(0.0, 1.0, lower_closed=True, upper_closed=True),
    "BEINF0": Support(0.0, 1.0, lower_closed=True),
    "PO": Support(0.0, None, lower_closed=True),
    "ZAGA": Support(0.0, None, lower_closed=True),
}

#: Expected ondil column order (all 22 match FamilyDefinition.parameters).
ONDIL_PARAMETER_ORDER: dict[str, tuple[str, ...]] = {
    "NO": ("mu", "sigma"),
    "NO2": ("mu", "sigma"),
    "GU": ("mu", "sigma"),
    "RG": ("mu", "sigma"),
    "LO": ("mu", "sigma"),
    "PE": ("mu", "sigma", "nu"),
    "JSUo": ("mu", "sigma", "nu", "tau"),
    "ST3": ("mu", "sigma", "nu", "tau"),
    "SST": ("mu", "sigma", "nu", "tau"),
    "TF": ("mu", "sigma", "nu"),
    "GA": ("mu", "sigma"),
    "IG": ("mu", "sigma"),
    "IGAMMA": ("mu", "sigma"),
    "LOGNO": ("mu", "sigma"),
    "LOGNO2": ("mu", "sigma"),
    "WEI": ("mu", "sigma"),
    "EXP": ("mu",),
    "BE": ("mu", "sigma"),
    "BEINF": ("mu", "sigma", "nu", "tau"),
    "BEINF0": ("mu", "sigma", "nu"),
    "PO": ("mu",),
    "ZAGA": ("mu", "sigma", "nu"),
}


def _call_dpqr(
    family: FamilyDefinition,
    attr: str,
    leading: tuple[Any, ...],
    params: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> Any:
    """Call a family d/p/q function the way the canonical adapter does.

    Reimplemented here only for the log-space tail (p(..., log_p=True)), which
    the frozen protocol does not expose; everything else goes through the
    adapter so the two dispatch rules cannot drift.
    """
    fn = getattr(family, attr)
    if fn is None:
        raise NotImplementedError(f"Family {family.name!r} does not define {attr!r}")
    extra = {} if extra is None else dict(extra)
    try:
        signature = inspect.signature(fn)
    except (TypeError, ValueError):
        signature = None
    if signature is not None:
        names = signature.parameters
        accepts_keywords = any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in names.values()
        ) or all(key in names for key in params)
        if accepts_keywords:
            return fn(*leading, **dict(params), **extra)
    ordered = tuple(params[key] for key in family.parameters if key in params)
    return fn(*leading, *ordered, **extra)


#: Families whose log-density does not follow the section 5 boundary rule yet.
#: They clamp the response with maximum(y, eps) so optimisers never see -inf;
#: the price is a finite (or nan) log-density *outside* the support, where R
#: returns -Inf.  Fixing them means replacing a load-bearing clamp, so the
#: contract ratchets instead: the set may only shrink (test_ondil_contract.py
#: fails on any family that is not listed here).
KNOWN_BOUNDARY_VIOLATIONS: dict[str, str] = {
    "IG": "nan at y=0 (0 * inf from the clamped 1.5*log(y) term)",
    "IGAMMA": "-1.1e16 instead of -inf (y clamped to eps)",
    "LOGNO": "nan at y=0 (log(y) of the clamp times 0)",
    "LOGNO2": "-2463 instead of -inf (y clamped to eps)",
    "ZAGA": "0.0 instead of -inf below 0 (rule applied on the density scale)",
}


class OndilLink:
    """Expose an OmniLSS Link through the ondil five-method interface."""

    def __init__(self, name: str, link: Link, support: Support) -> None:
        self.name = name
        self._link = link
        self.support = support

    def link(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(self._link.transform(jnp.asarray(x)))

    def inverse(self, eta: np.ndarray) -> np.ndarray:
        return np.asarray(self._link.inverse(jnp.asarray(eta)))

    def inverse_derivative(self, eta: np.ndarray) -> np.ndarray:
        return np.asarray(self._link.derivative(jnp.asarray(eta)))

    def link_derivative(self, x: np.ndarray) -> np.ndarray:
        # d eta / d mu = 1 / (d mu / d eta) evaluated at eta = g(mu).
        return np.asarray(
            jax.grad(lambda value: jnp.sum(self._link.transform(value)))(jnp.asarray(x))
        )

    def link_second_derivative(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(
            jax.grad(jax.grad(lambda value: jnp.sum(self._link.transform(value))))(
                jnp.asarray(x)
            )
        )


class OndilDistributionView:
    """Read-only numpy view of a family under the ondil Distribution surface."""

    def __init__(self, family: FamilyDefinition) -> None:
        self.family = family
        self._adapter = as_distribution_protocol(family)
        self._parameters = self._adapter.parameters()
        self.estimable = tuple(parameter.name for parameter in self._parameters)
        self.parameter_names = dict(enumerate(self.estimable))
        self.parameter_shape = {index: "scalar" for index in range(len(self.estimable))}
        self.corresponding_gamlss = family.name
        self.is_discrete = family.type == "Discrete"
        self.distribution_support = FAMILY_SUPPORT.get(family.name, Support())
        self.parameter_support = {
            index: _constraint_support(parameter.constraint)
            for index, parameter in enumerate(self._parameters)
        }
        self.links = {
            index: OndilLink(
                _LINK_NAMES.get(type(parameter.link).__name__, "unsupported"),
                parameter.link,
                _constraint_support(parameter.constraint),
            )
            for index, parameter in enumerate(self._parameters)
        }
        self.n_params = len(self.estimable)

    def theta_to_params(self, theta: np.ndarray) -> tuple[np.ndarray, ...]:
        return tuple(self._param_dict(theta).values())

    def _param_dict(self, theta: np.ndarray) -> dict[str, np.ndarray]:
        array = np.atleast_2d(np.asarray(theta, dtype=float))
        if array.shape[-1] != self.n_params:
            raise ValueError(
                f"{self.family.name}: theta has {array.shape[-1]} columns, "
                f"expected {self.n_params} ({self.estimable})"
            )
        return {name: array[:, index] for index, name in enumerate(self.estimable)}

    @staticmethod
    def _numpy(value: Any) -> np.ndarray:
        return np.asarray(value, dtype=float)

    def loglikelihood(self, y: np.ndarray, theta: np.ndarray) -> np.ndarray:
        return self.logpdf(y, theta)

    def logpdf(self, y: np.ndarray, theta: np.ndarray) -> np.ndarray:
        return self._numpy(self._adapter.logpdf(y, self._param_dict(theta)))

    def pdf(self, y: np.ndarray, theta: np.ndarray) -> np.ndarray:
        return np.exp(self.logpdf(y, theta))

    def cdf(self, y: np.ndarray, theta: np.ndarray) -> np.ndarray:
        return self._numpy(self._adapter.cdf(y, self._param_dict(theta)))

    def logcdf(self, y: np.ndarray, theta: np.ndarray) -> np.ndarray:
        params = self._param_dict(theta)
        return self._numpy(_call_dpqr(self.family, "p", (y,), params, {"log_p": True}))

    def ppf(self, q: np.ndarray, theta: np.ndarray) -> np.ndarray:
        return self._numpy(self._adapter.ppf(q, self._param_dict(theta)))

    def quantile(self, q: np.ndarray, theta: np.ndarray) -> np.ndarray:
        return self.ppf(q, theta)

    def _check_param(self, param: int) -> str:
        if param >= self.n_params or param < 0:
            raise ValueError(
                f"{self.family.name} has {self.n_params} parameters; got {param}"
            )
        return self.estimable[param]

    def dl1_dp1(self, y: np.ndarray, theta: np.ndarray, param: int = 0) -> np.ndarray:
        name = self._check_param(param)
        score = self._adapter.score(y, self._param_dict(theta))
        return self._numpy(score[name])

    def dl2_dp2(self, y: np.ndarray, theta: np.ndarray, param: int = 0) -> np.ndarray:
        name = self._check_param(param)
        hessian = self._adapter.hessian(y, self._param_dict(theta))
        return self._numpy(hessian[name][name])

    def dl2_dpp(
        self, y: np.ndarray, theta: np.ndarray, params: tuple[int, int] = (0, 1)
    ) -> np.ndarray:
        if params[0] == params[1]:
            raise ValueError("Cross derivatives must use different parameters.")
        first = self._check_param(params[0])
        second = self._check_param(params[1])
        hessian = self._adapter.hessian(y, self._param_dict(theta))
        return self._numpy(hessian[first][second])

    def initial_values(self, y: np.ndarray) -> np.ndarray:
        initial = self._adapter.init_params(y)
        rows = int(np.size(y))
        return np.column_stack([
            [np.broadcast_to(self._numpy(initial[name]), (rows,)) for name in self.estimable]
        ])


def _constraint_support(constraint: Any) -> Support:
    kind = type(constraint).__name__
    if kind == "Positive":
        return Support(0.0, None)
    if kind == "UnitInterval":
        return Support(0.0, 1.0)
    return Support()


def as_ondil(family: FamilyDefinition | str) -> OndilDistributionView:
    """Wrap a family (or family name) in the ondil-shaped numpy view."""
    if isinstance(family, str):
        from .distribution_registry import resolve

        family = resolve(family)
    return OndilDistributionView(family)


def theta_from_params(
    family: FamilyDefinition | str, params: dict[str, Any], *, rows: int | None = None
) -> np.ndarray:
    """Stack a params dict into a (rows, n_params) theta matrix."""
    view = as_ondil(family)
    missing = [name for name in view.estimable if name not in params]
    if missing:
        raise KeyError(f"{view.family.name}: missing parameters {missing}")
    columns = []
    count = rows
    for name in view.estimable:
        value = np.asarray(params[name], dtype=float)
        if value.ndim == 0:
            value = np.asarray([float(value)]) if count is None else np.full(count, float(value))
        if count is None:
            count = int(value.shape[0])
        columns.append(np.broadcast_to(value, (count,)))
    return np.column_stack(columns)


def params_from_theta(
    family: FamilyDefinition | str, theta: np.ndarray
) -> dict[str, np.ndarray]:
    """Split a theta matrix back into the canonical params dict."""
    return as_ondil(family)._param_dict(theta)


#: Parameter values used by the support probe.  Fixed constants, not fitted
#: initials: several families reject their data-based initial values (BE sigma
#: must stay below 1, TF/PE need nu > 0), and the probe must not depend on
#: whether an initialiser happens to land inside the valid region.
_PROBE_VALUES = {"mu": 0.5, "sigma": 0.5, "nu": 3.0, "tau": 2.0, "pi": 0.5}


def _probe_theta(view: OndilDistributionView) -> np.ndarray:
    return np.asarray(
        [_PROBE_VALUES.get(name, 0.5) for name in view.estimable], dtype=float
    )


def probe_support(family: FamilyDefinition | str) -> dict[str, Any]:
    """Measure the log-density just inside / at / outside the declared support.

    Raw numbers plus a validity flag.  The tests assert the invariants (finite
    inside, -inf outside, atoms finite at closed bounds) for the families whose
    probe parameters are valid; a family that lands on a nan reports
    probe_valid=False instead of pretending the probe succeeded.
    """
    view = as_ondil(family)
    support = view.distribution_support
    inside, at_lower, below = support.probe_points()
    if view.is_discrete:
        lower = 0.0 if support.lower is None else support.lower
        inside, at_lower, below = lower + 1.0, lower, lower - 1.0
    values = np.array([inside, at_lower, below])
    theta = np.tile(_probe_theta(view), (values.size, 1))
    logpdf = view.logpdf(values, theta)
    return {
        "support": support.normalized(),
        "theta": [float(value) for value in theta[0]],
        "points": [float(value) for value in values],
        "logpdf": [float(value) for value in logpdf],
        "probe_valid": bool(np.isfinite(logpdf[0])),
        "boundary_ok": _boundary_ok(view, support, logpdf),
        "known_boundary_violation": view.family.name in KNOWN_BOUNDARY_VIOLATIONS,
    }


def _boundary_ok(view: OndilDistributionView, support: Support, logpdf: np.ndarray) -> bool:
    """Check the section 5 rule: finite inside, -inf at open / below the support."""
    inside, at_lower, below = (float(value) for value in logpdf)
    if not np.isfinite(inside):
        return False
    if support.lower is None:
        return True
    if np.isfinite(at_lower) != support.lower_closed:
        return False
    return below == -np.inf


def contract_report(family: FamilyDefinition | str) -> dict[str, Any]:
    """Per-family contract report (the scan tool writes these to an artifact)."""
    view = as_ondil(family)
    expected = ONDIL_PARAMETER_ORDER.get(view.family.name)
    mismatches: list[str] = []
    if expected is not None and tuple(expected) != tuple(view.estimable):
        mismatches.append(
            f"parameter order {tuple(view.estimable)} != ondil {tuple(expected)}"
        )
    links = {}
    for index, link in view.links.items():
        name = view.estimable[index]
        if link.name == "unsupported":
            mismatches.append(f"link for {name} has no ondil counterpart")
        links[name] = {
            "omnilss": type(view._parameters[index].link).__name__,
            "ondil": link.name,
            "support": link.support.normalized(),
        }
    methods = {
        "logpdf": "supported",
        "pdf": "supported",
        "cdf": "supported",
        "logcdf": "supported",
        "ppf": "supported",
        "dl1_dp1": "supported",
        "dl2_dp2": "supported",
        "dl2_dpp": "supported" if view.n_params > 1 else "n/a (single parameter)",
        "initial_values": "supported",
        "rvs": "unsupported (JAX key, not scipy RNG)",
        "crps": "unsupported",
    }
    if view.is_discrete:
        methods["logpmf"] = "supported"
        methods["pmf"] = "supported"
        methods["logpdf"] = "unsupported (discrete: use logpmf)"
    elif view.family.type == "Mixed":
        methods["logpdf"] = "supported (atoms in the log-density)"
        methods["logpmf"] = "unsupported (Mixed)"
        methods["pmf"] = "unsupported (Mixed)"
    else:
        methods["logpmf"] = "unsupported (continuous)"
        methods["pmf"] = "unsupported (continuous)"
    return {
        "family": view.family.name,
        "type": view.family.type,
        "parameters": list(view.family.parameters),
        "estimable": list(view.estimable),
        "theta_order": {str(index): name for index, name in view.parameter_names.items()},
        "order_matches_ondil": expected is None or tuple(expected) == tuple(view.estimable),
        "ondil_counterpart": expected is not None,
        "distribution_support": view.distribution_support.normalized(),
        "parameter_support": {
            name: view.parameter_support[index].normalized()
            for index, name in enumerate(view.estimable)
        },
        "links": links,
        "methods": methods,
        "mismatches": mismatches,
    }
