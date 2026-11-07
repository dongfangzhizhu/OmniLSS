"""R gamlss.mx mixture density / CDF helpers plus the MX/NP control lists.

R source: R/gamlss.mx/R/gamlssMX-10-11-07.R (MX.control, dMX, pMX) and
R/gamlss.mx/R/gamlssNP.R (NP.control).

dMX()/pMX() evaluate a mixture of gamlss families: component i contributes
pi_i * dFAM_i(y, params_i) and the components are summed per observation.  The
parameter lists are positional, exactly like R, and each component only
receives the parameters its family declares (mu, sigma, nu, tau in that order).

R's probability check is dead code --
    if (!(9.9999999999 > sump) && (sump < 1.00000000001)) stop(...)
can never be true because the two halves are mutually exclusive -- so pi is not
validated there either.  The port keeps that behaviour and documents it instead
of adding a check R does not have.

The two control constructors reproduce R's guards, including the two that do
not actually work: MX.control's cc guard warns but leaves cc untouched (it
assigns to c.crit), and NP.control's EMcc guard does the same.
"""
from __future__ import annotations

import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from .distributions3 import cdf as _cdf
from .distributions3 import pdf as _pdf
from .distribution_registry import resolve

_PARAMETER_ORDER = ("mu", "sigma", "nu", "tau")


def _warn(message: str) -> None:
    warnings.warn(message, UserWarning, stacklevel=3)


def mx_control(
    cc: float = 0.0001,
    n_cyc: int = 200,
    trace: bool | float = False,
    seed: int | None = None,
    plot: bool = True,
    sample: float | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """R MX.control: EM settings for gamlssMX().

    R's cc guard is a no-op (it writes to a different variable name), so the
    warning is raised but the value is kept; n.cyc, by contrast, really is
    replaced by 100 (R's message says 20 -- the message is wrong, not the code).
    """

    if cc <= 0:
        _warn(
            "the value of cc supplied is zero or negative the default value of 0.0001 was used instead"
        )
    if n_cyc < 1:
        _warn(
            "the value of no cycles supplied is zero or negative the default value of 20 was used instead"
        )
        n_cyc = 100
    if not isinstance(trace, bool):
        if isinstance(trace, (int, float)) and float(trace) <= 0:
            _warn(
                "the value of trace supplied is less or equal t zero the default of 1 was used instead"
            )
            trace = 1
    if seed is None:
        seed = int(np.random.default_rng().integers(0, 100000))
    if sample is not None:
        if sample > 1 or sample < 0:
            sample = 0.1
            _warn(
                "the value of sample supplied is not probability the default of 0.10 was used instead"
            )
    plot = plot if isinstance(plot, bool) else True
    return {
        "cc": cc,
        "n.cyc": n_cyc,
        "trace": trace,
        "seed": seed,
        "plot": plot,
        "sample": sample,
    }


def np_control(
    EMcc: float = 0.001,
    EMn_cyc: int = 200,
    damp: bool = True,
    trace: bool = True,
    plot_opt: int = 3,
    **kwargs: Any,
) -> dict[str, Any]:
    """R NP.control: EM settings for gamlssNP().

    Both of R's guards here are no-ops: EMcc's writes to c.crit and EMn.cyc's
    writes to n.cyc, so neither value is actually replaced (only the warning is
    emitted).  The port reproduces that.
    """

    if EMcc <= 0:
        _warn(
            "the value of cc supplied is zero or negative the default value of 0.001 was used instead"
        )
    if EMn_cyc < 1:
        # R assigns n.cyc here, not EMn.cyc, so the value is kept unchanged.
        _warn(
            "the value of no cycles supplied is zero or negative the default value of 200 was used instead"
        )
    if not isinstance(trace, bool):
        if isinstance(trace, (int, float)) and float(trace) <= 0:
            _warn(
                "the value of trace supplied is less or equal t zero the default of 1 was used instead"
            )
            trace = 1
    return {
        "EMcc": EMcc,
        "EMn.cyc": EMn_cyc,
        "trace": bool(trace),
        "damp": bool(damp),
        "plot.opt": plot_opt,
    }


def _as_components(value: Any, count: int, name: str) -> list[Any]:
    if value is None:
        # R's defaults only matter for the parameters a family actually uses;
        # an unused list is simply never indexed.
        return [None] * count
    if isinstance(value, (list, tuple)):
        items = list(value)
    elif isinstance(value, np.ndarray) and value.ndim == 1 and value.shape[0] == count:
        items = list(value)
    else:
        items = [value]
    if len(items) != count:
        raise ValueError(f"{name} must have {count} entries, got {len(items)}")
    return items


def _mixture(
    y: Any,
    mu: Any,
    sigma: Any,
    nu: Any,
    tau: Any,
    pi: Any,
    family: Any,
    log: bool,
    use_cdf: bool,
) -> np.ndarray:
    values = np.asarray(y, dtype=float)
    component_count = len(pi)
    families = _as_components(family, component_count, "family")
    probabilities = _as_components(pi, component_count, "pi")
    supplied = {"mu": mu, "sigma": sigma, "nu": nu, "tau": tau}
    resolved = {name: _as_components(value, component_count, name) for name, value in supplied.items()}

    density = np.ones((values.shape[0], component_count))
    for index in range(component_count):
        definition = resolve(str(families[index])) if not hasattr(families[index], "pdf") else families[index]
        parameters = list(definition.parameters)
        missing = [name for name in parameters if name in _PARAMETER_ORDER and resolved[name][index] is None]
        if missing:
            raise ValueError(f"{definition.name} needs {missing[0]} for component {index + 1}")
        ordered = [
            np.asarray(resolved[name][index], dtype=float)
            for name in _PARAMETER_ORDER
            if name in parameters
        ]
        component = _cdf(definition, values, *ordered) if use_cdf else _pdf(definition, values, *ordered)
        density[:, index] = float(probabilities[index]) * np.asarray(component, dtype=float)
    total = density.sum(axis=1)
    return np.log(total) if log else total


def dmx(
    y: Any,
    mu: Any = None,
    sigma: Any = None,
    nu: Any = None,
    tau: Any = None,
    pi: Sequence[float] = (0.2, 0.8),
    family: Any = ("NO", "NO"),
    log: bool = False,
    **kwargs: Any,
) -> np.ndarray:
    """R dMX: mixture density of gamlss families, one row per observation."""

    return _mixture(y, mu, sigma, nu, tau, pi, family, log, use_cdf=False)


def pmx(
    q: Any,
    mu: Any = None,
    sigma: Any = None,
    nu: Any = None,
    tau: Any = None,
    pi: Sequence[float] = (0.2, 0.8),
    family: Any = ("NO", "NO"),
    log: bool = False,
    **kwargs: Any,
) -> np.ndarray:
    """R pMX: mixture CDF of gamlss families, one row per observation."""

    return _mixture(q, mu, sigma, nu, tau, pi, family, log, use_cdf=True)


# R spellings (the coverage tool normalises "MX.control" to MX_control).
dMX = dmx
pMX = pmx
MX_control = mx_control
NP_control = np_control


# ---------------------------------------------------------------------------
# get_pdfMX / get_cdfMX / get_pdfNP / get_cdfNP
# ---------------------------------------------------------------------------
#
# R source: R/gamlss.mx/R/gamlssMX-10-11-07.R:710,768 and
#           R/gamlss.mx/R/gamlssNP.R:806,860.
#
# These four accessors take a *fitted* mixture object and an observation number
# and return a function of y: the fitted pdf/cdf of that observation.  They do
# no fitting themselves -- they collect the fitted parameter values of each
# component at that observation, plus the mixing probabilities, and close over
# a call to dMX()/pMX().  Because gamlssMX()/gamlssNP() are not ported yet, the
# fitted object is represented here by MixtureFit, which carries exactly the
# fields R's accessors read (K, family/orig.family, models$parameters,
# fit vectors, prob, model.pi, bd).
#
# Two R behaviours are reproduced deliberately:
#
# * ``if (any(npar==npar[1])) nopar <- npar[1] else stop(...)`` is dead code --
#   the first element always equals itself -- so components with different
#   parameter counts are *not* rejected; nopar is simply npar[1].
# * fitted.gamlssNP forces K=1 for every parameter except mu, so non-mu
#   parameters are shared by all NP components.  MX components keep their own.
#
# The binomial (bd) branch is not ported: dMX()/pMX() here do not accept bd,
# and R's .gamlss.bi.list families need it.  Encountering one raises
# NotImplementedError rather than silently returning a wrong density.

#: R's ``.gamlss.bi.list`` (live R 4.6.0, gamlss namespace).
_BINOMIAL_FAMILIES = frozenset(
    {
        "BI", "Binomial", "BB", "Beta Binomial",
        "ZIBI", "ZIBB", "ZABI", "ZABB", "DBI",
        "BItr", "BBtr", "ZIBItr", "ZIBBtr", "ZABItr", "ZABBtr", "DBItr",
    }
)


@dataclass(frozen=True)
class MixtureFit:
    """R gamlssMX / gamlssNP fitted object, reduced to what the accessors read.

    ``fitted_values`` maps a parameter name ("mu", "sigma", "nu", "tau") to
    either a flat length-(K*N) vector in R's ``mu.fv`` layout, a (K, N) array,
    or a length-N vector shared by every component.  ``parameters`` is a
    per-component sequence of parameter names for MX, or a single sequence for
    NP (R's ``object$parameters``).
    """

    K: int
    N: int
    family: Sequence[str]
    prob: Any
    fitted_values: Mapping[str, Any]
    parameters: Any
    kind: str = "gamlssMX"
    orig_family: str | None = None
    model_pi: bool = False
    bd: Any = None


def _coerce_fit(object: Any) -> MixtureFit:
    if isinstance(object, MixtureFit):
        return object
    if isinstance(object, Mapping):
        return MixtureFit(**dict(object))
    raise TypeError(
        "object must be a MixtureFit (or a mapping with the same fields)"
    )


def _fitted_at(fit: MixtureFit, name: str, component: int, observation: int,
               is_np: bool) -> float:
    values = np.asarray(fit.fitted_values[name], dtype=np.float64)
    if values.ndim == 1:
        if values.size == fit.K * fit.N:
            values = values.reshape(fit.K, fit.N)
        elif values.size == fit.N:
            values = np.broadcast_to(values, (fit.K, fit.N))
        else:
            raise ValueError(
                f"{name} fitted values must have K*N or N entries, got {values.size}"
            )
    # R fitted.gamlssNP: `if (what!="mu") K=1` -- non-mu parameters are shared.
    source = 0 if (is_np and name != "mu") else component
    return float(values[source, observation - 1])


def _collect_mixture(object: Any, observation: Any, expected_kind: str):
    fit = _coerce_fit(object)
    if fit.kind != expected_kind:
        raise ValueError(f"the object should be an {expected_kind} object")
    obs = int(observation)
    if obs < 1 or obs > fit.N:
        raise IndexError(f"observation must be between 1 and {fit.N}, got {obs}")
    is_np = expected_kind == "gamlssNP"
    K = int(fit.K)

    if is_np:
        declared = [tuple(fit.parameters)] * K
    else:
        declared = [tuple(names) for names in fit.parameters]

    lists: dict[str, list[float]] = {}
    for name in _PARAMETER_ORDER:
        values = [
            _fitted_at(fit, name, i, obs, is_np)
            for i in range(K)
            if name in declared[i]
        ]
        if values:
            lists[name] = values

    probabilities = np.asarray(fit.prob, dtype=np.float64)
    if not is_np and fit.model_pi:
        if probabilities.ndim != 2:
            raise ValueError("prob must be an (N, K) array when model_pi=True")
        pi = [float(probabilities[obs - 1, i]) for i in range(K)]
    else:
        flat = probabilities.ravel()
        pi = [float(flat[i]) for i in range(K)]

    # R: `if (any(npar==npar[1])) nopar <- npar[1] else stop(...)` -- always
    # true, so differing component parameter counts are silently accepted.
    nopar = len(declared[0])
    family = (
        [fit.orig_family] * K if is_np else [str(name) for name in fit.family]
    )
    return lists, pi, nopar, family, fit.bd


def _plotfun(use_cdf: bool, lists, pi, nopar, family, bd):
    first = str(family[0]) if family else ""
    if bd is not None or first in _BINOMIAL_FAMILIES:
        raise NotImplementedError(
            "binomial (bd) mixtures are not ported: dMX/pMX need a bd argument"
        )
    kernel = pmx if use_cdf else dmx

    def plotfun(y: Any) -> np.ndarray:
        kwargs: dict[str, Any] = {"pi": pi, "family": family}
        for name in _PARAMETER_ORDER[:nopar]:
            if name in lists:
                kwargs[name] = lists[name]
        return kernel(y, **kwargs)

    return plotfun


def get_pdfMX(object: Any = None, observation: int = 1):
    """R get_pdfMX: return the fitted pdf of one observation of a gamlssMX fit."""

    lists, pi, nopar, family, bd = _collect_mixture(object, observation, "gamlssMX")
    return _plotfun(False, lists, pi, nopar, family, bd)


def get_cdfMX(object: Any = None, observation: int = 1):
    """R get_cdfMX: return the fitted cdf of one observation of a gamlssMX fit."""

    lists, pi, nopar, family, bd = _collect_mixture(object, observation, "gamlssMX")
    return _plotfun(True, lists, pi, nopar, family, bd)


def get_pdfNP(object: Any = None, observation: int = 1):
    """R get_pdfNP: return the fitted pdf of one observation of a gamlssNP fit."""

    lists, pi, nopar, family, bd = _collect_mixture(object, observation, "gamlssNP")
    return _plotfun(False, lists, pi, nopar, family, bd)


def get_cdfNP(object: Any = None, observation: int = 1):
    """R get_cdfNP: return the fitted cdf of one observation of a gamlssNP fit."""

    lists, pi, nopar, family, bd = _collect_mixture(object, observation, "gamlssNP")
    return _plotfun(True, lists, pi, nopar, family, bd)


__all__ = [
    "MX_control",
    "NP_control",
    "MixtureFit",
    "dMX",
    "dmx",
    "get_cdfMX",
    "get_cdfNP",
    "get_pdfMX",
    "get_pdfNP",
    "mx_control",
    "np_control",
    "pMX",
    "pmx",
]