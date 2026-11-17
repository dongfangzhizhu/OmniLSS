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
    # R accepts a scalar (dMX(0, ...) returns a length-1 vector); the array
    # bookkeeping below needs at least one dimension, so normalise here instead
    # of failing on y.shape[0].
    values = np.atleast_1d(np.asarray(y, dtype=float))
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
    # Fitted-object payload (D2.247).  The accessors only need the fields above;
    # a real gamlssMX/gamlssNP fit also carries its components and diagnostics.
    models: Any = None
    deviance: float | None = None
    post_prob: Any = None
    df_fit: float | None = None
    aic: float | None = None
    sbc: float | None = None
    residuals: Any = None
    seed: int | None = None
    extra: Any = None
    mass_points: Any = None
    ebp: Any = None
    em_iter: int | None = None


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



# ---------------------------------------------------------------------------
# gamlssMX / gamlssMXfits / plotMP (D2.247, R gamlss.mx)
# ---------------------------------------------------------------------------

def _has_missing(data: Mapping[str, Any]) -> bool:
    for value in data.values():
        array = np.asarray(value)
        if array.dtype.kind in "fiu" and not np.all(np.isfinite(array)):
            return True
    return False


def _response_values(formula: str, data: Mapping[str, Any]) -> np.ndarray:
    from .formula_parser import parse_formula

    name = parse_formula(formula).response
    if name not in data:
        raise ValueError(f"the response {name!r} is not in data")
    return np.asarray(data[name], dtype=np.float64)


def _component_values(model: Any, name: str) -> np.ndarray:
    if name not in model.fitted_values:
        raise NotImplementedError(
            f"component parameter {name!r} is fixed (for example a binomial bd); "
            "mixtures of such families are not ported"
        )
    return np.asarray(model.fitted_values[name], dtype=np.float64)


def _component_density(model: Any, y: np.ndarray) -> np.ndarray:
    values = [_component_values(model, name) for name in model.family.parameters]
    return np.asarray(model.family.d(y, *values), dtype=np.float64)


def _component_cdf(model: Any, y: np.ndarray) -> np.ndarray:
    values = [_component_values(model, name) for name in model.family.parameters]
    return np.asarray(model.family.p(y, *values), dtype=np.float64)


def _normal_quantile(values: np.ndarray) -> np.ndarray:
    from scipy.special import ndtri

    return np.asarray(ndtri(np.clip(values, 1e-300, 1.0 - 1e-16)), dtype=np.float64)


def gamlssMX(
    formula: str | Sequence[str] | None = None,
    pi_formula: str = "~1",
    family: Any = "NO",
    weights: Any = None,
    K: int = 2,
    prob: Any = None,
    data: Mapping[str, Any] | None = None,
    control: Mapping[str, Any] | None = None,
    g_control: Any = None,
    zero_component: bool = False,
    **kwargs: Any,
) -> MixtureFit:
    """R gamlssMX: EM fit of a K-component gamlss mixture.

    Ported branch: constant prior probabilities (pi.formula = ~1) without the
    zero component.  Modelled priors need R's multinom and the zero component is
    a point mass at zero; both raise NotImplementedError instead of being
    approximated.  The returned MixtureFit also carries the component models,
    the posterior matrix, the EM deviance and the AIC/SBC.
    """

    from . import gamlss

    if data is None:
        raise ValueError("the data argument is needed")
    if _has_missing(data):
        raise ValueError("The data contains NA's, use data = na.omit(mydata)")
    if zero_component:
        raise NotImplementedError(
            "zero.component=TRUE adds a point mass at zero; that branch of R's "
            "gamlssMX is not ported"
        )
    if str(pi_formula).replace(" ", "") not in {"~1"}:
        raise NotImplementedError(
            "modelled prior probabilities (pi.formula with covariates) need R's "
            "multinom; only pi.formula=~1 is ported"
        )
    if formula is None:
        raise ValueError("formula is required")
    K = int(K)
    if K < 1:
        raise ValueError("K must be a positive integer")
    if isinstance(formula, (list, tuple)):
        formulas = list(formula)
    else:
        formulas = [formula] * K
    if isinstance(family, (list, tuple)):
        families = list(family)
    else:
        families = [family] * K
    if len(formulas) != K:
        raise ValueError("the length of the list for the formula is not equal the number of components")
    if len(families) != K:
        raise ValueError("the length of the list for the family is not equal the number of components")

    settings = dict(control) if isinstance(control, Mapping) else mx_control()
    n_cyc = int(settings.get("n.cyc", 200))
    cc = float(settings.get("cc", 1e-4))
    seed = settings.get("seed")
    sample = settings.get("sample")
    step_probability = min(0.5, 0.10 if sample is None else float(sample))

    y = _response_values(str(formulas[0]), data)
    N = int(y.size)
    pweights = np.ones(N, dtype=np.float64) if weights is None else np.asarray(weights, dtype=np.float64)
    if prob is None:
        proportions = np.full(K, 1.0 / K)
    else:
        proportions = np.asarray(prob, dtype=np.float64).ravel()
        if proportions.size != K:
            raise ValueError("the length of prob must be K")
        proportions = proportions / proportions.sum()

    rng = np.random.default_rng(seed)
    W = np.ones((N, K), dtype=np.float64)
    models: list[Any] = []
    for index in range(K):
        mask = (rng.random(N) < step_probability).astype(np.float64)
        models.append(
            gamlss(
                str(formulas[index]),
                family=families[index],
                data=dict(data),
                weights=mask * pweights,
                control=g_control,
            )
        )
        W[:, index] = _component_density(models[index], y)
    totals = W.sum(axis=1)
    W = np.where(W / totals[:, None] < 1e-10, 0.0, W / totals[:, None])
    proportions = W.mean(axis=0)
    new_deviance = -2.0 * np.sum(np.log(totals))
    old_deviance = new_deviance + 1.0
    iterations = 1
    while abs(old_deviance - new_deviance) > cc and iterations < n_cyc:
        for index in range(K):
            models[index] = gamlss(
                str(formulas[index]),
                family=families[index],
                data=dict(data),
                weights=W[:, index] * pweights,
                control=g_control,
            )
            W[:, index] = proportions[index] * _component_density(models[index], y)
        totals = W.sum(axis=1)
        W = np.where(W / totals[:, None] < 1e-10, 0.0, W / totals[:, None])
        old_deviance = new_deviance
        new_deviance = -2.0 * np.sum(np.log(totals))
        proportions = W.mean(axis=0)
        iterations += 1

    cdf = np.zeros((N, K), dtype=np.float64)
    for index in range(K):
        cdf[:, index] = proportions[index] * _component_cdf(models[index], y)

    parameter_sets = [tuple(model.family.parameters) for model in models]
    fitted_values: dict[str, Any] = {}
    for name in _PARAMETER_ORDER:
        if all(name in names for names in parameter_sets):
            fitted_values[name] = np.vstack(
                [np.asarray(model.fitted_values[name], dtype=np.float64) for model in models]
            )

    df_fit = float(sum(float(model.df_fit) for model in models)) + (K - 1)
    return MixtureFit(
        K=K,
        N=N,
        family=[model.family.name for model in models],
        prob=proportions,
        fitted_values=fitted_values,
        parameters=parameter_sets,
        kind="gamlssMX",
        models=tuple(models),
        deviance=float(new_deviance),
        post_prob=W,
        df_fit=df_fit,
        aic=float(new_deviance) + 2.0 * df_fit,
        sbc=float(new_deviance) + np.log(N) * df_fit,
        residuals=_normal_quantile(cdf.sum(axis=1)),
        seed=None if seed is None else int(seed),
    )


def _np_weights(p: np.ndarray, logf: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, float]:
    """R weightslogl.calc.w: posterior weights and the mixture deviance."""

    with np.errstate(divide="ignore"):
        log_p = np.where(p > 0, np.log(p), -np.inf)
    log_pf = logf + log_p[None, :]
    pf = np.exp(log_pf)
    total = pf.sum(axis=1)
    posterior = pf / total[:, None]
    deviance = float(-2.0 * np.sum(weights * np.log(total)))
    return posterior, deviance


def gamlssNP(
    formula: str | None = None,
    random: str = "~1",
    family: Any = "NO",
    data: Mapping[str, Any] | None = None,
    K: int = 4,
    mixture: str = "np",
    tol: float = 0.5,
    weights: Any = None,
    pluginz: Any = None,
    control: Mapping[str, Any] | None = None,
    g_control: Any = None,
    **kwargs: Any,
) -> Any:
    """R gamlssNP: nonparametric ML mixture over K mass points.

    Ported branch: mixture="np" with random=~1 (intercept mass points).  The
    Gaussian-quadrature branch (mixture="gq"), random slopes and repeated
    measures (random=~x|id) raise NotImplementedError rather than being
    approximated.  K=1 returns the ordinary gamlss fit, exactly as R does.
    """

    from . import gamlss
    from .distribution_registry import resolve

    if data is None:
        raise ValueError("the data argument is needed")
    if _has_missing(data):
        raise ValueError("The data contains NA's, use data = na.omit(mydata)")
    if str(mixture) != "np":
        raise NotImplementedError('only mixture="np" is ported; "gq" is quadrature-based')
    if str(random).replace(" ", "") != "~1":
        raise NotImplementedError("only random=~1 (intercept mass points) is ported")
    if formula is None:
        raise ValueError("formula is required")
    K = int(K)
    if K < 1:
        raise ValueError("K must be a positive integer")

    settings = dict(control) if isinstance(control, Mapping) else np_control()
    em_cycles = int(settings.get("EMn.cyc", 200))
    em_tolerance = float(settings.get("EMcc", 0.001))

    fam = resolve(family) if not hasattr(family, "parameters") else family
    y = _response_values(str(formula), data)
    N = int(y.size)
    pweights = np.ones(N, dtype=np.float64) if weights is None else np.asarray(weights, dtype=np.float64)

    initial = gamlss(str(formula), family=fam, data=dict(data), weights=pweights, control=g_control)
    if K == 1:
        return initial

    nodes, node_weights = np.polynomial.hermite_e.hermegauss(K)
    order = np.argsort(-nodes)
    nodes = np.asarray(nodes, dtype=np.float64)[order]
    proportions = np.asarray(node_weights, dtype=np.float64)[order]
    proportions = proportions / proportions.sum()
    z_blocks = np.repeat(-nodes, N)

    mu_linear = np.asarray(initial.linear_predictors["mu"], dtype=np.float64)
    sigma_initial = np.asarray(initial.fitted_values.get("sigma", np.ones(N)), dtype=np.float64)
    if pluginz is None:
        # R recycles the length-N sigma against the length-K*N z vector.
        scale = (
            np.tile(sigma_initial, K) * z_blocks if "sigma" in fam.parameters else z_blocks
        )
        eta = np.tile(mu_linear, K) + float(tol) * np.asarray(scale, dtype=np.float64)
    else:
        eta = np.tile(mu_linear, K) + np.repeat(
            np.asarray(pluginz, dtype=np.float64) - mu_linear[0], K
        )
    mu_initial = np.asarray(fam.link_inverses["mu"](eta), dtype=np.float64)

    def _expanded_values(model: Any, name: str) -> np.ndarray:
        values = np.asarray(model.fitted_values[name], dtype=np.float64)
        return np.tile(values, K) if values.size == N else values

    def _log_density(mu_values: np.ndarray, model: Any) -> np.ndarray:
        args = []
        for name in fam.parameters:
            if name == "mu":
                args.append(mu_values)
            else:
                args.append(_expanded_values(model, name))
        density = np.asarray(fam.d(np.tile(y, K), *args), dtype=np.float64)
        with np.errstate(divide="ignore"):
            log_values = np.log(np.maximum(density, 1e-300))
        return np.clip(log_values, -700.0, None)

    log_f = _log_density(mu_initial, initial).reshape(K, N).T
    # The mixture deviance is per observation: weights stay length N (R's
    # reported G.deviance equals this, not the length-K*N recycling).
    posterior, deviance = _np_weights(proportions, log_f, pweights)
    ww = posterior.T.reshape(-1)
    kindex = np.repeat(np.arange(K), N)

    expanded = {name: np.tile(np.asarray(values), K) for name, values in data.items()}
    expanded["MASS"] = np.array([f"MASS{index + 1}" for index in kindex])
    expanded_formula = f"{formula} + MASS"
    weights_expanded = np.tile(pweights, K)

    deviance_history = [deviance]
    converged = False
    iteration = 1
    model = initial
    while iteration <= em_cycles and (not converged or iteration <= 9):
        model = gamlss(
            expanded_formula,
            family=fam,
            data=expanded,
            weights=ww * weights_expanded,
            control=g_control,
            start_from=None if iteration == 1 else model,
        )
        mu_expanded = np.asarray(model.fitted_values["mu"], dtype=np.float64)
        log_f = _log_density(mu_expanded, model).reshape(K, N).T
        proportions = np.array(
            [float(ww[kindex == index].mean()) for index in range(K)], dtype=np.float64
        )
        posterior, deviance = _np_weights(proportions, log_f, pweights)
        ww = posterior.T.reshape(-1)
        previous = deviance_history[-1]
        deviance_history.append(deviance)
        converged = abs(deviance_history[-1] - previous) < em_tolerance
        iteration += 1

    mu_linear_blocks = np.asarray(model.linear_predictors["mu"], dtype=np.float64).reshape(K, N)
    ebp = np.sum(ww.reshape(K, N).T * mu_linear_blocks.T, axis=1)
    cdf_expanded = np.asarray(_component_cdf(model, np.tile(y, K)), dtype=np.float64).reshape(K, N).T
    residuals = _normal_quantile(cdf_expanded @ proportions)
    posterior_residuals = _normal_quantile(np.sum(ww.reshape(K, N).T * cdf_expanded, axis=1))

    fitted_values: dict[str, Any] = {}
    for name in fam.parameters:
        values = np.asarray(model.fitted_values[name], dtype=np.float64)
        if values.size == K * N:
            fitted_values[name] = values.reshape(K, N)
        else:
            fitted_values[name] = np.broadcast_to(values, (K, N)).copy()

    labels = list((model.terms.get("mu") or {}).get("term_labels", []))
    coefficients = np.asarray(model.coefficients["mu"], dtype=np.float64).ravel()
    # term_labels excludes the intercept, so label i maps to coefficient i + 1.
    mass_indices = [
        index + 1 for index, label in enumerate(labels) if "MASS" in str(label)
    ]
    mass_points = np.concatenate(
        [[coefficients[0]], coefficients[0] + coefficients[mass_indices]]
    ) if mass_indices else np.full(K, coefficients[0])

    df_fit = float(model.df_fit) + (K - 1)
    return MixtureFit(
        K=K,
        N=N,
        family=[fam.name] * K,
        prob=proportions,
        fitted_values=fitted_values,
        parameters=list(fam.parameters),
        kind="gamlssNP",
        orig_family=fam.name,
        models=(model,),
        deviance=float(deviance_history[-1]),
        post_prob=posterior,
        df_fit=df_fit,
        aic=float(deviance_history[-1]) + 2.0 * df_fit,
        sbc=float(deviance_history[-1]) + np.log(N) * df_fit,
        residuals=residuals,
        mass_points=mass_points,
        ebp=ebp,
        em_iter=iteration - 1,
    )


def gamlssMXfits(
    n: int = 5,
    seed: int | None = None,
    **kwargs: Any,
) -> MixtureFit:
    """R gamlssMXfits: run gamlssMX n times from fresh random starts, keep the best.

    R generates the seeds with runif and rebuilds MX.control(seed=...) for every
    attempt, discarding a caller-supplied control; the port does the same (pass
    n.cyc and friends through kwargs, not through control).
    """

    from dataclasses import replace

    count = int(n)
    if count < 1:
        raise ValueError("n must be a positive integer")
    rng = np.random.default_rng(seed)
    seeds = [int(value) for value in rng.integers(0, 10000, size=count)]
    models: list[MixtureFit] = []
    deviances: list[float] = []
    for value in seeds:
        model = gamlssMX(control=mx_control(seed=value), **kwargs)
        models.append(model)
        deviances.append(float(model.deviance))
    best = int(np.argmin(deviances))
    return replace(
        models[best],
        extra={"dev": deviances, "seed": seeds, "which": best},
    )


@dataclass(frozen=True)
class MixturePersp:
    """Coordinates produced by plotMP (R's persp + trans3d point/stem layout)."""

    x_grid: Any
    y_grid: Any
    z: Any
    points: Any
    stems: Any
    theta: float
    phi: float
    expand: float
    col: str
    xlab: str
    ylab: str
    figure: Any = None


def plotMP(
    x: Any = None,
    y: Any = None,
    prob: Any = None,
    theta: float = 20,
    phi: float = 20,
    expand: float = 0.5,
    col: str = "lightblue",
    xlab: str = "intercept",
    ylab: str = "slope",
    **kwargs: Any,
) -> MixturePersp:
    """R plotMP: 3-D scatter of mass points with probability stems.

    R draws this with base graphics persp/trans3d.  The port returns the same
    geometry (20x20 grid covering the 10% padded range, one stem per mass point)
    and, when matplotlib is installed, also renders a 3-D figure on it.  Pixel
    parity with R graphics is explicitly out of scope (DEVELOPMENT_PLAN P4).
    """

    xs = np.asarray(x, dtype=np.float64).ravel()
    ys = np.asarray(y, dtype=np.float64).ravel()
    ps = np.asarray(prob, dtype=np.float64).ravel()
    if not (xs.size == ys.size == ps.size):
        raise ValueError("x, y and prob not have equal length")
    if xs.size == 0:
        raise ValueError("x, y and prob must not be empty")
    x_pad = (xs.max() - xs.min()) * 0.1
    y_pad = (ys.max() - ys.min()) * 0.1
    x_grid = np.linspace(xs.min() - x_pad, xs.max() + x_pad, 20)
    y_grid = np.linspace(ys.min() - y_pad, ys.max() + y_pad, 20)
    z = np.zeros((x_grid.size, y_grid.size), dtype=np.float64)
    stems = tuple(
        ((float(xi), float(yi), 0.0), (float(xi), float(yi), float(pi)))
        for xi, yi, pi in zip(xs, ys, ps)
    )
    figure = None
    try:  # optional rendering; the geometry above is the contract
        import matplotlib

        matplotlib.use("Agg", force=False)
        import matplotlib.pyplot as plt

        figure = plt.figure()
        axes = figure.add_subplot(111, projection="3d")
        axes.plot_surface(*np.meshgrid(x_grid, y_grid), z, alpha=0.1, color=col)
        axes.scatter(xs, ys, ps, color="red")
        for start, end in stems:
            axes.plot(
                [start[0], end[0]], [start[1], end[1]], [start[2], end[2]],
                color="red",
            )
        axes.set_xlabel(xlab)
        axes.set_ylabel(ylab)
        axes.set_zlabel("probability")
        axes.view_init(elev=float(theta), azim=float(phi))
    except Exception:  # pragma: no cover - matplotlib is optional
        figure = None
    return MixturePersp(
        x_grid=x_grid,
        y_grid=y_grid,
        z=z,
        points=np.column_stack([xs, ys, ps]),
        stems=stems,
        theta=float(theta),
        phi=float(phi),
        expand=float(expand),
        col=col,
        xlab=xlab,
        ylab=ylab,
        figure=figure,
    )


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
    "MixturePersp",
    "gamlssMX",
    "gamlssMXfits",
    "gamlssNP",
    "mx_control",
    "np_control",
    "pMX",
    "plotMP",
    "pmx",
]