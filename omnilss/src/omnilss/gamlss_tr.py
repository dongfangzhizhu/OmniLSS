"""R gamlss.tr truncated-distribution helpers (trun.d / trun.p / trun.q / trun.r).

R source: R/gamlss.tr/R/trun.{d,p,q,r}-*.R.

Each helper returns a closure over (family, type, par), exactly like R: for a
base family with density d and CDF p, the truncated distribution keeps only the
mass inside the truncation interval and renormalises.  The varying-truncation
mode is not ported yet and raises NotImplementedError.
"""

from __future__ import annotations

import builtins
from dataclasses import dataclass
from typing import Any, Callable

import jax.numpy as jnp
import numpy as np

from .distribution_registry import resolve
from .families import FamilyDefinition

_TYPES = ("left", "right", "both")

#: Step of the forward difference used to differentiate the truncated log-density.
#: gamlss's numeric.deriv default is sqrt(eps) * min(1, |value|); the generated
#: dld* functions in gamlss.dist use delta = 0.001, which is the convention kept
#: here (the truncated density is smooth, so the step only has to be small).
_SCORE_DELTA = 1.0e-3


def _resolve(family: Any) -> FamilyDefinition:
    return family if isinstance(family, FamilyDefinition) else resolve(family)


def _is_discrete(fam: FamilyDefinition) -> bool:
    return str(fam.type).strip().lower().startswith("discrete")


def _pv(fam: FamilyDefinition, params: dict[str, Any]) -> list[Any]:
    try:
        return [params[name] for name in fam.parameters]
    except KeyError as exc:  # pragma: no cover - defensive
        raise KeyError(
            f"missing family parameter {exc.args[0]!r} for {fam.name}"
        ) from exc


def _check_par(par: Any, trunc_type: str) -> tuple[float, ...]:
    if trunc_type not in _TYPES:
        raise ValueError(f"type must be one of {_TYPES}, got {trunc_type!r}")
    values = tuple(np.atleast_1d(np.asarray(par, dtype=float)).tolist())
    if trunc_type == "both" and len(values) != 2:
        raise ValueError("the length of par should be 2")
    if trunc_type != "both" and len(values) != 1:
        raise ValueError("the length of par should be 1")
    return values


def _unsupported(varying: bool) -> None:
    if varying:
        raise NotImplementedError("varying truncation is not ported yet")


def trun_d(
    par: Any, family: Any = "NO", type: str = "left", varying: bool = False
) -> Callable[..., Any]:
    """R trun.d: truncated density closure fun(x, log=False, **params)."""

    _unsupported(varying)
    fam = _resolve(family)
    par_values = _check_par(par, type)
    discrete = _is_discrete(fam)
    p0 = par_values[0]
    p1 = par_values[1] if len(par_values) == 2 else None

    def fun(x: Any, log: bool = False, **params: Any) -> Any:
        xs = jnp.asarray(x, dtype=jnp.float64)
        pv = _pv(fam, params)
        log_density = fam.d(xs, *pv, log=True)
        if type == "left":
            norm = jnp.log(1.0 - fam.p(p0, *pv))
            outside = (xs <= p0) if discrete else (xs < p0)
        elif type == "right":
            base = p0 - 1 if discrete else p0
            norm = jnp.log(fam.p(base, *pv))
            outside = (xs >= p0) if discrete else (xs > p0)
        else:
            if discrete:
                norm = jnp.log(fam.p(p1, *pv) - fam.p(p0, *pv))
                outside = (xs <= p0) | (xs >= p1)
            else:
                norm = jnp.log(fam.p(p1, *pv) - fam.p(p0, *pv))
                outside = (xs < p0) | (xs > p1)
        out = log_density - norm
        if log:
            return jnp.where(outside, jnp.nan, out)
        return jnp.where(outside, 0.0, jnp.exp(out))

    return fun


def trun_p(
    par: Any, family: Any = "NO", type: str = "left", varying: bool = False
) -> Callable[..., Any]:
    """R trun.p: truncated CDF closure fun(q, lower_tail=True, log_p=False, **params)."""

    _unsupported(varying)
    fam = _resolve(family)
    par_values = _check_par(par, type)
    discrete = _is_discrete(fam)
    p0 = par_values[0]
    p1 = par_values[1] if len(par_values) == 2 else None

    def fun(q: Any, lower_tail: bool = True, log_p: bool = False, **params: Any) -> Any:
        qs = jnp.asarray(q, dtype=jnp.float64)
        pv = _pv(fam, params)
        if type == "left":
            cof = (fam.p(qs, *pv) - fam.p(p0, *pv)) / (1.0 - fam.p(p0, *pv))
            outside = (qs <= p0) if discrete else (qs < p0)
        elif type == "right":
            base = p0 - 1 if discrete else p0
            cof = fam.p(qs, *pv) / fam.p(base, *pv)
            outside = (qs >= p0) if discrete else (qs > p0)
        else:
            if discrete:
                cof = (fam.p(qs, *pv) - fam.p(p0, *pv)) / (
                    fam.p(p1 - 1, *pv) - fam.p(p0, *pv)
                )
                outside = (qs <= p0) | (qs >= p1)
            else:
                cof = (fam.p(qs, *pv) - fam.p(p0, *pv)) / (
                    fam.p(p1, *pv) - fam.p(p0, *pv)
                )
                outside = (qs < p0) | (qs > p1)
        cof = cof if lower_tail else 1.0 - cof
        if log_p:
            return jnp.where(outside, jnp.nan, jnp.log(cof))
        return jnp.where(outside, 0.0, cof)

    return fun


def trun_q(
    par: Any, family: Any = "NO", type: str = "left", varying: bool = False
) -> Callable[..., Any]:
    """R trun.q: truncated quantile closure fun(p, lower_tail=True, log_p=False, **params)."""

    _unsupported(varying)
    fam = _resolve(family)
    par_values = _check_par(par, type)
    discrete = _is_discrete(fam)
    p0 = par_values[0]
    p1 = par_values[1] if len(par_values) == 2 else None

    def fun(
        prob: Any, lower_tail: bool = True, log_p: bool = False, **params: Any
    ) -> Any:
        prob = jnp.asarray(prob, dtype=jnp.float64)
        if log_p:
            prob = jnp.exp(prob)
        if bool(jnp.any(prob <= 0.0)) or bool(jnp.any(prob >= 1.0)):
            raise ValueError("p must be between 0 and 1")
        prob = prob if lower_tail else 1.0 - prob
        pv = _pv(fam, params)
        if type == "left":
            pp = fam.p(p0, *pv)
            return fam.q(pp + prob * (1.0 - pp), *pv)
        if type == "right":
            pp = fam.p(p0 - 1, *pv) if discrete else fam.p(p0, *pv)
            return fam.q(prob * pp, *pv)
        pp1 = fam.p(p0, *pv)
        pp2 = fam.p(p1 - 1, *pv) if discrete else fam.p(p1, *pv)
        return fam.q(prob * (pp2 - pp1) + pp1, *pv)

    return fun


def trun_r(
    par: Any, family: Any = "NO", type: str = "left", varying: bool = False
) -> Callable[..., Any]:
    """R trun.r: truncated random-generation closure fun(n, **params).

    R draws runif(n); exact stream parity is impossible, so this returns a
    numerically correct inverse-CDF sample (same distribution, different stream).
    """

    _unsupported(varying)
    fam = _resolve(family)
    par_values = _check_par(par, type)
    discrete = _is_discrete(fam)
    p0 = par_values[0]
    p1 = par_values[1] if len(par_values) == 2 else None

    def fun(n: int, **params: Any) -> Any:
        if n <= 0:
            raise ValueError("n must be a positive integer")
        n = int(np.ceil(n))
        u = jnp.asarray(np.random.uniform(size=n), dtype=jnp.float64)
        pv = _pv(fam, params)
        if type == "left":
            pp = fam.p(p0, *pv)
            return fam.q(pp + u * (1.0 - pp), *pv)
        if type == "right":
            pp = fam.p(p0 - 1, *pv) if discrete else fam.p(p0, *pv)
            return fam.q(u * pp, *pv)
        pp1 = fam.p(p0, *pv)
        pp2 = fam.p(p1 - 1, *pv) if discrete else fam.p(p1, *pv)
        return fam.q(u * (pp2 - pp1) + pp1, *pv)

    return fun


@dataclass(frozen=True)
class TruncatedFamily(FamilyDefinition):
    """A truncated family: a FamilyDefinition carrying the truncation spec."""

    truncation_type: str = "left"
    truncation_parameters: tuple[float, ...] = ()
    base_name: str = "NO"
    base_family: Any = None
    extra_name: str = "tr"

    def support(self) -> tuple[float | None, float | None]:
        """(lower, upper) bounds of the truncated support; None means open."""

        if self.truncation_type == "left":
            return (float(self.truncation_parameters[0]), None)
        if self.truncation_type == "right":
            return (None, float(self.truncation_parameters[0]))
        return (
            float(self.truncation_parameters[0]),
            float(self.truncation_parameters[1]),
        )

    def y_valid(self, y: Any) -> Any:
        """Response support implied by the truncation masks in trun.d/trun.p.

        Continuous truncation keeps the boundaries (outside means strictly
        beyond them); discrete truncation excludes the boundary value, because
        the mass at the cut belongs to the removed tail.
        """

        values = jnp.asarray(y, dtype=jnp.float64)
        lower, upper = self.support()
        discrete = str(self.type).strip().lower().startswith("discrete")
        result = jnp.asarray(True)
        if lower is not None:
            result = result & (values >= lower + 1.0 if discrete else values >= lower)
        if upper is not None:
            result = result & (values < upper if discrete else values <= upper)
        return result


def trun(
    par: Any = 0,
    family: Any = "NO",
    type: str = "left",
    extra_name: str = "tr",
    local: bool = True,
    delta: Any = None,
    varying: bool = False,
    **kwargs: Any,
) -> TruncatedFamily:
    """R trun(): build the truncated family object.

    local is accepted for call-site compatibility (R uses it to choose the frame
    that receives the generated d/p); Python returns them on the family object.
    """

    base = _resolve(family)
    par_values = _check_par(par, type)
    d = trun_d(par_values, base, type, varying=varying)
    p = trun_p(par_values, base, type, varying=varying)
    # R's trun() builds the fitting family (which carries no q/r), while
    # gen.trun() assigns qNOtr/rNOtr from trun.q/trun.r.  This family object
    # exposes q/r directly, so they must be the truncation-aware ones -- the base
    # quantile would silently return values outside the truncated support.
    q = trun_q(par_values, base, type, varying=varying)
    r = trun_r(par_values, base, type, varying=varying)

    def g_dev_inc(y: Any, **params: Any) -> Any:
        return -2.0 * d(y, log=True, **params)

    # R's gen.trun() family has dldm / d2ldm2 = NULL: gamlss fits those families in
    # gamlssML() by direct numerical maximisation (nlminb), not by IRLS on analytic
    # scores.  The IRLS engine here needs a score, so use gamlss's own numeric.deriv
    # convention -- a forward difference of the *truncated* log-density with delta
    # 0.001 -- and the -dld^2 observed-information form for the second derivative
    # (the convention R uses for numerically differentiated families).
    def _numeric_derivatives(density: Callable[..., Any], names: tuple[str, ...]):
        def make_score(name: str):
            def score(*, y: Any, **params: Any) -> Any:
                base_value = density(y, log=True, **params)
                bumped = dict(params)
                bumped[name] = params[name] + _SCORE_DELTA
                return (density(y, log=True, **bumped) - base_value) / _SCORE_DELTA

            return score

        def make_hessian(name: str):
            def hessian(*, y: Any, **params: Any) -> Any:
                first = make_score(name)(y=y, **params)
                return -(first**2)

            return hessian

        return (
            {name: make_score(name) for name in names},
            {name: make_hessian(name) for name in names},
        )

    numeric_scores, numeric_hessians = _numeric_derivatives(d, tuple(base.parameters))

    return TruncatedFamily(
        name=f"{base.name}{extra_name}",
        parameters=tuple(base.parameters),
        g_dev_inc=g_dev_inc,
        type=base.type,
        links=base.links,
        link_functions=base.link_functions,
        link_inverses=base.link_inverses,
        link_derivatives=base.link_derivatives,
        score_functions=numeric_scores,
        hessian_functions=numeric_hessians,
        fixed_parameters=base.fixed_parameters,
        d=d,
        p=p,
        q=q,
        r=r,
        constructor_parameters=base.constructor_parameters,
        fixed_parameter_defaults=base.fixed_parameter_defaults,
        truncation_type=type,
        truncation_parameters=tuple(float(value) for value in par_values),
        base_name=base.name,
        base_family=base,
        extra_name=str(extra_name),
    )


def gen_trun(
    par: Any = 0,
    family: Any = "NO",
    type: str = "left",
    extra_name: str = "tr",
    **kwargs: Any,
) -> dict[str, Any]:
    """R gen.trun: return the generated d/p/q/r plus the family object.

    R assigns dNOtr / pNOtr / qNOtr / rNOtr / NOtr into the caller frame;
    Python returns a mapping carrying both those names and the generic keys.
    """

    generated = trun(
        par=par, family=family, type=type, extra_name=extra_name, local=False
    )
    base = generated.base_name
    suffix = generated.extra_name
    d_name, p_name, q_name, r_name = (
        f"d{base}{suffix}",
        f"p{base}{suffix}",
        f"q{base}{suffix}",
        f"r{base}{suffix}",
    )
    fam_name = generated.name
    return {
        "d": generated.d,
        "p": generated.p,
        "q": generated.q,
        "r": generated.r,
        "family": generated,
        d_name: generated.d,
        p_name: generated.p,
        q_name: generated.q,
        r_name: generated.r,
        fam_name: generated,
    }


# ---------------------------------------------------------------------------
# fitTail / fitTailAll -- R/gamlss.tr/R/FitTrunTail.R
# ---------------------------------------------------------------------------
#
# R's fitTail() selects the upper/lower tail of y, builds a *truncated* family
# whose cut point is the last observation outside the tail, and fits it with
# gamlssML().  Two details matter for parity:
#
# * the tail side and the truncation side are opposite: type="right" (upper
#   tail) needs trun(..., type="left");
# * when the cut point coincides with the first value inside the tail, R nudges
#   it down by 1/100 of the local span so the first tail point is not lost.
#
# R fits with gamlssML(), i.e. nlminb on the link scale.  Live R 4.6.0's
# gamlssML is pathological on truncated families (it runs mu -> 0 for WEI3tr,
# NOtr, GAtr, ... because the truncation normalisation 1 - F(par) collapses),
# so its fitted numbers are NOT usable as a golden.  We keep R's likelihood
# semantics -- link-scale numerical maximisation of the truncated density --
# but start from the D2.209-corrected initial values and use the D2.211
# closure-AD family for the density, which converges to the actual MLE.  The
# live-R checks in the tests therefore pin the *structure* exactly (cut point,
# tail selection, fitTailAll sequence) and evaluate our coefficients through
# live R's own d*/p* kernels, while R's own failed fit is recorded as a
# known R-side defect.

_TAIL_SIDES = ("right", "left")


def _match_tail_side(value: Any) -> str:
    """R match.arg(type) with partial matching for the tail side."""
    if not isinstance(value, str):
        raise ValueError(f"type must be one of {_TAIL_SIDES}, got {value!r}")
    lowered = value.lower()
    matches = [side for side in _TAIL_SIDES if side.startswith(lowered)]
    if len(matches) != 1:
        raise ValueError(f"type must be one of {_TAIL_SIDES}, got {value!r}")
    return matches[0]


def _as_scalar(value: Any) -> float:
    return float(np.ravel(np.asarray(value, dtype=np.float64))[0])


@dataclass(frozen=True)
class TailFit:
    """Result of R ``gamlss.tr::fitTail()`` (intercept-only ML tail fit)."""

    family_name: str
    parameters: dict[str, float]
    parameter_names: tuple[str, ...]
    deviance: float
    aic: float
    converged: bool
    y: np.ndarray
    threshold: float
    howmany: int
    type: str
    truncation_type: str
    fitted_values: dict[str, np.ndarray]

    def fitted(self, parameter: str) -> np.ndarray:
        """R fitted(m1, \"mu\"): the intercept-only fitted vector."""
        return self.fitted_values[parameter]

    @property
    def n(self) -> int:
        return int(self.y.size)


@dataclass(frozen=True)
class TailAllFit:
    """Result of R ``gamlss.tr::fitTailAll()`` (Hill-type series of fits)."""

    matrix: np.ndarray
    rownames: tuple[str, ...]
    colnames: tuple[str, ...]
    start: int
    end: int
    frequency: int
    howmany: int
    npar: int


def _tail_selection(
    y: Any, percentage: float, howmany: Any, side: str
) -> tuple[np.ndarray, float, int]:
    """R fitTail's tailFun() plus the cut-point (mY) calculation."""
    values = np.asarray(y, dtype=np.float64).ravel()
    n = int(values.size)
    if n == 0:
        raise ValueError("y must contain at least one value")
    count = (
        int(np.floor(n * (float(percentage) / 100.0)))
        if howmany is None
        else int(howmany)
    )
    if count <= 5:
        raise ValueError("too few observations selected")
    ordered = np.sort(values)
    selected = ordered[-count:] if side == "right" else ordered[:count]
    if selected.size <= 5:
        raise ValueError("too few observations selected")

    # R (1-based): mY <- ys[(n - count)]; mYp1 <- ys[(n + 1 - count)]
    k = n - count
    mY = float(ordered[k - 1])
    mYp1 = float(ordered[k])
    if mY == mYp1:
        # R: ys[((n-count)-5):((n-count)+5)] -- inclusive 1-based endpoints.
        window = ordered[k - 6 : k + 5]
        mY -= float(np.sum(np.diff(window))) / 100.0
    return selected, mY, count


def _fit_tail_ml(
    selected: np.ndarray,
    base: FamilyDefinition,
    threshold: float,
    truncation_type: str,
    weights: np.ndarray | None,
    start_values: dict[str, Any],
) -> tuple[dict[str, float], float, bool]:
    """Maximise the truncated log-likelihood on the link scale, as gamlssML does."""
    from scipy.optimize import minimize

    from .ad import build_ad_family
    from .fitting import gamlss_ml

    truncated = trun(par=threshold, family=base, type=truncation_type)
    # D2.211: differentiate the *truncated* density, not a name-keyed rebuild
    # of the base family.
    ad_family = build_ad_family(source_family=truncated)
    initial = gamlss_ml("y ~ 1", family=truncated, data={"y": selected})

    current = {
        name: _as_scalar(initial.fitted_values[name]) for name in truncated.parameters
    }
    for name, value in start_values.items():
        if name in current:
            current[name] = float(value)

    estimable = tuple(truncated.estimable_parameters)
    fixed = {
        name: current[name] for name in truncated.parameters if name not in estimable
    }
    x0 = np.array(
        [
            _as_scalar(truncated.link_functions[name](np.asarray([current[name]])))
            for name in estimable
        ],
        dtype=np.float64,
    )

    def unpack(eta: np.ndarray) -> dict[str, float]:
        values = dict(fixed)
        for name, value in zip(estimable, eta):
            values[name] = _as_scalar(
                truncated.link_inverses[name](np.asarray([value]))
            )
        return values

    def objective_and_gradient(eta: np.ndarray) -> tuple[float, np.ndarray]:
        values = unpack(eta)
        kwargs = {"y": selected, **values}
        deviance = np.asarray(ad_family.g_dev_inc(**kwargs), dtype=np.float64)
        multiplier = 1.0 if weights is None else weights
        total = float(np.sum(deviance * multiplier))
        # The truncated likelihood is unbounded in some corners of the
        # parameter space (R's gamlssML runs mu -> 0 there).  A finite
        # objective plus an analytic gradient keeps the optimiser inside the
        # well-behaved region instead of letting finite differences step into
        # a NaN and abort with "ABNORMAL".
        if not np.isfinite(total):
            return 1.0e10, np.zeros_like(eta)
        gradient = []
        for name, value in zip(estimable, eta):
            score = np.asarray(
                ad_family.score_functions[name](**kwargs), dtype=np.float64
            )
            dparam_deta = _as_scalar(
                truncated.link_derivatives[name](np.asarray([value]))
            )
            gradient.append(-2.0 * float(np.sum(score * multiplier)) * dparam_deta)
        return total, np.asarray(gradient, dtype=np.float64)

    result = minimize(
        objective_and_gradient,
        x0,
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": 1000, "ftol": 1.0e-14, "gtol": 1.0e-10},
    )
    deviance, _gradient = objective_and_gradient(result.x)
    fitted = unpack(result.x)
    return fitted, deviance, bool(result.success)


def fitTail(
    y: Any,
    family: Any = "WEI3",
    percentage: float = 10,
    howmany: Any = None,
    type: str | tuple[str, ...] = "right",
    mu_start: Any = None,
    sigma_start: Any = None,
    nu_start: Any = None,
    tau_start: Any = None,
    weights: Any = None,
    **kwargs: Any,
) -> TailFit:
    """R gamlss.tr::fitTail(): ML fit of a truncated family to a data tail.

    Parameters mirror R: ``percentage`` (or ``howmany``) selects the tail,
    ``type`` is the tail side ("right"/"left"), and ``family`` is any GAMLSS
    family.  ``mu_start``/``sigma_start``/``nu_start``/``tau_start`` override
    the family's initial values (R's ``*.start``).  Extra ``**kwargs`` are
    accepted like R's ``...`` (passed on to the ML fit) but ignored here.

    Note
    ----
    Live R's ``gamlssML`` frequently fails to converge for generated truncated
    families (mu -> 0), so R's own fitted numbers are not a reliable golden;
    this port keeps R's link-scale ML semantics and converges to the actual
    maximum.  See the module note above.
    """
    side = _match_tail_side(type if isinstance(type, str) else "right")
    base = _resolve(family)
    selected, threshold, count = _tail_selection(y, percentage, howmany, side)
    truncation_type = "left" if side == "right" else "right"

    w = None if weights is None else np.asarray(weights, dtype=np.float64).ravel()
    if w is not None and w.size != selected.size:
        raise ValueError("weights must have the same length as the selected tail")

    starts = {
        "mu": mu_start,
        "sigma": sigma_start,
        "nu": nu_start,
        "tau": tau_start,
    }
    starts = {name: value for name, value in starts.items() if value is not None}
    fitted, deviance, converged = _fit_tail_ml(
        selected, base, threshold, truncation_type, w, starts
    )

    names = tuple(base.parameters)
    fitted_vectors = {
        name: np.full(selected.shape, fitted[name], dtype=np.float64) for name in names
    }
    return TailFit(
        family_name=base.name,
        parameters={name: fitted[name] for name in names},
        parameter_names=names,
        deviance=deviance,
        aic=deviance + 2.0 * len(names),
        converged=converged,
        y=selected,
        threshold=threshold,
        howmany=count,
        type=side,
        truncation_type=truncation_type,
        fitted_values=fitted_vectors,
    )


def fitTailAll(
    y: Any,
    family: Any = "WEI3",
    percentage: float = 10,
    howmany: Any = None,
    type: str | tuple[str, ...] = "right",
    plot: bool = True,
    print: bool = True,
    save: bool = False,
    start: int = 5,
    trace: int = 0,
    **kwargs: Any,
) -> TailAllFit:
    """R gamlss.tr::fitTailAll(): a Hill-type sequence of tail fits.

    R loops ``per`` from ``howmany`` down to ``npar + start + 1`` and collects
    the fitted parameters into a time series.  ``plot``/``save``/``print`` are
    accepted for call-site parity (R only uses ``trace`` for printing).
    """
    side = _match_tail_side(type if isinstance(type, str) else "right")
    base = _resolve(family)
    values = np.asarray(y, dtype=np.float64).ravel()
    count = (
        int(np.floor(values.size * (float(percentage) / 100.0)))
        if howmany is None
        else int(howmany)
    )
    npar = len(base.parameters)
    rest = count - (npar + start)
    if rest <= 0:
        raise ValueError("too few observations selected")

    names = tuple(base.parameters)
    matrix = np.zeros((rest, npar), dtype=np.float64)
    pers = list(range(count, npar + start, -1))
    for index, per in enumerate(pers):
        try:
            fit = fitTail(y, family=family, howmany=per, type=side, **kwargs)
            row = [fit.parameters.get(name, np.nan) for name in names]
        except Exception:
            row = [np.nan] * npar
        matrix[index, :] = row
        if trace == 1:
            builtins.print(index + 1, "..")
        elif trace == 2:
            builtins.print(index + 1, row, "..")

    return TailAllFit(
        matrix=matrix,
        rownames=tuple(str(per) for per in pers),
        colnames=names,
        start=npar + start + 1,
        end=count,
        frequency=1,
        howmany=count,
        npar=npar,
    )


__all__ = [
    "TruncatedFamily",
    "gen_trun",
    "trun",
    "trun_d",
    "trun_p",
    "trun_q",
    "trun_r",
    "TailFit",
    "TailAllFit",
    "fitTail",
    "fitTailAll",
]
