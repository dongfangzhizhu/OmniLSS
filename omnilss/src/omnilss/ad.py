"""JAX Auto-Differentiation support for GAMLSS Families."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import fields as _dataclass_fields
from dataclasses import is_dataclass as _is_dataclass
from typing import Any

import jax
import jax.numpy as jnp

from .families import FamilyDefinition


def _prepare_jax_args(args: tuple[Any, ...]) -> tuple[list[jnp.ndarray], bool]:
    """Broadcast scalar parameters to match vector inputs for vmapped calls.

    Optimized version with reduced overhead.
    """
    # Fast path for common case: all arrays of same size
    arrays = [jnp.asarray(arg, dtype=jnp.float64) for arg in args]

    # Quick check if all same size
    sizes = [int(arr.size) if arr.ndim > 0 else 1 for arr in arrays]
    target_size = max(sizes)

    if all(s == target_size for s in sizes):
        # All same size - fast path
        scalar_output = target_size == 1
        prepared = [
            jnp.reshape(arr, (-1,)) if arr.ndim > 0 else jnp.reshape(arr, (1,))
            for arr in arrays
        ]
        return prepared, scalar_output

    # Slow path: need broadcasting
    scalar_output = target_size == 1 and all(arr.ndim == 0 for arr in arrays)
    prepared: list[jnp.ndarray] = []

    for arr in arrays:
        flat = jnp.reshape(arr, (-1,)) if arr.ndim > 0 else jnp.reshape(arr, (1,))
        if flat.size == target_size:
            prepared.append(flat)
        elif flat.size == 1:
            prepared.append(jnp.repeat(flat, target_size))
        else:
            raise ValueError(
                "Arguments must be broadcastable to a common length; "
                f"got sizes {sizes}"
            )

    return prepared, scalar_output


def _log_density_from_family(
    family: FamilyDefinition,
    parameters: tuple[str, ...],
) -> Callable[..., jnp.ndarray]:
    """Wrap an existing family's density closure as a scalar log-density.

    This is what lets build_ad_family differentiate a family that was
    *constructed at run time* (trun(), cens(), a user-supplied
    FamilyDefinition) instead of a hand-written scalar kernel.  The closure is
    called the way every family density is called -- family.d(y, log=True,
    mu=..., sigma=...) -- so anything the family already encodes (truncation
    normalisation, fixed shape parameters, ...) is part of the differentiated
    expression.

    Scalar in, scalar out: build_ad_family maps this with jax.vmap, and every
    family density returns a length-1 array for scalar inputs.
    """

    density = getattr(family, "d", None)
    if not callable(density):
        raise TypeError(
            f"Family {getattr(family, 'name', family)!r} has no d() closure to "
            "differentiate; pass log_pdf_func= instead."
        )

    def log_pdf(y: Any, *params: Any) -> jnp.ndarray:
        kwargs = dict(zip(parameters, params))
        value = jnp.asarray(density(y, log=True, **kwargs), dtype=jnp.float64)
        return jnp.reshape(value, ())

    return log_pdf


def _hessian_from_score(
    score: Callable[..., Any],
) -> Callable[..., jnp.ndarray]:
    """gamlss scoring second derivative: d2l/dx2 := -(dl/dx)**2.

    This is the convention R uses for every family it differentiates
    numerically (gen.trun, gen.cens): the result is the expected negative
    information, so it is never positive and IRLS working weights stay
    positive.  A raw AD second derivative is the observed information, which
    can change sign (and does, for truncated densities).
    """

    def hessian(*args: Any, **kwargs: Any) -> jnp.ndarray:
        value = jnp.asarray(score(*args, **kwargs), dtype=jnp.float64)
        return -(value**2)

    return hessian


def build_ad_family(
    family_class: type[FamilyDefinition] | None = None,
    name: str | None = None,
    parameters: tuple[str, ...] | None = None,
    log_pdf_func: Callable[..., jnp.ndarray] | None = None,
    type_: str | None = None,
    links: Mapping[str, str] | None = None,
    link_functions: Mapping[str, Callable[..., Any]] | None = None,
    link_inverses: Mapping[str, Callable[..., Any]] | None = None,
    link_derivatives: Mapping[str, Callable[..., Any]] | None = None,
    fixed_parameters: tuple[str, ...] | None = None,
    fixed_parameter_defaults: Mapping[str, Any] | None = None,
    d: Callable[..., Any] | None = None,
    p: Callable[..., Any] | None = None,
    q: Callable[..., Any] | None = None,
    r: Callable[..., Any] | None = None,
    constructor_parameters: Mapping[str, Any] | None = None,
    score_overrides: Mapping[str, Callable[..., Any]] | None = None,
    hessian_overrides: Mapping[str, Callable[..., Any]] | None = None,
    source_family: FamilyDefinition | None = None,
    hessian_from_score: bool | None = None,
) -> FamilyDefinition:
    """Build a FamilyDefinition dynamically using JAX auto-differentiation (AD).

    This factory massively accelerates adding new families to OmniLSS.
    By just providing the raw mathematical scalar `log_pdf_func`, JAX AD provides
    exact analytic derivatives for score and hessian functions mapping over arrays.

    Parameters
    ----------
    source_family : FamilyDefinition, optional
        An existing family whose own density closure is differentiated.
        This is the way to AD a family that is built at run time --
        trun(), cens(), or a caller-defined FamilyDefinition -- because
        such families have no hand-written scalar kernel and no entry in
        any name-keyed registry.  When given, any of name / parameters /
        family_class / log_pdf_func that is left as None is taken from
        source_family, log_pdf_func defaults to source_family.d called in
        log mode, and the structural metadata (links, type, fixed
        parameters, constructor parameters, p/q/r) is inherited unless the
        caller overrides it.  The result is therefore the same distribution
        as source_family, with an exact AD score instead of whatever
        derivative set the source carried.
    hessian_from_score : bool, optional
        Replace each AD second derivative with the gamlss scoring form
        -(score)**2.  Defaults to True when source_family is given, False for
        the hand-written-kernel path.  The reason: IRLS needs the expected
        negative information, and the raw second derivative of a
        log-density is only that when the log-density is concave.  For a
        truncated family it is routinely positive (WEI3tr at mu=4.96 has
        d2l/dmu2 = +0.099), which makes the working weights negative and the
        fit diverge -- R sidesteps this by giving every numerically
        differentiated family d2ldx2 = -dldx**2.  Caller-supplied
        hessian_overrides still win.
    d, p, q, r : callable, optional
        Explicit dpqr functions. If provided, they override the auto-generated
        placeholders and are JIT-compiled for performance.
    score_overrides, hessian_overrides : mapping, optional
        Explicit derivatives, keyed by parameter name.  Where present they
        replace the AD-derived ones -- and they are *also* what lets the
        still-compiled AD path warm up safely.

        The reason this exists: AD succeeds for a first derivative and fails
        for the second whenever the log-density reaches a special function
        whose derivative JAX does not implement.  Two real cases in the
        Box-Cox cluster:

        * ``dBCT`` reaches ``betainc(a=tau/2, b=1/2, ...)``; ``betainc`` has
          no gradient in ``a``/``b``.  A ``custom_jvp`` (D2.13) patches the
          *first* derivative, but differentiating that JVP again -- which is
          exactly what the Hessian does -- re-enters ``betainc`` and raises
          ``Betainc gradient with respect to a and b not supported``.
        * ``dBCPE`` reaches ``gammainc(1/tau, s)``; differentiating in ``a``
          raises ``Differentiation rule for 'igamma_grad_a' not implemented``.

        Both families have R's closed-form derivatives transcribed already, so
        the fix is to hand them in rather than to teach JAX more special
        functions.  Scores/Hessians here are *better* than AD's anyway: they
        are R's own formulas, which is what parity means for this cluster
        (see ``_DERIV_TOLERANCE`` in ``tools/rs_family_parity_scan.py`` for the
        cases where R's own are only approximate).
    """
    if source_family is not None:
        # Everything the caller did not override comes from the family whose
        # density we are differentiating.  In particular p/q/r: an AD-ified
        # copy of a truncated family must keep the *truncated* CDF, otherwise
        # residuals silently revert to the base distribution.
        if family_class is None:
            family_class = type(source_family)
        if name is None:
            name = source_family.name
        if parameters is None:
            parameters = tuple(source_family.parameters)
        if log_pdf_func is None:
            log_pdf_func = _log_density_from_family(source_family, tuple(parameters))
        if type_ is None:
            type_ = source_family.type
        if links is None:
            links = source_family.links
        if link_functions is None:
            link_functions = source_family.link_functions
        if link_inverses is None:
            link_inverses = source_family.link_inverses
        if link_derivatives is None:
            link_derivatives = source_family.link_derivatives
        if fixed_parameters is None:
            fixed_parameters = source_family.fixed_parameters
        if fixed_parameter_defaults is None:
            fixed_parameter_defaults = source_family.fixed_parameter_defaults
        if constructor_parameters is None:
            constructor_parameters = source_family.constructor_parameters
        if d is None:
            d = source_family.d
        if p is None:
            p = source_family.p
        if q is None:
            q = source_family.q
        if r is None:
            r = source_family.r

    if type_ is None:
        type_ = "Continuous"
    if (
        family_class is None
        or name is None
        or parameters is None
        or log_pdf_func is None
    ):
        raise TypeError(
            "build_ad_family needs either (family_class, name, parameters, "
            "log_pdf_func) or source_family=<FamilyDefinition>"
        )

    # Pre-compile the vmapped log_pdf for reuse across g_dev_inc and _d
    _vmap_log_pdf = jax.jit(jax.vmap(log_pdf_func))

    # Pre-compile vmapped gradients and hessians for each parameter
    _vmap_grads = {}
    _vmap_hess = {}
    for i, par in enumerate(parameters):
        idx = i + 1  # argnums=i+1 because y is arg 0
        _vmap_grads[par] = jax.jit(jax.vmap(jax.grad(log_pdf_func, argnums=idx)))
        _vmap_hess[par] = jax.jit(
            jax.vmap(jax.grad(jax.grad(log_pdf_func, argnums=idx), argnums=idx))
        )

    # Warm-up compilation with dummy data to avoid first-call overhead
    # This significantly improves performance for small to medium datasets
    #
    # Parameters with a caller-supplied derivative are excluded: the warm-up
    # traces the AD Hessian even when the result is never used, and that trace
    # is exactly what raises on families whose log-density touches a special
    # function JAX cannot differentiate twice (see the docstring).  Building
    # the closures eagerly costs nothing, so nothing is lost by skipping them.
    derived_scores = dict(score_overrides or {})
    derived_hessians = dict(hessian_overrides or {})
    ad_derived = [
        par
        for par in parameters
        if par not in derived_scores and par not in derived_hessians
    ]
    try:
        dummy_size = 10
        dummy_args = [
            jnp.ones(dummy_size, dtype=jnp.float64) * 0.5
            for _ in range(len(parameters) + 1)
        ]
        _ = _vmap_log_pdf(*dummy_args)
        for par in ad_derived:
            _ = _vmap_grads[par](*dummy_args)
            _ = _vmap_hess[par](*dummy_args)
    except Exception:
        # If warm-up fails, continue without it (compilation will happen on first real call)
        pass

    def _g_dev_inc(*args, **kwargs) -> jnp.ndarray:
        # Support both positional and keyword arguments
        if kwargs:
            args = [kwargs.get("y")]
            for param in parameters:
                args.append(kwargs.get(param))

        args_1d, scalar_output = _prepare_jax_args(tuple(args))
        res = _vmap_log_pdf(*args_1d)
        if scalar_output:
            return -2.0 * res[0]
        return -2.0 * res

    score_functions = {}
    hessian_functions = {}

    for par in parameters:

        def make_grad(p_name):
            vmap_grad = _vmap_grads[p_name]

            def grad_func(*args, **kwargs):
                if kwargs:
                    args = [kwargs.get("y")]
                    for param in parameters:
                        args.append(kwargs.get(param))
                args_1d, scalar_output = _prepare_jax_args(tuple(args))
                res = vmap_grad(*args_1d)
                if scalar_output:
                    return res[0]
                return res

            return grad_func

        def make_hess(p_name):
            vmap_h = _vmap_hess[p_name]

            def hessian_func(*args, **kwargs):
                if kwargs:
                    args = [kwargs.get("y")]
                    for param in parameters:
                        args.append(kwargs.get(param))
                args_1d, scalar_output = _prepare_jax_args(tuple(args))
                res = vmap_h(*args_1d)
                if scalar_output:
                    return res[0]
                return res

            return hessian_func

        score_functions[par] = make_grad(par)
        hessian_functions[par] = make_hess(par)

    # Caller-supplied derivatives win.  They are keyed by parameter name and
    # expect the same signature as the AD ones -- (y, mu, sigma, ...) or the
    # keyword form -- so they drop straight in.
    score_functions.update(derived_scores)
    hessian_functions.update(derived_hessians)

    if hessian_from_score is None:
        # A family that only exists as a closure has no analytic d2ldx2 to
        # inherit, so use the gamlss numeric-derivative convention.
        hessian_from_score = source_family is not None
    if hessian_from_score:
        for par in parameters:
            if par in derived_hessians:
                continue  # an explicit override is always the answer
            hessian_functions[par] = _hessian_from_score(score_functions[par])

    # We also inject `pdf` into the class dynamically
    # since FamilyDefinition lacks it by default but tests/plot tools need it
    def _pdf(self, *args):
        args_1d, scalar_output = _prepare_jax_args(tuple(args))
        res = _vmap_log_pdf(*args_1d)
        if scalar_output:
            return jnp.exp(res[0])
        return jnp.exp(res)

    def _d_auto(*args, log=False):
        """Density/PMF function (auto-generated from log_pdf_func)."""
        args_1d, scalar_output = _prepare_jax_args(tuple(args))
        res = _vmap_log_pdf(*args_1d)
        if scalar_output:
            return res[0] if log else jnp.exp(res[0])
        return res if log else jnp.exp(res)

    def _p_placeholder(*args, lower_tail=True, log_p=False):
        """CDF function (placeholder - returns NaN for distributions without analytical CDF)."""
        q_val = jnp.asarray(args[0] if args else 0.0, dtype=jnp.float64)
        return jnp.full_like(q_val, jnp.nan)

    def _q_placeholder(*args, lower_tail=True, log_p=False):
        """Quantile function (placeholder - returns NaN for distributions without analytical quantile)."""
        p_val = jnp.asarray(args[0] if args else 0.5, dtype=jnp.float64)
        return jnp.full_like(p_val, jnp.nan)

    def _r_placeholder(key, n, *args):
        """Random generation (placeholder - returns NaN for distributions without implementation)."""
        return jnp.full((n,), jnp.nan, dtype=jnp.float64)

    # Use provided dpqr functions if available, otherwise use auto-generated ones
    _d_final = d if d is not None else _d_auto
    _p_final = p if p is not None else _p_placeholder
    _q_final = q if q is not None else _q_placeholder
    _r_final = r if r is not None else _r_placeholder

    # A family subclass carries extra dataclass fields that describe *what it
    # is* -- TruncatedFamily.truncation_type/base_name, CensoredFamily
    # .censoring_type, and whatever a caller-defined subclass adds.  Copying
    # them is what keeps the AD-ified object usable by the rest of the
    # pipeline: without base_name, the start-value mapping no longer sees a
    # truncated family and falls back to the generic target (exp(y) on a log
    # link).
    extra_fields: dict[str, Any] = {}
    if (
        source_family is not None
        and family_class is type(source_family)
        and _is_dataclass(source_family)
    ):
        known = {
            "name",
            "parameters",
            "g_dev_inc",
            "type",
            "links",
            "link_functions",
            "link_inverses",
            "link_derivatives",
            "score_functions",
            "hessian_functions",
            "fixed_parameters",
            "d",
            "p",
            "q",
            "r",
            "constructor_parameters",
            "fixed_parameter_defaults",
        }
        for field in _dataclass_fields(source_family):
            if field.name not in known:
                extra_fields[field.name] = getattr(source_family, field.name)

    return family_class(
        name=name,
        parameters=parameters,
        g_dev_inc=_g_dev_inc,
        type=type_,
        links=links,
        link_functions=link_functions,
        link_inverses=link_inverses,
        link_derivatives=link_derivatives,
        score_functions=score_functions,
        hessian_functions=hessian_functions,
        fixed_parameters=fixed_parameters,
        fixed_parameter_defaults=fixed_parameter_defaults,
        constructor_parameters=constructor_parameters,
        d=_d_final,  # Add d function
        p=_p_final,  # Add p function
        q=_q_final,  # Add q function
        r=_r_final,  # Add r function
        **extra_fields,
    )
