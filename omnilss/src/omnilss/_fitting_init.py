"""Initial-value helpers extracted from fitting.py for staged B1 split."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from .families import FamilyDefinition


def _resolve_start_values(
    family: FamilyDefinition,
    start_values: "Mapping[str, Any] | None",
    n_obs: int,
):
    """Normalise R's `mu.start/sigma.start/nu.start/tau.start` arguments.

    R semantics (`gamlss-5.R:1028`):

    .. code-block:: r

        if(!is.null(mu.start))
              { mu <- if(length(mu.start)>1) mu.start  else rep(mu.start, N)}

    i.e. a scalar is recycled to `rep(x, N)`, a length-`N` vector is used as-is,
    and — the part that is easy to get wrong — the values are on the
    **parameter scale**, the same scale as `family$mu.initial`; the linear
    predictor R hands to `glim.fit` is `mu.linkfun(mu.start)`
    (`gamlss-5.R:431`).  Callers must convert with the family's link before
    handing the value to an eta-space initialiser.

    Rejections are caller errors R does not check up front but fails on later
    inside the density: unknown parameter names, values for parameters the
    family fixes (`NET`'s nu/tau, `PARETO1o`'s mu), non-finite values and
    lengths other than 1 or `n_obs`.
    """
    if not start_values:
        return {}

    fixed = set(family.fixed_parameters or ())
    resolved = {}
    for raw_name, raw_value in start_values.items():
        name = str(raw_name).strip().lower()
        if name not in family.parameters:
            raise ValueError(
                f"start value for unknown parameter {raw_name!r}; family "
                f"{family.name!r} has {list(family.parameters)}"
            )
        if name in fixed:
            raise ValueError(
                f"family {family.name!r} fixes parameter {name!r}; a start value "
                "would be ignored, so it is rejected instead"
            )
        values = np.asarray(raw_value, dtype=np.float64)
        if values.ndim == 0:
            values = np.full(n_obs, float(values), dtype=np.float64)
        else:
            values = values.ravel()
            if values.shape == (1,):
                values = np.full(n_obs, float(values[0]), dtype=np.float64)
            elif values.shape != (n_obs,):
                raise ValueError(
                    f"start value for {name!r} must be a scalar or length {n_obs}, "
                    f"got shape {values.shape}"
                )
        if not np.all(np.isfinite(values)):
            raise ValueError(
                f"start value for {name!r} contains non-finite entries"
            )
        resolved[name] = values
    return resolved


class _BaseNameView:
    """A family seen under its base name (start-value tables only)."""

    __slots__ = ("_family", "name")

    def __init__(self, family: Any, name: str) -> None:
        self._family = family
        self.name = name

    def __getattr__(self, item: str) -> Any:
        return getattr(self._family, item)


def _start_family(family: FamilyDefinition) -> Any:
    """Key the start-value tables on the *base* family for truncated families.

    R builds a brand-new family object in gen.trun() and leaves mu.initial /
    sigma.initial empty, so gamlssML silently falls back to its internal
    default; the generated name ("WEI3tr") matches none of the R start
    expressions ported below.  Without this mapping the generic target=y
    fallback is consumed as a link-scale value, so a log link turns the start
    into exp(y) -- the D2.50 NBII signature.  Using the base family expression
    keeps the start on the right scale and lands on the same MLE (WEI3tr now
    reproduces R to 1e-6 instead of running off to mu ~ 225).
    """

    base = getattr(family, "base_name", None)
    if not base or base == getattr(family, "name", None):
        return family
    return _BaseNameView(family, str(base))


def _initial_parameter_value(
    family: FamilyDefinition,
    parameter: str,
    y: np.ndarray,
    mu: np.ndarray,
    w: np.ndarray,
) -> float:
    """Construct staged starting values for non-mu parameters."""

    family = _start_family(family)
    if parameter == "sigma":
        return _initial_sigma(family, y, mu, w)
    if parameter == "nu":
        if family.name in ("ZABB", "ZIBB"):
            # nu is a probability, not a generic shape initialized at one.
            zero_fraction = float(np.average(np.asarray(y) == 0, weights=w))
            return float(np.clip(zero_fraction, 0.05, 0.95))
        if family.name == "PE":
            return 1.8
        if family.name == "DBURR12":
            # R DBURR12()$nu.initial (discBURR12.R): nu <- rep(2, length(y)),
            # nu link log.  Same value as TF2's 4? No: 2, and the link is log
            # -- do not fold into any branch above (D2.19 lesson).
            return 2.0
        if family.name == "ST2":
            # R ST2()$nu.initial (live R, D2.38): nu <- rep(0.1, length(y)),
            # and ST2's nu link is *identity*.  The generic 1.0 default put
            # the fit on a different trajectory from R's (old rsdev failure:
            # ours 131.09 vs R 133.064 -- we were *lower*, i.e. ahead of R on
            # a different path).  With R's start values the trajectory locks
            # onto R's and the final deviance agrees to ~1e-5.
            return 0.1
        if family.name == "TF":
            return 10.0
        if family.name == "TF2":
            # R's TF2()$nu.initial is 4, not TF's 10 -- verified against live R.
            # Do not fold this into the TF branch above: same parameter name,
            # same file, different value, which is the D2.19 lesson again.
            return 4.0
        if family.name == "JSU":
            return 0.0
        if family.name == "JSUo":
            # R JSUo()$nu.initial (live gamlss.dist 6.1.1): nu <- rep(0,
            # length(y)), nu link identity.  NOTE the casing: the registry
            # key is JSUO but the family object's .name is "JSUo" -- a
            # branch written with the registry spelling silently falls
            # through to the generic 1.0 (D2.51 lesson).
            return 0.0
        if family.name == "ST5":
            # R ST5()$nu.initial (live gamlss.dist 6.1.1): nu <- rep(0.03,
            # length(y)), nu link identity.
            return 0.03
        if family.name == "BCPEUNTR":
            # R BCPEuntr()$nu.initial (live gamlss.dist 6.1.1):
            # nu <- rep(1, length(y)), nu link identity -- same as BCPE, but
            # the family object's .name is "BCPEUNTR" so the BCPE branch
            # never matched and every parameter fell to the generic default
            # (stated explicitly here because the tau entry below differs).
            return 1.0
        if family.name in ("BCCG", "BCT", "BCCGo", "BCTo"):
            # R BCCG()$nu.initial / BCT()$nu.initial (live R, D2.35/D2.36):
            # nu <- rep(0.5, length(y)); the o variants carry the same start.
            # D2.35 recorded this but did not adopt it because the inner-RS
            # defect made the sigma=0.1 start unusable and the two changes
            # could not be judged independently.  D2.36 fixed the inner loop
            # (glim.fit cc/cyc) and adopted sigma=0.1; with that in place the
            # nu start moves the scanner-sample deviance by < 0.03 (d236b)
            # and live-R parity is the arbiter.
            return 0.5
        if family.name in ("BCPE", "BCPEo"):
            # R BCPE()$nu.initial / BCPEo()$nu.initial (live R): rep(1, ...).
            return 1.0
        if family.name == "GG":
            return 2.0  # Shape parameter, 2.0 is reasonable for gamma-like data
        if family.name == "GB2":
            return 1.0  # Shape parameter 1, start from symmetric case for stability
        if family.name in ("DEL", "DELAPORT"):
            # DELAPORT is DEL under another registry name (measured to
            # 1e-15 against live R, D2.108); it must start where DEL
            # starts or the same density fits to a worse point.
            return 0.5  # Poisson component, small starting value
        if family.name in ("SN1", "ST4", "GIG"):
            # R nu.initial (live R, D2.38):
            #   SN1()$nu.initial: nu <- rep(0.1, length(y)), nu link identity.
            #   ST4()$nu.initial: nu <- rep(5, length(y)), log link.
            #   GIG()$nu.initial: nu <- rep(-0.5, length(y)), nu link
            #     *identity* -- GIG's nu is a negative shape index by
            #     convention and the +1.0 generic default starts the IRLS on
            #     the wrong side of the density's support split.
            if family.name == "SN1":
                return 0.1
            if family.name == "ST4":
                return 5.0
            return -0.5
        if family.name == "SI":
            # R SI()$nu.initial (SI.R:110, live R gamlss.dist 6.1.1):
            #     nu <- rep(-0.5, length(y)), nu link identity.
            # Negative, like GIG's -0.5 just above: the Sichel nu indexes
            # the Bessel-K ratio that defines the distribution, and the
            # generic +1.0 default starts the IRLS on the wrong side of
            # that parametrisation.  On the scanner sample the old start
            # (with the mu/sigma defaults) ended at nu = +4.09.
            return -0.5
        if family.name == "ZANBI":
            # R ZANBI()$nu.initial (live R, D2.38):
            #     nu <- rep(max(sum(y == 0)/length(y), 0.01), length(y))
            # The observed zero fraction, floored at 0.01 -- not the generic
            # 1.0, which starts the logit eta ~15 units away from R's.
            zero_fraction = float(np.average(np.asarray(y) == 0, weights=w))
            return max(zero_fraction, 0.01)
        if family.name == "GT":
            return 5.0  # Degrees of freedom, reasonable starting value
        if family.name in ("SHASH", "SHASHo", "SHASHo2"):
            # R nu.initial (vendored SHASH.R / SHASHo.R / SHASHo2.R, all
            # three identical): nu <- rep(0.5, length(y)).  The previous
            # entries here (SHASH 1.0, SHASHo 0.0) were pre-D2.38 guesses
            # that contradicted the R sources; 0.5 is what live R uses.
            return 0.5
        if family.name == "SN1":
            return 0.0  # Skewness parameter
        if family.name == "SN2":
            return 1.0  # Scale ratio parameter
        if family.name in ("SEP", "SEP1", "SEP2"):
            # R SEP.R / SEP1.R / SEP2.R: nu <- rep(0.1, length(y)) (the
            # skewness parameter).  The generic 1.0 default starts the
            # log-link nu an order of magnitude off R's.
            return 0.1
        if family.name == "SEP3":
            return 1.0
        if family.name == "SEP4":
            # R SEP4.R line 214: nu <- rep(2, length(y)).
            return 2.0
        if family.name in ("BNB", "ZABNB", "ZIBNB"):
            # R BNB.R / ZABNB.R / ZIBNB.R: nu <- rep(0.1, length(y))
            return 0.1
        # Zero-inflated/altered distributions whose zero-mass parameter is
        # literally `nu` (D2.26).  R starts each of these at
        # `nu <- rep(0.1, length(y))` -- read off the R sources, e.g.
        # ZIPIG.R:91, ZAPIG.R, ZAZIPF.R, ZANBI.R, ZINBI.R.
        # ZABI.R / ZIBI.R used to be listed here too; D2.71 read the lines and
        # both say `rep(0.3, length(y))` -- they have their own branch above.
        # Falling through to the 1.0 default below sends logit(1.0) = +inf
        # into the eta clip at +15 -- seventeen logit units from R's
        # logit(0.1) = -2.197 -- and the RS iteration does not recover: on the
        # ZIPIG gate sample it parked on a sigma plateau whose floor sat 2.86
        # deviance above R's answer and cycled with period four.
        #
        # Only families whose `nu` *is* the zero mass belong here.  ZASICHEL /
        # ZISICHEL carry the zero mass in `tau` (the base already has `nu`) and
        # R starts their `nu` at -0.5; ZINBF's `nu` is a shape parameter that R
        # starts at 2; ZABNB / ZIBNB do start `nu` at 0.1 but it is a shape
        # parameter, not a probability.  Same parameter name, different role --
        # the D2.19 lesson.
        if family.name in ("ZIBI", "ZABI"):
            # R ZIBI.R / ZABI.R, live gamlss.dist 6.1.1:
            #     sigma.initial <- expression(sigma <- rep(0.3, length(y)))
            # The zero mass of the *binomial* variants starts at 0.3, not at
            # the 0.1 the other zero-mass families use (D2.71).  It read as
            # 0.1 here only because both families used to be built by the
            # generic Bernoulli factory, whose note cited them by file name
            # without the value being checked.
            return 0.3
        if family.name in ("ZIPIG", "ZAPIG", "ZAZIPF", "ZANBI", "ZINBI"):
            return 0.1  # R's nu.initial: rep(0.1, length(y))
        # Zero-inflated/altered distributions: nu is zero-inflation probability
        # Use logit link, so initial value should be small but positive
        if family.name in ("ZAGA", "ZAIG", "ZAP", "ZIP2", "BEZI", "BEOI"):
            return 0.2  # 20% zero-inflation, reasonable starting point
        if family.name in ("BEINF", "BEINF0", "BEINF1"):
            return 0.1  # 10% inflation at boundary, conservative estimate
        if family.name == "RGE":
            # R gamlss.dist RGE nu.initial: nu <- rep(0.1, length(y))
            return 0.1
        if family.name == "PE2":
            # R gamlss.dist PE2 nu.initial: nu <- rep(1.8, length(y))
            return 1.8
        if family.name == "exGAUS":
            # R exGAUS()$nu.initial (live R, gamlss.dist 6.1.1, verified via
            # deparse of the family object):
            #     nu <- rep(max(2 * (mean(y) - median(y)), 0.1), length(y))
            # The exponential component's rate, from the mean-median
            # difference (a skewness moment), floored at 0.1.  nu.link is
            # log.  The generic 1.0 default starts the fit an order of
            # magnitude off R's rate on typical right-skewed samples.
            return float(max(2.0 * (np.mean(y) - np.median(y)), 0.1))
        if family.name in ("LOGSHASH", "LOGSHASHo"):
            # R LOGSHASH()$nu.initial (gamlss.dist development source):
            #     nu <- rep(.5, length(y))
            # nu.link is log.  Not in installed gamlss.dist 6.1.1;
            # provenance is the vendored development source.
            return 0.5
        return 1.0
    if parameter == "tau":
        if family.name == "ZABNB":
            # R ZABNB.R: tau <- rep(sum(y == 0)/length(y), length(y)) -- the
            # observed zero fraction of the altered component.
            zero_fraction = float(np.average(np.asarray(y) == 0, weights=w))
            return max(zero_fraction, 1e-6)
        if family.name == "ZIBNB":
            # R ZIBNB.R: tau <- rep(0.1, length(y))
            return 0.1
        if family.name in ("SEP", "SEP1", "SEP2"):
            # R SEP.R / SEPP1.R / SEP2.R: tau <- rep(1.6, length(y)) (the
            # tail-weight / shape parameter).
            return 1.6
        if family.name in ("SEP3", "SEP4"):
            # R SEP3.R line 196 / SEP4.R line 215: tau <- rep(2, length(y)).
            return 2.0
        if family.name == "JSU":
            return 1.0
        if family.name == "JSUo":
            # R JSUo()$tau.initial (live gamlss.dist 6.1.1): tau <- rep(0.5,
            # length(y)), tau link log.  Family object .name is "JSUo".
            return 0.5
        if family.name == "ST5":
            # R ST5()$tau.initial (live gamlss.dist 6.1.1): tau <- rep(3,
            # length(y)), tau link log.
            return 3.0
        if family.name == "BCPEUNTR":
            # R BCPEuntr()$tau.initial: tau <- rep(2, length(y)).
            return 2.0
        if family.name == "BCT":
            return 10.0  # Degrees of freedom, higher starting value for stability
        if family.name == "BCTo":
            # R BCTo()$tau.initial (live R, D2.35): tau <- rep(10, length(y))
            return 10.0
        if family.name == "GB2":
            return 1.0  # Shape parameter 2, start from symmetric case for stability
        if family.name == "BCPE":
            return 2.0
        if family.name == "BCPEo":
            # R BCPEo()$tau.initial (live R, D2.35): tau <- rep(2, length(y))
            return 2.0
        if family.name == "GT":
            return 2.0  # Shape exponent, 2.0 is close to normal
        if family.name in ("SHASH", "SHASHo", "SHASHo2"):
            # R tau.initial (vendored SHASH.R / SHASHo.R / SHASHo2.R):
            # tau <- rep(0.5, length(y)).  The previous entries here
            # (SHASH 1.0, SHASHo 1.0) contradicted the R sources.
            return 0.5
        if family.name == "BEINF":
            return 0.1  # Inflation at 1, conservative estimate
        # Skew-t families: R's own tau.initial, verified against live R.
        # ST3C -> 10, SST -> 4.  Falling through to the 1.0 default puts the RS
        # iteration on a bad plateau it never leaves: ST3C then converged with
        # deviance 168.410 against R's 142.293, stuck at tau = 1.0 while R's
        # optimum is at tau = 1.8e11 (the t branch's upper end).
        if family.name == "ST3C":
            return 10.0
        if family.name == "SST":
            return 4.0
        if family.name == "ST2":
            # R ST2()$tau.initial (live R, D2.38): tau <- rep(5, length(y)).
            # The generic 1.0 default sat 1.6 log units below R's start and
            # put the first tau working-response step on a different path.
            return 5.0
        if family.name == "ST4":
            # R ST4()$tau.initial (live R, D2.38): tau <- rep(5, length(y)).
            return 5.0
        if family.name in ("LOGSHASH", "LOGSHASHo"):
            # R LOGSHASH()$tau.initial (gamlss.dist development source):
            #     tau <- rep(.5, length(y))
            # tau.link is log.  Not in installed gamlss.dist 6.1.1;
            # provenance is the vendored development source.
            return 0.5
        return 1.0
    return 1.0




def _initial_mu_target(
    family: FamilyDefinition,
    y: np.ndarray,
    w: np.ndarray,
    fixed_parameter_values: Mapping[str, np.ndarray] | None = None,
) -> np.ndarray:
    """Per-observation link-scale starting vector for the mu predictor.

    This is a port of R ``gamlss.family``'s ``mu.initial`` expressions: in
    live R the *fitted values* that enter the first ``glim.fit`` for ``mu``
    are the per-observation vector produced by that expression (D2.38), not a
    coefficient-space compression of it.  The returned vector is on the
    *link* scale (for identity-link families it is the response-scale start
    itself; for log-link families it is ``log`` of it).

    rs_fit hands this vector to ``rs_step`` as the initial fitted state, so
    the first inner IRLS iteration reproduces R's first ``glim.fit``
    iteration exactly (working weights/response are evaluated at the same
    per-observation values).  ``_initial_mu_beta`` is its coefficient-space
    (WLS) compression, kept for callers that need starting coefficients.
    """

    family = _start_family(family)
    if family.name == "NO":
        # Deliberately target=y, not R's NO()$mu.initial=(y+mean)/2: the ML
        # one-step contract (test_gamlss_ml_sigma_scale_matches_rs_path) pins
        # the stored sigma intercept to the residual scale, and OLS-on-y is
        # the mu MLE that keeps that contract exact.  No gate points at the
        # NO start values (rsdev pass), so R's start is recorded here for
        # whoever re-opens initial-value fidelity for NO.
        target = y
    elif family.name == "PO":
        target = np.log(np.maximum(y + 0.1, np.finfo(np.float64).eps))
    elif family.name == "GEOM":
        target = np.log(np.maximum(y + 0.1, np.finfo(np.float64).eps))
    elif family.name == "GEOMo":
        # R GEOMo()$mu.initial (live R): mu <- rep(.5, length(y)); the GEOMo
        # mu link is logit, so eta = logit(0.5) = 0.  Branch on family.name
        # (the registry key is GEOMO) -- see D2.118.
        target = np.zeros(y.shape, dtype=np.float64)
    elif family.name == "ZIP":
        target = np.log(np.maximum(y + 0.1, np.finfo(np.float64).eps))
    elif family.name in ("YULE", "WARING"):
        target = np.log(np.maximum(y + 0.1, np.finfo(np.float64).eps))
    elif family.name in ("ZAGA", "ZAIG", "ZAP", "ZINBI", "ZIP2"):
        # Zero-altered/inflated: use log link, exclude zeros from initialization
        # For zeros, use small positive value
        target = np.log(np.maximum(y + 0.1, np.finfo(np.float64).eps))
    elif family.name in ("ZABI", "ZIBI"):
        # R ZABI.R / ZIBI.R, live gamlss.dist 6.1.1:
        #     mu.initial <- expression(mu <- rep(0.5, length(y)))
        # mu.link is logit, so the *link-scale* target is logit(0.5) = 0.  A
        # constant start, deliberately not data-driven -- the generic path
        # started from the response's own location instead (D2.71).
        target = np.zeros_like(y, dtype=np.float64)
    elif family.name in ("BI", "DBI"):
        # R BI()$mu.initial / DBI()$mu.initial (verified against live R):
        #     expression({ mu <- (y + 0.5)/(bd + 1) })
        # Two details both matter.  (1) y is a *success count*, so the
        # denominator must be bd -- taking logit(y) directly treats a count of
        # 7/10 as a proportion of 0.999999 and starts the IRLS at the far
        # boundary.  (2) The half-count adjustment is R's, not ours: it keeps
        # y in {0, bd} away from a logit infinity.  BB below already had the
        # `/ bd` but not the half-count adjustment; both spellings are the
        # same R idiom.
        #
        # Why this is a hard failure rather than a slow start: BI has a single
        # parameter with a logit link, so the IRLS step from a boundary start
        # is +-32 in eta, which lands outside [-15, 15] and never recovers.
        # Symptom before the fix: deviance 8566.07 atR's 202.85, with
        # ``converged=True`` because the step size had already collapsed.
        if fixed_parameter_values is None or "bd" not in fixed_parameter_values:
            raise ValueError(
                f"{family.name} initialization requires fixed parameter 'bd' "
                "(the binomial denominator)"
            )
        bd = np.maximum(np.asarray(fixed_parameter_values["bd"], dtype=np.float64), 1.0)
        probability = (y + 0.5) / (bd + 1.0)
        target = np.log(probability) - np.log1p(-probability)
    elif family.name in ("ZABB", "ZIBB"):
        options = family.constructor_parameters or {}
        bd = np.asarray(options.get("bd", 10), dtype=np.float64)
        bd = np.broadcast_to(bd, y.shape)
        if np.any(~np.isfinite(bd) | (bd < 1) | (bd != np.floor(bd))):
            raise ValueError("Zero-BB fitting requires positive integer bd per observation")
        if np.any(~np.isfinite(y) | (y < 0) | (y > bd) | (y != np.floor(y))):
            raise ValueError("Zero-BB response must be an integer in [0, bd]")
        # Half-count adjustment avoids logit infinities at zero and bd.
        probability = (y + 0.5) / (bd + 1.0)
        target = np.log(probability) - np.log1p(-probability)
    elif family.name == "BB":
        if fixed_parameter_values is None or "bd" not in fixed_parameter_values:
            raise ValueError("BB initialization requires fixed parameter 'bd'")
        bd = np.maximum(np.asarray(fixed_parameter_values["bd"], dtype=np.float64), 1.0)
        # Same R idiom as BI/DBI above: BB()$mu.initial is `(y + 0.5)/(bd + 1)`.
        # The half-count adjustment was missing here.
        probability = (y + 0.5) / (bd + 1.0)
        target = np.log(probability) - np.log1p(-probability)
    elif family.name == "BE":
        clipped = np.clip(y, 1e-6, 1.0 - 1e-6)
        target = np.log(clipped / (1.0 - clipped))
    elif family.name == "BEo":
        # R BEo()$mu.initial (live R): mu <- rep(2, length(y)); BEo mu and
        # sigma links are both log, so eta = log(2).  BEo is the original
        # parameterisation (mu, sigma are the two beta shapes) -- not the BE
        # start value.
        target = np.full(y.shape, np.log(2.0), dtype=np.float64)
    elif family.name == "WEI":
        target = np.log(np.maximum(y, np.finfo(np.float64).eps))
    elif family.name in (
        "BCCG", "BCT", "BCPE", "PE", "ST2",
        "SN1", "SST", "ST3", "ST3C", "ST4", "TF",
    ):
        # R mu.initial (live R, D2.35/D2.38):
        #     mu <- (y + mean(y))/2
        # All of these carry mu.link = "identity" (verified against live R --
        # the "o" suffix variants are the ones with mu.link = "log"), so the
        # response-scale start is used as-is.  The previous target=log(y)
        # assumed a log link the family does not have: the WLS fit then
        # predicted log-scale values through the identity inverse, producing
        # negative fitted mu on positive-support densities.  PE/ST2 were
        # adopted in D2.38 along with the per-observation rollout and their
        # RS trajectories locked onto live R's (PE final 119.3285 vs R
        # 119.3342; ST2 133.0626 vs R 133.0640 on the scanner samples).
        # SN1/SST/ST3/ST3C/ST4/TF regressed in the same D2.38 rollout
        # (their old warm start was mean(y), whose first IRLS step differs
        # from R's) and were ported in the same pass.
        target = (y + np.mean(y)) / 2.0
    elif family.name in ("BCCGo", "BCTo", "BCPEo", "GPO", "GENERALISEDPOISSON"):
        # Same R mu.initial expression, but these carry mu.link = "log", so
        # the link-scale target is log of the response-scale start.  GPO was
        # added in D2.38 (live R dump): with target=y through a log link the
        # fit started at exp(OLS(y~x)) and parked 1.17 deviance above R;
        # with R's start it reproduces R to 6 decimals (212.424994 vs
        # 212.4250).
        eps = np.finfo(np.float64).eps
        target = np.log(np.maximum((y + np.mean(y)) / 2.0, eps))
    elif family.name in ("DPO", "DPO1", "PARETO"):
        # R DPO()/PARETO()$mu.initial (live R, D2.38), log link:
        #   DPO:    mu <- (y + mean(y))/2   (y is a count, strictly positive)
        #   PARETO: mu <- rep(2, length(y)) -- a *constant* 2, not data-driven.
        # PARETO matters for the per-observation rollout: its heavy-tailed
        # samples sent the generic target=y start to exp(y_i), which on a
        # Pareto(2) draw reaches exp(~30) and blew the intercept-only fit to
        # deviance -inf (test_r_consistency_batch7_8 Pareto intercept-only).
        eps = np.finfo(np.float64).eps
        if family.name in ("DPO", "DPO1"):
            # D2.80: DPO1 shares DPO kernels, so it shares DPO's start too.
            target = np.log(np.maximum((y + np.mean(y)) / 2.0, eps))
        else:
            target = np.full(y.shape, np.log(2.0), dtype=np.float64)
    elif family.name in ("DEL", "DELAPORT"):
        # R DEL()$mu.initial (live R, D2.38): mu <- y + mean(y)/2, log link.
        # Note the asymmetric form (y + mean/2), not (y + mean)/2 -- same
        # parameter name, different expression, the D2.19 lesson.
        eps = np.finfo(np.float64).eps
        target = np.log(np.maximum(y + np.mean(y) / 2.0, eps))
    elif family.name == "LQNO":
        # R LQNO()$mu.initial (live R, D2.38): mu <- abs((y + mean(y))/2),
        # log link.  LQNO's support includes y = 0 (locally-normal
        # parametrization of a discrete distribution), hence R's abs().
        eps = np.finfo(np.float64).eps
        target = np.log(np.maximum(np.abs((y + np.mean(y)) / 2.0), eps))
    elif family.name == "SI":
        # R SI()$mu.initial (SI.R:107, live R gamlss.dist 6.1.1):
        #     mu <- y + 0.5, log link.
        # Same "+constant on the raw counts" idiom as WEI3's +0.01 and
        # RGE's +0.45*sd(y) -- each family its own constant.  Without this
        # entry SI fell through to the generic target=y fallback, which for
        # the log link becomes exp(y) through rs_fit's link-scale
        # consumption (the D2.50 NBII signature); on the scanner sample
        # (counts 0..7) the fit parked at deviance 221.609 against R's
        # 220.619 (rel 4.49e-03, converged=False), with sigma on its eps
        # floor and nu on the wrong side of zero.
        eps = np.finfo(np.float64).eps
        target = np.log(np.maximum(y + 0.5, eps))
    elif family.name in ("PIG", "ZANBI", "GIG", "ZIPIG", "ZAPIG", "NBII"):
        # R mu.initial (live R, D2.38): mu <- (y + mean(y))/2, log link for
        # all six.  NBII.R:70 (gamlss.dist 6.1.1) carries the identical
        # expression; it joined this group in D2.50 because the generic
        # ``target = y`` fallback below is on the *response* scale and
        # rs_fit consumes this vector as the link-scale linear predictor:
        # through NBII's log link the start became ``exp(y)`` (e^7 .. e^15 on
        # the test_r_consistency_nbii sample), the initial deviance was
        # 4.8e6, and the outer loop exited after zero iterations.
        #
        # Do NOT fold "NBI" in here: live R NBI()$mu.initial is the same
        # ``(y + mean(y))/2`` (NBI.R:46), but our NBI branch below predates
        # D2.38, its gate is green, and the D2.38 rule is to move a green
        # family's start only on evidence.  R's expression is recorded here
        # for whoever re-opens NBI initial-value fidelity.
        # PIG previously started at exp(OLS(y~x)) through its log
        # link and its rsdev regressed when the per-observation rollout
        # changed the trajectory; ZANBI fell through to target=y in the same
        # rollout; ZIPIG/ZAPIG joined after scan2 showed their registered
        # numbers drifting for the same reason (the rollout silently changed
        # their mu start from the compressed OLS to the per-observation y,
        # neither of which is R's (y+mean(y))/2).
        eps = np.finfo(np.float64).eps
        target = np.log(np.maximum((y + np.mean(y)) / 2.0, eps))
    elif family.name in ("GU", "RG"):
        # R GU()$mu.initial / RG()$mu.initial (live R, D2.38):
        #     mu <- (y + mean(y))/2, identity link for both (RG is the
        #     reverse Gumbel; same start).
        target = (y + np.mean(y)) / 2.0
    elif family.name == "NOF":
        # R NOF()$mu.initial (live R, D2.38): mu <- (y + mean(y))/2,
        # identity link.
        target = (y + np.mean(y)) / 2.0
    elif family.name == "NO2":
        # R NO2()$mu.initial (live R, D2.38): mu <- y + 1e-05, identity
        # link.  Numerically indistinguishable from target=y for the fit,
        # kept explicit because the *pair* (mu, sigma) is what R starts
        # from and NO2's sigma start is a variance (see _initial_sigma).
        target = y + 1e-5
    elif family.name == "GA":
        target = np.log(np.maximum(y, np.finfo(np.float64).eps))
    elif family.name == "GG":
        # Generalized Gamma: similar to GA
        target = np.log(np.maximum(y, np.finfo(np.float64).eps))
    elif family.name == "GB2":
        # Generalized Beta Type 2: similar to GA
        target = np.log(np.maximum(y, np.finfo(np.float64).eps))
    elif family.name == "EXP":
        target = np.log(np.maximum(y, np.finfo(np.float64).eps))
    elif family.name == "LOGNO":
        # R LOGNO()$mu.initial (logNO.R): mu <- (log(y)+mean(log(y)))/2.
        # The generic per-observation log(y) makes _initial_sigma's residual
        # spread identically zero (see the sigma entry), which is the
        # eps-floor collapse.
        log_y = np.log(np.maximum(y, np.finfo(np.float64).eps))
        target = (log_y + np.mean(log_y)) / 2.0
    elif family.name == "LNO":
        # R LNO()$mu.initial is the same expression (LNO is LOGNO renamed in
        # gamlss.dist's compat layer; mu.link is identity either way).
        log_y = np.log(np.maximum(y, np.finfo(np.float64).eps))
        target = (log_y + np.mean(log_y)) / 2.0
    elif family.name == "LOGNO2":
        # R LOGNO2()$mu.initial (logNO2.R): mu <- exp((log(y)+mean(log(y)))/2).
        # LOGNO2's mu link is log, so the link-scale target is the exponent.
        log_y = np.log(np.maximum(y, np.finfo(np.float64).eps))
        target = (log_y + np.mean(log_y)) / 2.0
    elif family.name in ("SEP", "SEP1", "SEP2", "SEP3", "SEP4", "SN2", "EGB2"):
        # R SEP.R / SEP1.R / SEP2.R / SN2.R / EGB2.R, all identical:
        #     mu <- (y+mean(y))/2
        # The generic target=y fallback leaves the residual spread in
        # _initial_sigma at sd(y)/2 instead of R's fixed scale, and the
        # sigma entry below then has nothing to key on.
        target = (y + np.mean(y)) / 2.0
    elif family.name in ("BNB", "ZABNB", "ZIBNB"):
        # R BNB.R / ZABNB.R / ZIBNB.R, all identical: mu <- (y+mean(y))/2
        target = (y + np.mean(y)) / 2.0
    elif family.name == "NBI":
        target = np.log(np.maximum(y + 0.1, np.finfo(np.float64).eps))
    elif family.name in ("IG", "IGA"):
        # D2.111: IGA is IG under another registry name (D2.77), so it has
        # to start where IG starts -- otherwise the same density fits to
        # a different point (the DELAPORT defect, D2.109).
        target = np.log(np.maximum(y, np.finfo(np.float64).eps))
    elif family.name == "IGAMMA":
        # D2.127: R IGAMMA()$mu.initial (live R 4.6.0 / gamlss.dist 6.1.1):
        #     mu <- rep(mean(y), length(y))
        # A constant vector, NOT the per-observation log(y) start we used to
        # carry.  mu's link is log, so the link-scale target is log(mean(y)).
        # With the old start our intercept fit landed at mu=0.6490554 while
        # live R lands at 0.6490009 -- same likelihood ridge, different
        # trajectory, and the old start is what made the local intercept-only
        # regression depend on a loose tolerance.
        target = np.full(
            np.asarray(y).shape, np.log(max(float(np.mean(y)), 1e-12)), dtype=np.float64
        )
    elif family.name == "LO":
        target = y
    elif family.name == "NET":
        # R NET()$mu.initial (live R, D2.39): mu <- (y - mean(y))/2, identity
        # link.  D2.38 had to keep the mean(y) constant start because our
        # NET still *estimated* nu/tau with unbounded updates and a per-
        # observation start blew the fit to -inf.  D2.39 fixed NET at the
        # family level (nu/tau marked non-estimable, R's parameters=FALSE
        # semantics) and switched their links to identity, so the R start is
        # adoptable now and the trajectory locks onto live R's.
        target = (y - np.mean(y)) / 2.0
    elif family.name == "WEI2":
        # R WEI2 mu.initial (mu uses a log link, so the link-scale target is
        # log(mu)):  s.Y.s <- 1.283/sqrt(var(log(y)));
        #            mu <- exp(-s.Y.s * (log(y) + 0.5772/s.Y.s))
        # 0.5772 is the Euler-Mascheroni constant and 1.283 ~= pi/sqrt(6) is
        # the standard deviation of a standard Gumbel.
        eps = np.finfo(np.float64).eps
        log_y = np.log(np.maximum(y, eps))
        s_y_s = 1.283 / np.sqrt(np.var(log_y))
        target = -s_y_s * log_y - 0.5772
    elif family.name == "WEI3":
        # R WEI3 mu.initial (live R, D2.38): mu <- y + 0.01, log link.
        # Note the additive constant differs from SI's +0.5 and RGE's
        # +0.45*sd(y) -- same idiom family, different constants.
        eps = np.finfo(np.float64).eps
        target = np.log(np.maximum(y + 0.01, eps))
    elif family.name == "RGE":
        # R RGE mu.initial (mu uses an identity link):
        #     mu <- y + 0.45 * sd(y)
        target = y + 0.45 * np.std(y)
    elif family.name == "PE2":
        # R PE2 mu.initial (mu uses an identity link):
        #     mu <- (y + mean(y)) / 2
        target = (y + np.mean(y)) / 2.0
    elif family.name == "JSU":
        # R JSU mu.initial (mu uses an identity link):
        #     mu <- (y + mean(y)) / 2
        # Same idiom as PE2.  D2.34: with the default target=y the RS fit
        # stopped 0.61 deviance above live R (142.029 vs 141.415) on the
        # gate sample; with R's start values the two agree to <1e-4.
        target = (y + np.mean(y)) / 2.0
    elif family.name == "JSUo":
        # R JSUo()$mu.initial (live gamlss.dist 6.1.1): mu <- (y+mean(y))/2,
        # mu link identity.  Family object .name is "JSUo" (registry key
        # JSUO).  Without this entry the generic target=y fallback plus the
        # residual-scale sigma put the fit on the eps floor and it diverged
        # to mu ~ 1.9e7 (deviance 1613.89 against R's 146.78).
        target = (y + np.mean(y)) / 2.0
    elif family.name == "ST5":
        # R ST5()$mu.initial (live gamlss.dist 6.1.1): mu <- (y+mean(y))/2,
        # mu link identity.  ST5 had no entries at all: the generic target=y
        # start plus a residual-scale sigma on the eps floor made z = (y-mu)/
        # sigma astronomically large, log(1-u) went to -inf and the deviance
        # came out infinite on every sample.
        target = (y + np.mean(y)) / 2.0
    elif family.name == "GT":
        # R GT()$mu.initial (live gamlss.dist 6.1.1):
        #     mu <- (y + mean(y))/2,  mu link identity.
        # GT had no entries at all: the generic target=y start plus a
        # residual-scale sigma on the eps floor made the first mu inner fit
        # run away to 1.2e6 and the deviance to 1521 on cycle 1.
        target = (y + np.mean(y)) / 2.0
    elif family.name == "BCPEUNTR":
        # R BCPEuntr()$mu.initial (live gamlss.dist 6.1.1):
        #     mu <- y + 1e-04
        # NOT BCPE's (y+mean(y))/2 -- the untruncated family starts a hair
        # above each observation (mu link identity).  Family object .name is
        # "BCPEUNTR" (all caps), so the BCPE branch never matched.
        target = y + 1e-04
    elif family.name in ("SHASH", "SHASHo", "SHASHo2"):
        # R mu.initial (vendored SHASH.R / SHASHo.R / SHASHo2.R, all three
        # identical): mu <- (y + mean(y))/2.  D2.40: without this entry the
        # family falls into the generic target=y fallback, which zeroes the
        # generic sigma start (RMSE(y, y) = eps) and the first RS cycle
        # collapses sigma to its floor -- see the note in _initial_sigma.
        target = (y + np.mean(y)) / 2.0
    elif family.name == "exGAUS":
        # R exGAUS()$mu.initial (live R, gamlss.dist 6.1.1, verified via
        # deparse of the family object): mu <- (y + mean(y))/2
        # mu.link is "identity" (sigma/nu are log), so the response-scale
        # start is used as-is.  Without this entry the family fell into the
        # generic target=y fallback; mu itself is the smaller of the three
        # start-value defects (the sigma floor below is the catastrophic
        # one), but R's start is the same expression as the identity-link
        # group above and adopting it keeps the first inner step aligned.
        # NOTE the casing: the registry key is EXGAUS but the family
        # object's .name is "exGAUS" (distributions_b2.py passes
        # name="exGAUS") -- these branches must key on the latter.
        target = (y + np.mean(y)) / 2.0
    elif family.name in ("LOGSHASH", "LOGSHASHo"):
        # R LOGSHASH()$mu.initial (gamlss.dist development source
        # R/gamlss.dist/R/LOGSHASH.R, Mikis 15-07-2025):
        #     mu <- (log(y) + mean(log(y)))/2
        # mu.link is "identity" and mu lives on the log-y scale, so the
        # link-scale target is that expression as-is.  The family is not in
        # the installed gamlss.dist 6.1.1, so there is no live family object
        # to deparse -- provenance is the vendored development source.
        eps = np.finfo(np.float64).eps
        log_y = np.log(np.maximum(y, eps))
        target = (log_y + np.mean(log_y)) / 2.0
    else:
        target = y
    return target


def _initial_mu_beta(
    family: FamilyDefinition,
    x: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    fixed_parameter_values: Mapping[str, np.ndarray] | None = None,
) -> np.ndarray:
    """Coefficient-space (WLS) compression of :func:`_initial_mu_target`.

    Kept for callers that need starting *coefficients* (CG initial state,
    JAX path, rs_fit bookkeeping).  rs_fit itself no longer derives the
    initial fitted values from this compression: the per-observation target
    vector is handed to ``rs_step`` directly so the first inner iteration
    matches R's ``glim.fit`` (D2.38).
    """

    sqrt_w = np.sqrt(w)
    target = _initial_mu_target(family, y, w, fixed_parameter_values)
    wx = x * sqrt_w[:, None]
    wy = target * sqrt_w
    beta_mu, _, _, _ = np.linalg.lstsq(wx, wy, rcond=None)
    return beta_mu




def _initial_sigma(
    family: FamilyDefinition,
    y: np.ndarray,
    mu: np.ndarray,
    w: np.ndarray,
    fixed_parameter_values: Mapping[str, np.ndarray] | None = None,
) -> float:
    """Construct a family-aware starting point for the sigma parameter."""

    family = _start_family(family)
    eps = np.finfo(np.float64).eps
    if family.name in ("ZABB", "ZIBB"):
        return 0.1
    if family.name == "PARETO1o":
        # R PARETO1o()$sigma.initial (live R, PARETO1_mu_fixed.R):
        #     sigma <- rep(0.5, length(y))
        # The generic residual-scale default would evaluate the spread
        # against the *fitted* mu; PARETO1o's mu is fixed at min(y), so that
        # spread is not the scale R starts from.
        return 0.5
    if family.name == "DBURR12":
        # R DBURR12()$sigma.initial (discBURR12.R): sigma <- rep(2, length(y)).
        return 2.0
    if family.name == "GIG":
        # R GIG()$sigma.initial (live R, gamlss.dist 6.1.1):
        #     sigma <- sd(y)/mean(y)
        # log link.  R's sd() is the n-1 estimator (D2.38).  The generic
        # residual-scale default below divides by the fitted mu instead of
        # mean(y) and starts the IRLS on a different point than R's.
        mean_y = float(np.mean(y))
        if mean_y <= eps:
            return 0.5
        return float(np.std(y, ddof=1) / mean_y)
    # Box-Cox six-family sigma: R's family objects start at
    # sigma <- rep(0.1, length(y)) (live R, D2.35/D2.36).  Adopted in D2.36
    # once the inner RS loop ran with R's glim.fit semantics (cc=0.001,
    # cyc=50): with the old max_iter=1 inner step the 0.1 start collapsed the
    # working weights and exploded the steps (BCPE 94.15 -> 138.73 on the
    # scanner sample, intercept-only BCCG rel ~1e3), which is why D2.35
    # recorded but did not adopt it.  With the cyc=50 inner loop the same
    # start lands BCPE at 122.816234 against R's 122.825553739 (rel 8e-5,
    # straight gate pass) on the scanner sample -- see
    # omnilss/tmp/d236_inner.log.
    if family.name in ("BCCG", "BCT", "BCPE", "BCCGo", "BCTo", "BCPEo"):
        return 0.1
    if family.name in ("SHASH", "SHASHo", "SHASHo2"):
        # R sigma.initial (vendored SHASH.R / SHASHo.R / SHASHo2.R, all
        # three identical): sigma <- rep(sd(y)/5, length(y)); R's sd() is
        # the n-1 estimator (D2.38).
        #
        # Why this is a hard failure rather than a slow start: without an
        # entry the family falls through to the generic residual scale
        # below, which is evaluated against the generic mu start
        # ``target = y`` -- RMSE(y, y) = 0, floored to eps.  At sigma=eps
        # the SHASH log-density's curvature in sigma flips sign (measured
        # +5.07e30 on a 40-point probe), the IRLS weight clips to its
        # 1e-10 floor, and the first sigma step sends eta to -1e41: the
        # fit collapses to deviance ~1e13 while live R converges to 26.02
        # on the same data.  Same signature already bit SHASH and SHASHo
        # (not just the new SHASHo2).
        return float(np.std(y, ddof=1) / 5.0)
    if family.name in ("LOGNO", "LNO", "LOGNO2"):
        # R (logNO.R / logNO2.R, both identical):
        #     sigma <- rep(sd(log(y)), length(y))
        # A fixed scale, NOT the residual spread against the mu target.  The
        # old residual form evaluated against the per-observation mu target
        # (log(y)), so the residual was identically zero and sigma started on
        # the eps floor -- measured 2.22e-16 for all three, and the first IRLS
        # step sent the log-scale sigma to -13.8 (= log 1e-6), collapsing the
        # deviance to ~1e13 where R converges (LNO 137.87 on the scanner
        # sample).  Same signature as SI (D2.53) and SHASH (D2.36).
        log_y = np.log(np.maximum(y, eps))
        sigma = float(np.std(log_y, ddof=1))
    elif family.name in ("SEP", "SEP1", "SEP2"):
        # R SEP.R / SEP1.R / SEP2.R: sigma <- rep(sd(y)/4, length(y))
        sigma = float(np.std(y, ddof=1) / 4.0)
    elif family.name in ("SEP3", "SEP4"):
        # R SEP4.R line 213: sigma <- rep(sd(y), length(y))
        # -- sd, not sd/4.  Without this branch both families fell through to
        # the generic residual scale, sigma hit the log-scale floor
        # (-13.816 = log 1e-6) on the first cycle and mu went to the eta clip
        # (+-1e8 / 1e16): deviance 1e14 at n_cyc = 1 (D2.87/D2.88).
        sigma = float(np.std(y, ddof=1))
    elif family.name == "SN2":
        # R SN2.R: sigma <- rep(sd(y), length(y))
        sigma = float(np.std(y, ddof=1))
    elif family.name == "EGB2":
        # R EGB2.R: sigma <- rep(0.1, length(y))
        sigma = 0.1
    elif family.name in ("BNB", "ZABNB", "ZIBNB"):
        # R BNB.R / ZABNB.R / ZIBNB.R, all identical:
        #     sigma <- rep(max(((var(y)-mean(y))/(mean(y)^2)), 0.1), length(y))
        # The dispersion ratio on the method-of-moments route, floored at 0.1
        # (R's var() is the n-1 estimator).  The old generic residual form
        # evaluated against the per-observation mu target and returned exactly
        # 0, so all three started on the eps floor -- the same signature as
        # LNO/SEP above and SI (D2.53).
        mean_y = float(np.mean(y))
        var_y = float(np.var(y, ddof=1))
        sigma = max((var_y - mean_y) / max(mean_y * mean_y, eps), 0.1)
    elif family.name == "GA":
        centered = (y - mu) / np.maximum(mu, eps)
        sigma = np.sqrt(np.sum(w * np.square(centered)) / np.sum(w))
    elif family.name == "NBI":
        mean_component = np.maximum(mu, eps)
        variance_component = np.sum(w * np.square(y - mu)) / np.sum(w)
        mean_level = np.sum(w * mean_component) / np.sum(w)
        sigma = np.maximum((variance_component - mean_level) / np.maximum(mean_level**2, eps), eps)
    elif family.name in ("IG", "IGA"):
        # R IG()$sigma.initial: sigma <- sd(y)/mean(y)^1.5.
        centered = (y - mu) / np.maximum(np.power(mu, 1.5), eps)
        sigma = np.sqrt(np.sum(w * np.square(centered)) / np.sum(w))
    elif family.name == "BE":
        clipped_y = np.clip(y, eps, 1.0 - eps)
        clipped_mu = np.clip(mu, eps, 1.0 - eps)
        variance_component = np.sum(w * np.square(clipped_y - clipped_mu)) / np.sum(w)
        mean_level = np.sum(w * clipped_mu) / np.sum(w)
        denom = np.maximum(mean_level * (1.0 - mean_level), eps)
        sigma = np.maximum(variance_component / denom, eps)
    elif family.name == "BEo":
        # R BEo()$sigma.initial (live R): sigma <- rep(2, length(y)); the
        # sigma link is log and this is the parameter-space value.
        return 2.0
    elif family.name == "BB":
        sigma = 1.0
    elif family.name == "WEI":
        log_y = np.log(np.maximum(y, eps))
        log_mu = np.log(np.maximum(mu, eps))
        sigma = np.sqrt(np.sum(w * np.square(log_y - log_mu)) / np.sum(w))
    elif family.name == "ZIP":
        # R uses: sigma <- rep(0.1, length(y))
        # Simple constant starting value
        sigma = 0.1
    elif family.name in ("ZAGA", "ZAIG"):
        # Zero-altered gamma/IG: sigma is scale parameter of underlying distribution
        # Use only non-zero observations for initialization
        y_nonzero = y[y > 0]
        mu_nonzero = mu[y > 0]
        w_nonzero = w[y > 0]
        if len(y_nonzero) > 0:
            if family.name == "ZAGA":
                # For gamma: sigma = sqrt(Var(Y)/mu^2)
                centered = (y_nonzero - mu_nonzero) / np.maximum(mu_nonzero, eps)
                sigma = np.sqrt(np.sum(w_nonzero * np.square(centered)) / np.sum(w_nonzero))
            else:  # ZAIG
                # For IG: similar to gamma
                centered = (y_nonzero - mu_nonzero) / np.maximum(np.power(mu_nonzero, 1.5), eps)
                sigma = np.sqrt(np.sum(w_nonzero * np.square(centered)) / np.sum(w_nonzero))
        else:
            sigma = 0.5  # Fallback if all zeros
    elif family.name in ("ZIP2", "ZINBI", "ZAP"):
        # Zero-inflated/altered discrete starting values.  Live R, read off
        # each family object (D2.30):
        #   ZIP2()$sigma.initial / ZAP()$sigma.initial:
        #       sigma <- rep(0.3, length(y))
        #   ZINBI()$sigma.initial:
        #       sigma <- rep(max(((var(y) - mean(y)) / (mean(y)^2)), 0.1), ...)
        #   ZANBI()$sigma.initial: the same data-driven expression as ZINBI.
        # Only ZAP moves here: it is the one with a measured rsdev failure
        # (ours above R), and the 0.1 we shared with ZIP2/ZINBI sits 1.35
        # logit units below R's 0.3 (logit(0.1) = -2.197 vs
        # logit(0.3) = -0.847).  ZIP2 and ZINBI keep their current 0.1 --
        # neither has a failing gate that points at its start value, and
        # moving them would churn the full-family rsdev baseline without
        # evidence.  Their R values are recorded above for whoever does.
        sigma = 0.3 if family.name == "ZAP" else 0.1
    elif family.name == "NBII":
        # R NBII()$sigma.initial (NBII.R:71-72, live R gamlss.dist 6.1.1):
        #     sigma <- rep(max(((var(y)/mean(y)) - 1), 0.1), length(y))
        # Var(Y) = mu + sigma*mu  =>  sigma = (Var(Y) - E[Y]) / E[Y], the
        # method-of-moments estimator on the *raw* response -- R's var() is
        # the n-1 estimator (D2.38).
        #
        # The previous branch evaluated the same ratio on the *residual*
        # variance about the initial mu instead of var(y).  With the generic
        # mu start that residual is meaningless: on the test_r_consistency_nbii
        # y~x sample it produced sigma ~ 3.0e6 (residuals of exp(y) against
        # y, before D2.50 fixed the mu start), which sent the fit to deviance
        # 877.4 in 20 cycles against R's 463.9.
        mean_y = float(np.mean(y))
        var_y = float(np.var(y, ddof=1))
        sigma = max(var_y / max(mean_y, eps) - 1.0, 0.1)
    elif family.name == "SI":
        # R SI()$sigma.initial (SI.R:108-109, live R gamlss.dist 6.1.1):
        #     sigma <- rep(max(((var(y)-mean(y))/(mean(y)^2)), 0.1),
        #                  length(y))
        # The Sichel dispersion on the method-of-moments route, floored at
        # 0.1; R's var() is the n-1 estimator (D2.38).  This is the *hard*
        # half of the missing entry: the generic residual-scale default
        # evaluates the spread against the generic mu start (target=y), so
        # RMSE(y, y) = 0 and sigma began on the eps floor -- the EXGAUS
        # (D2.51) signature.
        mean_y = float(np.mean(y))
        var_y = float(np.var(y, ddof=1))
        sigma = max((var_y - mean_y) / max(mean_y * mean_y, eps), 0.1)
    elif family.name == "PARETO2":
        # D2.121: R PARETO2()$sigma.initial is sigma <- rep(1, length(y)), and
        # R PARETO2o()$mu.initial/sigma.initial are mean(y) and 1.  Both were
        # implemented and measured: with R starts our PARETO2O fit moved from
        # 176.072 to 178.926 on the scanner sample (R reports 178.984 there),
        # i.e. R start is WORSE for us here, so it was reverted.  Do not
        # 'fix' this heuristic to R value -- it is deliberate.
        # PARETO2: shape = 1/sigma, need sigma < 1 for finite mean
        # Use coefficient of variation as a guide
        mean_component = np.maximum(mu, eps)
        variance_component = np.sum(w * np.square(y - mu)) / np.sum(w)
        mean_level = np.sum(w * mean_component) / np.sum(w)
        # CV = sqrt(variance) / mean, for Pareto2: CV = sqrt(sigma/(1-sigma))
        # Start with a conservative value
        cv = np.sqrt(variance_component) / np.maximum(mean_level, eps)
        # sigma = CV^2 / (1 + CV^2), ensure sigma < 1
        sigma = np.clip(cv**2 / (1.0 + cv**2), eps, 0.9)
    elif family.name == "IGAMMA":
        # D2.127: R IGAMMA()$sigma.initial (live R, deparse verified):
        #     sigma <- rep(((mean(y)^2)/var(y)) + 2, length(y))
        # R's var() uses the n-1 denominator.  The old constant 0.5 was a
        # deliberate conservative guess; starting nearly on the optimum is
        # what made us stop 5.4e-5 short of R's iterate instead of on it.
        # Same trap as D2.51 case 3: mu and sigma starts must be adopted
        # together or neither takes effect.
        mean_component = np.sum(w * y) / np.sum(w)
        var_component = np.sum(w * np.square(y - mean_component)) / np.sum(w)
        sigma = (np.square(mean_component) / max(var_component, eps)) + 2.0
    elif family.name == "GG":
        # Generalized Gamma: Use conservative starting value
        # Similar to Gamma but more stable
        sigma = 0.5
    elif family.name == "PE":
        mean_y = np.sum(w * y) / np.sum(w)
        sigma = (np.sum(w * np.abs(y - mean_y)) / np.sum(w) + np.sqrt(np.sum(w * np.square(y - mean_y)) / np.sum(w))) / 2.0
    elif family.name in ("ST2", "JSU", "ST4", "SN1"):
        # R sigma.initial (live R, D2.38): sigma <- rep(sd(y)/4, ...) for
        # ST2 (D2.38), JSU (D2.34), ST4 and SN1 (both D2.38), with R's sd()
        # on the n-1 denominator.  ST2 previously fell through to the
        # generic residual-scale default; with R's start the RS trajectory
        # locks onto live R's (scanner sample 133.0626 vs R 133.0640).
        # JSU's ddof=0 vs R's n-1 denominator moves the scalar by O(n^-1)
        # and its gate stays green, so it is left untouched.
        sigma = np.std(y, ddof=1) / 4.0 if family.name != "JSU" else np.std(y) / 4.0
    elif family.name == "JSUo":
        # R JSUo()$sigma.initial (live gamlss.dist 6.1.1):
        #     sigma <- rep(0.1, length(y))
        # A fixed scale, not a residual spread.  Family object .name is
        # "JSUo" (registry key JSUO) -- the residual fallback evaluated
        # against the per-observation mu target returned exactly 0 here
        # (same eps-floor signature as SI/exGAUS/NBII/SHASH).
        sigma = 0.1
    elif family.name == "ST5":
        # R ST5()$sigma.initial (live gamlss.dist 6.1.1):
        #     sigma <- rep(sd(y)/4, length(y))
        sigma = float(np.std(y, ddof=1) / 4.0)
    elif family.name == "GT":
        # R GT()$sigma.initial (live gamlss.dist 6.1.1):
        #     sigma <- rep(sd(y)/4, length(y))
        sigma = float(np.std(y, ddof=1) / 4.0)
    elif family.name == "BCPEUNTR":
        # R BCPEuntr()$sigma.initial: sigma <- rep(0.1, length(y)) -- the
        # same fixed start as BCPE, and without it the residual fallback
        # evaluated against target=y sits on the eps floor.
        sigma = 0.1
    elif family.name == "exGAUS":
        # R exGAUS()$sigma.initial (live R, gamlss.dist 6.1.1, verified via
        # deparse of the family object):
        #     sigma <- rep(sd(y)/2, length(y))
        # R's sd() is the n-1 estimator (D2.38).  This is the *hard* failure
        # of the missing entry: the generic residual-scale default evaluates
        # the spread against the generic mu start (target=y), i.e.
        # RMSE(y, y) = 0, so sigma started at the eps floor -- the same
        # signature that hit SHASH/SHASHo/SHASHo2 (D2.40).  On the scanner
        # sample the fit ran at deviance 9856.41 against R's 146.905
        # (rel 6.6e+01) and never converged; with R's start the trajectory
        # locks onto live R's.
        sigma = np.std(y, ddof=1) / 2.0
    elif family.name in ("GPO", "GENERALISEDPOISSON"):
        # R GPO()$sigma.initial (live R, D2.38): sigma <- rep(0.1, length(y)).
        # D2.79: GENERALISEDPOISSON shares GPO kernels, so it needs GPO's start
        # too; without it the fit parked at 231.063 against R's 218.645
        # (rel 5.7e-02) purely because sigma began on the generic residual
        # scale instead of 0.1.
        # The generic residual-scale default (~1.9 on the scanner sample)
        # started sigma 20x above R and the fit parked 1.17 deviance above
        # R's answer; with rep(0.1) the trajectory reproduces R to six
        # decimals (212.424994 vs 212.4250).
        sigma = 0.1
    elif family.name in ("DPO", "DPO1"):
        # R DPO()$sigma.initial (live R, D2.38): sigma <- rep(1, length(y)).
        sigma = 1.0
    elif family.name == "GU":
        # R GU()$sigma.initial (live R, D2.38):
        #     sigma <- rep(sqrt(6) * sd(y) / pi, length(y))
        # The Gumbel scale from the n-1 sd.  The generic residual-scale
        # default is biased for GU's heavy right tail and the RS trajectory
        # diverged from R's.
        sigma = np.sqrt(6.0) * np.std(y, ddof=1) / np.pi
    elif family.name in ("DEL", "DELAPORT", "LQNO"):
        # R DEL()$sigma.initial (live R, D2.38):
        #     sigma <- rep(max((var(y) - mean(y)) / mean(y)^2, 0.1), ...)
        # R LQNO()$sigma.initial (live R, D2.38):
        #     sigma <- rep(abs(var(y)/mean(y) - 1) / mean(y), length(y))
        # R's var() uses the n-1 denominator.  DEL clamps at 0.1 (a
        # variance/mean ratio below 0.1 under-disperses the start); LQNO
        # takes the absolute value instead (its mean can be near zero).
        mean_y = np.mean(y)
        var_y = np.var(y, ddof=1)
        if family.name in ("DEL", "DELAPORT"):
            sigma = max((var_y - mean_y) / max(mean_y**2, eps), 0.1)
        else:
            sigma = abs(var_y / max(mean_y, eps) - 1.0) / max(mean_y, eps)
    elif family.name in ("PIG",):
        # R PIG()$sigma.initial (live R, D2.38): the same
        # max((var(y) - mean(y)) / mean(y)^2, 0.1) overdispersion start as
        # DEL.  The generic residual-scale default ignores PIG's
        # variance-mean coupling and the RS trajectory diverged from R's.
        mean_y = np.mean(y)
        var_y = np.var(y, ddof=1)
        sigma = max((var_y - mean_y) / max(mean_y**2, eps), 0.1)
    elif family.name == "NO2":
        # R NO2()$sigma.initial (live R, D2.38): sigma <- rep(var(y), ...).
        # A *variance*, not an sd -- through sigma's log link the two differ
        # by a factor of 2 in eta (0.69 log units), which is exactly the
        # kind of half-scale start the trajectory does not walk off.
        sigma = np.var(y, ddof=1)
    elif family.name == "NOF":
        # R NOF()$sigma.initial (live R, D2.38):
        #     sigma <- rep(0.2 * sd(y) / sqrt(abs(mean(y))), length(y))
        sigma = 0.2 * np.std(y, ddof=1) / np.sqrt(abs(np.mean(y)))
    elif family.name in ("YULE", "WARING"):
        # R initializes WARING at sigma = 1; using a much smaller value sends
        # the fit too close to the boundary before the first update.
        if family.name == "WARING":
            sigma = 1.0
        else:
            sigma = 2.0  # YULE
    elif family.name in ("WEI2", "WEI3"):
        # R WEI2/WEI3 sigma.initial (live R, D2.38):
        #     s.Y.s <- 1.283/sqrt(var(log(y))); sigma <- rep(s.Y.s, ...)
        # (Weibull shape parameter; R's var() uses the n-1 denominator.)
        eps = np.finfo(np.float64).eps
        log_y = np.log(np.maximum(y, eps))
        sigma = 1.283 / np.sqrt(np.var(log_y))
    elif family.name == "RGE":
        # R RGE sigma.initial: sigma <- rep(0.78 * sd(y), length(y))
        sigma = 0.78 * np.std(y)
    elif family.name == "PE2":
        # R PE2 sigma.initial: sigma <- (abs(y - mean(y)) + sd(y)) / 2
        # R evaluates this per observation; collapsing to its response-scale
        # mean gives the same intercept-only start as mean(|y-m|)+sd)/2.
        mean_y = np.mean(y)
        sigma = (np.mean(np.abs(y - mean_y)) + np.std(y)) / 2.0
    elif family.name in ("LOGSHASH", "LOGSHASHo"):
        # R LOGSHASH()$sigma.initial (gamlss.dist development source
        # R/gamlss.dist/R/LOGSHASH.R, Mikis 15-07-2025):
        #     sigma <- rep(sd(log(y))/5, length(y))
        # R's sd() is the n-1 estimator (D2.38), and the spread is on the
        # log-y scale -- the same expression family as the S-hash trio
        # (sd(y)/5 there, sd(log(y))/5 here because LOGSHASH is their
        # log-transform).  Not in installed gamlss.dist 6.1.1; provenance
        # is the vendored development source.
        eps = np.finfo(np.float64).eps
        log_y = np.log(np.maximum(y, eps))
        sigma = np.std(log_y, ddof=1) / 5.0
    else:
        sigma = np.sqrt(np.sum(w * np.square(y - mu)) / np.sum(w))
    # Note: NO uses a log link for sigma, so its link-scale coefficient is
    # legitimately negative whenever sigma < 1 (the converged RS value for
    # typical data is negative too).  Only guard against a non-positive
    # residual scale here; forcing sigma >= 1 broke the gamlss_ml contract
    # that the stored intercept equals the moment estimate / RS result.
    return max(float(sigma), eps)




def _initial_sigma_fitted(
    family: FamilyDefinition,
    y: np.ndarray,
    mu: np.ndarray,
    w: np.ndarray,
    fixed_parameter_values: Mapping[str, np.ndarray] | None = None,
) -> np.ndarray:
    """Per-observation sigma starting vector (R's ``sigma.initial`` state).

    Most R families start sigma at ``rep(<scalar>, length(y))`` -- for those
    this is just the scalar broadcast to a vector, matching the response
    scale that enters R's first ``glim.fit`` for sigma.  PE / PE2 are the
    exceptions with a genuinely per-observation expression
    ``(abs(y - mean(y)) + sd(y)) / 2`` (live R, D2.38): R's ``sd()`` uses the
    n-1 denominator, and the per-observation spread is *not* cosmetic -- it
    sets the working weights of mu's and nu's first inner steps, which is
    what locked the PE trajectory onto R's bit-for-bit (initial deviance
    91.394523 = live R on the scanner sample).
    """

    if family.name in ("PE", "PE2"):
        mean_y = float(np.mean(y))
        sd_y = float(np.std(y, ddof=1))
        sigma_vec = (np.abs(y - mean_y) + sd_y) / 2.0
    else:
        sigma_vec = np.full(
            np.asarray(y).shape,
            _initial_sigma(family, y, mu, w, fixed_parameter_values),
            dtype=np.float64,
        )
    return np.maximum(sigma_vec, 1e-6)




__all__ = [
    "_initial_parameter_value",
    "_initial_mu_beta",
    "_initial_mu_target",
    "_initial_sigma",
    "_initial_sigma_fitted",
]
