"""Initial-value helpers extracted from fitting.py for staged B1 split."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from .families import FamilyDefinition


def _initial_parameter_value(
    family: FamilyDefinition,
    parameter: str,
    y: np.ndarray,
    mu: np.ndarray,
    w: np.ndarray,
) -> float:
    """Construct staged starting values for non-mu parameters."""

    if parameter == "sigma":
        return _initial_sigma(family, y, mu, w)
    if parameter == "nu":
        if family.name in ("ZABB", "ZIBB"):
            # nu is a probability, not a generic shape initialized at one.
            zero_fraction = float(np.average(np.asarray(y) == 0, weights=w))
            return float(np.clip(zero_fraction, 0.05, 0.95))
        if family.name == "PE":
            return 1.8
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
        if family.name == "DEL":
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
        if family.name == "ZANBI":
            # R ZANBI()$nu.initial (live R, D2.38):
            #     nu <- rep(max(sum(y == 0)/length(y), 0.01), length(y))
            # The observed zero fraction, floored at 0.01 -- not the generic
            # 1.0, which starts the logit eta ~15 units away from R's.
            zero_fraction = float(np.average(np.asarray(y) == 0, weights=w))
            return max(zero_fraction, 0.01)
        if family.name == "GT":
            return 5.0  # Degrees of freedom, reasonable starting value
        if family.name == "SHASH":
            return 1.0  # Skewness parameter
        if family.name == "SHASHo":
            return 0.0  # Skewness parameter (original parameterization)
        if family.name == "SN1":
            return 0.0  # Skewness parameter
        if family.name == "SN2":
            return 1.0  # Scale ratio parameter
        # Zero-inflated/altered distributions whose zero-mass parameter is
        # literally `nu` (D2.26).  R starts each of these at
        # `nu <- rep(0.1, length(y))` -- read off the R sources, e.g.
        # ZIPIG.R:91, ZAPIG.R, ZAZIPF.R, ZABI.R, ZANBI.R, ZINBI.R, ZIBI.R.
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
        if family.name in ("ZIPIG", "ZAPIG", "ZIBI", "ZAZIPF", "ZABI",
                           "ZANBI", "ZINBI"):
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
        return 1.0
    if parameter == "tau":
        if family.name == "JSU":
            return 1.0
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
        if family.name == "SHASH":
            return 1.0  # Kurtosis parameter
        if family.name == "SHASHo":
            return 1.0  # Kurtosis parameter
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
    elif family.name == "ZIP":
        target = np.log(np.maximum(y + 0.1, np.finfo(np.float64).eps))
    elif family.name in ("YULE", "WARING"):
        target = np.log(np.maximum(y + 0.1, np.finfo(np.float64).eps))
    elif family.name in ("ZAGA", "ZAIG", "ZAP", "ZINBI", "ZIP2"):
        # Zero-altered/inflated: use log link, exclude zeros from initialization
        # For zeros, use small positive value
        target = np.log(np.maximum(y + 0.1, np.finfo(np.float64).eps))
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
    elif family.name in ("BCCGo", "BCTo", "BCPEo", "GPO"):
        # Same R mu.initial expression, but these carry mu.link = "log", so
        # the link-scale target is log of the response-scale start.  GPO was
        # added in D2.38 (live R dump): with target=y through a log link the
        # fit started at exp(OLS(y~x)) and parked 1.17 deviance above R;
        # with R's start it reproduces R to 6 decimals (212.424994 vs
        # 212.4250).
        eps = np.finfo(np.float64).eps
        target = np.log(np.maximum((y + np.mean(y)) / 2.0, eps))
    elif family.name in ("DPO", "PARETO"):
        # R DPO()/PARETO()$mu.initial (live R, D2.38), log link:
        #   DPO:    mu <- (y + mean(y))/2   (y is a count, strictly positive)
        #   PARETO: mu <- rep(2, length(y)) -- a *constant* 2, not data-driven.
        # PARETO matters for the per-observation rollout: its heavy-tailed
        # samples sent the generic target=y start to exp(y_i), which on a
        # Pareto(2) draw reaches exp(~30) and blew the intercept-only fit to
        # deviance -inf (test_r_consistency_batch7_8 Pareto intercept-only).
        eps = np.finfo(np.float64).eps
        if family.name == "DPO":
            target = np.log(np.maximum((y + np.mean(y)) / 2.0, eps))
        else:
            target = np.full(y.shape, np.log(2.0), dtype=np.float64)
    elif family.name == "DEL":
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
    elif family.name in ("PIG", "ZANBI", "GIG", "ZIPIG", "ZAPIG"):
        # R mu.initial (live R, D2.38): mu <- (y + mean(y))/2, log link for
        # all five.  PIG previously started at exp(OLS(y~x)) through its log
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
        target = np.log(np.maximum(y, np.finfo(np.float64).eps))
    elif family.name == "LNO":
        # LNO is alias for LOGNO (identity link for mu)
        target = np.log(np.maximum(y, np.finfo(np.float64).eps))
    elif family.name == "NBI":
        target = np.log(np.maximum(y + 0.1, np.finfo(np.float64).eps))
    elif family.name == "IG":
        target = np.log(np.maximum(y, np.finfo(np.float64).eps))
    elif family.name == "IGAMMA":
        target = np.log(np.maximum(y, np.finfo(np.float64).eps))
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

    eps = np.finfo(np.float64).eps
    if family.name in ("ZABB", "ZIBB"):
        return 0.1
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
    if family.name == "LOGNO":
        log_y = np.log(np.maximum(y, eps))
        sigma = np.sqrt(np.sum(w * np.square(log_y - mu)) / np.sum(w))
    elif family.name == "LNO":
        # LNO is alias for LOGNO
        log_y = np.log(np.maximum(y, eps))
        sigma = np.sqrt(np.sum(w * np.square(log_y - mu)) / np.sum(w))
    elif family.name == "GA":
        centered = (y - mu) / np.maximum(mu, eps)
        sigma = np.sqrt(np.sum(w * np.square(centered)) / np.sum(w))
    elif family.name == "NBI":
        mean_component = np.maximum(mu, eps)
        variance_component = np.sum(w * np.square(y - mu)) / np.sum(w)
        mean_level = np.sum(w * mean_component) / np.sum(w)
        sigma = np.maximum((variance_component - mean_level) / np.maximum(mean_level**2, eps), eps)
    elif family.name == "IG":
        centered = (y - mu) / np.maximum(np.power(mu, 1.5), eps)
        sigma = np.sqrt(np.sum(w * np.square(centered)) / np.sum(w))
    elif family.name == "BE":
        clipped_y = np.clip(y, eps, 1.0 - eps)
        clipped_mu = np.clip(mu, eps, 1.0 - eps)
        variance_component = np.sum(w * np.square(clipped_y - clipped_mu)) / np.sum(w)
        mean_level = np.sum(w * clipped_mu) / np.sum(w)
        denom = np.maximum(mean_level * (1.0 - mean_level), eps)
        sigma = np.maximum(variance_component / denom, eps)
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
        # NBII: size = mu/sigma, Var(Y) = mu(1 + sigma)
        # Therefore: sigma = Var(Y)/mu - 1
        mean_component = np.maximum(mu, eps)
        variance_component = np.sum(w * np.square(y - mu)) / np.sum(w)
        mean_level = np.sum(w * mean_component) / np.sum(w)
        # sigma = (variance - mean) / mean
        sigma = np.maximum((variance_component - mean_level) / np.maximum(mean_level, eps), eps)
    elif family.name == "PARETO2":
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
        # IGAMMA: Use a simple fixed starting value
        # R's formula gives values that are too large and unstable
        # Start with a conservative small value
        sigma = 0.5
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
    elif family.name == "GPO":
        # R GPO()$sigma.initial (live R, D2.38): sigma <- rep(0.1, length(y)).
        # The generic residual-scale default (~1.9 on the scanner sample)
        # started sigma 20x above R and the fit parked 1.17 deviance above
        # R's answer; with rep(0.1) the trajectory reproduces R to six
        # decimals (212.424994 vs 212.4250).
        sigma = 0.1
    elif family.name == "DPO":
        # R DPO()$sigma.initial (live R, D2.38): sigma <- rep(1, length(y)).
        sigma = 1.0
    elif family.name == "GU":
        # R GU()$sigma.initial (live R, D2.38):
        #     sigma <- rep(sqrt(6) * sd(y) / pi, length(y))
        # The Gumbel scale from the n-1 sd.  The generic residual-scale
        # default is biased for GU's heavy right tail and the RS trajectory
        # diverged from R's.
        sigma = np.sqrt(6.0) * np.std(y, ddof=1) / np.pi
    elif family.name in ("DEL", "LQNO"):
        # R DEL()$sigma.initial (live R, D2.38):
        #     sigma <- rep(max((var(y) - mean(y)) / mean(y)^2, 0.1), ...)
        # R LQNO()$sigma.initial (live R, D2.38):
        #     sigma <- rep(abs(var(y)/mean(y) - 1) / mean(y), length(y))
        # R's var() uses the n-1 denominator.  DEL clamps at 0.1 (a
        # variance/mean ratio below 0.1 under-disperses the start); LQNO
        # takes the absolute value instead (its mean can be near zero).
        mean_y = np.mean(y)
        var_y = np.var(y, ddof=1)
        if family.name == "DEL":
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
