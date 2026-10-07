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
        if family.name == "TF":
            return 10.0
        if family.name == "TF2":
            # R's TF2()$nu.initial is 4, not TF's 10 -- verified against live R.
            # Do not fold this into the TF branch above: same parameter name,
            # same file, different value, which is the D2.19 lesson again.
            return 4.0
        if family.name == "JSU":
            return 0.0
        if family.name == "BCCG":
            return 1.0
        if family.name == "BCT":
            return 1.0
        if family.name == "BCPE":
            return 1.0
        if family.name == "GG":
            return 2.0  # Shape parameter, 2.0 is reasonable for gamma-like data
        if family.name == "GB2":
            return 1.0  # Shape parameter 1, start from symmetric case for stability
        if family.name == "DEL":
            return 0.5  # Poisson component, small starting value
        if family.name == "NET":
            return 1.0  # Threshold k1, 1.0 is reasonable starting value
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
        # Zero-inflated/altered distributions: nu is zero-inflation probability
        # Use logit link, so initial value should be small but positive
        if family.name in ("ZAGA", "ZAIG", "ZAP", "ZINBI", "ZIP2", "BEZI", "BEOI"):
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
        if family.name == "GB2":
            return 1.0  # Shape parameter 2, start from symmetric case for stability
        if family.name == "BCPE":
            return 2.0
        if family.name == "NET":
            return 2.0  # Threshold k2, must be >= nu, 2.0 > 1.0
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
        return 1.0
    return 1.0




def _initial_mu_beta(
    family: FamilyDefinition,
    x: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    fixed_parameter_values: Mapping[str, np.ndarray] | None = None,
) -> np.ndarray:
    """Construct a family-aware starting point for the mu predictor."""

    sqrt_w = np.sqrt(w)
    if family.name == "NO":
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
    elif family.name == "BI":
        clipped = np.clip(y, 1e-6, 1.0 - 1e-6)
        target = np.log(clipped / (1.0 - clipped))
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
        clipped = np.clip(y / bd, 1e-6, 1.0 - 1e-6)
        target = np.log(clipped / (1.0 - clipped))
    elif family.name == "BE":
        clipped = np.clip(y, 1e-6, 1.0 - 1e-6)
        target = np.log(clipped / (1.0 - clipped))
    elif family.name == "WEI":
        target = np.log(np.maximum(y, np.finfo(np.float64).eps))
    elif family.name == "BCCG":
        target = np.log(np.maximum(y, np.finfo(np.float64).eps))
    elif family.name == "BCT":
        target = np.log(np.maximum(y, np.finfo(np.float64).eps))
    elif family.name == "BCPE":
        target = np.log(np.maximum(y, np.finfo(np.float64).eps))
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
        # NET: mu uses identity link
        target = y
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
    elif family.name == "RGE":
        # R RGE mu.initial (mu uses an identity link):
        #     mu <- y + 0.45 * sd(y)
        target = y + 0.45 * np.std(y)
    elif family.name == "PE2":
        # R PE2 mu.initial (mu uses an identity link):
        #     mu <- (y + mean(y)) / 2
        target = (y + np.mean(y)) / 2.0
    else:
        target = y
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
        # Zero-inflated/altered discrete: use simple starting value
        sigma = 0.1
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
    elif family.name == "NET":
        # NET: Use conservative starting value
        sigma = 1.0
    elif family.name == "PE":
        mean_y = np.sum(w * y) / np.sum(w)
        sigma = (np.sum(w * np.abs(y - mean_y)) / np.sum(w) + np.sqrt(np.sum(w * np.square(y - mean_y)) / np.sum(w))) / 2.0
    elif family.name in ("YULE", "WARING"):
        # R initializes WARING at sigma = 1; using a much smaller value sends
        # the fit too close to the boundary before the first update.
        if family.name == "WARING":
            sigma = 1.0
        else:
            sigma = 2.0  # YULE
    elif family.name == "WEI2":
        # R WEI2 sigma.initial: s.Y.s <- 1.283/sqrt(var(log(y)));
        #                       sigma <- rep(s.Y.s, length(y))
        # (Weibull shape parameter; response scale, log link downstream.)
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




__all__ = ["_initial_parameter_value", "_initial_mu_beta", "_initial_sigma"]
