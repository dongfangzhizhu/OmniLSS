"""R GAMLSS family/link framework migration.

R sources:
- `R/gamlss.dist/R/gamlss-family.R`: `as.family`, `gamlss.family`,
  `gamlss.family.default`, `as.gamlss.family`, `checklink`, `print.gamlss.family`
- `R/gamlss.dist/R/make-link-gamlss.R`: `make.link.gamlss`, `show.link`

The link closures reproduce live R's `linkfun` / `linkinv` / `mu.eta` /
`valideta` semantics, including R's epsilon clamps, so a ported call site gets
the same numbers.  `show_link` returns the permissible link list per parameter
(the data R's `show.link` returns), generated from live R and stored in
:mod:`omnilss._family_link_options`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import jax.numpy as jnp
from jax.scipy.stats import cauchy, norm

from ._family_link_options import FAMILY_LINK_OPTIONS
from .distribution_registry import resolve
from .families import FamilyDefinition

_F = jnp.float64
_EPS = jnp.finfo(_F).eps


def _arr(x: Any) -> Any:
    return jnp.asarray(x, dtype=_F)


@dataclass(frozen=True)
class LinkGamlss:
    """Python counterpart of R's `link-gamlss` object.

    Mirrors the five slots R builds in `make.link.gamlss`: `linkfun`,
    `linkinv`, `mu.eta`, `valideta` and `name`.
    """

    name: str
    linkfun: Callable[[Any], Any]
    linkinv: Callable[[Any], Any]
    mu_eta: Callable[[Any], Any]
    valideta: Callable[[Any], Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "linkfun": self.linkfun,
            "linkinv": self.linkinv,
            "mu.eta": self.mu_eta,
            "valideta": self.valideta,
        }


def _always_valid(eta: Any):
    return jnp.asarray(True)


def _positive_eta(eta: Any):
    return jnp.all(_arr(eta) > 0.0)


def _nonzero_eta(eta: Any):
    return jnp.all(_arr(eta) != 0.0)


def _bounded_link(name: str, shift: tuple[float, float]) -> LinkGamlss:
    """Bounded logit link used by R's `[-1,1]`, `(0,2]`, `(1,3]`, `(0,5]`.

    R's own `mu.eta` branch for these links contains a typo
    (`.Machine$souble.eps`) that would raise; we implement the mathematically
    intended derivative `e * (s1 - s0) / (1 + e)^2`, which is what R plainly
    meant.  `linkfun`/`linkinv`/`valideta` match R exactly.
    """

    s0, s1 = float(shift[0]), float(shift[1])

    def linkfun(mu: Any):
        mu = _arr(mu)
        return jnp.log((mu - s0) / (s1 - mu))

    def linkinv(eta: Any):
        eta = _arr(eta)
        thresh = -jnp.log(_EPS)
        eta = jnp.minimum(jnp.maximum(eta, -thresh), thresh)
        e = jnp.exp(eta)
        return (s1 * e + s0) / (1.0 + e)

    def mu_eta(eta: Any):
        eta = _arr(eta)
        thresh = -jnp.log(_EPS)
        eta = jnp.minimum(jnp.maximum(eta, -thresh), thresh)
        e = jnp.exp(eta)
        return e * (s1 - s0) / jnp.square(1.0 + e)

    return LinkGamlss(name, linkfun, linkinv, mu_eta, _always_valid)


def _logit() -> LinkGamlss:
    def linkfun(mu: Any):
        mu = jnp.clip(_arr(mu), _EPS, 1.0 - _EPS)
        return jnp.log(mu / (1.0 - mu))

    def linkinv(eta: Any):
        thresh = -jnp.log(_EPS)
        eta = jnp.clip(_arr(eta), -thresh, thresh)
        return 1.0 / (1.0 + jnp.exp(-eta))

    def mu_eta(eta: Any):
        m = linkinv(eta)
        return jnp.maximum(m * (1.0 - m), _EPS)

    return LinkGamlss("logit", linkfun, linkinv, mu_eta, _always_valid)


def _probit() -> LinkGamlss:
    def linkfun(mu: Any):
        return norm.ppf(_arr(mu))

    def linkinv(eta: Any):
        thresh = -norm.ppf(_EPS)
        eta = jnp.clip(_arr(eta), -thresh, thresh)
        return norm.cdf(eta)

    def mu_eta(eta: Any):
        return jnp.maximum(norm.pdf(_arr(eta)), _EPS)

    return LinkGamlss("probit", linkfun, linkinv, mu_eta, _always_valid)


def _cauchit() -> LinkGamlss:
    def linkfun(mu: Any):
        return cauchy.ppf(_arr(mu))

    def linkinv(eta: Any):
        thresh = -cauchy.ppf(_EPS)
        eta = jnp.clip(_arr(eta), -thresh, thresh)
        return cauchy.cdf(eta)

    def mu_eta(eta: Any):
        return jnp.maximum(cauchy.pdf(_arr(eta)), _EPS)

    return LinkGamlss("cauchit", linkfun, linkinv, mu_eta, _always_valid)


def _cloglog() -> LinkGamlss:
    def linkfun(mu: Any):
        return jnp.log(-jnp.log(1.0 - _arr(mu)))

    def linkinv(eta: Any):
        eta = _arr(eta)
        value = -jnp.expm1(-jnp.exp(eta))
        return jnp.maximum(jnp.minimum(value, 1.0 - _EPS), _EPS)

    def mu_eta(eta: Any):
        eta = jnp.minimum(_arr(eta), 700.0)
        return jnp.maximum(jnp.exp(eta) * jnp.exp(-jnp.exp(eta)), _EPS)

    return LinkGamlss("cloglog", linkfun, linkinv, mu_eta, _always_valid)


def _identity() -> LinkGamlss:
    def linkfun(mu: Any):
        return _arr(mu)

    def linkinv(eta: Any):
        return _arr(eta)

    def mu_eta(eta: Any):
        return jnp.ones_like(_arr(eta))

    return LinkGamlss("identity", linkfun, linkinv, mu_eta, _always_valid)


def _log() -> LinkGamlss:
    def linkfun(mu: Any):
        return jnp.log(_arr(mu))

    def linkinv(eta: Any):
        return jnp.maximum(jnp.exp(_arr(eta)), _EPS)

    def mu_eta(eta: Any):
        return jnp.maximum(jnp.exp(_arr(eta)), _EPS)

    return LinkGamlss("log", linkfun, linkinv, mu_eta, _always_valid)


def _sqrt() -> LinkGamlss:
    def linkfun(mu: Any):
        return jnp.power(_arr(mu), 0.5)

    def linkinv(eta: Any):
        return jnp.square(_arr(eta))

    def mu_eta(eta: Any):
        return 2.0 * _arr(eta)

    return LinkGamlss("sqrt", linkfun, linkinv, mu_eta, _positive_eta)


def _inverse_square() -> LinkGamlss:
    def linkfun(mu: Any):
        return 1.0 / jnp.square(_arr(mu))

    def linkinv(eta: Any):
        return 1.0 / jnp.power(_arr(eta), 0.5)

    def mu_eta(eta: Any):
        return -1.0 / (2.0 * jnp.power(_arr(eta), 1.5))

    return LinkGamlss("1/mu^2", linkfun, linkinv, mu_eta, _positive_eta)


def _square() -> LinkGamlss:
    def linkfun(mu: Any):
        return jnp.square(_arr(mu))

    def linkinv(eta: Any):
        return jnp.power(_arr(eta), 0.5)

    def mu_eta(eta: Any):
        return 0.5 * jnp.power(_arr(eta), -0.5)

    return LinkGamlss("mu^2", linkfun, linkinv, mu_eta, _positive_eta)


def _shifted_log(name: str, shift: float, delta: float, base: float) -> LinkGamlss:
    """R's shifted-log links.

    R builds these asymmetrically, so they do not all round-trip:
    - logshiftto0 / Slog: linkfun = log(mu - 1e-5), linkinv = 1e-5 + exp(eta)
      (round-trips exactly);
    - logshiftto1 / 2: linkfun = log(mu - shift + 1e-5), linkinv = shift + exp(eta)
      (round-trips to mu + 1e-5).
    We reproduce both branches exactly.
    """

    def linkfun(mu: Any):
        return jnp.log(_arr(mu) - shift + delta)

    def linkinv(eta: Any):
        return base + jnp.maximum(_EPS, jnp.exp(_arr(eta)))

    def mu_eta(eta: Any):
        return jnp.maximum(_EPS, jnp.exp(_arr(eta)))

    return LinkGamlss(name, linkfun, linkinv, mu_eta, _always_valid)


def _inverse() -> LinkGamlss:
    def linkfun(mu: Any):
        return 1.0 / _arr(mu)

    def linkinv(eta: Any):
        return 1.0 / _arr(eta)

    def mu_eta(eta: Any):
        return -1.0 / jnp.square(_arr(eta))

    return LinkGamlss("inverse", linkfun, linkinv, mu_eta, _nonzero_eta)


_LINK_FACTORIES: dict[str, Callable[[], LinkGamlss]] = {
    "logit": _logit,
    "probit": _probit,
    "cauchit": _cauchit,
    "cloglog": _cloglog,
    "identity": _identity,
    "log": _log,
    "sqrt": _sqrt,
    "1/mu^2": _inverse_square,
    "mu^2": _square,
    "inverse": _inverse,
    "logshiftto0": lambda: _shifted_log("logshiftto0", 0.0, -1e-5, 1e-5),
    "logshiftto1": lambda: _shifted_log("logshiftto1", 1.0, 1e-5, 1.0),
    "logshiftto2": lambda: _shifted_log("logshiftto2", 2.0, 1e-5, 2.0),
    "slog": lambda: _shifted_log("Slog", 0.0, -1e-5, 1e-5),
    "[-1,1]": lambda: _bounded_link("[-1,1]", (-1.0 - 1e-10, 1.0 + 1e-10)),
    "(0,2]": lambda: _bounded_link("(0,2]", (0.0, 2.0 + 1e-10)),
    "(1,3]": lambda: _bounded_link("(1,3]", (1.0, 3.0 + 1e-10)),
    "(0,5]": lambda: _bounded_link("(0,5]", (0.0, 5.0 + 1e-10)),
}


def make_link_gamlss(link: str | LinkGamlss) -> LinkGamlss:
    """R's `make.link.gamlss`: return a `link-gamlss` object for a link name.

    Mirrors the R switch: named links plus the bounded / shifted variants.
    `power(...)` and numeric links are deprecated in R and rejected here with
    the same intent; `own` needs user-defined functions and raises.
    """

    if isinstance(link, LinkGamlss):
        return link
    if not isinstance(link, str):
        raise ValueError(
            "numeric / non-character links (R's deprecated power()/number form) "
            "are not supported; pass a link name such as 'log'"
        )
    key = link.strip()
    lowered = key.lower()
    if lowered.startswith("power"):
        raise ValueError(
            "calling make_link_gamlss('power(...)') is deprecated in R and unsupported here"
        )
    if lowered == "own":
        raise NotImplementedError(
            "the 'own' link requires user-defined own.linkfun/own.linkinv/"
            "own.mu.eta/own.valideta functions; construct a LinkGamlss directly"
        )
    factory = _LINK_FACTORIES.get(key) or _LINK_FACTORIES.get(lowered)
    if factory is None:
        raise ValueError(f'"{link}" link not recognised')
    return factory()


def _link_options_for(name: str) -> dict[str, list[str]] | None:
    if name in FAMILY_LINK_OPTIONS:
        return FAMILY_LINK_OPTIONS[name]
    lowered = name.lower()
    for key, value in FAMILY_LINK_OPTIONS.items():
        if key.lower() == lowered:
            return value
    return None


def as_family(object: Any) -> FamilyDefinition:
    """R's `as.family`: coerce a name / callable / family object to a family."""

    if isinstance(object, FamilyDefinition):
        return object
    if isinstance(object, str):
        return resolve(object)
    if callable(object):
        return as_family(object())
    raise TypeError("The object argument is invalid")


def gamlss_family_default(object: Any) -> FamilyDefinition:
    """R's `gamlss.family.default`: the coercion used by `gamlss.family`."""

    if object is None:
        return resolve("NO")
    if isinstance(object, FamilyDefinition):
        return object
    if isinstance(object, str):
        return resolve(object)
    if callable(object):
        return gamlss_family_default(object())
    raise ValueError("The object argument is invalid")


def gamlss_family(object: Any) -> FamilyDefinition:
    """R's `gamlss.family` (S3 generic collapsed to a single dispatcher)."""

    return gamlss_family_default(object)


def as_gamlss_family(object: Any) -> FamilyDefinition:
    """R's `as.gamlss.family`."""

    if isinstance(object, FamilyDefinition):
        return object
    return gamlss_family_default(object)


def show_link(family: Any = "NO") -> dict[str, list[str]]:
    """R's `show.link`: permissible link names for each family parameter.

    Returns the `link.List` each family declares (exactly what R returns),
    looked up case-insensitively.  For a family that is not in the generated
    table, falls back to the family's declared default links so the call never
    silently returns an empty mapping.
    """

    fam = as_gamlss_family(family)
    options = _link_options_for(fam.name)
    if options is not None:
        return {parameter: list(links) for parameter, links in options.items()}
    fallback = dict(fam.links or {})
    if fallback:
        return {parameter: [link] for parameter, link in fallback.items()}
    return {parameter: [] for parameter in fam.parameters}


def checklink(
    which_link: str | None = None,
    which_dist: str | None = None,
    link: Any = None,
    link_list: list[str] | tuple[str, ...] | None = None,
) -> LinkGamlss:
    """R's `checklink`: resolve `link` to a `link-gamlss` object.

    R validates that the four arguments are supplied, then builds the link from
    a character name (even one outside `link_list`) or accepts an existing
    `link-gamlss` object; anything else raises with the "not available"
    message.  We replicate that behaviour, including the message shape.
    """

    if which_link is None:
        raise ValueError("The parameter link name has not been defined.")
    if which_dist is None:
        raise ValueError("The distribution has not been defined.")
    if link is None:
        raise ValueError("The link has not been defined.")
    if link_list is None:
        raise ValueError("The list of links has not been defined.")

    if isinstance(link, LinkGamlss):
        return link
    if isinstance(link, str):
        return make_link_gamlss(link)
    raise ValueError(
        f'"{which_link}" link "{link!r}" not available, for "{which_dist}" '
        f"family; available links are {list(link_list)}"
    )


def family_summary(family: Any) -> str:
    """Human-readable summary, mirroring R's `print.gamlss.family`."""

    fam = as_gamlss_family(family)
    lines = [f"GAMLSS Family: {fam.name}"]
    links = dict(fam.links or {})
    for parameter in fam.parameters:
        if parameter in links:
            label = f"{parameter:<5}"
            lines.append(f"Link function for {parameter:<5}: {links[parameter]}")
    return "\n".join(lines)


__all__ = [
    "LinkGamlss",
    "as_family",
    "as_gamlss_family",
    "checklink",
    "family_summary",
    "gamlss_family",
    "gamlss_family_default",
    "make_link_gamlss",
    "show_link",
]
