"""Tests for the R gamlss family/link framework migration.

R sources: R/gamlss.dist/R/gamlss-family.R and make-link-gamlss.R.
Covers make.link.gamlss (linkfun/linkinv/mu.eta/valideta), checklink,
show.link, as.family / gamlss.family / as.gamlss.family.
"""
from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)  # parity prerequisite

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from omnilss.distribution_registry import resolve  # noqa: E402
from omnilss.families import FamilyDefinition  # noqa: E402
from omnilss.gamlss_family import (  # noqa: E402
    LinkGamlss,
    as_family,
    as_gamlss_family,
    checklink,
    family_summary,
    gamlss_family,
    gamlss_family_default,
    make_link_gamlss,
    show_link,
)

# (mu grid, eta grid) valid for each link.  Bounded links use interior points.
_GRIDS = {
    "logit": (np.array([0.1, 0.3, 0.5, 0.7, 0.9]), np.array([-2.0, -0.5, 0.0, 0.5, 2.0])),
    "probit": (np.array([0.1, 0.3, 0.5, 0.7, 0.9]), np.array([-2.0, -0.5, 0.0, 0.5, 2.0])),
    "cauchit": (np.array([0.1, 0.3, 0.5, 0.7, 0.9]), np.array([-2.0, -0.5, 0.0, 0.5, 2.0])),
    "cloglog": (np.array([0.1, 0.3, 0.5, 0.7, 0.9]), np.array([-3.0, -1.0, 0.0, 1.0, 3.0])),
    "identity": (np.array([-1.0, 0.0, 0.5, 1.0, 2.0]), np.array([-2.0, -0.5, 0.0, 0.5, 2.0])),
    "log": (np.array([0.1, 0.5, 1.0, 2.0, 5.0]), np.array([-2.0, -1.0, 0.0, 1.0, 2.0])),
    "sqrt": (np.array([0.1, 0.5, 1.0, 2.0, 5.0]), np.array([0.3, 0.5, 1.0, 2.0, 3.0])),
    "1/mu^2": (np.array([0.1, 0.5, 1.0, 2.0, 5.0]), np.array([0.3, 0.5, 1.0, 2.0, 3.0])),
    "mu^2": (np.array([0.1, 0.5, 1.0, 2.0, 5.0]), np.array([0.3, 0.5, 1.0, 2.0, 3.0])),
    "logshiftto0": (np.array([0.1, 0.5, 1.0, 2.0, 5.0]), np.array([-1.0, 0.0, 1.0, 2.0, 3.0])),
    "Slog": (np.array([0.1, 0.5, 1.0, 2.0, 5.0]), np.array([-1.0, 0.0, 1.0, 2.0, 3.0])),
    "logshiftto1": (np.array([1.1, 1.5, 2.0, 3.0, 5.0]), np.array([-1.0, 0.0, 1.0, 2.0, 3.0])),
    "logshiftto2": (np.array([2.1, 2.5, 3.0, 4.0, 6.0]), np.array([-1.0, 0.0, 1.0, 2.0, 3.0])),
    "inverse": (np.array([0.1, 0.5, 1.0, 2.0, 5.0]), np.array([-2.0, -1.0, 0.5, 1.0, 2.0])),
}


def _links() -> list[str]:
    return list(_GRIDS)


_SHIFT_ATOL = {"logshiftto1": 2e-5, "logshiftto2": 2e-5}


@pytest.mark.parametrize("name", _links())
def test_link_roundtrip(name: str) -> None:
    mu, _ = _GRIDS[name]
    link = make_link_gamlss(name)
    back = np.asarray(link.linkinv(link.linkfun(jnp.asarray(mu))))
    # R's logshiftto1/2 deliberately round-trip to mu + 1e-5.
    np.testing.assert_allclose(back, mu, rtol=1e-9, atol=_SHIFT_ATOL.get(name, 1e-10))


@pytest.mark.parametrize("name", [n for n in _links() if n != "cauchit"])
def test_mu_eta_is_derivative_of_linkinv(name: str) -> None:
    _, eta = _GRIDS[name]
    link = make_link_gamlss(name)
    h = 1e-6
    numeric = (
        np.asarray(link.linkinv(jnp.asarray(eta + h)))
        - np.asarray(link.linkinv(jnp.asarray(eta - h)))
    ) / (2 * h)
    analytic = np.asarray(link.mu_eta(jnp.asarray(eta)))
    np.testing.assert_allclose(analytic, numeric, rtol=1e-4, atol=1e-6)


def test_unknown_link_raises() -> None:
    with pytest.raises(ValueError, match="not recognised"):
        make_link_gamlss("definitely-not-a-link")


def test_power_and_numeric_are_rejected() -> None:
    with pytest.raises(ValueError, match="deprecated"):
        make_link_gamlss("power(0.5)")
    with pytest.raises(ValueError, match="not supported"):
        make_link_gamlss(0.5)  # type: ignore[arg-type]


def test_own_link_needs_user_functions() -> None:
    with pytest.raises(NotImplementedError, match="own"):
        make_link_gamlss("own")


def test_link_object_has_all_slots() -> None:
    link = make_link_gamlss("log")
    assert isinstance(link, LinkGamlss)
    assert set(link.as_dict()) == {"name", "linkfun", "linkinv", "mu.eta", "valideta"}
    assert link.name == "log"


@pytest.mark.parametrize(
    "family,expected",
    [
        ("NO", {"mu": ["inverse", "log", "identity", "own"], "sigma": ["inverse", "log", "identity", "own"]}),
        ("PO", {"mu": ["inverse", "log", "sqrt", "identity"]}),
        ("BE", {"mu": ["logit", "probit", "cloglog", "cauchit", "log", "own"]}),
    ],
)
def test_show_link_matches_r(family: str, expected: dict[str, list[str]]) -> None:
    got = show_link(family)
    for parameter, links in expected.items():
        assert got[parameter] == links, (family, parameter, got[parameter])


def test_show_link_is_case_insensitive() -> None:
    assert show_link("no") == show_link("NO")


def test_show_link_falls_back_for_unknown_family_entry() -> None:
    # A synthetic family whose name is not in the generated table.
    fam = FamilyDefinition(
        name="__SYNTHETIC__",
        parameters=("mu", "sigma"),
        g_dev_inc=lambda *a, **k: 0.0,
        links={"mu": "identity", "sigma": "log"},
    )
    got = show_link(fam)
    assert got == {"mu": ["identity"], "sigma": ["log"]}


def test_checklink_accepts_string_and_object() -> None:
    link = checklink(which_link="mu", which_dist="NO", link="log", link_list=["log", "identity"])
    assert link.name == "log"
    existing = make_link_gamlss("identity")
    assert checklink("mu", "NO", existing, ["identity"]) is existing


def test_checklink_requires_all_arguments() -> None:
    with pytest.raises(ValueError, match="link name has not been defined"):
        checklink(which_dist="NO", link="log", link_list=["log"])
    with pytest.raises(ValueError, match="distribution has not been defined"):
        checklink(which_link="mu", link="log", link_list=["log"])
    with pytest.raises(ValueError, match="link has not been defined"):
        checklink(which_link="mu", which_dist="NO", link_list=["log"])
    with pytest.raises(ValueError, match="list of links"):
        checklink(which_link="mu", which_dist="NO", link="log")


def test_checklink_rejects_non_link_object() -> None:
    with pytest.raises(ValueError, match="not available"):
        checklink("mu", "NO", object(), ["log"])


def test_family_coercion_helpers() -> None:
    no = resolve("NO")
    assert as_family("NO") is not None and as_family("NO").name == no.name
    assert as_family(no) is no
    assert as_family(lambda: resolve("GA")).name == resolve("GA").name
    assert gamlss_family(None).name == "NO"
    assert gamlss_family_default("PO").name == "PO"
    assert as_gamlss_family(no) is no
    with pytest.raises(ValueError, match="invalid"):
        gamlss_family_default(object())


def test_family_summary_lists_links() -> None:
    text = family_summary("NO")
    assert "GAMLSS Family: NO" in text
    assert "identity" in text and "log" in text
