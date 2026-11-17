"""R gamlss.dist family generation: Family.d/p/q/r, Family, gen.Family, GAMLSS.

Every transformed density, CDF and quantile is compared against live R 4.6.0 +
gamlss.dist 6.1.1 (Family.d / Family.p / Family.q sourced from genFamily.R).
"""
from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from omnilss.families import FamilyDefinition  # noqa: E402
from omnilss.distribution_registry import resolve as _resolve_family  # noqa: E402
from omnilss.family_generator import (  # noqa: E402
    Family,
    GAMLSS,
    GAMLSSDistribution,
    GeneratedFamily,
    family_d,
    family_p,
    family_q,
    family_r,
    gen_Family,
)

_REPO = Path(__file__).resolve().parents[2]

_CASES = [
    ("NO", "log", 0.0, 1.0, [0.2, 0.5, 1.0, 2.5, 8.0], [0.1, 0.5, 0.9]),
    ("NO", "logit", 0.0, 1.0, [0.05, 0.2, 0.5, 0.8, 0.95], [0.1, 0.5, 0.9]),
    ("GU", "log", 0.0, 1.0, [0.2, 0.5, 1.0, 2.5, 8.0], [0.1, 0.5, 0.9]),
    ("LO", "logit", 0.0, 1.0, [0.05, 0.2, 0.5, 0.8, 0.95], [0.1, 0.5, 0.9]),
]

_RSCRIPT = """suppressPackageStartupMessages(library(gamlss.dist))
source('R/gamlss.dist/R/genFamily.R')
library(jsonlite)
args <- commandArgs(trailingOnly = TRUE)
cs <- fromJSON(args[1], simplifyDataFrame = FALSE)
out <- list()
for (i in seq_along(cs)) {
  c0 <- cs[[i]]
  fd <- Family.d(family = c0$fam, type = c0$type)
  fp <- Family.p(family = c0$fam, type = c0$type)
  fq <- Family.q(family = c0$fam, type = c0$type)
  out[[i]] <- list(
    d = as.numeric(fd(c0$x, mu = c0$mu, sigma = c0$sigma, log = TRUE)),
    p = as.numeric(fp(c0$x, mu = c0$mu, sigma = c0$sigma)),
    q = as.numeric(fq(c0$p, mu = c0$mu, sigma = c0$sigma))
  )
}
cat(toJSON(out, auto_unbox = FALSE, digits = 17))
"""


def test_invalid_type_raises() -> None:
    with pytest.raises(ValueError, match="should be one of"):
        family_d("NO", "sqrt")


@pytest.mark.parametrize("family,type_,mu,sigma,x,probs", _CASES)
def test_density_integrates_to_one(family, type_, mu, sigma, x, probs) -> None:
    # Change variables: with x = g(z) the density on the z scale is
    # f_x(g(z)) * g'(z), which must integrate to one over the whole line.
    grid = np.linspace(-9.0, 9.0, 60001)
    z = jnp.asarray(grid, dtype=jnp.float64)
    if type_ == "log":
        xs = jnp.exp(z)
        jacobian = np.asarray(jnp.exp(z), dtype=float)
    else:
        xs = 1.0 / (1.0 + jnp.exp(-z))
        ez = jnp.exp(-z)
        jacobian = np.asarray(ez / (1.0 + ez) ** 2, dtype=float)
    dens = np.asarray(
        family_d(family, type_)(xs, mu=mu, sigma=sigma, log=False), dtype=float
    )
    assert np.all(np.isfinite(dens))
    # Exact identity: the density on the z scale is the base density.
    base = _resolve_family(family).d(z, mu, sigma)
    np.testing.assert_allclose(dens * jacobian, np.asarray(base, dtype=float), rtol=1e-10, atol=1e-12)
    # Normalisation up to the truncation error of the finite grid.
    area = np.trapezoid(dens * jacobian, grid)
    assert area == pytest.approx(1.0, abs=5e-4)


@pytest.mark.parametrize("family,type_,mu,sigma,x,probs", _CASES)
def test_cdf_quantile_roundtrip(family, type_, mu, sigma, x, probs) -> None:
    p = jnp.asarray(probs, dtype=jnp.float64)
    values = np.asarray(family_q(family, type_)(p, mu=mu, sigma=sigma), dtype=float)
    back = np.asarray(
        family_p(family, type_)(jnp.asarray(values), mu=mu, sigma=sigma), dtype=float
    )
    np.testing.assert_allclose(back, np.asarray(probs), atol=1e-10)


def test_family_object_is_a_family_definition() -> None:
    generated = Family("NO", "log")
    assert isinstance(generated, FamilyDefinition)
    assert isinstance(generated, GeneratedFamily)
    assert generated.name == "logNO"
    assert generated.parameters == ("mu", "sigma")
    assert generated.transform_type == "log"
    assert generated.base_name == "NO"


def test_family_rejects_discrete_base() -> None:
    with pytest.raises(ValueError, match="should be continuous"):
        Family("PO", "log")


def test_y_valid_uses_the_transform_support() -> None:
    assert bool(Family("NO", "log").y_valid([0.5, 2.0]))
    assert not bool(Family("NO", "log").y_valid([-0.5, 2.0]))
    assert bool(Family("NO", "logit").y_valid([0.2, 0.8]))
    assert not bool(Family("NO", "logit").y_valid([1.2]))


def test_gen_family_returns_r_style_names() -> None:
    generated = gen_Family("NO", "log")
    for key in ("d", "p", "q", "r", "family", "dlogNO", "plogNO", "qlogNO", "rlogNO", "logNO"):
        assert key in generated
    assert generated["family"] is generated["logNO"]
    assert generated["dlogNO"] is generated["family"].d


def test_random_generator_length_and_reproducibility() -> None:
    key = jax.random.PRNGKey(7)
    first = np.asarray(family_r("NO", "log")(50, key=key, mu=0.0, sigma=1.0), dtype=float)
    second = np.asarray(family_r("NO", "log")(50, key=key, mu=0.0, sigma=1.0), dtype=float)
    assert first.shape == (50,)
    assert np.all(first > 0.0)
    np.testing.assert_allclose(first, second)


def test_gamlss_distribution_binds_parameters() -> None:
    dist = GAMLSS("NO", mu=0.0, sigma=1.0)
    assert isinstance(dist, GAMLSSDistribution)
    assert dist.family_name == "NO"
    assert dist.pdf(0.0) == pytest.approx(0.3989422804014327, rel=1e-12)
    assert float(dist.cdf(0.0)) == pytest.approx(0.5, abs=1e-12)
    assert float(dist.quantile(0.5)) == pytest.approx(0.0, abs=1e-9)
    draws = np.asarray(dist.random(20, key=jax.random.PRNGKey(0)), dtype=float)
    assert draws.shape == (20,)


def test_gamlss_distribution_warns_on_foreign_parameter() -> None:
    with pytest.warns(UserWarning, match="not a parameter"):
        GAMLSS("NO", mu=0.0, sigma=1.0, tau=3.0)


def test_gamlss_distribution_reports_missing_mean() -> None:
    dist = GAMLSS("NO", mu=0.0, sigma=1.0)
    with pytest.raises(NotImplementedError, match="mean is not implemented"):
        dist.mean()


@pytest.mark.slow
def test_transformed_families_match_live_r() -> None:
    if shutil.which("Rscript") is None:
        pytest.skip("Rscript not available")
    if not (_REPO / "R" / "gamlss.dist" / "R" / "genFamily.R").exists():
        pytest.skip("vendored genFamily.R not present")
    payload = [
        {"fam": f, "type": t, "mu": m, "sigma": s, "x": x, "p": pr}
        for (f, t, m, s, x, pr) in _CASES
    ]
    with tempfile.TemporaryDirectory() as tmp:
        pay = Path(tmp) / "p.json"
        pay.write_text(json.dumps(payload), encoding="utf-8")
        rscript = Path(tmp) / "ref.R"
        rscript.write_text(_RSCRIPT, encoding="utf-8")
        proc = subprocess.run(
            ["Rscript", str(rscript), str(pay)],
            capture_output=True,
            text=True,
            timeout=300,
            cwd=str(_REPO),
        )
    if proc.returncode != 0:
        pytest.skip(f"R failed: {proc.stderr[-300:]}")
    reference = json.loads(proc.stdout.strip().splitlines()[-1])
    for index, (f, t, m, s, x, pr) in enumerate(_CASES):
        xs = jnp.asarray(x, dtype=jnp.float64)
        ps = jnp.asarray(pr, dtype=jnp.float64)
        np.testing.assert_allclose(
            np.asarray(family_d(f, t)(xs, mu=m, sigma=s, log=True), dtype=float),
            np.asarray(reference[index]["d"], dtype=float),
            rtol=1e-10,
            atol=1e-12,
            err_msg=f"{f} {t} density",
        )
        np.testing.assert_allclose(
            np.asarray(family_p(f, t)(xs, mu=m, sigma=s), dtype=float),
            np.asarray(reference[index]["p"], dtype=float),
            rtol=1e-10,
            atol=1e-12,
            err_msg=f"{f} {t} cdf",
        )
        np.testing.assert_allclose(
            np.asarray(family_q(f, t)(ps, mu=m, sigma=s), dtype=float),
            np.asarray(reference[index]["q"], dtype=float),
            rtol=1e-10,
            atol=1e-12,
            err_msg=f"{f} {t} quantile",
        )


def test_generated_derivatives_see_the_transformed_response() -> None:
    """D2.243: score/hessian must be evaluated at transform(y).

    R's Family() rewrites y to log(y) inside the derivative bodies.  Before the
    wrapper was added the generated family mixed a transformed deviance with
    gradients at the raw response, and an RS fit returned deviance 413.47 where
    R returns 140.38 on the same sample.
    """
    base = _resolve_family("NO")
    log_family = Family("NO", "log")
    y = jnp.asarray([0.5, 1.0, 2.5], dtype=jnp.float64)
    mu = jnp.asarray([0.3, 0.3, 0.3], dtype=jnp.float64)
    sigma = jnp.asarray([0.8, 0.8, 0.8], dtype=jnp.float64)
    np.testing.assert_allclose(
        np.asarray(log_family.score_functions["mu"](y, mu, sigma), dtype=float),
        np.asarray(base.score_functions["mu"](jnp.log(y), mu, sigma), dtype=float),
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        np.asarray(log_family.score_functions["sigma"](y, mu, sigma), dtype=float),
        np.asarray(base.score_functions["sigma"](jnp.log(y), mu, sigma), dtype=float),
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        np.asarray(log_family.hessian_functions["mu"](y, mu, sigma), dtype=float),
        np.asarray(base.hessian_functions["mu"](jnp.log(y), mu, sigma), dtype=float),
        rtol=0.0,
        atol=0.0,
    )


@pytest.mark.parametrize(
    "transform,seed,mu_scale,sigma_scale,deviance,mu_coef,sigma_coef",
    [
        (
            "log", 42, 0.3, 0.4,
            140.3830665004935,
            (0.43908389769441319, 0.36313077671600119),
            (-0.98996841514882172,),
        ),
        (
            "logit", 77, 0.5, 0.35,
            -188.50696591334733,
            (0.14193788197275153, 0.46144047154206708),
            (-0.99366711370308858,),
        ),
    ],
)
def test_generated_family_fit_matches_r_rs(
    transform, seed, mu_scale, sigma_scale, deviance, mu_coef, sigma_coef
) -> None:
    """Live R 4.6.0 + gamlss.dist 6.1.1: gen.Family("NO", type) then RS(n.cyc=100).

    Values captured from the R run recorded in the D2.243 log.  Before the
    derivative fix this fit disagreed by deviance 413.47 vs 140.38 (log).
    """
    import numpy as np

    from omnilss import gamlss

    rng = np.random.default_rng(seed)
    n = 80 if transform == "log" else 90
    x = rng.normal(size=n)
    if transform == "log":
        y = np.exp(0.5 + mu_scale * x + rng.normal(scale=sigma_scale, size=n))
    else:
        z = 0.2 + mu_scale * x + rng.normal(scale=sigma_scale, size=n)
        y = 1.0 / (1.0 + np.exp(-z))
    model = gamlss(
        "y ~ x", family=Family("NO", transform), data={"y": y, "x": x},
        method="RS", max_iter=100,
    )
    assert model.converged
    assert float(model.g_dev) == pytest.approx(deviance, rel=2e-4, abs=1e-6)
    np.testing.assert_allclose(
        np.ravel(np.asarray(model.coefficients["mu"], dtype=float)),
        np.asarray(mu_coef, dtype=float),
        rtol=2e-4,
        atol=2e-4,
    )
    np.testing.assert_allclose(
        np.ravel(np.asarray(model.coefficients["sigma"], dtype=float)),
        np.asarray(sigma_coef, dtype=float),
        rtol=2e-4,
        atol=2e-4,
    )
