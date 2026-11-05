"""p/q/r functions for the R 'o' (original) parameterisation families.

Every CDF and quantile is compared against live R 4.6.0 + gamlss.dist 6.1.1.
R defines pBCCGo == pBCCG, pBCPEo == pBCPE and pBCTo == pBCT, so those delegate
to the registry family directly.
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

from omnilss.o_variant_distributions import (  # noqa: E402
    pBCCGo,
    pBCPEo,
    pBCTo,
    pBEo,
    pGEOMo,
    pJSUo,
    pPARETO2o,
    pSHASHo,
    qBCCGo,
    qBCPEo,
    qBCTo,
    qBEo,
    qGEOMo,
    qJSUo,
    qPARETO2o,
    qSHASHo,
    rBCCGo,
    rBCPEo,
    rBCTo,
    rBEo,
    rGEOMo,
    rPARETO2o,
    rSHASHo,
)

# family -> (p, q, params, x grid, continuous?)
_CASES = {
    "BCCGo": (pBCCGo, qBCCGo, {"mu": 2.0, "sigma": 0.5, "nu": 0.5}, [0.5, 1, 2, 3, 4], True),
    "BCPEo": (
        pBCPEo,
        qBCPEo,
        {"mu": 2.0, "sigma": 0.5, "nu": 0.5, "tau": 2.0},
        [0.5, 1, 2, 3, 4],
        True,
    ),
    "BCTo": (pBCTo, qBCTo, {"mu": 2.0, "sigma": 0.5, "nu": 0.5, "tau": 2.0}, [0.5, 1, 2, 3, 4], True),
    "BEo": (pBEo, qBEo, {"mu": 0.5, "sigma": 0.2}, [0.1, 0.3, 0.5, 0.7, 0.9], True),
    "GEOMo": (pGEOMo, qGEOMo, {"mu": 0.3}, [0, 1, 2, 3, 4], False),
    "JSUo": (pJSUo, qJSUo, {"mu": 0.0, "sigma": 1.0, "nu": 0.5, "tau": 2.0}, [-2, -1, 0, 1, 2], True),
    "PARETO2o": (pPARETO2o, qPARETO2o, {"mu": 1.0, "sigma": 0.5}, [1.5, 2, 3, 4, 5], True),
    "SHASHo": (pSHASHo, qSHASHo, {"mu": 0.0, "sigma": 1.0, "nu": 1.0, "tau": 2.0}, [-2, -1, 0, 1, 2], True),
}

_PROBS = [0.1, 0.3, 0.5, 0.7, 0.9]

_RSCRIPT = """suppressPackageStartupMessages(library(gamlss.dist))
library(jsonlite)
args <- commandArgs(trailingOnly = TRUE)
p <- fromJSON(args[1], simplifyDataFrame = FALSE)
out <- list()
for (fam in names(p)) {
  pf <- get(paste0("p", fam))
  qf <- get(paste0("q", fam))
  par <- as.list(p[[fam]]$params)
  out[[fam]] <- list(
    p = as.numeric(do.call(pf, c(list(q = p[[fam]]$x), par))),
    q = as.numeric(do.call(qf, c(list(p = p[[fam]]$probs), par))),
    lower = as.numeric(do.call(pf, c(list(q = p[[fam]]$x), par, list(lower.tail = FALSE)))),
    logp = as.numeric(do.call(pf, c(list(q = p[[fam]]$x), par, list(log.p = TRUE))))
  )
}
cat(toJSON(out, auto_unbox = FALSE, digits = 17))
"""


def test_lower_tail_and_log_p_are_consistent() -> None:
    for name, (pfunc, _qfunc, params, x, _cont) in _CASES.items():
        # PARETO2o is covered separately: R itself ignores log.p there.
        q = jnp.asarray(x, dtype=jnp.float64)
        lower = np.asarray(pfunc(q, **params), dtype=float)
        upper = np.asarray(pfunc(q, **params, lower_tail=False), dtype=float)
        logged = np.asarray(pfunc(q, **params, log_p=True), dtype=float)
        np.testing.assert_allclose(lower + upper, 1.0, atol=1e-12, err_msg=name)
        np.testing.assert_allclose(logged, np.log(lower), atol=1e-12, err_msg=name)


@pytest.mark.parametrize("family", [n for n, c in _CASES.items() if c[4]])
def test_cdf_quantile_roundtrip(family: str) -> None:
    _pfunc, qfunc, params, _x, _cont = _CASES[family]
    probs = jnp.asarray(_PROBS, dtype=jnp.float64)
    values = np.asarray(qfunc(probs, **params), dtype=float)
    back = np.asarray(_pfunc(jnp.asarray(values), **params), dtype=float)
    np.testing.assert_allclose(back, np.asarray(_PROBS), atol=1e-9, err_msg=family)


def test_geometric_probabilities_sum_to_one() -> None:
    cdf = [float(pGEOMo(k, 0.3)) for k in range(0, 600)]
    pmf = np.diff(np.concatenate(([0.0], np.asarray(cdf, dtype=float))))
    assert np.all(pmf >= -1e-15)
    assert float(np.sum(pmf)) == pytest.approx(1.0, abs=1e-9)


def test_random_generators_return_requested_length() -> None:
    key = jax.random.PRNGKey(0)
    assert np.asarray(rBCCGo(7, 2.0, 0.5, 0.5, key=key)).shape == (7,)
    assert np.asarray(rBCPEo(7, 2.0, 0.5, 0.5, 2.0, key=key)).shape == (7,)
    assert np.asarray(rBCTo(7, 2.0, 0.5, 0.5, 2.0, key=key)).shape == (7,)
    assert np.asarray(rBEo(7, 0.5, 0.2, key=key)).shape == (7,)
    assert np.asarray(rGEOMo(7, 0.3, key=key)).shape == (7,)
    assert np.asarray(rPARETO2o(7, 1.0, 0.5, key=key)).shape == (7,)
    assert np.asarray(rSHASHo(7, 0.0, 1.0, 1.0, 2.0, key=key)).shape == (7,)


def test_random_generators_are_reproducible_with_a_key() -> None:
    key = jax.random.PRNGKey(42)
    first = np.asarray(rBCTo(5, 2.0, 0.5, 0.5, 2.0, key=key), dtype=float)
    second = np.asarray(rBCTo(5, 2.0, 0.5, 0.5, 2.0, key=key), dtype=float)
    np.testing.assert_allclose(first, second)


def test_random_generators_validate_parameters() -> None:
    with pytest.raises(ValueError, match="n must be a positive integer"):
        rBCCGo(0, 2.0, 0.5, 0.5)
    with pytest.raises(ValueError, match="mu must be positive"):
        rBCPEo(5, -1.0, 0.5, 0.5, 2.0)
    with pytest.raises(ValueError, match="tau must be positive"):
        rBCTo(5, 2.0, 0.5, 0.5, -2.0)


def test_geometric_draws_are_integers() -> None:
    draws = np.asarray(rGEOMo(50, 0.3, key=jax.random.PRNGKey(1)))
    assert draws.dtype.kind in {"i", "u"}
    assert np.all(draws >= 0)


@pytest.mark.slow
def test_pqr_matches_live_r() -> None:
    if shutil.which("Rscript") is None:
        pytest.skip("Rscript not available")
    payload = {
        name: {"x": x, "params": params, "probs": _PROBS}
        for name, (_p, _q, params, x, _cont) in _CASES.items()
    }
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
        )
    if proc.returncode != 0:
        pytest.skip(f"R failed: {proc.stderr[-300:]}")
    reference = json.loads(proc.stdout.strip().splitlines()[-1])
    for name, (pfunc, qfunc, params, x, _cont) in _CASES.items():
        xs = jnp.asarray(x, dtype=jnp.float64)
        probs = jnp.asarray(_PROBS, dtype=jnp.float64)
        np.testing.assert_allclose(
            np.asarray(pfunc(xs, **params), dtype=float),
            np.asarray(reference[name]["p"], dtype=float),
            rtol=1e-11,
            atol=1e-12,
            err_msg=f"{name} cdf",
        )
        np.testing.assert_allclose(
            np.asarray(qfunc(probs, **params), dtype=float),
            np.asarray(reference[name]["q"], dtype=float),
            rtol=1e-10,
            atol=1e-11,
            err_msg=f"{name} quantile",
        )
        np.testing.assert_allclose(
            np.asarray(pfunc(xs, **params, lower_tail=False), dtype=float),
            np.asarray(reference[name]["lower"], dtype=float),
            rtol=1e-11,
            atol=1e-12,
            err_msg=f"{name} upper tail",
        )
        if name == "PARETO2o":
            # R's pPARETO2o ignores log.p because of the cdf < - log(cdf)
            # typo; our value is the log of the CDF R returns.
            np.testing.assert_allclose(
                np.asarray(pfunc(xs, **params, log_p=True), dtype=float),
                np.log(np.asarray(reference[name]["p"], dtype=float)),
                rtol=1e-11,
                atol=1e-12,
                err_msg="PARETO2o log p (intent, R typo documented)",
            )
            continue
        np.testing.assert_allclose(
            np.asarray(pfunc(xs, **params, log_p=True), dtype=float),
            np.asarray(reference[name]["logp"], dtype=float),
            rtol=1e-11,
            atol=1e-12,
            err_msg=f"{name} log p",
        )
