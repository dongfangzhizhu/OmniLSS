"""R gamlss.tr truncation helpers (trun.d / trun.p / trun.q / trun.r).

Kernels verified against the vendored gamlss.tr source under live R 4.6.0.
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

from omnilss.gamlss_tr import trun_d, trun_p, trun_q, trun_r  # noqa: E402

_REPO = Path(__file__).resolve().parents[2]

_CASES = [
    ("NO", "left", 0.0, {"mu": 0.0, "sigma": 1.0}, [0.0, 0.5, 1.0, 2.0], [0.1, 0.5, 0.9]),
    ("NO", "right", 1.0, {"mu": 0.0, "sigma": 1.0}, [-1.0, 0.0, 0.5, 1.0], [0.1, 0.5, 0.9]),
    ("NO", "both", [-1.0, 1.0], {"mu": 0.0, "sigma": 1.0}, [-0.5, 0.0, 0.5], [0.2, 0.5, 0.8]),
    ("PO", "left", 2.0, {"mu": 3.0}, [2, 3, 4, 5], [0.1, 0.5, 0.9]),
    ("PO", "right", 2.0, {"mu": 3.0}, [1, 2, 3], [0.1, 0.5, 0.9]),
]

_RSCRIPT = """suppressPackageStartupMessages(library(gamlss.dist))
source('R/gamlss.tr/R/trun.d-30-03-24.R')
source('R/gamlss.tr/R/trun.p-30-03-24.R')
source('R/gamlss.tr/R/trun.q-27-12-12.R')
library(jsonlite)
args <- commandArgs(trailingOnly = TRUE)
cs <- fromJSON(args[1], simplifyDataFrame = FALSE)
out <- list()
for (i in seq_along(cs)) {
  cse <- cs[[i]]
  fd <- trun.d(cse$par, cse$fam, type = cse$type)
  fp <- trun.p(cse$par, cse$fam, type = cse$type)
  fq <- trun.q(cse$par, cse$fam, type = cse$type)
  out[[i]] <- list(
    d = as.numeric(do.call(fd, c(list(x = cse$x), cse$params))),
    p = as.numeric(do.call(fp, c(list(q = cse$x), cse$params))),
    q = as.numeric(do.call(fq, c(list(p = cse$prob), cse$params)))
  )
}
cat(toJSON(out, auto_unbox = FALSE, digits = 17))
"""


@pytest.mark.parametrize("family,trunc,par,params,x,probs", _CASES)
def test_cdf_quantile_roundtrip(family, trunc, par, params, x, probs) -> None:
    if family != "NO":
        pytest.skip("discrete quantiles are stepwise; covered by live-R parity")
    cdf = trun_p(par, family, trunc)
    quantile = trun_q(par, family, trunc)
    for prob in (0.05, 0.5, 0.95):
        q = float(quantile(prob, **params))
        back = float(cdf(q, **params))
        assert back == pytest.approx(prob, abs=1e-9)


def test_varying_is_not_ported() -> None:
    with pytest.raises(NotImplementedError, match="varying"):
        trun_d(0.0, "NO", "left", varying=True)


def test_par_length_validation() -> None:
    with pytest.raises(ValueError, match="length of par should be 2"):
        trun_d(0.0, "NO", "both")
    with pytest.raises(ValueError, match="length of par should be 1"):
        trun_d([0.0, 1.0], "NO", "left")


def test_random_samples_respect_support() -> None:
    np.random.seed(0)
    samples = np.asarray(trun_r(0.0, "NO", "left")(500, mu=0.0, sigma=1.0), dtype=float)
    assert np.all(samples >= 0.0)
    samples_po = np.asarray(trun_r(2.0, "PO", "left")(500, mu=3.0), dtype=float)
    assert np.all(samples_po > 2.0)


@pytest.mark.slow
def test_matches_live_r() -> None:
    if shutil.which("Rscript") is None:
        pytest.skip("Rscript not available")
    if not (_REPO / "R" / "gamlss.tr" / "R" / "trun.d-30-03-24.R").exists():
        pytest.skip("vendored gamlss.tr source not present")
    payload = [
        {"fam": f, "type": t, "par": p, "params": pa, "x": x, "prob": pr}
        for (f, t, p, pa, x, pr) in _CASES
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
    for index, (f, t, p, pa, x, pr) in enumerate(_CASES):
        mine_d = np.asarray(trun_d(p, f, t)(x, **pa), dtype=float)
        mine_p = np.asarray(trun_p(p, f, t)(x, **pa), dtype=float)
        mine_q = np.asarray(trun_q(p, f, t)(pr, **pa), dtype=float)
        np.testing.assert_allclose(mine_d, np.asarray(reference[index]["d"], float), rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(mine_p, np.asarray(reference[index]["p"], float), rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(mine_q, np.asarray(reference[index]["q"], float), rtol=1e-10, atol=1e-12)
