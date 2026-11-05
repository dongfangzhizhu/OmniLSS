"""R gamlss.cens helpers (cens.d / cens.p / cens.q / cens / gen.cens).

cens.d verified against the vendored gamlss.cens source under live R 4.6.0.
cens.p event branch is deterministic and matches the base CDF; the randomised
branch is checked by range only (R uses runif, so stream parity is impossible).
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

from omnilss.distribution_registry import resolve  # noqa: E402
from omnilss.gamlss_cens import (  # noqa: E402
    CensoredFamily,
    cens,
    cens_d,
    cens_p,
    cens_q,
    gen_cens,
)

_REPO = Path(__file__).resolve().parents[2]

_CASES = [
    ("right", {"time": [1, 2, 3, 4], "status": [1, 0, 1, 0]}, 0.0, 1.0),
    ("left", {"time": [1, 2, 3, 4], "status": [1, 0, 1, 0]}, 0.0, 1.0),
    ("interval", {"time": [1, 2, 3, 4], "time2": [2, 3, 4, 5], "status": [0, 1, 2, 3]}, 0.0, 1.0),
]

_RSCRIPT = """suppressPackageStartupMessages(library(gamlss.dist))
suppressPackageStartupMessages(library(survival))
source("R/gamlss.cens/R/cens.d.R")
library(jsonlite)
args <- commandArgs(trailingOnly = TRUE)
cs <- fromJSON(args[1], simplifyDataFrame = FALSE)
out <- list()
for (i in seq_along(cs)) {
  cse <- cs[[i]]
  if (cse$type == "interval") {
    s <- cbind(cse$time, cse$time2, cse$status); colnames(s) <- c("t1", "t2", "status")
  } else {
    s <- cbind(cse$time, cse$status); colnames(s) <- c("time", "status")
  }
  class(s) <- "Surv"
  f <- cens.d("NO", type = cse$type)
  out[[i]] <- as.numeric(f(s, mu = cse$mu, sigma = cse$sigma, log = TRUE))
}
cat(toJSON(out, auto_unbox = FALSE, digits = 17))
"""


def test_invalid_type_raises() -> None:
    with pytest.raises(ValueError, match="type must be one of"):
        cens_d("NO", "counting")


def test_surv_accepts_matrix() -> None:
    y_dict = {"time": [1.0, 2.0], "status": [1, 0]}
    y_mat = np.array([[1.0, 1.0], [2.0, 0.0]])
    a = cens_d("NO", "right")(y_dict, mu=0.0, sigma=1.0, log=True)
    b = cens_d("NO", "right")(y_mat, mu=0.0, sigma=1.0, log=True)
    np.testing.assert_allclose(np.asarray(a), np.asarray(b))


def test_cens_p_event_branch_is_base_cdf() -> None:
    fam = resolve("NO")
    y = {"time": [1.0, 2.0], "status": [1, 1]}
    got = np.asarray(cens_p("NO", "right")(y, mu=0.0, sigma=1.0), dtype=float)
    expected = np.asarray(fam.p(jnp.asarray([1.0, 2.0]), 0.0, 1.0), dtype=float)
    np.testing.assert_allclose(got, expected, rtol=1e-12)


def test_cens_p_censored_branches_are_in_range() -> None:
    fam = resolve("NO")
    y = np.array([[1.0, 0.0], [2.0, 0.0]])
    cdf = np.asarray(fam.p(jnp.asarray([1.0, 2.0]), 0.0, 1.0), dtype=float)
    right = np.asarray(cens_p("NO", "right")(y, mu=0.0, sigma=1.0), dtype=float)
    left = np.asarray(cens_p("NO", "left")(y, mu=0.0, sigma=1.0), dtype=float)
    assert np.all(right >= cdf - 1e-12) and np.all(right <= 1.0)
    assert np.all(left <= cdf + 1e-12) and np.all(left >= 0.0)


def test_cens_q_is_base_quantile() -> None:
    fam = resolve("NO")
    q = cens_q("NO")
    a = np.asarray(q(jnp.asarray([0.1, 0.5, 0.9]), 0.0, 1.0), dtype=float)
    b = np.asarray(fam.q(jnp.asarray([0.1, 0.5, 0.9]), 0.0, 1.0), dtype=float)
    np.testing.assert_allclose(a, b)


def test_cens_builds_family_object() -> None:
    cf = cens("NO", "right")
    assert isinstance(cf, CensoredFamily)
    assert cf.name == "NOrc"
    assert cf.parameters == ("mu", "sigma")
    y = {"time": [1.0, 2.0, 3.0], "status": [1, 0, 1]}
    loglik = np.asarray(cf.loglik(y, mu=0.0, sigma=1.0), dtype=float)
    # g_dev_inc is elementwise (the fitting engine sums it with weights).
    assert float(np.sum(cf.g_dev_inc(y=y, mu=0.0, sigma=1.0))) == pytest.approx(
        float(-2.0 * np.sum(loglik))
    )


def test_gen_cens_returns_generated_names() -> None:
    generated = gen_cens("NO", "right")
    for key in ("d", "p", "q", "family", "dNOrc", "pNOrc", "qNOrc", "NOrc"):
        assert key in generated
    assert generated["family"] is generated["NOrc"]


@pytest.mark.slow
def test_cens_d_matches_live_r() -> None:
    if shutil.which("Rscript") is None:
        pytest.skip("Rscript not available")
    if not (_REPO / "R" / "gamlss.cens" / "R" / "cens.d.R").exists():
        pytest.skip("vendored gamlss.cens source not present")
    payload = [
        {"type": t, "time": c["time"], "time2": c.get("time2", c["time"]), "status": c["status"], "mu": mu, "sigma": sg}
        for (t, c, mu, sg) in _CASES
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
    for index, (t, c, mu, sg) in enumerate(_CASES):
        y = {"time": c["time"], "status": c["status"]}
        if "time2" in c:
            y["time2"] = c["time2"]
        mine = np.asarray(cens_d("NO", t)(y, mu=mu, sigma=sg, log=True), dtype=float)
        expected = np.asarray(reference[index], dtype=float)
        np.testing.assert_allclose(mine, expected, rtol=1e-10, atol=1e-12)
