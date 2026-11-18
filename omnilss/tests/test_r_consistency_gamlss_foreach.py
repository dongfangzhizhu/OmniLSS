"""Live-R consistency evidence for gamlss.foreach (D2.265).

fitQR is deterministic: R fits SEP3 with sigma = tau = 1 and nu = sqrt((1-q)/q),
so the reported deviance is comparable. The bootstraps are random procedures and
are covered by test_gamlss_foreach.py instead.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest

from omnilss import fitQR

_REPO = Path(__file__).resolve().parents[2]
_R_SCRIPT = """
suppressPackageStartupMessages(library(gamlss))
suppressPackageStartupMessages(library(gamlss.dist))
suppressPackageStartupMessages(library(jsonlite))
args <- commandArgs(trailingOnly = TRUE)
d <- fromJSON(args[1])
df <- data.frame(y = d$y, x = d$x)
out <- list()
for (q in d$quantiles) {
  nu <- sqrt((1 - q) / q)
  fit <- gamlss(y ~ x, family = SEP3, data = df, sigma.fix = TRUE, sigma.start = 1,
                nu.fix = TRUE, nu.start = nu, tau.fix = TRUE, tau.start = 1,
                control = gamlss.control(trace = FALSE, n.cyc = 100))
  out[[length(out) + 1]] <- list(q = q, dev = deviance(fit),
                                 mu = as.numeric(coef(fit, "mu")))
}
cat(toJSON(out, digits = 17, auto_unbox = TRUE))
"""


def _payload() -> dict:
    rng = np.random.default_rng(3)
    n = 120
    x = rng.normal(size=n)
    y = 1.0 + 0.5 * x + (0.4 + 0.2 * np.abs(x)) * rng.normal(size=n)
    return {"y": y.tolist(), "x": x.tolist(), "quantiles": [0.25, 0.5, 0.75]}


def _live_r(payload: dict) -> list[dict]:
    if shutil.which("Rscript") is None:
        pytest.skip("Rscript not available")
    with tempfile.TemporaryDirectory() as tmp:
        data_file = Path(tmp) / "qr.json"
        data_file.write_text(json.dumps(payload), encoding="utf-8")
        script = Path(tmp) / "ref.R"
        script.write_text(_R_SCRIPT, encoding="utf-8")
        proc = subprocess.run(
            ["Rscript", str(script), str(data_file)],
            capture_output=True,
            text=True,
            timeout=900,
            cwd=str(_REPO),
        )
    if proc.returncode != 0:
        pytest.skip(f"R failed: {proc.stderr[-300:]}")
    return json.loads(proc.stdout.strip().splitlines()[-1])


@pytest.mark.slow
def test_fitqr_deviance_matches_live_r():
    payload = _payload()
    reference = _live_r(payload)
    data = {"y": np.asarray(payload["y"], dtype=float), "x": np.asarray(payload["x"], dtype=float)}
    models = fitQR(
        "y ~ x", data, quantile=[0.25, 0.5, 0.75], max_iter=100
    )
    assert len(models) == len(reference)
    for model, expected in zip(models, reference):
        assert float(model.g_dev) == pytest.approx(float(expected["dev"]), rel=1e-5)
        # R reports the deviance of its last inner fit; the outer coefficients may
        # sit at a neighbouring kink, so coefficients are only required to be close.
        np.testing.assert_allclose(
            np.ravel(np.asarray(model.coefficients["mu"], dtype=float)),
            np.asarray(expected["mu"], dtype=float), atol=0.08,
        )
