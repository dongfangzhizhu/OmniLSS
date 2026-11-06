"""R gamlss.mx dMX / pMX and the MX/NP control lists.

The mixture densities are compared with the vendored gamlss.mx sources loaded
directly (the package is not installed) for two normal components, three
components, a normal + gamma mixture and a Poisson mixture.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest

from omnilss.mixtures import dmx, mx_control, np_control, pmx

_REPO = Path(__file__).resolve().parents[2]

_RSCRIPT = """suppressPackageStartupMessages(library(gamlss))
library(jsonlite)
.gamlss.bi.list <- c("BI", "Binomial", "BB", "Beta Binomial")
source("R/gamlss.mx/R/gamlssMX-10-11-07.R")
source("R/gamlss.mx/R/gamlssNP.R")
out <- list()
out$d_nono <- as.numeric(dMX(c(0, 1, 2), mu = list(1, 5), sigma = list(1, 2), pi = list(0.2, 0.8)))
out$p_nono <- as.numeric(pMX(c(0, 1, 2), mu = list(1, 5), sigma = list(1, 2), pi = list(0.2, 0.8)))
out$d_log <- as.numeric(dMX(c(0, 1, 2), mu = list(1, 5), sigma = list(1, 2), pi = list(0.2, 0.8), log = TRUE))
out$p_log <- as.numeric(pMX(c(0, 1, 2), mu = list(1, 5), sigma = list(1, 2), pi = list(0.2, 0.8), log = TRUE))
out$d_three <- as.numeric(dMX(c(0.5, 1, 2), mu = list(1, 2, 5), sigma = list(1, 2, 1),
                            pi = list(0.3, 0.3, 0.4), family = list("NO", "NO", "NO")))
out$d_mixed <- as.numeric(dMX(c(0.5, 1, 2), mu = list(1, 2.5), sigma = list(1, 1),
                           pi = list(0.4, 0.6), family = list("NO", "GA")))
out$p_mixed <- as.numeric(pMX(c(0.5, 1, 2), mu = list(1, 2.5), sigma = list(1, 1),
                           pi = list(0.4, 0.6), family = list("NO", "GA")))
out$d_po <- as.numeric(dMX(c(0, 1, 2), mu = list(1, 4), pi = list(0.5, 0.5),
                          family = list("PO", "PO")))
out$p_po <- as.numeric(pMX(c(0, 1, 2), mu = list(1, 4), pi = list(0.5, 0.5),
                          family = list("PO", "PO")))
out$mx_default <- MX.control(seed = 1)
out$mx_guard <- suppressWarnings(MX.control(cc = 0, n.cyc = 0, trace = -1, sample = 2, seed = 1))
out$np_default <- NP.control()
out$np_guard <- suppressWarnings(NP.control(EMcc = 0, EMn.cyc = 0, trace = -1))
cat("JSONSTART"); cat(toJSON(out, auto_unbox = TRUE, digits = 17))
"""


def _norm_pdf(y: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    return np.exp(-0.5 * ((y - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))


def test_dmx_sums_the_weighted_component_densities() -> None:
    y = np.array([0.0, 1.0, 2.0])
    expected = 0.2 * _norm_pdf(y, 1.0, 1.0) + 0.8 * _norm_pdf(y, 5.0, 2.0)
    np.testing.assert_allclose(dmx(y, [1, 5], [1, 2], pi=[0.2, 0.8]), expected, rtol=1e-14)


def test_pmx_sums_the_weighted_component_cdfs() -> None:
    from scipy.stats import norm

    y = np.array([0.0, 1.0, 2.0])
    expected = 0.2 * norm.cdf(y, 1.0, 1.0) + 0.8 * norm.cdf(y, 5.0, 2.0)
    np.testing.assert_allclose(
        pmx(y, [1, 5], [1, 2], pi=[0.2, 0.8]), expected, rtol=1e-12, atol=1e-14
    )


def test_the_log_flag_takes_the_log_of_the_mixture() -> None:
    y = np.array([0.0, 1.0, 2.0])
    plain = dmx(y, [1, 5], [1, 2], pi=[0.2, 0.8])
    logged = dmx(y, [1, 5], [1, 2], pi=[0.2, 0.8], log=True)
    np.testing.assert_allclose(logged, np.log(plain), rtol=1e-14)
    np.testing.assert_allclose(
        pmx(y, [1, 5], [1, 2], pi=[0.2, 0.8], log=True),
        np.log(pmx(y, [1, 5], [1, 2], pi=[0.2, 0.8])),
        rtol=1e-14,
    )


def test_three_components_and_mixed_families() -> None:
    y = np.array([0.5, 1.0, 2.0])
    three = dmx(y, [1, 2, 5], [1, 2, 1], pi=[0.3, 0.3, 0.4], family=["NO", "NO", "NO"])
    manual = (
        0.3 * _norm_pdf(y, 1.0, 1.0)
        + 0.3 * _norm_pdf(y, 2.0, 2.0)
        + 0.4 * _norm_pdf(y, 5.0, 1.0)
    )
    np.testing.assert_allclose(three, manual, rtol=1e-14)
    mixed = dmx(y, [1, 2.5], [1, 1], pi=[0.4, 0.6], family=["NO", "GA"])
    assert np.all(mixed > 0.0)
    assert mixed.shape == (3,)


def test_poisson_mixture_matches_its_closed_form() -> None:
    from scipy.stats import poisson

    y = np.array([0.0, 1.0, 2.0])
    expected = 0.5 * poisson.pmf(y, 1.0) + 0.5 * poisson.pmf(y, 4.0)
    np.testing.assert_allclose(dmx(y, [1, 4], pi=[0.5, 0.5], family=["PO", "PO"]), expected, rtol=1e-13)


def test_component_lists_must_match_the_probabilities() -> None:
    y = np.array([0.0, 1.0])
    with pytest.raises(ValueError, match="must have 2 entries"):
        dmx(y, [1], [1], pi=[0.5, 0.5])
    with pytest.raises(ValueError, match="needs sigma"):
        dmx(y, [1, 5], pi=[0.5, 0.5])


def test_mx_control_defaults_and_guards() -> None:
    settings = mx_control(seed=1)
    assert settings == {
        "cc": 0.0001,
        "n.cyc": 200,
        "trace": False,
        "seed": 1,
        "plot": True,
        "sample": None,
    }
    with pytest.warns(UserWarning, match="value of cc supplied"):
        guarded = mx_control(cc=0.0, seed=1)
    # R assigns to c.crit, so cc itself is left alone.
    assert guarded["cc"] == 0.0
    with pytest.warns(UserWarning, match="no cycles supplied"):
        assert mx_control(n_cyc=0, seed=1)["n.cyc"] == 100
    with pytest.warns(UserWarning, match="not probability"):
        assert mx_control(sample=2.0, seed=1)["sample"] == 0.1
    with pytest.warns(UserWarning, match="value of trace supplied"):
        assert mx_control(trace=-1, seed=1)["trace"] == 1


def test_np_control_guards_are_no_ops() -> None:
    assert np_control() == {
        "EMcc": 0.001,
        "EMn.cyc": 200,
        "trace": True,
        "damp": True,
        "plot.opt": 3,
    }
    with pytest.warns(UserWarning, match="value of cc supplied"):
        guarded = np_control(EMcc=0.0, EMn_cyc=0, trace=-1)
    # Both guards write to the wrong variable in R, so the values survive.
    assert guarded["EMcc"] == 0.0
    assert guarded["EMn.cyc"] == 0
    assert guarded["trace"] is True


@pytest.mark.slow
def test_mixtures_match_live_r() -> None:
    if shutil.which("Rscript") is None:
        pytest.skip("Rscript not available")
    if not (_REPO / "R" / "gamlss.mx" / "R" / "gamlssMX-10-11-07.R").exists():
        pytest.skip("vendored gamlss.mx not present")
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "mx.R"
        path.write_text(_RSCRIPT, encoding="utf-8")
        proc = subprocess.run(
            ["Rscript", str(path)], capture_output=True, text=True, timeout=300, cwd=str(_REPO)
        )
    if "JSONSTART" not in proc.stdout:
        pytest.skip(f"R failed: {proc.stderr[-300:]}")
    ref = json.loads(proc.stdout.split("JSONSTART", 1)[1].strip())

    y = np.array([0.0, 1.0, 2.0])
    positive = np.array([0.5, 1.0, 2.0])
    cases = {
        "d_nono": dmx(y, [1, 5], [1, 2], pi=[0.2, 0.8], family=["NO", "NO"]),
        "p_nono": pmx(y, [1, 5], [1, 2], pi=[0.2, 0.8], family=["NO", "NO"]),
        "d_log": dmx(y, [1, 5], [1, 2], pi=[0.2, 0.8], family=["NO", "NO"], log=True),
        "p_log": pmx(y, [1, 5], [1, 2], pi=[0.2, 0.8], family=["NO", "NO"], log=True),
        "d_three": dmx(positive, [1, 2, 5], [1, 2, 1], pi=[0.3, 0.3, 0.4], family=["NO"] * 3),
        "d_mixed": dmx(positive, [1, 2.5], [1, 1], pi=[0.4, 0.6], family=["NO", "GA"]),
        "p_mixed": pmx(positive, [1, 2.5], [1, 1], pi=[0.4, 0.6], family=["NO", "GA"]),
        "d_po": dmx(y, [1, 4], pi=[0.5, 0.5], family=["PO", "PO"]),
        "p_po": pmx(y, [1, 4], pi=[0.5, 0.5], family=["PO", "PO"]),
    }
    for key, mine in cases.items():
        np.testing.assert_allclose(
            mine, np.asarray(ref[key], dtype=float), rtol=1e-12, atol=1e-14, err_msg=key
        )

    settings = mx_control(seed=1)
    for name in ("cc", "n.cyc", "trace", "seed", "plot"):
        assert settings[name] == ref["mx_default"][name], name
    with pytest.warns(UserWarning):
        guarded = mx_control(cc=0.0, n_cyc=0, trace=-1, sample=2.0, seed=1)
    for name, value in ref["mx_guard"].items():
        if name == "sample":
            assert guarded[name] == pytest.approx(float(value))
        else:
            assert guarded[name] == value, name

    assert np_control() == ref["np_default"]
    with pytest.warns(UserWarning):
        assert np_control(EMcc=0.0, EMn_cyc=0, trace=-1) == ref["np_guard"]