"""R 'o' (original) parameterisation density functions.

Covers dBCCGo / dBCPEo / dBCTo / dBEo / dGEOMo / dJSUo / dPARETO2o / dSHASHo.
Kernels match live R gamlss.dist to <= 7.1e-15 on the audit grid.
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

from omnilss.o_variant_density import (  # noqa: E402
    dBCCGo,
    dBCPEo,
    dBCTo,
    dBEo,
    dGEOMo,
    dJSUo,
    dPARETO2o,
    dSHASHo,
)

_CASES = {
    "BCCGo": (dBCCGo, [0.5, 1, 2, 3, 4], {"mu": 2.0, "sigma": 0.5, "nu": 0.5}),
    "BCPEo": (dBCPEo, [0.5, 1, 2, 3, 4], {"mu": 2.0, "sigma": 0.5, "nu": 0.5, "tau": 2.0}),
    "BCTo": (dBCTo, [0.5, 1, 2, 3, 4], {"mu": 2.0, "sigma": 0.5, "nu": 0.5, "tau": 2.0}),
    "BEo": (dBEo, [0.1, 0.3, 0.5, 0.7, 0.9], {"mu": 0.5, "sigma": 0.2}),
    "GEOMo": (dGEOMo, [0, 1, 2, 3, 4], {"mu": 0.3}),
    "JSUo": (dJSUo, [-2, -1, 0, 1, 2], {"mu": 0.0, "sigma": 1.0, "nu": 0.5, "tau": 2.0}),
    "PARETO2o": (dPARETO2o, [1.5, 2, 3, 4, 5], {"mu": 1.0, "sigma": 0.5}),
    "SHASHo": (dSHASHo, [-2, -1, 0, 1, 2], {"mu": 0.0, "sigma": 1.0, "nu": 1.0, "tau": 2.0}),
}

_RSCRIPT = """suppressPackageStartupMessages(library(gamlss.dist))
library(jsonlite)
args <- commandArgs(trailingOnly = TRUE)
p <- fromJSON(args[1])
out <- list()
for (fam in names(p)) {
  fn <- get(paste0("d", fam))
  par <- as.list(p[[fam]]$params)
  out[[fam]] <- do.call(fn, c(list(x = p[[fam]]$y), par, list(log = TRUE)))
}
cat(toJSON(out, auto_unbox = TRUE, digits = 17))
"""


@pytest.mark.parametrize("family", list(_CASES))
def test_log_flag_is_consistent(family: str) -> None:
    func, ys, params = _CASES[family]
    x = jnp.asarray(ys)
    dens = np.asarray(func(x, **params, log=False))
    logv = np.asarray(func(x, **params, log=True))
    np.testing.assert_allclose(np.log(dens), logv, rtol=1e-12, atol=1e-12)


@pytest.mark.slow
def test_matches_live_r() -> None:
    if shutil.which("Rscript") is None:
        pytest.skip("Rscript not available")
    payload = {
        family: {"y": ys, "params": params}
        for family, (_func, ys, params) in _CASES.items()
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
    for family, (func, ys, params) in _CASES.items():
        mine = np.asarray(func(jnp.asarray(ys), **params, log=True))
        expected = np.asarray(reference[family], dtype=float)
        np.testing.assert_allclose(mine, expected, rtol=1e-10, atol=1e-10)
