"""R gamlss.dist meanBEZI / meanBEOI.

The R bodies are one-liners over the fitted mu and nu.  R's own function needs a
gamlss object, so the live-R check evaluates the same expression on the same
mu/nu vectors and compares with the Python result.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from omnilss.distribution_registry import resolve  # noqa: E402
from omnilss.family_means import meanBEOI, meanBEZI  # noqa: E402

_MU = [0.2, 0.5, 0.8]
_NU = [0.1, 0.0, 0.4]
_SIGMA = [0.2, 0.2, 0.2]


class _Model:
    def __init__(self, name: str, mu=_MU, nu=_NU) -> None:
        self.family = resolve(name)
        self.fitted_values = {
            "mu": jnp.asarray(mu, dtype=jnp.float64),
            "sigma": jnp.asarray(_SIGMA, dtype=jnp.float64),
            "nu": jnp.asarray(nu, dtype=jnp.float64),
        }


def test_mean_bezi_is_one_minus_nu_times_mu() -> None:
    model = _Model("BEZI")
    expected = [(1.0 - n) * m for m, n in zip(_MU, _NU)]
    np.testing.assert_allclose(np.asarray(meanBEZI(model), dtype=float), expected, rtol=1e-12)


def test_mean_beoi_is_nu_plus_one_minus_nu_times_mu() -> None:
    model = _Model("BEOI")
    expected = [n + (1.0 - n) * m for m, n in zip(_MU, _NU)]
    np.testing.assert_allclose(np.asarray(meanBEOI(model), dtype=float), expected, rtol=1e-12)


def test_family_guards_match_r_messages() -> None:
    with pytest.raises(ValueError, match="do not have a BEZI distribution"):
        meanBEZI(_Model("BEOI"))
    with pytest.raises(ValueError, match="do not have a BEOI distribution"):
        meanBEOI(_Model("BEZI"))


def test_missing_fitted_parameter_is_reported() -> None:
    class Broken:
        family = "BEZI"
        fitted_values = {"mu": jnp.asarray([0.5])}

    with pytest.raises(ValueError, match="does not have fitted values"):
        meanBEZI(Broken())


def test_accepts_a_string_family_name() -> None:
    class Strings:
        family = "BEZI"
        fitted_values = {
            "mu": jnp.asarray(_MU),
            "nu": jnp.asarray(_NU),
        }

    np.testing.assert_allclose(
        np.asarray(meanBEZI(Strings()), dtype=float),
        np.asarray(_Model("BEZI").fitted_values["mu"]) * (1.0 - np.asarray(_NU)),
        rtol=1e-12,
    )


@pytest.mark.slow
def test_formula_matches_live_r() -> None:
    if shutil.which("Rscript") is None:
        pytest.skip("Rscript not available")
    script = f"""mu <- c({", ".join(map(str, _MU))})
nu <- c({", ".join(map(str, _NU))})
cat(jsonlite::toJSON(list(bezi = as.numeric((1 - nu) * mu),
                          beoi = as.numeric(nu + (1 - nu) * mu)),
                     auto_unbox = FALSE, digits = 17))"""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "means.R"
        path.write_text(script, encoding="utf-8")
        proc = subprocess.run(
            ["Rscript", str(path)], capture_output=True, text=True, timeout=300
        )
    if proc.returncode != 0:
        pytest.skip(f"R failed: {proc.stderr[-300:]}")
    reference = json.loads(proc.stdout.strip().splitlines()[-1])
    np.testing.assert_allclose(
        np.asarray(meanBEZI(_Model("BEZI")), dtype=float),
        np.asarray(reference["bezi"], dtype=float),
        rtol=1e-14,
    )
    np.testing.assert_allclose(
        np.asarray(meanBEOI(_Model("BEOI")), dtype=float),
        np.asarray(reference["beoi"], dtype=float),
        rtol=1e-14,
    )
