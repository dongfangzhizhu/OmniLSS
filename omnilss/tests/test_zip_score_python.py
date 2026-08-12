"""ZIP score/Hessian functions against R reference values (Phase 4.4).

Converts the previous print-only script into real assertions.  Expected
values are the R gamlss.dist ZIP score/Hessian results recorded historically.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import jax.numpy as jnp

from omnilss.distributions import ZIP

MU = 2.0
SIGMA = 0.3

# (y, dldm, dlds, d2ldm2, d2lds2) — R reference values
R_REFERENCE = [
    (0, -0.2400, 2.1905, -0.0576, -4.7983),
    (1, -0.5000, -1.4286, -0.2500, -2.0408),
    (2, 0.0000, -1.4286, -0.0000, -2.0408),
    (3, 0.5000, -1.4286, -0.2500, -2.0408),
]


@pytest.mark.parametrize(
    "y, exp_dldm, exp_dlds, exp_d2ldm2, exp_d2lds2", R_REFERENCE
)
def test_zip_score_hessian_matches_r(
    y, exp_dldm, exp_dlds, exp_d2ldm2, exp_d2lds2
) -> None:
    family = ZIP()
    ya, ma, sa = jnp.array([y]), jnp.array([MU]), jnp.array([SIGMA])

    dldm = float(family.score_functions["mu"](ya, ma, sa)[0])
    dlds = float(family.score_functions["sigma"](ya, ma, sa)[0])
    d2ldm2 = float(family.hessian_functions["mu"](ya, ma, sa)[0])
    d2lds2 = float(family.hessian_functions["sigma"](ya, ma, sa)[0])

    np.testing.assert_allclose(dldm, exp_dldm, atol=1e-4, err_msg="dldm")
    np.testing.assert_allclose(dlds, exp_dlds, atol=1e-4, err_msg="dlds")
    np.testing.assert_allclose(d2ldm2, exp_d2ldm2, atol=1e-4, err_msg="d2ldm2")
    np.testing.assert_allclose(d2lds2, exp_d2lds2, atol=1e-4, err_msg="d2lds2")
