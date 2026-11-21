# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for gamlss.dist gen.Family skeleton compatibility constructors (D2.280/B2a).

Covers all 17 skeleton constructors exported from
``omnilss.distributions_gen_skeleton_compat`` that correspond to R's
``gamlss.dist`` skeleton templates, and verifies they remain classified as
``COMPAT_ALIAS`` in the fail-closed capability registry.
"""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from omnilss.distributions_gen_skeleton_compat import (
    GenSkeletonCompatFamily,
    binom_1_31,
    binom_2_33,
    binom_3_33,
    contR01_2_13,
    contR01_4_33,
    contR_2_12,
    contR_3_11,
    contR_4_13,
    contRplus_1_1,
    contRplus_3_13,
    contRplus_4_33,
    count_1_22,
    count_2_32,
    count_2_32R,
    count_2_33,
    count_3_32,
    count_3_33,
)
from omnilss.family_capabilities import (
    ImplementationStatus,
    family_capability_names,
    get_family_capability,
)

SKELETON_CASES = [
    (binom_1_31, "BINOM_1_31", ("mu",), np.array([0.0, 1.0])),
    (binom_2_33, "BINOM_2_33", ("mu",), np.array([0.0, 1.0])),
    (binom_3_33, "BINOM_3_33", ("mu",), np.array([0.0, 1.0])),
    (contR01_2_13, "CONTR01_2_13", ("mu", "sigma"), np.array([0.2, 0.5, 0.8])),
    (contR01_4_33, "CONTR01_4_33", ("mu", "sigma"), np.array([0.2, 0.5, 0.8])),
    (contR_2_12, "CONTR_2_12", ("mu", "sigma"), np.array([-1.0, 0.0, 1.0])),
    (contR_3_11, "CONTR_3_11", ("mu", "sigma"), np.array([-1.0, 0.0, 1.0])),
    (contR_4_13, "CONTR_4_13", ("mu", "sigma"), np.array([-1.0, 0.0, 1.0])),
    (contRplus_1_1, "CONTRPLUS_1_1", ("mu", "sigma"), np.array([0.5, 1.5, 3.0])),
    (contRplus_3_13, "CONTRPLUS_3_13", ("mu", "sigma"), np.array([0.5, 1.5, 3.0])),
    (contRplus_4_33, "CONTRPLUS_4_33", ("mu", "sigma"), np.array([0.5, 1.5, 3.0])),
    (count_1_22, "COUNT_1_22", ("mu",), np.array([0.0, 1.0, 3.0])),
    (count_2_32, "COUNT_2_32", ("mu",), np.array([0.0, 1.0, 3.0])),
    (count_2_32R, "COUNT_2_32R", ("mu",), np.array([0.0, 1.0, 3.0])),
    (count_2_33, "COUNT_2_33", ("mu",), np.array([0.0, 1.0, 3.0])),
    (count_3_32, "COUNT_3_32", ("mu",), np.array([0.0, 1.0, 3.0])),
    (count_3_33, "COUNT_3_33", ("mu",), np.array([0.0, 1.0, 3.0])),
]

_PARAM_DEFAULTS = {
    "mu": 0.5,
    "sigma": 0.4,
}


@pytest.mark.parametrize("factory,expected_name,expected_params,y_vals", SKELETON_CASES)
def test_gen_skeleton_compat_constructor(factory, expected_name, expected_params, y_vals):
    fam = factory()
    assert isinstance(fam, GenSkeletonCompatFamily)
    assert fam.name == expected_name
    assert tuple(fam.parameters) == expected_params
    assert set(fam.links.keys()) == set(expected_params)
    assert set(fam.link_functions.keys()) == set(expected_params)
    assert set(fam.link_inverses.keys()) == set(expected_params)

    kwargs = {"y": jnp.asarray(y_vals, dtype=jnp.float64)}
    for par in fam.parameters:
        kwargs[par] = jnp.full(y_vals.shape, _PARAM_DEFAULTS[par], dtype=jnp.float64)
    dev = np.asarray(fam.g_dev_inc(**kwargs), dtype=np.float64)
    assert dev.shape == y_vals.shape
    assert np.all(np.isfinite(dev))

    if expected_name in set(family_capability_names()):
        cap = get_family_capability(expected_name)
        assert cap.implementation_status is ImplementationStatus.COMPAT_ALIAS
        assert not cap.is_production_safe
