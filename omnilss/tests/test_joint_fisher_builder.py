from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from omnilss.derivatives import build_joint_fisher_from_blocks


def test_build_joint_fisher_from_blocks_assembles_cross_terms():
    slices = {"mu": slice(0, 2), "sigma": slice(2, 3)}
    blocks = {
        ("mu", "mu"): jnp.array([[4.0, 1.0], [1.0, 3.0]]),
        ("mu", "sigma"): jnp.array([[0.2], [0.4]]),
        ("sigma", "mu"): jnp.array([[0.2, 0.4]]),
        ("sigma", "sigma"): jnp.array([[2.0]]),
    }

    fisher = build_joint_fisher_from_blocks(blocks, slices)

    expected = np.array([[4.0, 1.0, 0.2], [1.0, 3.0, 0.4], [0.2, 0.4, 2.0]])
    np.testing.assert_allclose(np.asarray(fisher), expected, atol=1e-12)


def test_build_joint_fisher_from_blocks_rejects_bad_shape():
    slices = {"mu": slice(0, 2), "sigma": slice(2, 3)}
    bad = {
        ("mu", "mu"): jnp.array([[1.0]]),
    }
    with pytest.raises(ValueError, match="expected"):
        build_joint_fisher_from_blocks(bad, slices)

