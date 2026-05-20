from __future__ import annotations

import jax.numpy as jnp

from omnilss.ad import build_ad_family
from omnilss.families import FamilyDefinition
from omnilss.runtime import DeterministicPolicy, RuntimeTolerancePolicy, SeedManager


class _TmpFamily(FamilyDefinition):
    pass


def _gaussian_log_pdf(y, mu, sigma):
    z = (y - mu) / sigma
    return -0.5 * jnp.log(2.0 * jnp.pi) - jnp.log(sigma) - 0.5 * z * z


def test_build_ad_family_does_not_monkey_patch_class_state():
    assert "d" not in _TmpFamily.__dict__
    family = build_ad_family(
        family_class=_TmpFamily,
        name="TMP",
        parameters=("mu", "sigma"),
        log_pdf_func=_gaussian_log_pdf,
    )
    assert family.d is not None
    assert "d" not in _TmpFamily.__dict__


def test_runtime_deterministic_policy_seed_manager_is_reproducible():
    policy = DeterministicPolicy(dtype="float64", tolerance=RuntimeTolerancePolicy())
    assert policy.deterministic_optimizer_order is True

    manager = SeedManager(seed=17)
    a = manager.rng().normal(size=5)
    b = manager.rng().normal(size=5)
    assert (a == b).all()
