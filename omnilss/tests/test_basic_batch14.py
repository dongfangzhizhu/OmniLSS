"""Basic tests for batch 14 distributions."""

import jax.numpy as jnp
import pytest


def test_batch14_family_instantiates() -> None:
    """Batch-14 core families should resolve and expose parameters."""
    from omnilss.distributions import resolve_family

    for family_name in ("SHASHo", "SHASHo2", "JSUo", "ST5", "BCPEo", "BCTo"):
        fam = resolve_family(family_name)
        assert fam is not None
        assert fam.name == family_name
        assert len(fam.parameters) >= 1


@pytest.mark.parametrize("family_name", ["SHASHo", "SHASHo2"])
def test_batch14_family_loglikelihood_finite(family_name: str) -> None:
    """Log-likelihood increment call should return finite values."""
    from omnilss.distributions import resolve_family

    fam = resolve_family(family_name)
    y = jnp.array([0.5, 1.0, 1.5, 2.0])
    params = {p: jnp.ones(4) for p in fam.parameters}
    ll = fam.g_dev_inc(y=y, **params)
    assert jnp.all(jnp.isfinite(ll))
