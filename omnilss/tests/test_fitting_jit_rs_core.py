import numpy as np
import jax.numpy as jnp

from omnilss.fitting_jit import create_jit_rs_no_core


def test_jit_rs_no_core_runs_end_to_end():
    rng = np.random.default_rng(0)
    y = 1.5 + 0.7 * rng.normal(size=128)
    fn = create_jit_rs_no_core(max_iter=20, tol=1e-6)
    out = fn(jnp.asarray(y))
    assert out["mu"].shape[0] == y.shape[0]
    assert out["sigma"].shape[0] == y.shape[0]
    assert float(out["g_dev"]) > 0.0
