"""R/Python Consistency Tests for GAMLSS Families."""

import unittest
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from omnilss.distributions import resolve_family
from tests.rbus.core import RTestBus


class TestFamilyConsistency(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.bus = RTestBus()
        except RuntimeError as e:
            raise unittest.SkipTest(f"RTestBus not available: {e}")

    def test_NO_family_consistency(self):
        """Test Normal distribution (NO) density against R."""
        y = np.linspace(-3, 3, 100)
        mu = np.linspace(-1, 1, 100)
        sigma = np.linspace(0.5, 2.0, 100)

        # R evaluation
        r_d = self.bus.eval_family(
            family="NO",
            func_type="d",
            args={"x": y.tolist(), "mu": mu.tolist(), "sigma": sigma.tolist()}
        )
        
        # JAX evaluation
        family = resolve_family("NO")
        jax_d = np.asarray(jnp.exp(-0.5 * family.g_dev_inc(jnp.array(y), jnp.array(mu), jnp.array(sigma))))

        np.testing.assert_allclose(jax_d, r_d, rtol=1e-5, atol=1e-8)

    def test_PO_family_consistency(self):
        """Test Poisson distribution (PO) density against R."""
        y = np.arange(0, 100)
        mu = np.linspace(0.1, 10, 100)

        r_d = self.bus.eval_family(
            family="PO",
            func_type="d",
            args={"x": y.tolist(), "mu": mu.tolist()}
        )
        
        family = resolve_family("PO")
        jax_d = np.asarray(jnp.exp(-0.5 * family.g_dev_inc(jnp.array(y), jnp.array(mu))))

        np.testing.assert_allclose(jax_d, r_d, rtol=1e-5, atol=1e-8)

    def test_GA_family_consistency(self):
        """Test Gamma distribution (GA) density against R."""
        y = np.linspace(0.1, 5, 100)
        mu = np.linspace(0.5, 2.0, 100)
        sigma = np.linspace(0.1, 1.0, 100)

        r_d = self.bus.eval_family(
            family="GA",
            func_type="d",
            args={"x": y.tolist(), "mu": mu.tolist(), "sigma": sigma.tolist()}
        )
        
        family = resolve_family("GA")
        jax_d = np.asarray(jnp.exp(-0.5 * family.g_dev_inc(jnp.array(y), jnp.array(mu), jnp.array(sigma))))

        np.testing.assert_allclose(jax_d, r_d, rtol=1e-5, atol=1e-8)


    def test_GU_family_consistency(self):
        y = np.linspace(-5, 10, 100)
        mu = np.linspace(-2, 2, 100)
        sigma = np.linspace(0.5, 3.0, 100)
        r_d = self.bus.eval_family("GU", "d", {"x": y.tolist(), "mu": mu.tolist(), "sigma": sigma.tolist()})
        family = resolve_family("GU")
        jax_d = np.asarray(jnp.exp(-0.5 * family.g_dev_inc(jnp.array(y), jnp.array(mu), jnp.array(sigma))))
        np.testing.assert_allclose(jax_d, r_d, rtol=1e-5, atol=1e-8)

    def test_RG_family_consistency(self):
        y = np.linspace(-5, 10, 100)
        mu = np.linspace(-2, 2, 100)
        sigma = np.linspace(0.5, 3.0, 100)
        r_d = self.bus.eval_family("RG", "d", {"x": y.tolist(), "mu": mu.tolist(), "sigma": sigma.tolist()})
        family = resolve_family("RG")
        jax_d = np.asarray(jnp.exp(-0.5 * family.g_dev_inc(jnp.array(y), jnp.array(mu), jnp.array(sigma))))
        np.testing.assert_allclose(jax_d, r_d, rtol=1e-5, atol=1e-8)

    def test_IGAMMA_family_consistency(self):
        y = np.linspace(0.1, 5, 100)
        mu = np.linspace(0.5, 2.0, 100)
        sigma = np.linspace(0.1, 1.0, 100)
        r_d = self.bus.eval_family("IGAMMA", "d", {"x": y.tolist(), "mu": mu.tolist(), "sigma": sigma.tolist()})
        family = resolve_family("IGAMMA")
        jax_d = np.asarray(jnp.exp(-0.5 * family.g_dev_inc(jnp.array(y), jnp.array(mu), jnp.array(sigma))))
        np.testing.assert_allclose(jax_d, r_d, rtol=1e-5, atol=1e-8)

    def test_PARETO2_family_consistency(self):
        y = np.linspace(0.1, 10, 100)
        mu = np.linspace(1.0, 5.0, 100)
        sigma = np.linspace(0.5, 2.0, 100)
        r_d = self.bus.eval_family("PARETO2", "d", {"x": y.tolist(), "mu": mu.tolist(), "sigma": sigma.tolist()})
        family = resolve_family("PARETO2")
        jax_d = np.asarray(jnp.exp(-0.5 * family.g_dev_inc(jnp.array(y), jnp.array(mu), jnp.array(sigma))))
        np.testing.assert_allclose(jax_d, r_d, rtol=1e-5, atol=1e-8)

    def test_NBII_family_consistency(self):
        y = np.arange(0, 100)
        mu = np.linspace(0.1, 10, 100)
        sigma = np.linspace(0.1, 2.0, 100)
        r_d = self.bus.eval_family("NBII", "d", {"x": y.tolist(), "mu": mu.tolist(), "sigma": sigma.tolist()})
        family = resolve_family("NBII")
        jax_d = np.asarray(jnp.exp(-0.5 * family.g_dev_inc(jnp.array(y), jnp.array(mu), jnp.array(sigma))))
        np.testing.assert_allclose(jax_d, r_d, rtol=1e-5, atol=1e-8)

    def test_NO2_family_consistency(self):
        y = np.linspace(-3, 3, 100)
        mu = np.linspace(-1, 1, 100)
        sigma = np.linspace(0.25, 4.0, 100)  # sigma here is variance
        r_d = self.bus.eval_family("NO2", "d", {"x": y.tolist(), "mu": mu.tolist(), "sigma": sigma.tolist()})
        family = resolve_family("NO2")
        jax_d = np.asarray(jnp.exp(-0.5 * family.g_dev_inc(jnp.array(y), jnp.array(mu), jnp.array(sigma))))
        np.testing.assert_allclose(jax_d, r_d, rtol=1e-5, atol=1e-8)

    def test_LOGNO2_family_consistency(self):
        y = np.linspace(0.1, 5, 100)
        mu = np.linspace(0.5, 3.0, 100)
        sigma = np.linspace(0.1, 1.5, 100)
        r_d = self.bus.eval_family("LOGNO2", "d", {"x": y.tolist(), "mu": mu.tolist(), "sigma": sigma.tolist()})
        family = resolve_family("LOGNO2")
        jax_d = np.asarray(jnp.exp(-0.5 * family.g_dev_inc(jnp.array(y), jnp.array(mu), jnp.array(sigma))))
        np.testing.assert_allclose(jax_d, r_d, rtol=1e-5, atol=1e-8)

    def test_PE_family_consistency(self):
        y = np.linspace(-3, 3, 100)
        mu = np.linspace(-1, 1, 100)
        sigma = np.linspace(0.5, 2.0, 100)
        nu = np.linspace(0.5, 3.0, 100)
        r_d = self.bus.eval_family("PE", "d", {"x": y.tolist(), "mu": mu.tolist(), "sigma": sigma.tolist(), "nu": nu.tolist()})
        family = resolve_family("PE")
        jax_d = np.asarray(jnp.exp(-0.5 * family.g_dev_inc(jnp.array(y), jnp.array(mu), jnp.array(sigma), jnp.array(nu))))
        np.testing.assert_allclose(jax_d, r_d, rtol=1e-5, atol=1e-8)

    def test_SIMPLEX_family_consistency(self):
        y = np.linspace(0.01, 0.99, 100)
        mu = np.linspace(0.1, 0.9, 100)
        sigma = np.linspace(0.1, 2.0, 100)
        r_d = self.bus.eval_family("SIMPLEX", "d", {"x": y.tolist(), "mu": mu.tolist(), "sigma": sigma.tolist()})
        family = resolve_family("SIMPLEX")
        jax_d = np.asarray(jnp.exp(-0.5 * family.g_dev_inc(jnp.array(y), jnp.array(mu), jnp.array(sigma))))
        np.testing.assert_allclose(jax_d, r_d, rtol=1e-5, atol=1e-8)

    def test_exGAUS_family_consistency(self):
        y = np.linspace(0, 10, 100)
        mu = np.linspace(1, 5, 100)
        sigma = np.linspace(0.5, 2.0, 100)
        nu = np.linspace(0.5, 3.0, 100)
        r_d = self.bus.eval_family("exGAUS", "d", {"x": y.tolist(), "mu": mu.tolist(), "sigma": sigma.tolist(), "nu": nu.tolist()})
        family = resolve_family("EXGAUS")
        jax_d = np.asarray(jnp.exp(-0.5 * family.g_dev_inc(jnp.array(y), jnp.array(mu), jnp.array(sigma), jnp.array(nu))))
        np.testing.assert_allclose(jax_d, r_d, rtol=1e-4, atol=1e-7)


    def test_SHASH_family_consistency(self):
        y = np.linspace(-3, 3, 100)
        mu = np.zeros(100)
        sigma = np.ones(100)
        nu = np.linspace(0.2, 1.5, 100)
        tau = np.linspace(0.3, 1.5, 100)
        r_d = self.bus.eval_family("SHASH", "d", {"x": y.tolist(), "mu": mu.tolist(), "sigma": sigma.tolist(), "nu": nu.tolist(), "tau": tau.tolist()})
        family = resolve_family("SHASH")
        jax_d = np.asarray(jnp.exp(-0.5 * family.g_dev_inc(jnp.array(y), jnp.array(mu), jnp.array(sigma), jnp.array(nu), jnp.array(tau))))
        np.testing.assert_allclose(jax_d, r_d, rtol=1e-5, atol=1e-8)

    def test_SHASHo_family_consistency(self):
        y = np.linspace(-3, 3, 100)
        mu = np.zeros(100)
        sigma = np.ones(100)
        nu = np.linspace(-1.0, 1.0, 100)
        tau = np.linspace(0.5, 2.0, 100)
        r_d = self.bus.eval_family("SHASHo", "d", {"x": y.tolist(), "mu": mu.tolist(), "sigma": sigma.tolist(), "nu": nu.tolist(), "tau": tau.tolist()})
        family = resolve_family("SHASHO")
        jax_d = np.asarray(jnp.exp(-0.5 * family.g_dev_inc(jnp.array(y), jnp.array(mu), jnp.array(sigma), jnp.array(nu), jnp.array(tau))))
        np.testing.assert_allclose(jax_d, r_d, rtol=1e-5, atol=1e-8)

    def test_SN1_family_consistency(self):
        y = np.linspace(-3, 3, 100)
        mu = np.zeros(100)
        sigma = np.ones(100)
        nu = np.linspace(-2.0, 2.0, 100)
        r_d = self.bus.eval_family("SN1", "d", {"x": y.tolist(), "mu": mu.tolist(), "sigma": sigma.tolist(), "nu": nu.tolist()})
        family = resolve_family("SN1")
        jax_d = np.asarray(jnp.exp(-0.5 * family.g_dev_inc(jnp.array(y), jnp.array(mu), jnp.array(sigma), jnp.array(nu))))
        np.testing.assert_allclose(jax_d, r_d, rtol=1e-5, atol=1e-8)

    def test_SN2_family_consistency(self):
        y = np.linspace(-3, 3, 100)
        mu = np.zeros(100)
        sigma = np.ones(100)
        nu = np.linspace(0.5, 3.0, 100)
        r_d = self.bus.eval_family("SN2", "d", {"x": y.tolist(), "mu": mu.tolist(), "sigma": sigma.tolist(), "nu": nu.tolist()})
        family = resolve_family("SN2")
        jax_d = np.asarray(jnp.exp(-0.5 * family.g_dev_inc(jnp.array(y), jnp.array(mu), jnp.array(sigma), jnp.array(nu))))
        np.testing.assert_allclose(jax_d, r_d, rtol=1e-5, atol=1e-8)

    def test_GT_family_consistency(self):
        y = np.linspace(-3, 3, 100)
        mu = np.zeros(100)
        sigma = np.ones(100)
        nu = np.linspace(1.0, 10.0, 100)
        tau = np.linspace(0.5, 3.0, 100)
        r_d = self.bus.eval_family("GT", "d", {"x": y.tolist(), "mu": mu.tolist(), "sigma": sigma.tolist(), "nu": nu.tolist(), "tau": tau.tolist()})
        family = resolve_family("GT")
        jax_d = np.asarray(jnp.exp(-0.5 * family.g_dev_inc(jnp.array(y), jnp.array(mu), jnp.array(sigma), jnp.array(nu), jnp.array(tau))))
        np.testing.assert_allclose(jax_d, r_d, rtol=1e-5, atol=1e-8)


    def test_BEINF_family_consistency(self):
        y = np.array([0.0, 0.2, 0.5, 0.8, 1.0])
        mu = np.linspace(0.2, 0.8, 5)
        sigma = np.linspace(0.1, 0.5, 5)
        nu = np.linspace(0.1, 0.5, 5)
        tau = np.linspace(0.1, 0.5, 5)
        r_d = self.bus.eval_family("BEINF", "d", {"x": y.tolist(), "mu": mu.tolist(), "sigma": sigma.tolist(), "nu": nu.tolist(), "tau": tau.tolist()})
        family = resolve_family("BEINF")
        jax_d = np.asarray(jnp.exp(-0.5 * family.g_dev_inc(jnp.array(y), jnp.array(mu), jnp.array(sigma), jnp.array(nu), jnp.array(tau))))
        np.testing.assert_allclose(jax_d, r_d, rtol=1e-5, atol=1e-8)

    def test_BEINF0_family_consistency(self):
        y = np.array([0.0, 0.2, 0.5, 0.8])
        mu = np.linspace(0.2, 0.8, 4)
        sigma = np.linspace(0.1, 0.5, 4)
        nu = np.linspace(0.1, 0.5, 4)
        r_d = self.bus.eval_family("BEINF0", "d", {"x": y.tolist(), "mu": mu.tolist(), "sigma": sigma.tolist(), "nu": nu.tolist()})
        family = resolve_family("BEINF0")
        jax_d = np.asarray(jnp.exp(-0.5 * family.g_dev_inc(jnp.array(y), jnp.array(mu), jnp.array(sigma), jnp.array(nu))))
        np.testing.assert_allclose(jax_d, r_d, rtol=1e-5, atol=1e-8)

    def test_BEINF1_family_consistency(self):
        y = np.array([0.2, 0.5, 0.8, 1.0])
        mu = np.linspace(0.2, 0.8, 4)
        sigma = np.linspace(0.1, 0.5, 4)
        nu = np.linspace(0.1, 0.5, 4)
        r_d = self.bus.eval_family("BEINF1", "d", {"x": y.tolist(), "mu": mu.tolist(), "sigma": sigma.tolist(), "nu": nu.tolist()})
        family = resolve_family("BEINF1")
        jax_d = np.asarray(jnp.exp(-0.5 * family.g_dev_inc(jnp.array(y), jnp.array(mu), jnp.array(sigma), jnp.array(nu))))
        np.testing.assert_allclose(jax_d, r_d, rtol=1e-5, atol=1e-8)

    def test_BEZI_family_consistency(self):
        y = np.array([0.0, 0.2, 0.5, 0.8])
        mu = np.linspace(0.2, 0.8, 4)
        sigma = np.linspace(1.0, 5.0, 4)  # sigma is precision
        nu = np.linspace(0.1, 0.5, 4)
        r_d = self.bus.eval_family("BEZI", "d", {"x": y.tolist(), "mu": mu.tolist(), "sigma": sigma.tolist(), "nu": nu.tolist()})
        family = resolve_family("BEZI")
        jax_d = np.asarray(jnp.exp(-0.5 * family.g_dev_inc(jnp.array(y), jnp.array(mu), jnp.array(sigma), jnp.array(nu))))
        np.testing.assert_allclose(jax_d, r_d, rtol=1e-5, atol=1e-8)

    def test_BEOI_family_consistency(self):
        y = np.array([0.2, 0.5, 0.8, 1.0])
        mu = np.linspace(0.2, 0.8, 4)
        sigma = np.linspace(1.0, 5.0, 4)  # sigma is precision
        nu = np.linspace(0.1, 0.5, 4)
        r_d = self.bus.eval_family("BEOI", "d", {"x": y.tolist(), "mu": mu.tolist(), "sigma": sigma.tolist(), "nu": nu.tolist()})
        family = resolve_family("BEOI")
        jax_d = np.asarray(jnp.exp(-0.5 * family.g_dev_inc(jnp.array(y), jnp.array(mu), jnp.array(sigma), jnp.array(nu))))
        np.testing.assert_allclose(jax_d, r_d, rtol=1e-5, atol=1e-8)


    def test_ZAGA_family_consistency(self):
        y = np.array([0.0, 0.5, 1.0, 2.0])
        mu = np.linspace(0.5, 2.0, 4)
        sigma = np.linspace(0.1, 1.0, 4)
        nu = np.linspace(0.1, 0.5, 4)
        r_d = self.bus.eval_family("ZAGA", "d", {"x": y.tolist(), "mu": mu.tolist(), "sigma": sigma.tolist(), "nu": nu.tolist()})
        family = resolve_family("ZAGA")
        jax_d = np.asarray(jnp.exp(-0.5 * family.g_dev_inc(jnp.array(y), jnp.array(mu), jnp.array(sigma), jnp.array(nu))))
        np.testing.assert_allclose(jax_d, r_d, rtol=1e-5, atol=1e-8)

    def test_ZAIG_family_consistency(self):
        y = np.array([0.0, 0.5, 1.0, 2.0])
        mu = np.linspace(0.5, 2.0, 4)
        sigma = np.linspace(0.1, 1.0, 4)
        nu = np.linspace(0.1, 0.5, 4)
        r_d = self.bus.eval_family("ZAIG", "d", {"x": y.tolist(), "mu": mu.tolist(), "sigma": sigma.tolist(), "nu": nu.tolist()})
        family = resolve_family("ZAIG")
        jax_d = np.asarray(jnp.exp(-0.5 * family.g_dev_inc(jnp.array(y), jnp.array(mu), jnp.array(sigma), jnp.array(nu))))
        np.testing.assert_allclose(jax_d, r_d, rtol=1e-5, atol=1e-8)

    def test_ZIP2_family_consistency(self):
        y = np.array([0, 1, 2, 5])
        mu = np.linspace(0.5, 5.0, 4)
        sigma = np.linspace(0.1, 0.5, 4)
        r_d = self.bus.eval_family("ZIP2", "d", {"x": y.tolist(), "mu": mu.tolist(), "sigma": sigma.tolist()})
        family = resolve_family("ZIP2")
        jax_d = np.asarray(jnp.exp(-0.5 * family.g_dev_inc(jnp.array(y), jnp.array(mu), jnp.array(sigma))))
        np.testing.assert_allclose(jax_d, r_d, rtol=1e-5, atol=1e-8)

    def test_ZINBI_family_consistency(self):
        y = np.array([0, 1, 2, 5])
        mu = np.linspace(0.5, 5.0, 4)
        sigma = np.linspace(0.1, 1.0, 4)
        nu = np.linspace(0.1, 0.5, 4)
        r_d = self.bus.eval_family("ZINBI", "d", {"x": y.tolist(), "mu": mu.tolist(), "sigma": sigma.tolist(), "nu": nu.tolist()})
        family = resolve_family("ZINBI")
        jax_d = np.asarray(jnp.exp(-0.5 * family.g_dev_inc(jnp.array(y), jnp.array(mu), jnp.array(sigma), jnp.array(nu))))
        np.testing.assert_allclose(jax_d, r_d, rtol=1e-5, atol=1e-8)

    def test_ZAP_family_consistency(self):
        y = np.array([0, 1, 2, 5])
        mu = np.linspace(0.5, 5.0, 4)
        sigma = np.linspace(0.1, 0.5, 4)
        r_d = self.bus.eval_family("ZAP", "d", {"x": y.tolist(), "mu": mu.tolist(), "sigma": sigma.tolist()})
        family = resolve_family("ZAP")
        jax_d = np.asarray(jnp.exp(-0.5 * family.g_dev_inc(jnp.array(y), jnp.array(mu), jnp.array(sigma))))
        np.testing.assert_allclose(jax_d, r_d, rtol=1e-5, atol=1e-8)

    # Batch 6: Discrete special series
    def test_PIG_family_consistency(self):
        y = np.array([1, 2, 5, 10])
        mu = np.linspace(1.0, 5.0, 4)
        sigma = np.linspace(0.1, 1.0, 4)
        r_d = self.bus.eval_family("PIG", "d", {"x": y.tolist(), "mu": mu.tolist(), "sigma": sigma.tolist()})
        family = resolve_family("PIG")
        jax_d = np.asarray(jnp.exp(-0.5 * family.g_dev_inc(jnp.array(y), jnp.array(mu), jnp.array(sigma))))
        np.testing.assert_allclose(jax_d, r_d, rtol=1e-5, atol=1e-8)

    def test_SICHEL_family_consistency(self):
        """Test SICHEL family consistency with R.
        
        Note: Now uses exact Bessel functions via scipy for accurate computation.
        """
        y = np.array([1, 2, 5, 10])
        mu = np.linspace(1.0, 5.0, 4)
        sigma = np.linspace(0.1, 1.0, 4)
        nu = np.linspace(-1.0, 1.0, 4)
        r_d = self.bus.eval_family("SICHEL", "d", {"x": y.tolist(), "mu": mu.tolist(), "sigma": sigma.tolist(), "nu": nu.tolist()})
        family = resolve_family("SICHEL")
        jax_d = np.asarray(jnp.exp(-0.5 * family.g_dev_inc(jnp.array(y), jnp.array(mu), jnp.array(sigma), jnp.array(nu))))
        np.testing.assert_allclose(jax_d, r_d, rtol=1e-5, atol=1e-8)

    def test_SI_family_consistency(self):
        """Test SI family consistency with R.
        
        Note: Now uses exact Bessel functions via scipy for accurate computation.
        Note: SI distribution in R gamlss.dist is defined for non-negative integers only.
        """
        y = np.array([0, 2, 5, 10])  # Changed from [-2, 0, 2, 5] to [0, 2, 5, 10]
        mu = np.linspace(1.0, 5.0, 4)
        sigma = np.linspace(1.0, 5.0, 4)
        r_d = self.bus.eval_family("SI", "d", {"x": y.tolist(), "mu": mu.tolist(), "sigma": sigma.tolist()})
        family = resolve_family("SI")
        jax_d = np.asarray(jnp.exp(-0.5 * family.g_dev_inc(jnp.array(y), jnp.array(mu), jnp.array(sigma))))
        np.testing.assert_allclose(jax_d, r_d, rtol=1e-5, atol=1e-8)

    def test_DPO_family_consistency(self):
        y = np.array([1, 2, 5, 10])
        mu = np.linspace(1.0, 5.0, 4)
        sigma = np.linspace(0.5, 2.0, 4)
        r_d = self.bus.eval_family("DPO", "d", {"x": y.tolist(), "mu": mu.tolist(), "sigma": sigma.tolist()})
        family = resolve_family("DPO")
        jax_d = np.asarray(jnp.exp(-0.5 * family.g_dev_inc(jnp.array(y), jnp.array(mu), jnp.array(sigma))))
        np.testing.assert_allclose(jax_d, r_d, rtol=1e-4, atol=1e-7)

    def test_DEL_family_consistency(self):
        y = np.array([1, 2, 5, 10])
        mu = np.linspace(2.0, 10.0, 4)
        sigma = np.linspace(0.5, 2.0, 4)
        nu = np.linspace(0.2, 0.8, 4)
        r_d = self.bus.eval_family("DEL", "d", {"x": y.tolist(), "mu": mu.tolist(), "sigma": sigma.tolist(), "nu": nu.tolist()})
        family = resolve_family("DEL")
        jax_d = np.asarray(jnp.exp(-0.5 * family.g_dev_inc(jnp.array(y), jnp.array(mu), jnp.array(sigma), jnp.array(nu))))
        np.testing.assert_allclose(jax_d, r_d, rtol=1e-5, atol=1e-8)

    def test_YULE_family_consistency(self):
        y = np.array([1, 2, 5, 10])
        mu = np.linspace(0.5, 3.0, 4)
        r_d = self.bus.eval_family("YULE", "d", {"x": y.tolist(), "mu": mu.tolist()})
        family = resolve_family("YULE")
        jax_d = np.asarray(jnp.exp(-0.5 * family.g_dev_inc(jnp.array(y), jnp.array(mu))))
        np.testing.assert_allclose(jax_d, r_d, rtol=1e-5, atol=1e-8)

    def test_WARING_family_consistency(self):
        y = np.array([1, 2, 5, 10])
        mu = np.linspace(1.0, 5.0, 4)
        sigma = np.linspace(1.0, 5.0, 4)
        r_d = self.bus.eval_family("WARING", "d", {"x": y.tolist(), "mu": mu.tolist(), "sigma": sigma.tolist()})
        family = resolve_family("WARING")
        jax_d = np.asarray(jnp.exp(-0.5 * family.g_dev_inc(jnp.array(y), jnp.array(mu), jnp.array(sigma))))
        np.testing.assert_allclose(jax_d, r_d, rtol=1e-5, atol=1e-8)

    # Batch 7: Multinomial and compound families
    def test_BB_family_consistency(self):
        y = np.array([1, 3, 5, 8])
        mu = np.linspace(0.2, 0.8, 4)
        sigma = np.linspace(0.1, 0.5, 4)
        bd = np.array([10, 10, 10, 10])
        r_d = self.bus.eval_family("BB", "d", {"x": y.tolist(), "mu": mu.tolist(), "sigma": sigma.tolist(), "bd": bd.tolist()})
        family = resolve_family("BB")
        jax_d = np.asarray(jnp.exp(-0.5 * family.g_dev_inc(jnp.array(y), jnp.array(mu), jnp.array(sigma), jnp.array(bd))))
        np.testing.assert_allclose(jax_d, r_d, rtol=1e-5, atol=1e-8)

    def test_BNB_family_consistency(self):
        y = np.array([1, 2, 5, 10])
        mu = np.linspace(1.0, 5.0, 4)
        sigma = np.linspace(0.5, 2.0, 4)
        nu = np.linspace(0.5, 2.0, 4)
        r_d = self.bus.eval_family("BNB", "d", {"x": y.tolist(), "mu": mu.tolist(), "sigma": sigma.tolist(), "nu": nu.tolist()})
        family = resolve_family("BNB")
        jax_d = np.asarray(jnp.exp(-0.5 * family.g_dev_inc(jnp.array(y), jnp.array(mu), jnp.array(sigma), jnp.array(nu))))
        np.testing.assert_allclose(jax_d, r_d, rtol=1e-5, atol=1e-8)

    def test_MN3_family_consistency(self):
        """Test MN3 categorical-family consistency with R."""
        y = np.array([1, 2, 3, 1])
        mu = np.linspace(0.5, 2.0, 4)
        sigma = np.linspace(0.5, 1.5, 4)
        r_d = self.bus.eval_family("MN3", "d", {"x": y.tolist(), "mu": mu.tolist(), "sigma": sigma.tolist()})
        family = resolve_family("MN3")
        jax_d = np.asarray(jnp.exp(-0.5 * family.g_dev_inc(jnp.array(y), jnp.array(mu), jnp.array(sigma))))
        np.testing.assert_allclose(jax_d, r_d, rtol=1e-5, atol=1e-8)

    def test_MN4_family_consistency(self):
        """Test MN4 categorical-family consistency with R."""
        y = np.array([1, 2, 3, 4])
        mu = np.linspace(0.5, 2.0, 4)
        sigma = np.linspace(0.5, 1.5, 4)
        nu = np.linspace(0.4, 1.0, 4)
        r_d = self.bus.eval_family(
            "MN4",
            "d",
            {"x": y.tolist(), "mu": mu.tolist(), "sigma": sigma.tolist(), "nu": nu.tolist()},
        )
        family = resolve_family("MN4")
        jax_d = np.asarray(
            jnp.exp(-0.5 * family.g_dev_inc(jnp.array(y), jnp.array(mu), jnp.array(sigma), jnp.array(nu)))
        )
        np.testing.assert_allclose(jax_d, r_d, rtol=1e-5, atol=1e-8)

    def test_MN5_family_consistency(self):
        """Test MN5 categorical-family consistency with R."""
        y = np.array([1, 2, 4, 5])
        mu = np.linspace(0.5, 2.0, 4)
        sigma = np.linspace(0.5, 1.5, 4)
        nu = np.linspace(0.4, 1.0, 4)
        tau = np.linspace(0.3, 0.9, 4)
        r_d = self.bus.eval_family(
            "MN5",
            "d",
            {
                "x": y.tolist(),
                "mu": mu.tolist(),
                "sigma": sigma.tolist(),
                "nu": nu.tolist(),
                "tau": tau.tolist(),
            },
        )
        family = resolve_family("MN5")
        jax_d = np.asarray(
            jnp.exp(
                -0.5
                * family.g_dev_inc(
                    jnp.array(y),
                    jnp.array(mu),
                    jnp.array(sigma),
                    jnp.array(nu),
                    jnp.array(tau),
                )
            )
        )
        np.testing.assert_allclose(jax_d, r_d, rtol=1e-5, atol=1e-8)

    # Batch 8: Additional continuous distributions
    def test_GG_family_consistency(self):
        y = np.array([0.5, 1.0, 2.0, 5.0])
        mu = np.linspace(1.0, 3.0, 4)
        sigma = np.linspace(0.3, 0.7, 4)
        nu = np.linspace(0.5, 2.0, 4)
        r_d = self.bus.eval_family("GG", "d", {"x": y.tolist(), "mu": mu.tolist(), "sigma": sigma.tolist(), "nu": nu.tolist()})
        family = resolve_family("GG")
        jax_d = np.asarray(jnp.exp(-0.5 * family.g_dev_inc(jnp.array(y), jnp.array(mu), jnp.array(sigma), jnp.array(nu))))
        np.testing.assert_allclose(jax_d, r_d, rtol=1e-5, atol=1e-8)

    def test_GB2_family_consistency(self):
        y = np.array([0.5, 1.0, 2.0, 5.0])
        mu = np.linspace(1.0, 3.0, 4)
        sigma = np.linspace(0.5, 1.5, 4)
        nu = np.linspace(0.5, 2.0, 4)
        tau = np.linspace(0.3, 1.0, 4)
        r_d = self.bus.eval_family("GB2", "d", {"x": y.tolist(), "mu": mu.tolist(), "sigma": sigma.tolist(), "nu": nu.tolist(), "tau": tau.tolist()})
        family = resolve_family("GB2")
        jax_d = np.asarray(jnp.exp(-0.5 * family.g_dev_inc(jnp.array(y), jnp.array(mu), jnp.array(sigma), jnp.array(nu), jnp.array(tau))))
        np.testing.assert_allclose(jax_d, r_d, rtol=1e-5, atol=1e-8)

    def test_PARETO_family_consistency(self):
        y = np.array([1.1, 2.0, 5.0, 10.0])
        mu = np.linspace(0.5, 3.0, 4)
        r_d = self.bus.eval_family("PARETO", "d", {"x": y.tolist(), "mu": mu.tolist()})
        family = resolve_family("PARETO")
        jax_d = np.asarray(jnp.exp(-0.5 * family.g_dev_inc(jnp.array(y), jnp.array(mu))))
        np.testing.assert_allclose(jax_d, r_d, rtol=1e-5, atol=1e-8)

    def test_NET_family_consistency(self):
        y = np.array([0.1, 0.5, 1.0, 2.0])
        mu = np.linspace(0.5, 1.5, 4)
        sigma = np.linspace(0.5, 2.0, 4)
        nu = np.linspace(1.0, 3.0, 4)
        tau = np.linspace(1.5, 3.0, 4)
        r_d = self.bus.eval_family("NET", "d", {"x": y.tolist(), "mu": mu.tolist(), "sigma": sigma.tolist(), "nu": nu.tolist(), "tau": tau.tolist()})
        family = resolve_family("NET")
        jax_d = np.asarray(jnp.exp(-0.5 * family.g_dev_inc(jnp.array(y), jnp.array(mu), jnp.array(sigma), jnp.array(nu), jnp.array(tau))))
        np.testing.assert_allclose(jax_d, r_d, rtol=1e-5, atol=1e-8)

    def test_LNO_family_consistency(self):
        y = np.array([0.5, 1.0, 2.0, 5.0])
        mu = np.linspace(1.0, 3.0, 4)
        sigma = np.linspace(0.1, 0.5, 4)
        nu = np.linspace(0.0, 0.5, 4)
        r_d = self.bus.eval_family("LNO", "d", {"x": y.tolist(), "mu": mu.tolist(), "sigma": sigma.tolist(), "nu": nu.tolist()})
        family = resolve_family("LNO")
        jax_d = np.asarray(jnp.exp(-0.5 * family.g_dev_inc(jnp.array(y), jnp.array(mu), jnp.array(sigma), jnp.array(nu))))
        np.testing.assert_allclose(jax_d, r_d, rtol=1e-5, atol=1e-8)


if __name__ == "__main__":
    unittest.main()
