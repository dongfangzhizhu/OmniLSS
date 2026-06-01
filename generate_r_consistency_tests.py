"""Generate R consistency test files for missing families."""
from pathlib import Path

# Template for R consistency test
TEST_TEMPLATE = '''"""R consistency checks for {family} distribution (d/p/q)."""

from __future__ import annotations

import unittest

import jax.numpy as jnp
import numpy as np

from omnilss.distribution_registry import resolve
from tests.rbus.core import RTestBus


class Test{family}Consistency(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            cls.bus = RTestBus()
        except RuntimeError as exc:
            raise unittest.SkipTest(f"RTestBus not available: {{exc}}")

    def test_{family_lower}_dpq_consistency(self) -> None:
        fam = resolve("{family}")

        for profile_name, x, mu, p in (
            (
                "central",
                {x_central},
                {mu_central},
                np.linspace(0.02, 0.98, {n_central}),
            ),
            (
                "tail",
                {x_tail},
                {mu_tail},
                np.concatenate([
                    np.geomspace(1e-5, 1e-2, {n_tail_half}),
                    1.0 - np.geomspace(1e-2, 1e-5, {n_tail_half}),
                ]),
            ),
        ):
            with self.subTest(profile=profile_name):
                r_d = self.bus.eval_family("{family}", "d", {{"x": x.tolist(), "mu": mu.tolist()}})
                r_p = self.bus.eval_family("{family}", "p", {{"q": x.tolist(), "mu": mu.tolist()}})
                r_q = self.bus.eval_family("{family}", "q", {{"p": p.tolist(), "mu": mu.tolist()}})

                py_d = np.asarray(fam.d(jnp.array(x), jnp.array(mu)))
                py_p = np.asarray(fam.p(jnp.array(x), jnp.array(mu)))
                py_q = np.asarray(fam.q(jnp.array(p), jnp.array(mu)))

                np.testing.assert_allclose(py_d, r_d, rtol=1e-5, atol=1e-7)
                np.testing.assert_allclose(py_p, r_p, rtol=1e-5, atol=1e-7)
                np.testing.assert_allclose(py_q, r_q, rtol=1e-5, atol=1e-7)


if __name__ == "__main__":
    unittest.main()
'''

# Configuration for different family types
FAMILY_CONFIGS = {
    # Discrete count distributions
    "discrete_count": {
        "x_central": "np.arange(0, 50)",
        "mu_central": "np.linspace(0.5, 10.0, 50)",
        "n_central": 50,
        "x_tail": "np.arange(0, 100)",
        "mu_tail": "np.linspace(0.1, 20.0, 100)",
        "n_tail_half": 50,
    },
    # Continuous positive distributions
    "continuous_positive": {
        "x_central": "np.linspace(0.01, 10.0, 80)",
        "mu_central": "np.linspace(0.5, 5.0, 80)",
        "n_central": 80,
        "x_tail": "np.linspace(0.001, 50.0, 120)",
        "mu_tail": "np.linspace(0.1, 20.0, 120)",
        "n_tail_half": 60,
    },
    # Continuous real line distributions
    "continuous_real": {
        "x_central": "np.linspace(-5.0, 5.0, 80)",
        "mu_central": "np.linspace(-2.0, 2.0, 80)",
        "n_central": 80,
        "x_tail": "np.linspace(-10.0, 10.0, 120)",
        "mu_tail": "np.linspace(-5.0, 5.0, 120)",
        "n_tail_half": 60,
    },
    # Bounded (0,1) distributions
    "bounded_01": {
        "x_central": "np.linspace(0.01, 0.99, 80)",
        "mu_central": "np.linspace(0.1, 0.9, 80)",
        "n_central": 80,
        "x_tail": "np.linspace(0.001, 0.999, 120)",
        "mu_tail": "np.linspace(0.05, 0.95, 120)",
        "n_tail_half": 60,
    },
}

# Map families to their types
FAMILY_TYPE_MAP = {
    # Discrete count
    "LG": "discrete_count",
    "ZIPF": "discrete_count",
    "ZALG": "discrete_count",
    "GPO": "discrete_count",
    "DELAPORT": "discrete_count",
    "DPO1": "discrete_count",
    "GP": "discrete_count",
    "GENERALISEDPOISSON": "discrete_count",
    "ZAPIG": "discrete_count",
    "ZIPIG": "discrete_count",
    "ZASICHEL": "discrete_count",
    "ZISICHEL": "discrete_count",
    "ZAZIPF": "discrete_count",
    "ZABB": "discrete_count",
    "ZIBB": "discrete_count",
    "ZABNB": "discrete_count",
    "ZIBNB": "discrete_count",
    
    # Continuous positive
    "GAF": "continuous_positive",
    "NOF": "continuous_positive",
    "RGE": "continuous_positive",
    "WEI2": "continuous_positive",
    "WEI3": "continuous_positive",
    "GB1": "continuous_positive",
    "PARETO1": "continuous_positive",
    "PARETO1O": "continuous_positive",
    "PIG2": "continuous_positive",
    "PE2": "continuous_positive",
    
    # Continuous real line
    "EGB2": "continuous_real",
    "LQNO": "continuous_real",
    "SST": "continuous_real",
    "ST5": "continuous_real",
    "SEP": "continuous_real",
    "SEP1": "continuous_real",
    "SEP2": "continuous_real",
    "SEP3": "continuous_real",
    "SEP4": "continuous_real",
    "LOGSHASH": "continuous_real",
    "LOGSHASHO": "continuous_real",
    "SHASHO": "continuous_real",
    "SHASHO2": "continuous_real",
    "JSUO": "continuous_real",
    
    # Bounded (0,1)
    "LOGITNO": "bounded_01",
}

def generate_test_file(family: str, output_dir: Path) -> None:
    """Generate a test file for the given family."""
    family_type = FAMILY_TYPE_MAP.get(family, "continuous_positive")
    config = FAMILY_CONFIGS[family_type]
    
    test_content = TEST_TEMPLATE.format(
        family=family,
        family_lower=family.lower(),
        x_central=config["x_central"],
        mu_central=config["mu_central"],
        n_central=config["n_central"],
        x_tail=config["x_tail"],
        mu_tail=config["mu_tail"],
        n_tail_half=config["n_tail_half"],
    )
    
    output_file = output_dir / f"test_r_consistency_{family.lower()}.py"
    output_file.write_text(test_content, encoding='utf-8')
    print(f"Generated: {output_file.name}")

def main():
    test_dir = Path(__file__).parent / "omnilss" / "tests"
    
    # Priority families to generate tests for
    priority_families = [
        # Zero-inflated/adjusted variants (high priority)
        "ZAPIG", "ZIPIG", "ZASICHEL", "ZISICHEL", "ZAZIPF",
        "ZABB", "ZIBB", "ZABNB", "ZIBNB",
        
        # Important continuous distributions
        "GAF", "NOF", "RGE", "WEI2", "WEI3", "GB1",
        "PARETO1", "PARETO1O", "PIG2", "PE2",
        
        # Skew/shape distributions
        "EGB2", "LQNO", "SST", "ST5",
        "SEP", "SEP1", "SEP2", "SEP3", "SEP4",
        "LOGSHASH", "LOGSHASHO", "SHASHO", "SHASHO2", "JSUO",
        
        # Count distributions
        "LG", "ZIPF", "ZALG", "GPO", "DELAPORT", "DPO1",
        "GP", "GENERALISEDPOISSON",
        
        # Bounded
        "LOGITNO",
    ]
    
    for family in priority_families:
        if family in FAMILY_TYPE_MAP:
            generate_test_file(family, test_dir)
    
    print(f"\nGenerated {len(priority_families)} test files")

if __name__ == "__main__":
    main()
