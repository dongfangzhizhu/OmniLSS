"""Generate correct R consistency tests based on family parameters."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "omnilss" / "src"))

from omnilss.distribution_registry import resolve

# Test template for 1-parameter families (mu only)
TEST_TEMPLATE_1PARAM = '''"""R consistency checks for {family} distribution (d/p/q)."""

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

# Test template for 2-parameter families (mu, sigma or mu, nu)
TEST_TEMPLATE_2PARAM = '''"""R consistency checks for {family} distribution (d/p/q)."""

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

        for profile_name, x, mu, {param2}, p in (
            (
                "central",
                {x_central},
                {mu_central},
                {param2_central},
                np.linspace(0.02, 0.98, {n_central}),
            ),
            (
                "tail",
                {x_tail},
                {mu_tail},
                {param2_tail},
                np.concatenate([
                    np.geomspace(1e-5, 1e-2, {n_tail_half}),
                    1.0 - np.geomspace(1e-2, 1e-5, {n_tail_half}),
                ]),
            ),
        ):
            with self.subTest(profile=profile_name):
                r_d = self.bus.eval_family("{family}", "d", {{"x": x.tolist(), "mu": mu.tolist(), "{param2}": {param2}.tolist()}})
                r_p = self.bus.eval_family("{family}", "p", {{"q": x.tolist(), "mu": mu.tolist(), "{param2}": {param2}.tolist()}})
                r_q = self.bus.eval_family("{family}", "q", {{"p": p.tolist(), "mu": mu.tolist(), "{param2}": {param2}.tolist()}})

                py_d = np.asarray(fam.d(jnp.array(x), jnp.array(mu), jnp.array({param2})))
                py_p = np.asarray(fam.p(jnp.array(x), jnp.array(mu), jnp.array({param2})))
                py_q = np.asarray(fam.q(jnp.array(p), jnp.array(mu), jnp.array({param2})))

                np.testing.assert_allclose(py_d, r_d, rtol=1e-5, atol=1e-7)
                np.testing.assert_allclose(py_p, r_p, rtol=1e-5, atol=1e-7)
                np.testing.assert_allclose(py_q, r_q, rtol=1e-5, atol=1e-7)


if __name__ == "__main__":
    unittest.main()
'''

# Configuration for different family types
FAMILY_CONFIGS = {
    "discrete_count": {
        "x_central": "np.arange(0, 50)",
        "mu_central": "np.linspace(0.5, 10.0, 50)",
        "sigma_central": "np.linspace(0.2, 2.0, 50)",
        "nu_central": "np.linspace(0.1, 0.9, 50)",
        "n_central": 50,
        "x_tail": "np.arange(0, 100)",
        "mu_tail": "np.linspace(0.1, 20.0, 100)",
        "sigma_tail": "np.linspace(0.1, 3.0, 100)",
        "nu_tail": "np.linspace(0.05, 0.95, 100)",
        "n_tail_half": 50,
    },
    "continuous_positive": {
        "x_central": "np.linspace(0.01, 10.0, 80)",
        "mu_central": "np.linspace(0.5, 5.0, 80)",
        "sigma_central": "np.linspace(0.2, 2.0, 80)",
        "nu_central": "np.linspace(0.1, 0.9, 80)",
        "n_central": 80,
        "x_tail": "np.linspace(0.001, 50.0, 120)",
        "mu_tail": "np.linspace(0.1, 20.0, 120)",
        "sigma_tail": "np.linspace(0.1, 3.0, 120)",
        "nu_tail": "np.linspace(0.05, 0.95, 120)",
        "n_tail_half": 60,
    },
}

# Map families to their types
FAMILY_TYPE_MAP = {
    # 1-parameter discrete count
    "PARETO1": "continuous_positive",
    "LG": "discrete_count",
    "ZIPF": "discrete_count",
    
    # 2-parameter families
    "ZAZIPF": "discrete_count",
    "GAF": "continuous_positive",
    "NOF": "continuous_positive",
    "WEI2": "continuous_positive",
    "WEI3": "continuous_positive",
    "PARETO1O": "continuous_positive",
    "PIG2": "continuous_positive",
    "ZALG": "discrete_count",
    "GPO": "discrete_count",
    "DPO1": "discrete_count",
    "GP": "discrete_count",
    "GENERALISEDPOISSON": "discrete_count",
    "LOGITNO": "continuous_positive",
}

def generate_test_file(family: str, output_dir: Path) -> None:
    """Generate a test file for the given family."""
    try:
        fam = resolve(family)
        params = fam.parameters
        n_params = len(params)
        
        family_type = FAMILY_TYPE_MAP.get(family, "continuous_positive")
        config = FAMILY_CONFIGS[family_type]
        
        if n_params == 1:
            # 1-parameter family
            test_content = TEST_TEMPLATE_1PARAM.format(
                family=family,
                family_lower=family.lower(),
                x_central=config["x_central"],
                mu_central=config["mu_central"],
                n_central=config["n_central"],
                x_tail=config["x_tail"],
                mu_tail=config["mu_tail"],
                n_tail_half=config["n_tail_half"],
            )
        elif n_params == 2:
            # 2-parameter family
            param2 = params[1]  # sigma or nu
            test_content = TEST_TEMPLATE_2PARAM.format(
                family=family,
                family_lower=family.lower(),
                param2=param2,
                x_central=config["x_central"],
                mu_central=config["mu_central"],
                param2_central=config[f"{param2}_central"],
                n_central=config["n_central"],
                x_tail=config["x_tail"],
                mu_tail=config["mu_tail"],
                param2_tail=config[f"{param2}_tail"],
                n_tail_half=config["n_tail_half"],
            )
        else:
            print(f"Skipping {family}: {n_params} parameters (not yet supported)")
            return
        
        output_file = output_dir / f"test_r_consistency_{family.lower()}.py"
        output_file.write_text(test_content, encoding='utf-8')
        print(f"Generated: {output_file.name} ({n_params} params: {params})")
        
    except Exception as e:
        print(f"Error generating test for {family}: {e}")

def main():
    test_dir = Path(__file__).parent / "omnilss" / "tests"
    
    # Priority: 1-2 parameter families
    simple_families = [
        # 1-parameter
        "PARETO1", "LG", "ZIPF",
        # 2-parameter
        "ZAZIPF", "GAF", "NOF", "WEI2", "WEI3", "PARETO1O", "PIG2",
        "ZALG", "GPO", "DPO1", "GP", "GENERALISEDPOISSON", "LOGITNO",
    ]
    
    print("Generating correct R consistency tests for simple families...")
    print("=" * 80)
    
    for family in simple_families:
        generate_test_file(family, test_dir)
    
    print("\n" + "=" * 80)
    print(f"Generated tests for {len(simple_families)} families")

if __name__ == "__main__":
    main()
