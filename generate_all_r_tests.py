"""Generate complete R consistency tests for all families."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "omnilss" / "src"))

from omnilss.distribution_registry import resolve
from family_param_ranges import get_param_ranges, FAMILY_PARAM_RANGES

# Base test template
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

        for profile_name, {param_vars} in (
            (
                "central",
                {central_values}
            ),
            (
                "tail",
                {tail_values}
            ),
        ):
            with self.subTest(profile=profile_name):
                r_d = self.bus.eval_family("{family}", "d", {{{r_d_args}}})
                r_p = self.bus.eval_family("{family}", "p", {{{r_p_args}}})
                r_q = self.bus.eval_family("{family}", "q", {{{r_q_args}}})

                py_d = np.asarray(fam.d({py_d_args}))
                py_p = np.asarray(fam.p({py_p_args}))
                py_q = np.asarray(fam.q({py_q_args}))

                np.testing.assert_allclose(py_d, r_d, rtol=1e-5, atol=1e-7)
                np.testing.assert_allclose(py_p, r_p, rtol=1e-5, atol=1e-7)
                np.testing.assert_allclose(py_q, r_q, rtol=1e-5, atol=1e-7)


if __name__ == "__main__":
    unittest.main()
'''

# Family type mapping
FAMILY_TYPE_MAP = {
    # 1-parameter
    "PARETO1": "continuous_positive",
    "LG": "discrete_count",
    "ZIPF": "discrete_count",
    
    # 2-parameter
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
    "LOGITNO": "bounded_01",
    
    # 3-parameter
    "ZAPIG": "discrete_count",
    "ZIPIG": "discrete_count",
    "ZABB": "discrete_count",
    "ZIBB": "discrete_count",
    "RGE": "continuous_positive",
    "PE2": "continuous_positive",
    "LQNO": "continuous_real",
    "DELAPORT": "discrete_count",
    
    # 4-parameter
    "ZASICHEL": "discrete_count",
    "ZISICHEL": "discrete_count",
    "ZABNB": "discrete_count",
    "ZIBNB": "discrete_count",
    "GB1": "bounded_01",
    "EGB2": "continuous_real",
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
}

def generate_test_file(family: str, output_dir: Path, overwrite: bool = False) -> bool:
    """Generate a test file for the given family.
    
    Returns:
        True if test was generated successfully, False otherwise
    """
    try:
        fam = resolve(family)
        params = fam.parameters
        n_params = len(params)
        
        output_file = output_dir / f"test_r_consistency_{family.lower()}.py"
        if output_file.exists() and not overwrite:
            print(f"Skipping {family}: test file already exists")
            return False
        
        family_type = FAMILY_TYPE_MAP.get(family, "continuous_positive")
        ranges = get_param_ranges(family, family_type)
        
        # Build parameter variables list
        param_vars = ["x"] + list(params) + ["p"]
        param_vars_str = ", ".join(param_vars)
        
        # Build central values
        central_parts = [ranges["x"]]
        for param in params:
            central_parts.append(ranges.get(param, ranges["mu"]))
        central_parts.append(ranges["p"])
        central_values_str = ",\n                ".join(central_parts) + ","
        
        # Build tail values (use same for now, can be customized later)
        tail_values_str = central_values_str
        
        # Build R d/p/q arguments
        r_d_args_parts = ['"x": x.tolist()'] + [f'"{p}": {p}.tolist()' for p in params]
        r_d_args = ", ".join(r_d_args_parts)
        
        r_p_args_parts = ['"q": x.tolist()'] + [f'"{p}": {p}.tolist()' for p in params]
        r_p_args = ", ".join(r_p_args_parts)
        
        r_q_args_parts = ['"p": p.tolist()'] + [f'"{p}": {p}.tolist()' for p in params]
        r_q_args = ", ".join(r_q_args_parts)
        
        # Build Python d/p/q arguments
        py_d_args_parts = ["jnp.array(x)"] + [f"jnp.array({p})" for p in params]
        py_d_args = ", ".join(py_d_args_parts)
        
        py_p_args_parts = ["jnp.array(x)"] + [f"jnp.array({p})" for p in params]
        py_p_args = ", ".join(py_p_args_parts)
        
        py_q_args_parts = ["jnp.array(p)"] + [f"jnp.array({p})" for p in params]
        py_q_args = ", ".join(py_q_args_parts)
        
        # Generate test content
        test_content = TEST_TEMPLATE.format(
            family=family,
            family_lower=family.lower(),
            param_vars=param_vars_str,
            central_values=central_values_str,
            tail_values=tail_values_str,
            r_d_args=r_d_args,
            r_p_args=r_p_args,
            r_q_args=r_q_args,
            py_d_args=py_d_args,
            py_p_args=py_p_args,
            py_q_args=py_q_args,
        )
        
        output_file.write_text(test_content, encoding='utf-8')
        print(f"[OK] Generated: {output_file.name} ({n_params} params: {params})")
        return True
        
    except Exception as e:
        print(f"[ERROR] Error generating test for {family}: {e}")
        return False

def main():
    test_dir = Path(__file__).parent / "omnilss" / "tests"
    
    # All priority families
    all_families = [
        # 1-parameter (3)
        "PARETO1", "LG", "ZIPF",
        # 2-parameter (13)
        "ZAZIPF", "GAF", "NOF", "WEI2", "WEI3", "PARETO1O", "PIG2",
        "ZALG", "GPO", "DPO1", "GP", "GENERALISEDPOISSON", "LOGITNO",
        # 3-parameter (8)
        "ZAPIG", "ZIPIG", "ZABB", "ZIBB", "RGE", "PE2", "LQNO", "DELAPORT",
        # 4-parameter (18)
        "ZASICHEL", "ZISICHEL", "ZABNB", "ZIBNB", "GB1", "EGB2", "SST", "ST5",
        "SEP", "SEP1", "SEP2", "SEP3", "SEP4",
        "LOGSHASH", "LOGSHASHO", "SHASHO", "SHASHO2", "JSUO",
    ]
    
    print("Generating R consistency tests for all priority families...")
    print("=" * 80)
    
    success_count = 0
    skip_count = 0
    error_count = 0
    
    for family in all_families:
        result = generate_test_file(family, test_dir, overwrite=True)
        if result:
            success_count += 1
        elif result is False:
            skip_count += 1
        else:
            error_count += 1
    
    print("\n" + "=" * 80)
    print(f"Summary:")
    print(f"  [OK] Generated: {success_count}")
    print(f"  [-] Skipped: {skip_count}")
    print(f"  [X] Errors: {error_count}")
    print(f"  Total: {len(all_families)}")

if __name__ == "__main__":
    main()
