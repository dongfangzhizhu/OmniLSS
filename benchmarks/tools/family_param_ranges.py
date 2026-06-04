"""Parameter range configurations for each family.

This module defines valid parameter ranges for R consistency tests.
Each family has specific constraints on its parameters.
"""

# Parameter range configurations by family
FAMILY_PARAM_RANGES = {
    # 1-parameter families
    "PARETO1": {
        "type": "continuous_positive",
        "x": "np.linspace(1.01, 50.0, 80)",  # x > 1 for Pareto
        "mu": "np.linspace(0.1, 5.0, 80)",  # shape parameter > 0
        "p": "np.linspace(0.02, 0.98, 80)",
    },
    "LG": {
        "type": "discrete_count",
        "x": "np.arange(1, 51)",  # x >= 1 for logarithmic, 50 elements
        "mu": "np.linspace(0.05, 0.95, 50)",  # 0 < mu < 1, 50 elements
        "p": "np.linspace(0.02, 0.98, 50)",
    },
    "ZIPF": {
        "type": "discrete_count",
        "x": "np.arange(1, 51)",  # x >= 1 for Zipf, 50 elements
        "mu": "np.linspace(0.5, 5.0, 50)",  # shape parameter > 0, 50 elements
        "p": "np.linspace(0.02, 0.98, 50)",
    },
    
    # 2-parameter families (mu, sigma)
    "GAF": {
        "type": "continuous_positive",
        "x": "np.linspace(0.01, 10.0, 80)",
        "mu": "np.linspace(0.5, 5.0, 80)",
        "sigma": "np.linspace(0.2, 2.0, 80)",
        "p": "np.linspace(0.02, 0.98, 80)",
    },
    "NOF": {
        "type": "continuous_positive",
        "x": "np.linspace(0.01, 10.0, 80)",
        "mu": "np.linspace(0.5, 5.0, 80)",
        "sigma": "np.linspace(0.2, 2.0, 80)",
        "p": "np.linspace(0.02, 0.98, 80)",
    },
    "WEI2": {
        "type": "continuous_positive",
        "x": "np.linspace(0.01, 10.0, 80)",
        "mu": "np.linspace(0.5, 5.0, 80)",
        "sigma": "np.linspace(0.2, 2.0, 80)",
        "p": "np.linspace(0.02, 0.98, 80)",
    },
    "WEI3": {
        "type": "continuous_positive",
        "x": "np.linspace(0.01, 10.0, 80)",
        "mu": "np.linspace(0.5, 5.0, 80)",
        "sigma": "np.linspace(0.2, 2.0, 80)",
        "p": "np.linspace(0.02, 0.98, 80)",
    },
    "PARETO1O": {
        "type": "continuous_positive",
        "x": "np.linspace(1.01, 50.0, 80)",
        "mu": "np.linspace(0.5, 5.0, 80)",
        "sigma": "np.linspace(0.2, 2.0, 80)",
        "p": "np.linspace(0.02, 0.98, 80)",
    },
    "PIG2": {
        "type": "continuous_positive",
        "x": "np.linspace(0.01, 10.0, 80)",
        "mu": "np.linspace(0.5, 5.0, 80)",
        "sigma": "np.linspace(0.2, 2.0, 80)",
        "p": "np.linspace(0.02, 0.98, 80)",
    },
    "GPO": {
        "type": "discrete_count",
        "x": "np.arange(0, 50)",
        "mu": "np.linspace(0.5, 10.0, 50)",
        "sigma": "np.linspace(0.1, 2.0, 50)",
        "p": "np.linspace(0.02, 0.98, 50)",
    },
    "DPO1": {
        "type": "discrete_count",
        "x": "np.arange(0, 50)",
        "mu": "np.linspace(0.5, 10.0, 50)",
        "sigma": "np.linspace(0.1, 2.0, 50)",
        "p": "np.linspace(0.02, 0.98, 50)",
    },
    "GP": {
        "type": "discrete_count",
        "x": "np.arange(0, 50)",
        "mu": "np.linspace(0.5, 10.0, 50)",
        "sigma": "np.linspace(-0.5, 0.5, 50)",  # theta parameter
        "p": "np.linspace(0.02, 0.98, 50)",
    },
    "GENERALISEDPOISSON": {
        "type": "discrete_count",
        "x": "np.arange(0, 50)",
        "mu": "np.linspace(0.5, 10.0, 50)",
        "sigma": "np.linspace(-0.5, 0.5, 50)",
        "p": "np.linspace(0.02, 0.98, 50)",
    },
    "LOGITNO": {
        "type": "bounded_01",
        "x": "np.linspace(0.01, 0.99, 80)",
        "mu": "np.linspace(0.1, 0.9, 80)",
        "sigma": "np.linspace(0.2, 2.0, 80)",
        "p": "np.linspace(0.02, 0.98, 80)",
    },
    
    # 2-parameter families (mu, nu)
    "ZAZIPF": {
        "type": "discrete_count",
        "x": "np.arange(0, 50)",  # includes 0 for zero-altered
        "mu": "np.linspace(0.5, 5.0, 50)",
        "nu": "np.linspace(0.1, 0.9, 50)",  # zero probability
        "p": "np.linspace(0.02, 0.98, 50)",
    },
    "ZALG": {
        "type": "discrete_count",
        "x": "np.arange(0, 50)",
        "mu": "np.linspace(0.05, 0.95, 50)",
        "nu": "np.linspace(0.1, 0.9, 50)",
        "p": "np.linspace(0.02, 0.98, 50)",
    },
    
    # 3-parameter families (mu, sigma, nu)
    "ZAPIG": {
        "type": "discrete_count",
        "x": "np.arange(0, 50)",
        "mu": "np.linspace(0.5, 10.0, 50)",
        "sigma": "np.linspace(0.2, 2.0, 50)",
        "nu": "np.linspace(0.1, 0.9, 50)",
        "p": "np.linspace(0.02, 0.98, 50)",
    },
    "ZIPIG": {
        "type": "discrete_count",
        "x": "np.arange(0, 50)",
        "mu": "np.linspace(0.5, 10.0, 50)",
        "sigma": "np.linspace(0.2, 2.0, 50)",
        "nu": "np.linspace(0.1, 0.9, 50)",
        "p": "np.linspace(0.02, 0.98, 50)",
    },
    "ZABB": {
        "type": "discrete_count",
        "x": "np.arange(0, 20)",  # binomial-like
        "mu": "np.linspace(0.1, 0.9, 20)",
        "sigma": "np.linspace(0.2, 2.0, 20)",
        "nu": "np.linspace(0.1, 0.9, 20)",
        "p": "np.linspace(0.02, 0.98, 20)",
    },
    "ZIBB": {
        "type": "discrete_count",
        "x": "np.arange(0, 20)",
        "mu": "np.linspace(0.1, 0.9, 20)",
        "sigma": "np.linspace(0.2, 2.0, 20)",
        "nu": "np.linspace(0.1, 0.9, 20)",
        "p": "np.linspace(0.02, 0.98, 20)",
    },
    "RGE": {
        "type": "continuous_positive",
        "x": "np.linspace(0.01, 10.0, 80)",
        "mu": "np.linspace(0.5, 5.0, 80)",
        "sigma": "np.linspace(0.2, 2.0, 80)",
        "nu": "np.linspace(0.1, 5.0, 80)",
        "p": "np.linspace(0.02, 0.98, 80)",
    },
    "PE2": {
        "type": "continuous_positive",
        "x": "np.linspace(0.01, 10.0, 80)",
        "mu": "np.linspace(0.5, 5.0, 80)",
        "sigma": "np.linspace(0.2, 2.0, 80)",
        "nu": "np.linspace(-2.0, 2.0, 80)",
        "p": "np.linspace(0.02, 0.98, 80)",
    },
    "LQNO": {
        "type": "continuous_real",
        "x": "np.linspace(-5.0, 5.0, 80)",
        "mu": "np.linspace(-2.0, 2.0, 80)",
        "sigma": "np.linspace(0.2, 2.0, 80)",
        "nu": "np.linspace(-2.0, 2.0, 80)",
        "p": "np.linspace(0.02, 0.98, 80)",
    },
    "DELAPORT": {
        "type": "discrete_count",
        "x": "np.arange(0, 50)",
        "mu": "np.linspace(0.5, 10.0, 50)",
        "sigma": "np.linspace(0.2, 2.0, 50)",
        "nu": "np.linspace(0.1, 5.0, 50)",
        "p": "np.linspace(0.02, 0.98, 50)",
    },
}

# Default ranges for families not explicitly configured
DEFAULT_RANGES = {
    "discrete_count": {
        "x": "np.arange(0, 50)",
        "mu": "np.linspace(0.5, 10.0, 50)",
        "sigma": "np.linspace(0.2, 2.0, 50)",
        "nu": "np.linspace(0.1, 0.9, 50)",
        "tau": "np.linspace(0.1, 5.0, 50)",
        "p": "np.linspace(0.02, 0.98, 50)",
    },
    "continuous_positive": {
        "x": "np.linspace(0.01, 10.0, 80)",
        "mu": "np.linspace(0.5, 5.0, 80)",
        "sigma": "np.linspace(0.2, 2.0, 80)",
        "nu": "np.linspace(-2.0, 2.0, 80)",
        "tau": "np.linspace(0.1, 5.0, 80)",
        "p": "np.linspace(0.02, 0.98, 80)",
    },
    "continuous_real": {
        "x": "np.linspace(-5.0, 5.0, 80)",
        "mu": "np.linspace(-2.0, 2.0, 80)",
        "sigma": "np.linspace(0.2, 2.0, 80)",
        "nu": "np.linspace(-2.0, 2.0, 80)",
        "tau": "np.linspace(0.1, 5.0, 80)",
        "p": "np.linspace(0.02, 0.98, 80)",
    },
    "bounded_01": {
        "x": "np.linspace(0.01, 0.99, 80)",
        "mu": "np.linspace(0.1, 0.9, 80)",
        "sigma": "np.linspace(0.2, 2.0, 80)",
        "nu": "np.linspace(0.1, 0.9, 80)",
        "tau": "np.linspace(0.1, 5.0, 80)",
        "p": "np.linspace(0.02, 0.98, 80)",
    },
}

def get_param_ranges(family: str, family_type: str = "continuous_positive") -> dict:
    """Get parameter ranges for a family.
    
    Args:
        family: Family name (e.g., "LG", "GAF")
        family_type: Type of family if not in FAMILY_PARAM_RANGES
        
    Returns:
        Dictionary with parameter range expressions
    """
    if family in FAMILY_PARAM_RANGES:
        return FAMILY_PARAM_RANGES[family]
    
    # Use default ranges based on type
    ranges = DEFAULT_RANGES.get(family_type, DEFAULT_RANGES["continuous_positive"]).copy()
    ranges["type"] = family_type
    return ranges
