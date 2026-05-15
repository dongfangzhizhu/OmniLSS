"""Distribution protocol boundary for OmniLSS core."""

from .adapters import FamilyDistributionAdapter, as_distribution_protocol
from .protocol import (
    Data,
    DistributionProtocol,
    Params,
    REQUIRED_DISTRIBUTION_METHODS,
    RandomKey,
    assert_distribution_protocol,
)

__all__ = [
    "Data",
    "DistributionProtocol",
    "FamilyDistributionAdapter",
    "Params",
    "REQUIRED_DISTRIBUTION_METHODS",
    "RandomKey",
    "as_distribution_protocol",
    "assert_distribution_protocol",
]
