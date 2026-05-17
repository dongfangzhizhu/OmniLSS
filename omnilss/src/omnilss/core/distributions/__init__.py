"""Distribution protocol boundary for OmniLSS core.

Current compatibility note
--------------------------
Runtime fitting still centers on ``FamilyDefinition`` from ``omnilss.families``.
The helpers exported here define the architecture-freeze protocol surface and
allow adapting ``FamilyDefinition`` via ``FamilyDistributionAdapter`` while the
codebase incrementally converges on a single canonical interface.
"""

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
