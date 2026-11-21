# SPDX-License-Identifier: GPL-3.0-or-later
"""Self-contained online / streaming GAMLSS estimation layer (P5-1c).

Built strictly on OmniLSS's own distribution contract
(:mod:`omnilss.ondil_contract`) with zero external dependencies on reference
repositories.
"""
from __future__ import annotations

from .estimator import OnlineGAMLSS
from .gram import OnlineGram
from .regularized import (
    OnlineRegularizedPath,
    RegularizedPathResult,
    online_coordinate_descent,
)

__all__ = [
    "OnlineGAMLSS",
    "OnlineGram",
    "OnlineRegularizedPath",
    "RegularizedPathResult",
    "online_coordinate_descent",
]
