"""R-aligned sequential multi-parameter GAIC interfaces.

R source reference:
- file: `gamlss/R/stepGAICAll-A-parallel.R`
- functions: `stepGAICAll.A`
"""

from .stepGAICAll_B_Parallel import (
    MultiParameterScopeResult,
    MultiParameterScopeRow,
    StepGAICAllResult,
    StepGAICAllStep,
    addterm_all,
    dropterm_all,
    step_gaic_all,
)

stepGAICAll_A = step_gaic_all

__all__ = [
    "MultiParameterScopeResult",
    "MultiParameterScopeRow",
    "StepGAICAllResult",
    "StepGAICAllStep",
    "addterm_all",
    "dropterm_all",
    "stepGAICAll_A",
    "step_gaic_all",
]
