"""R-aligned main fitting and control interfaces.

R source reference:
- file: `gamlss/R/gamlss-5.R`
- functions: `gamlss`, `gamlss.control`, `glim.control`
"""

from .controls import GAMLSSControl, GLIMControl, gamlss_control, glim_control
from .fitting import gamlss

gamlss_control_exact = gamlss_control
glim_control_exact = glim_control

__all__ = [
    "GAMLSSControl",
    "GLIMControl",
    "gamlss",
    "gamlss_control",
    "glim_control",
    "gamlss_control_exact",
    "glim_control_exact",
]
