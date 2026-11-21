"""R-aligned main fitting and control interfaces.

R source reference:
- file: `gamlss/R/gamlss-5.R`
- functions: `gamlss`, `gamlss.control`, `glim.control`, `gamlssNews`
"""

from __future__ import annotations

from pathlib import Path

from ._version import PACKAGE_VERSION
from .controls import GAMLSSControl, GLIMControl, gamlss_control, glim_control
from .fitting import gamlss

gamlss_control_exact = gamlss_control
glim_control_exact = glim_control


def gamlssNews(max_lines: int | None = None) -> str:
    """R reference: ``gamlss/R/gamlss-5.R::gamlssNews``.

    Return release news / changelog text for the installed OmniLSS version.
    """
    candidates = [
        Path(__file__).resolve().parents[3] / "CHANGELOG.md",
        Path(__file__).resolve().parents[2] / "CHANGELOG.md",
    ]
    for path in candidates:
        if path.is_file():
            text = path.read_text(encoding="utf-8", errors="replace")
            if max_lines is not None and max_lines > 0:
                return "\n".join(text.splitlines()[: int(max_lines)])
            return text
    return (
        f"OmniLSS {PACKAGE_VERSION} — Differentiable Distributional Modeling in JAX.\n"
        "See https://github.com/dongfangzhizhu/OmniLSS/blob/main/CHANGELOG.md for full release notes."
    )


gamlss_news = gamlssNews

__all__ = [
    "GAMLSSControl",
    "GLIMControl",
    "gamlss",
    "gamlssNews",
    "gamlss_news",
    "gamlss_control",
    "glim_control",
    "gamlss_control_exact",
    "glim_control_exact",
]
