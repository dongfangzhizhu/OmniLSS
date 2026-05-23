"""Generic compatibility builder for pending `gen.Family` skeleton exports.

R reference: `gamlss.dist/R/genFamily.R` skeleton families (`count_*`,
`binom_*`, `contR_*`, `contRplus_*`, `contR01_*`).
"""

from __future__ import annotations

from .distributions import NO, PO
from .families import FamilyDefinition


def build_gen_family_skeleton(skeleton: str) -> FamilyDefinition:
    """Build a compatibility family for a named R `gen.Family` skeleton.

    Current phase-1 strategy keeps skeleton names addressable and routes them to
    stable baseline kernels while dedicated implementations are developed.
    """
    key = skeleton.strip()
    if key.startswith("count_") or key.startswith("binom_"):
        base = PO()
        return FamilyDefinition(
            name=key.upper(),
            parameters=base.parameters,
            g_dev_inc=base.g_dev_inc,
            type=base.type,
            links=base.links,
            link_functions=base.link_functions,
            link_inverses=base.link_inverses,
            link_derivatives=base.link_derivatives,
            score_functions=base.score_functions,
            hessian_functions=base.hessian_functions,
            fixed_parameters=base.fixed_parameters,
            d=base.d,
            p=base.p,
            q=base.q,
            r=base.r,
        )
    if key.startswith("contR"):
        base = NO()
        return FamilyDefinition(
            name=key.upper(),
            parameters=base.parameters,
            g_dev_inc=base.g_dev_inc,
            type=base.type,
            links=base.links,
            link_functions=base.link_functions,
            link_inverses=base.link_inverses,
            link_derivatives=base.link_derivatives,
            score_functions=base.score_functions,
            hessian_functions=base.hessian_functions,
            fixed_parameters=base.fixed_parameters,
            d=base.d,
            p=base.p,
            q=base.q,
            r=base.r,
        )
    raise ValueError(f"Unsupported skeleton: {skeleton}")
