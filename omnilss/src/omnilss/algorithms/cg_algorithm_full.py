"""Week 2 CG building blocks: joint scoring matrix and stable solves."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import jax.numpy as jnp


Array = jnp.ndarray


@dataclass(frozen=True)
class JointScoringResult:
    """Joint scoring matrix assembly result."""

    matrix: Array
    score: Array
    block_slices: dict[str, slice]


def _stack_score_vector(scores: Mapping[str, Array]) -> tuple[Array, dict[str, slice]]:
    pieces = []
    slices: dict[str, slice] = {}
    cursor = 0
    for name, vec in scores.items():
        v = jnp.asarray(vec).reshape(-1)
        pieces.append(v)
        nxt = cursor + int(v.shape[0])
        slices[name] = slice(cursor, nxt)
        cursor = nxt
    return jnp.concatenate(pieces) if pieces else jnp.zeros((0,)), slices


def build_joint_scoring_matrix(
    scores: Mapping[str, Array],
    hessian_blocks: Mapping[tuple[str, str], Array],
    ridge: float = 1e-8,
) -> JointScoringResult:
    """Build block joint matrix ``H`` and stacked score vector.

    Parameters
    ----------
    scores:
        Score vectors keyed by parameter name.
    hessian_blocks:
        Observed Hessian blocks keyed by ``(row_param, col_param)``.
    ridge:
        Diagonal regularization added to the assembled matrix.
    """

    score_vec, block_slices = _stack_score_vector(scores)
    p_total = int(score_vec.shape[0])
    h = jnp.zeros((p_total, p_total), dtype=score_vec.dtype)

    for (pi, pj), block in hessian_blocks.items():
        if pi not in block_slices or pj not in block_slices:
            continue
        rs = block_slices[pi]
        cs = block_slices[pj]
        blk = jnp.asarray(block)
        expected = (rs.stop - rs.start, cs.stop - cs.start)
        if blk.shape != expected:
            raise ValueError(
                f"hessian block {(pi, pj)} has shape {blk.shape}, expected {expected}"
            )
        h = h.at[rs, cs].set(blk)

    h = 0.5 * (h + h.T)
    h = h + ridge * jnp.eye(p_total, dtype=h.dtype)
    return JointScoringResult(matrix=h, score=score_vec, block_slices=block_slices)


def solve_joint_system(
    scores: Mapping[str, Array],
    hessian_blocks: Mapping[tuple[str, str], Array],
    ridge: float = 1e-8,
) -> dict[str, Array]:
    """Solve ``H * delta = score`` and split ``delta`` by parameter blocks."""

    assembled = build_joint_scoring_matrix(scores, hessian_blocks, ridge=ridge)
    delta = jnp.linalg.solve(assembled.matrix, assembled.score)
    return {
        name: delta[sl]
        for name, sl in assembled.block_slices.items()
    }
