"""Week 2 CG building blocks: joint scoring matrix and stable solves."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

import jax.numpy as jnp


Array = jnp.ndarray


@dataclass(frozen=True)
class JointScoringResult:
    """Joint scoring matrix assembly result."""

    matrix: Array
    score: Array
    block_slices: dict[str, slice]


@dataclass(frozen=True)
class CGOuterStepResult:
    """Result for one CG outer step."""

    updated_eta: dict[str, Array]
    deltas: dict[str, Array]
    accepted_step_size: float
    old_global_deviance: float
    new_global_deviance: float


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
    """Build block joint matrix ``H`` and stacked score vector."""

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
    return {name: delta[sl] for name, sl in assembled.block_slices.items()}


def cg_outer_step(
    eta: Mapping[str, Array],
    scores: Mapping[str, Array],
    hessian_blocks: Mapping[tuple[str, str], Array],
    global_deviance_fn: Callable[[Mapping[str, Array]], float],
    ridge: float = 1e-8,
    step_size: float = 1.0,
    min_step_size: float = 1e-4,
    backtracking: float = 0.5,
) -> CGOuterStepResult:
    """Run one line-searched CG outer step.

    Uses joint Newton direction from ``solve_joint_system`` then performs
    backtracking line-search until global deviance decreases.
    """

    base_eta = {k: jnp.asarray(v) for k, v in eta.items()}
    deltas = solve_joint_system(scores=scores, hessian_blocks=hessian_blocks, ridge=ridge)

    old_dev = float(global_deviance_fn(base_eta))
    alpha = float(step_size)
    accepted_eta = base_eta
    accepted_dev = old_dev

    while alpha >= min_step_size:
        trial = {k: base_eta[k] + alpha * deltas[k] for k in base_eta}
        trial_dev = float(global_deviance_fn(trial))
        if trial_dev <= old_dev:
            accepted_eta = trial
            accepted_dev = trial_dev
            break
        alpha *= backtracking

    return CGOuterStepResult(
        updated_eta=accepted_eta,
        deltas=deltas,
        accepted_step_size=alpha,
        old_global_deviance=old_dev,
        new_global_deviance=accepted_dev,
    )
