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


@dataclass(frozen=True)
class CGRunResult:
    """Result for iterative CG outer-loop execution."""

    eta: dict[str, Array]
    converged: bool
    n_iter: int
    deviance_history: tuple[float, ...]
    step_sizes: tuple[float, ...]
    termination_reason: str


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
    """Run one line-searched CG outer step."""

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

    if accepted_dev >= old_dev:
        alpha = 0.0

    return CGOuterStepResult(
        updated_eta=accepted_eta,
        deltas=deltas,
        accepted_step_size=alpha,
        old_global_deviance=old_dev,
        new_global_deviance=accepted_dev,
    )


def run_cg_outer_loop(
    eta0: Mapping[str, Array],
    build_scores_hessian_fn: Callable[[Mapping[str, Array]], tuple[dict[str, Array], dict[tuple[str, str], Array]]],
    global_deviance_fn: Callable[[Mapping[str, Array]], float],
    max_outer: int = 50,
    c_crit: float = 1e-6,
    ridge: float = 1e-8,
    step_size: float = 1.0,
) -> CGRunResult:
    """Run simplified full-CG outer loop with relative deviance convergence."""

    eta = {k: jnp.asarray(v) for k, v in eta0.items()}
    dev_hist: list[float] = [float(global_deviance_fn(eta))]
    step_hist: list[float] = []

    converged = False
    for outer in range(1, max_outer + 1):
        scores, hessian_blocks = build_scores_hessian_fn(eta)
        step = cg_outer_step(
            eta=eta,
            scores=scores,
            hessian_blocks=hessian_blocks,
            global_deviance_fn=global_deviance_fn,
            ridge=ridge,
            step_size=step_size,
        )
        eta = step.updated_eta
        step_hist.append(step.accepted_step_size)
        dev_hist.append(step.new_global_deviance)

        prev = abs(dev_hist[-2]) + 1e-12
        rel_change = abs(dev_hist[-2] - dev_hist[-1]) / prev
        if rel_change < c_crit:
            converged = True
            return CGRunResult(
                eta=eta,
                converged=converged,
                n_iter=outer,
                deviance_history=tuple(dev_hist),
                step_sizes=tuple(step_hist),
                termination_reason="relative_deviance_converged",
            )

        if step.accepted_step_size == 0.0:
            return CGRunResult(
                eta=eta,
                converged=False,
                n_iter=outer,
                deviance_history=tuple(dev_hist),
                step_sizes=tuple(step_hist),
                termination_reason="no_progress_step_rejected",
            )


    return CGRunResult(
        eta=eta,
        converged=converged,
        n_iter=max_outer,
        deviance_history=tuple(dev_hist),
        step_sizes=tuple(step_hist),
        termination_reason="max_outer_reached",
    )
