"""Joint Fisher-information builders for multi-parameter GAMLSS models."""

from __future__ import annotations

from collections.abc import Mapping

import jax.numpy as jnp

Array = jnp.ndarray


def build_joint_fisher_from_blocks(
    block_map: Mapping[tuple[str, str], Array],
    slices: Mapping[str, slice],
    *,
    symmetrize: bool = True,
) -> Array:
    """Assemble a dense joint Fisher matrix from named parameter blocks."""
    total_size = max(s.stop for s in slices.values()) if slices else 0
    fisher = jnp.zeros((total_size, total_size), dtype=jnp.float64)

    for left, left_slice in slices.items():
        for right, right_slice in slices.items():
            block = block_map.get((left, right))
            if block is None:
                continue
            expected_shape = (
                left_slice.stop - left_slice.start,
                right_slice.stop - right_slice.start,
            )
            if block.shape != expected_shape:
                raise ValueError(
                    f"block {(left, right)} has shape {block.shape}, expected {expected_shape}"
                )
            fisher = fisher.at[left_slice, right_slice].set(block)

    if symmetrize and fisher.size:
        fisher = 0.5 * (fisher + fisher.T)
    return fisher
