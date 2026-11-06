"""R gamlss.spatial neighbourhood helpers: nb2nb() and nb2prec().

R source: R/gamlss.spatial/R/extraFunctions.R.

spdep stores an "nb" object as a positional list of neighbour index vectors
with a region.id attribute; gamlss wants a named list instead, which is what
nb2nb() builds.  nb2prec() then turns that named neighbourhood into the Gaussian
Markov random field precision matrix G: the diagonal holds the number of
neighbours and every neighbour pair gets -1.

G is returned as a pandas DataFrame with the region names on both axes, and is
already reordered to the covariate level order (R's G[levels(x), levels(x)]
block -- a no-op for the usual case, kept for parity).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class NeighbourNB:
    """spdep-style nb: positional adjacency plus the region ids."""

    neighbours: tuple[tuple[int, ...], ...]
    region_id: tuple[str, ...]

    def __post_init__(self) -> None:
        if len(self.neighbours) != len(self.region_id):
            raise ValueError("region.id and the neighbour list have different lengths")


def nb2nb(neighbour_nb: Any) -> dict[str, dict[str, tuple[int, ...]]]:
    """R nb2nb: name the positional neighbour list by region id."""

    if isinstance(neighbour_nb, NeighbourNB):
        neighbours = neighbour_nb.neighbours
        names = neighbour_nb.region_id
    else:
        neighbours = tuple(tuple(int(index) for index in row) for row in neighbour_nb)
        names = getattr(neighbour_nb, "region_id", None)
        if names is None:
            raise ValueError("the object is not a nb class")
    if len(names) != len(neighbours):
        raise ValueError("the object is not a nb class")
    return {"nb": {str(name): tuple(neighbours[index]) for index, name in enumerate(names)}}


def nb2prec(
    neighbour: Any,
    x: Any,
    area: Sequence[str] | None = None,
) -> pd.DataFrame:
    """R nb2prec: Gaussian Markov random field precision from a neighbourhood."""

    series = x if isinstance(x, pd.Series) else pd.Series(x)
    if not isinstance(series.dtype, pd.CategoricalDtype):
        raise ValueError("x must be a factor")
    data_levels = [str(level) for level in series.cat.categories]

    if area is None:
        knot_levels = list(data_levels)
    else:
        if not all(isinstance(level, (str, int, float)) for level in area):
            raise ValueError("area must be a factor or a chacacter vector")
        knot_levels = [str(level) for level in area]

    if len(data_levels) > len(knot_levels):
        raise ValueError("MRF basis dimension set too high")
    if any(level not in knot_levels for level in data_levels):
        raise ValueError("data contain regions that are not contained in the area specification")

    mapping = neighbour["nb"] if isinstance(neighbour, Mapping) else neighbour
    names = [str(name) for name in mapping]
    if sorted(names) != sorted(knot_levels):
        raise ValueError("mismatch between neighbour/polys supplied area names and data area names")

    size = len(knot_levels)
    position = {name: index for index, name in enumerate(knot_levels)}
    matrix = np.zeros((size, size))
    for name in names:
        neighbours = list(mapping[name])
        row = position[name]
        matrix[row, row] = float(len(neighbours))
        for neighbour_index in neighbours:
            column = position[names[int(neighbour_index) - 1]]
            matrix[row, column] = -1.0

    if np.any(matrix != matrix.T):
        raise ValueError("Something wrong with auto- penalty construction")

    if sorted(names) != names:
        order = [position[level] for level in data_levels]
        matrix = matrix[order, :][:, order]
        knot_levels = list(data_levels)

    return pd.DataFrame(matrix, index=knot_levels, columns=knot_levels)


__all__ = ["NeighbourNB", "nb2nb", "nb2prec"]