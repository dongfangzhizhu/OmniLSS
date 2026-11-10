"""R gamlss.spatial neighbourhood helpers: polys2nb(), nb2nb(), nb2prec().

R source: R/gamlss.spatial/R/extraFunctions.R.

spdep stores an "nb" object as a positional list of neighbour index vectors
with a region.id attribute; gamlss wants a named list instead, which is what
nb2nb() builds.  nb2prec() then turns that named neighbourhood into the Gaussian
Markov random field precision matrix G: the diagonal holds the number of
neighbours and every neighbour pair gets -1.

``polys2nb`` (ported from Simon Wood's mgcv) builds the neighbourhood directly
from polygon boundary coordinates: two areas are neighbours when they share at
least one vertex.  A bounding-box prefilter (four overlap clauses per axis)
cuts the O(n^2) vertex comparisons; ``uniquecombs``-style row de-duplication is
used both to strip duplicate vertices and to count shared ones as
``nrow(rbind(a, b)) - nrow(unique_rows(rbind(a, b)))``.  Numeric output is
live-R verified in ``omnilss/tests/test_polys2nb.py``.

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


@dataclass(frozen=True)
class PolyNB:
    """R ``polys2nb`` result: 1-based neighbour indices + bounding box.

    ``neighbours[k]`` holds the 1-based indices (R convention) of the areas
    sharing a vertex with area ``k``; ``region_id`` carries ``names(polys)``.
    ``xlim``/``ylim`` are ``(min, max)`` of all polygon bounding boxes.
    """

    neighbours: tuple[tuple[int, ...], ...]
    region_id: tuple[str, ...]
    xlim: tuple[float, float]
    ylim: tuple[float, float]

    def __post_init__(self) -> None:
        if len(self.neighbours) != len(self.region_id):
            raise ValueError("region.id and the neighbour list have different lengths")


def _unique_rows(points: np.ndarray) -> np.ndarray:
    """mgcv ``uniquecombs``: de-duplicate 2-D rows, keep first-occurrence order.

    R's ``unique.matrix`` (which mgcv::uniquecombs wraps) preserves the order
    of first appearance.  ``np.unique(..., axis=0)`` sorts lexicographically,
    which changes row order -- we only need the *count* of unique rows for the
    shared-vertex test, but keep first-order to mirror R exactly.
    """
    points = np.asarray(points, dtype=np.float64)
    if points.size == 0:
        return points.reshape(0, 2)
    _, first = np.unique(points, axis=0, return_index=True)
    return points[np.sort(first)]


def polys2nb(polys: Any) -> PolyNB:
    """R ``gamlss.spatial::polys2nb``: vertex-sharing neighbourhood of polygons.

    ``polys`` is a mapping (region name -> Nx2 coordinate array, optional NA
    separator rows) or a sequence of such arrays.  NA rows are dropped, then
    each polygon's vertices are de-duplicated.  Two areas are neighbours when
    their bounding boxes overlap AND ``rbind`` of their (unique) vertices has
    fewer unique rows than total rows (i.e. at least one shared vertex).

    R quirk reproduced: neighbours are defined by **shared vertices** only, so
    a vertex lying on another polygon's edge segment is NOT detected (documented
    warning in the R source).  Neighbour indices are 1-based (R convention).
    """
    if isinstance(polys, Mapping):
        names = tuple(str(k) for k in polys)
        raw = list(polys.values())
    else:
        raw = list(polys)
        names = tuple(str(i + 1) for i in range(len(raw)))

    cleaned: list[np.ndarray] = []
    lo1 = np.empty(len(raw))
    hi1 = np.empty(len(raw))
    lo2 = np.empty(len(raw))
    hi2 = np.empty(len(raw))
    for i, poly in enumerate(raw):
        arr = np.asarray(poly, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] != 2:
            raise ValueError("each polygon must be an Nx2 coordinate array")
        # R: polys[[i]][!is.na(rowSums(polys[[i]])),] -- drop rows with any NA
        arr = arr[~np.isnan(arr).any(axis=1)]
        lo1[i], hi1[i] = arr[:, 0].min(), arr[:, 0].max()
        lo2[i], hi2[i] = arr[:, 1].min(), arr[:, 1].max()
        cleaned.append(_unique_rows(arr))

    n = len(raw)
    neighbours: list[tuple[int, ...]] = []
    for k in range(n):
        # R's four-way bbox overlap clauses, per axis
        ol1 = ((lo1[k] <= hi1) & (lo1[k] >= lo1)) | ((hi1[k] <= hi1) & (hi1[k] >= lo1)) \
            | ((lo1 <= hi1[k]) & (lo1 >= lo1[k])) | ((hi1 <= hi1[k]) & (hi1 >= lo1[k]))
        ol2 = ((lo2[k] <= hi2) & (lo2[k] >= lo2)) | ((hi2[k] <= hi2) & (hi2[k] >= lo2)) \
            | ((lo2 <= hi2[k]) & (lo2 >= lo2[k])) | ((hi2 <= hi2[k]) & (hi2 >= lo2[k]))
        overlap = ol1 & ol2
        overlap[k] = False
        candidates = np.nonzero(overlap)[0]
        nb: list[int] = []
        cok = cleaned[k]
        for j in candidates:
            co = np.vstack((cleaned[j], cok))
            n_shared = co.shape[0] - _unique_rows(co).shape[0]
            if n_shared > 0:
                nb.append(int(j) + 1)   # 1-based R index
        # R: unique(nb[[k]]) -- keep first-occurrence order
        seen: set[int] = set()
        uniq = [v for v in nb if not (v in seen or seen.add(v))]
        neighbours.append(tuple(uniq))

    return PolyNB(
        neighbours=tuple(neighbours),
        region_id=names,
        xlim=(float(lo1.min()), float(hi1.max())),
        ylim=(float(lo2.min()), float(hi2.max())),
    )


__all__ = ["NeighbourNB", "PolyNB", "nb2nb", "nb2prec", "polys2nb"]