from __future__ import annotations

import numpy as np

from omnilss.histSmo import histSmo, histSmoC, histSmoO, histSmoP


def test_histSmo_variants_return_consistent_shapes() -> None:
    y = np.array([0.1, 0.4, 0.8, 1.2, 1.5, 2.0], dtype=np.float64)
    a = histSmo(y, bins=4)
    b = histSmoO(y, bins=4)
    c = histSmoC(y, bins=4)
    d = histSmoP(y, bins=4)

    for out in (a, b, c, d):
        assert out.counts.shape == (4,)
        assert out.bin_edges.shape == (5,)
        assert out.density.shape == (4,)
