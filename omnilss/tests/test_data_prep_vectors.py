"""R gamlss.prepdata helpers: y_distinct, y_factor, data_factor, high_val, low_val.

All five are deterministic and compared with the vendored R sources
(data_str.R, data_scale.R, data_cor.R) under live R.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from omnilss.data_prep import data_factor, high_val, low_val, y_distinct, y_factor

_REPO = Path(__file__).resolve().parents[2]

_RSCRIPT = """suppressPackageStartupMessages(library(gamlss))
source('R/gamlss.prepdata/R/data_str.R')
source('R/gamlss.prepdata/R/data_scale.R')
source('R/gamlss.prepdata/R/data_cor.R')
cat(jsonlite::toJSON(list(
  distinct_num = as.integer(length(table(c(1, 2, 2, 3, 3, 3)))),
  distinct_fac = as.integer(length(table(factor(c("a", "b", "a"), levels = c("a", "b", "c"))))),
  lower = levels(y_factor(factor(c("a", "a", "b", "c", "c", "c")), "lower")),
  higher = levels(y_factor(factor(c("a", "a", "b", "c", "c", "c")), "higher")),
  dfac = levels(data_factor(data.frame(g = factor(c("a", "a", "b", "c", "c", "c")), x = 1:6), "lower")$g),
  high = as.numeric(high_val(cbind(a = c(1, 0.95, 0.01), b = c(0.95, 1, 0.5), c = c(0.01, 0.5, 1)), 0.9, plot = FALSE)[, 3]),
  low = as.numeric(low_val(cbind(a = c(1, 0.95, 0.01), b = c(0.95, 1, 0.5), c = c(0.01, 0.5, 1)), 0.05, plot = FALSE)[, 3])
), auto_unbox = FALSE))
"""


def _corr() -> pd.DataFrame:
    return pd.DataFrame(
        [[1.0, 0.95, 0.01], [0.95, 1.0, 0.5], [0.01, 0.5, 1.0]],
        columns=["a", "b", "c"],
        index=["a", "b", "c"],
    )


def test_y_distinct_counts_values_and_levels() -> None:
    assert y_distinct([1, 2, 2, 3, 3, 3]) == 3
    assert y_distinct(pd.Series([1.0, np.nan, 2.0])) == 2
    # R counts factor levels, including unused ones.
    assert y_distinct(pd.Categorical(["a", "b", "a"], categories=["a", "b", "c"])) == 3


def test_y_factor_relevels_to_the_requested_frequency() -> None:
    series = pd.Series(pd.Categorical(["a", "a", "b", "c", "c", "c"]))
    assert list(y_factor(series, "lower").cat.categories) == ["b", "a", "c"]
    assert list(y_factor(series, "higher").cat.categories) == ["c", "a", "b"]


def test_y_factor_validates_input() -> None:
    with pytest.raises(ValueError, match="should be one of"):
        y_factor(pd.Series(pd.Categorical(["a", "b"])), "middle")
    with pytest.raises(ValueError, match="not a factor"):
        y_factor(pd.Series([1.0, 2.0]))
    with pytest.raises(ValueError, match="ordered"):
        y_factor(pd.Series(pd.Categorical(["a", "b"], ordered=True)))


def test_data_factor_relevels_every_unordered_factor() -> None:
    frame = pd.DataFrame(
        {
            "g": pd.Categorical(["a", "a", "b", "c", "c", "c"]),
            "h": pd.Categorical(["x", "y", "y", "x", "x", "x"], categories=["x", "y"]),
            "value": [1, 2, 3, 4, 5, 6],
        }
    )
    out = data_factor(frame, "higher")
    assert list(out["g"].cat.categories) == ["c", "a", "b"]
    assert out["value"].tolist() == [1, 2, 3, 4, 5, 6]


def test_high_val_returns_upper_triangle_pairs_above_the_threshold() -> None:
    pairs = high_val(_corr(), 0.9, print=False)
    assert pairs.to_dict("records") == [{"name1": "a", "name2": "b", "value": 0.95}]


def test_low_val_returns_upper_triangle_pairs_below_the_threshold() -> None:
    pairs = low_val(_corr(), 0.05, print=False)
    assert pairs.to_dict("records") == [{"name1": "a", "name2": "c", "value": 0.01}]


def test_high_and_low_val_validate_the_threshold() -> None:
    for bad in (0.0, 1.0, 1.5):
        with pytest.raises(ValueError, match="greater than  0 and lass than 1"):
            high_val(_corr(), bad, print=False)
        with pytest.raises(ValueError, match="greater than  0 and lass than 1"):
            low_val(_corr(), bad, print=False)


def test_diagonal_is_never_reported() -> None:
    # Upper triangle of the fixture is (a,b)=0.95, (a,c)=0.01, (b,c)=0.50.
    assert high_val(_corr(), 0.5, print=False).shape[0] == 1
    assert low_val(_corr(), 0.9, print=False).shape[0] == 2
    assert high_val(_corr(), 0.0 + 1e-6, print=False).shape[0] == 3


@pytest.mark.slow
def test_matches_live_r() -> None:
    if shutil.which("Rscript") is None:
        pytest.skip("Rscript not available")
    if not (_REPO / "R" / "gamlss.prepdata" / "R" / "data_cor.R").exists():
        pytest.skip("vendored gamlss.prepdata not present")
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "y.R"
        path.write_text(_RSCRIPT, encoding="utf-8")
        proc = subprocess.run(
            ["Rscript", str(path)],
            capture_output=True,
            text=True,
            timeout=300,
            cwd=str(_REPO),
        )
    if proc.returncode != 0:
        pytest.skip(f"R failed: {proc.stderr[-300:]}")
    ref = json.loads(proc.stdout.strip().splitlines()[-1])
    corr = _corr()
    series = pd.Series(pd.Categorical(["a", "a", "b", "c", "c", "c"]))

    def scalar(key: str) -> int:
        return int(np.asarray(ref[key]).ravel()[0])

    assert y_distinct([1, 2, 2, 3, 3, 3]) == scalar("distinct_num")
    assert (
        y_distinct(pd.Categorical(["a", "b", "a"], categories=["a", "b", "c"]))
        == scalar("distinct_fac")
    )
    assert list(y_factor(series, "lower").cat.categories) == list(ref["lower"])
    assert list(y_factor(series, "higher").cat.categories) == list(ref["higher"])
    frame = pd.DataFrame({"g": series, "x": [1, 2, 3, 4, 5, 6]})
    assert list(data_factor(frame, "lower")["g"].cat.categories) == list(ref["dfac"])
    assert high_val(corr, 0.9, print=False)["value"].tolist() == [
        float(value) for value in np.ravel(ref["high"])
    ]
    assert low_val(corr, 0.05, print=False)["value"].tolist() == [
        float(value) for value in np.ravel(ref["low"])
    ]
