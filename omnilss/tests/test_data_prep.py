"""R gamlss.prepdata data-shaping helpers.

Behaviour is compared with the vendored R sources
(R/gamlss.prepdata/R/data_dim.R, data_str.R, data_rm.R) sourced under live R.
R returns the data invisibly and prints summaries; the Python versions return
the shaped frame and the summary mapping.
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

from omnilss.data_prep import (
    data_dim,
    data_distinct,
    data_exclude_class,
    data_fac2num,
    data_int2num,
    data_names,
    data_na_obs,
    data_na_vars,
    data_omit,
    data_only_continuous,
    data_rm,
    data_rm1val,
    data_rmNAvars,
)

_REPO = Path(__file__).resolve().parents[2]

_RSCRIPT = """suppressPackageStartupMessages(library(gamlss))
source('R/gamlss.prepdata/R/data_dim.R')
source('R/gamlss.prepdata/R/data_str.R')
source('R/gamlss.prepdata/R/data_rm.R')
df <- data.frame(
  y = c(1, 2, NA, 4, 5),
  x = 1:5,
  g = factor(c("a", "b", "a", "b", "a")),
  k = rep(7L, 5))
out <- list(
  dims = as.integer(dim(df)),
  names = names(df),
  na_vars = as.integer(sapply(df, function(v) sum(is.na(v)))),
  na_obs_y = as.integer(which(is.na(df$y))),
  omit_rows = as.integer(nrow(data_omit(df))),
  distinct = as.integer(data_distinct(df, get.distinct = TRUE, print = FALSE)),
  rm1val_names = names(data_rm1val(df)),
  rm_names = names(data_rm(df, c("y", "k"))),
  excl_names = names(data_exclude_class(df)),
  cont_names = names(data_only_continuous(df)),
  fac2num = as.numeric(data_fac2num(df, "g")$g),
  fac2num_rows = as.integer(nrow(data_fac2num(df, "g"))),
  rmNA_names = names(data_rmNAvars(df)))
cat(jsonlite::toJSON(out, auto_unbox = FALSE))
"""


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "y": [1.0, 2.0, np.nan, 4.0, 5.0],
            "x": [1, 2, 3, 4, 5],
            "g": pd.Categorical(["a", "b", "a", "b", "a"]),
            "k": [7, 7, 7, 7, 7],
        }
    )


def test_rejects_non_frame_input() -> None:
    with pytest.raises(TypeError, match="data.frame"):
        data_dim([1, 2, 3], print=False)


def test_accepts_a_mapping_of_columns() -> None:
    summary = data_dim({"a": [1, 2, 3], "b": [4, 5, 6]}, print=False)
    assert summary["nrow"] == 3 and summary["ncol"] == 2


def test_data_dim_counts_missing_rows() -> None:
    summary = data_dim(_frame(), print=False)
    assert summary["nrow"] == 5
    assert summary["ncol"] == 4
    assert summary["rows_with_missing"] == 1
    assert summary["pct_missing_rows"] == pytest.approx(20.0)
    assert summary["na_per_column"]["y"] == 1


def test_data_names_reports_columns() -> None:
    assert data_names(_frame(), print=False) == ["y", "x", "g", "k"]


def test_data_na_vars_counts_per_column() -> None:
    assert data_na_vars(_frame(), print=False) == {"y": 1, "x": 0, "g": 0, "k": 0}


def test_data_na_obs_is_zero_based() -> None:
    assert data_na_obs(_frame(), print=False)["y"] == [2]


def test_data_omit_drops_rows_with_any_missing() -> None:
    out = data_omit(_frame(), print=False)
    assert out.shape[0] == 4
    assert not out.isna().any().any()


def test_data_distinct_counts_levels_and_values() -> None:
    counts = data_distinct(_frame(), get_distinct=True, print=False)
    assert counts == {"y": 4, "x": 5, "g": 2, "k": 1}


def test_data_distinct_warns_for_single_valued_columns() -> None:
    with pytest.warns(UserWarning, match="1 distinct value"):
        data_distinct(_frame(), get_distinct=True, print=False)


def test_data_distinct_returns_the_frame_by_default() -> None:
    out = data_distinct(_frame(), print=False)
    assert isinstance(out, pd.DataFrame)
    assert list(out.columns) == ["y", "x", "g", "k"]


def test_data_rm1val_drops_single_valued_columns() -> None:
    assert list(data_rm1val(_frame(), print=False).columns) == ["y", "x", "g"]


def test_data_rm_accepts_names_indices_and_scalars() -> None:
    assert list(data_rm(_frame(), ["y", "k"]).columns) == ["x", "g"]
    assert list(data_rm(_frame(), [1, 4]).columns) == ["x", "g"]
    assert list(data_rm(_frame(), "y").columns) == ["x", "g", "k"]


def test_data_exclude_class_and_only_continuous() -> None:
    assert list(data_exclude_class(_frame()).columns) == ["y", "x", "k"]
    assert list(data_only_continuous(_frame()).columns) == ["y", "x", "k"]


def test_data_fac2num_uses_one_based_codes_and_drops_na_rows() -> None:
    out = data_fac2num(_frame(), "g")
    assert out.shape[0] == 4
    assert out["g"].tolist() == [1.0, 2.0, 2.0, 1.0]


def test_data_fac2num_rejects_numeric_columns() -> None:
    with pytest.raises(ValueError, match="not character or factor"):
        data_fac2num(_frame(), "x")


def test_data_int2num_promotes_high_cardinality_columns() -> None:
    out = data_int2num(_frame(), 3, print=False)
    assert str(out["x"].dtype) == "float64"
    assert str(out["k"].dtype) == "int64"


def test_data_int2num_leaves_small_cardinality_alone() -> None:
    out = data_int2num(_frame(), 10, print=False)
    assert str(out["x"].dtype) == "int64"


def test_data_rmNAvars_drops_all_missing_columns() -> None:
    frame = pd.DataFrame({"a": [np.nan, np.nan], "b": [1.0, 2.0]})
    assert list(data_rmNAvars(frame).columns) == ["b"]


@pytest.mark.slow
def test_matches_live_r() -> None:
    if shutil.which("Rscript") is None:
        pytest.skip("Rscript not available")
    if not (_REPO / "R" / "gamlss.prepdata" / "R" / "data_dim.R").exists():
        pytest.skip("vendored gamlss.prepdata not present")
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "prep.R"
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
    frame = _frame()

    summary = data_dim(frame, print=False)
    assert [summary["nrow"], summary["ncol"]] == list(ref["dims"])
    assert data_names(frame, print=False) == list(ref["names"])
    assert list(data_na_vars(frame, print=False).values()) == list(ref["na_vars"])
    assert [index + 1 for index in data_na_obs(frame, print=False)["y"]] == list(ref["na_obs_y"])
    def scalar(key: str) -> int:
        return int(np.asarray(ref[key]).ravel()[0])

    assert data_omit(frame, print=False).shape[0] == scalar("omit_rows")
    assert list(data_distinct(frame, get_distinct=True, print=False).values()) == list(ref["distinct"])
    assert list(data_rm1val(frame, print=False).columns) == list(ref["rm1val_names"])
    assert list(data_rm(frame, ["y", "k"]).columns) == list(ref["rm_names"])
    assert list(data_exclude_class(frame).columns) == list(ref["excl_names"])
    assert list(data_only_continuous(frame).columns) == list(ref["cont_names"])
    assert data_fac2num(frame, "g")["g"].tolist() == list(ref["fac2num"])
    assert data_fac2num(frame, "g").shape[0] == scalar("fac2num_rows")
    assert list(data_rmNAvars(frame).columns) == list(ref["rmNA_names"])
