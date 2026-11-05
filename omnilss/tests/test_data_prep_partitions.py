"""R gamlss.prepdata partitioning and renaming helpers.

Renaming / selection / name shortening are deterministic and compared directly
with the vendored R sources.  The partition helpers draw random numbers, so R's
sample() stream cannot be reproduced; these are checked structurally (shape,
partition levels, fold sums, in-bag/out-of-bag complementarity) and for shape
equivalence with R.
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
    data_boot_index,
    data_boot_weights,
    data_cut,
    data_Kfold_index,
    data_Kfold_weights,
    data_part,
    data_part_list,
    data_rename,
    data_select,
    data_shorter_names,
)

_REPO = Path(__file__).resolve().parents[2]

_RSCRIPT = """suppressPackageStartupMessages(library(gamlss))
source('R/gamlss.prepdata/R/data_dim.R')
source('R/gamlss.prepdata/R/data_part.R')
df <- data.frame(alpha = 1:8, beta = 11:18)
part <- data_part(df, partition = 2, setseed = 1)
kfold <- data_Kfold_weights(df, K = 4, setseed = 1)
boot <- data_boot_index(df, B = 3, setseed = 1)
out <- list(
  rename = names(data_rename(df, "alpha", "a")),
  select = names(data_select(df, "beta")),
  shorter = names(data_shorter_names(df, max = 3)),
  part_rows = as.integer(nrow(part)),
  part_levels = levels(part$partition),
  kfold_dim = as.integer(dim(kfold)),
  kfold_rowsum = as.integer(rowSums(kfold)),
  kfold_colsum = as.integer(colSums(kfold)),
  boot_len = as.integer(length(boot$IB)),
  ib_len = as.integer(length(boot$IB[[1]])),
  oob_len = as.integer(length(boot$OOB[[1]])))
cat(jsonlite::toJSON(out, auto_unbox = FALSE))
"""


def _frame() -> pd.DataFrame:
    return pd.DataFrame({"alpha": list(range(1, 9)), "beta": list(range(11, 19))})


def test_rename_keeps_positions_and_validates_lengths() -> None:
    renamed = data_rename(_frame(), "alpha", "a")
    assert list(renamed.columns) == ["a", "beta"]
    with pytest.raises(ValueError, match="same length"):
        data_rename(_frame(), ["alpha"], ["a", "b"])
    with pytest.raises(ValueError, match="oldnames should be a character"):
        data_rename(_frame(), [1], ["a"])


def test_select_keeps_only_requested_columns() -> None:
    assert list(data_select(_frame(), ["beta"]).columns) == ["beta"]
    assert list(data_select(_frame(), 2).columns) == ["beta"]


def test_shorter_names_truncates_long_names() -> None:
    assert list(data_shorter_names(_frame(), max=3, print=False).columns) == ["alp", "bet"]
    renamed = data_shorter_names(_frame(), newnames=["a", "b"], print=False)
    assert list(renamed.columns) == ["a", "b"]
    with pytest.raises(ValueError, match="same length"):
        data_shorter_names(_frame(), newnames=["a"], print=False)


def test_cut_subsamples_rows() -> None:
    out = data_cut(_frame(), 0.5, seed=7, print=False)
    assert out.shape == (4, 2)
    assert data_cut(_frame(), 1.0, print=False).shape == (8, 2)


def test_part_appends_a_partition_column() -> None:
    out = data_part(_frame(), 2, setseed=3, print=False)
    assert out.shape == (8, 3)
    assert list(out["partition"].cat.categories) == ["train", "test"]
    three = data_part(_frame(), 3, setseed=3, print=False)
    assert list(three["partition"].cat.categories) == ["train", "valid", "test"]


def test_part_validates_inputs() -> None:
    with pytest.raises(ValueError, match="greater that one"):
        data_part(_frame(), 1)
    with pytest.raises(ValueError, match="less or equal that 20"):
        data_part(_frame(), 25)
    with pytest.raises(ValueError, match="add up to 1"):
        data_part(_frame(), 2, probs=[0.5, 0.6])
    with pytest.raises(ValueError, match="length of probs should be 2"):
        data_part(_frame(), 2, probs=[1.0])


def test_part_list_splits_the_rows() -> None:
    parts = data_part_list(_frame(), 2, setseed=3, print=False)
    assert set(parts) == {"train", "test"}
    assert sum(part.shape[0] for part in parts.values()) == 8
    assert all(list(part.columns) == ["alpha", "beta"] for part in parts.values())


def test_kfold_weights_shape_and_masks() -> None:
    weights = data_Kfold_weights(_frame(), K=4, setseed=1)
    assert weights.shape == (8, 4)
    assert np.all(weights.sum(axis=1) == 3.0)
    assert set(np.unique(weights)) <= {0.0, 1.0}
    assert np.all(weights.sum(axis=0) == 6.0)


def test_kfold_index_zeroes_the_in_fold_rows() -> None:
    weights = data_Kfold_weights(_frame(), K=4, setseed=1)
    index = data_Kfold_index(_frame(), K=4, setseed=1)
    assert index.shape == (8, 4)
    rows = np.arange(1, 9)[:, None]
    np.testing.assert_allclose(index, weights * rows)


def test_kfold_is_reproducible_with_the_seed() -> None:
    first = data_Kfold_weights(_frame(), K=4, setseed=11)
    second = data_Kfold_weights(_frame(), K=4, setseed=11)
    np.testing.assert_allclose(first, second)
    assert not np.allclose(first, data_Kfold_weights(_frame(), K=4, setseed=12))


def test_boot_index_in_bag_and_out_of_bag() -> None:
    result = data_boot_index(_frame(), B=5, setseed=2)
    assert len(result["IB"]) == 5 and len(result["OOB"]) == 5
    universe = set(range(1, 9))
    for in_bag, out_bag in zip(result["IB"], result["OOB"]):
        assert len(in_bag) == 8
        assert np.all(np.diff(in_bag) >= 0)
        assert set(out_bag) == universe - set(int(value) for value in np.unique(in_bag))


def test_boot_weights_are_frequencies() -> None:
    indices = data_boot_index(_frame(), B=5, setseed=2)["IB"]
    weights = data_boot_weights(_frame(), B=5, setseed=2)
    assert weights.shape == (8, 5)
    for column, in_bag in enumerate(indices):
        counts = np.bincount(np.asarray(in_bag, dtype=int), minlength=9)[1:]
        np.testing.assert_allclose(weights[:, column], counts)
        assert weights[:, column].sum() == 8


@pytest.mark.slow
def test_matches_live_r_structure() -> None:
    if shutil.which("Rscript") is None:
        pytest.skip("Rscript not available")
    if not (_REPO / "R" / "gamlss.prepdata" / "R" / "data_part.R").exists():
        pytest.skip("vendored gamlss.prepdata not present")
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "part.R"
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

    assert list(data_rename(frame, "alpha", "a").columns) == list(ref["rename"])
    assert list(data_select(frame, "beta").columns) == list(ref["select"])
    assert list(data_shorter_names(frame, max=3, print=False).columns) == list(ref["shorter"])

    assert data_part(frame, 2, setseed=1, print=False).shape[0] == int(ref["part_rows"][0])
    assert list(data_part(frame, 2, setseed=1, print=False)["partition"].cat.categories) == list(
        ref["part_levels"]
    )

    weights = data_Kfold_weights(frame, K=4, setseed=1)
    assert list(weights.shape) == list(ref["kfold_dim"])
    assert int(weights.sum(axis=1)[0]) == int(ref["kfold_rowsum"][0])
    assert int(weights.sum(axis=0)[0]) == int(ref["kfold_colsum"][0])

    boot = data_boot_index(frame, B=3, setseed=1)
    assert len(boot["IB"]) == int(ref["boot_len"][0])
    assert len(boot["IB"][0]) == int(ref["ib_len"][0])
    assert len(boot["OOB"][0]) == int(ref["oob_len"][0])
