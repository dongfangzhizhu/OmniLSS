"""R gamlss.prepdata structure helpers: data_str, data_formulae, data_cha2fac, data_few2fac.

Compared with the vendored R sources (data_str.R / data_scale.R) under live R.
R prints its summary as a side effect and returns the data invisibly; the Python
port returns the summary mapping and takes a print flag.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from omnilss.data_prep import data_cha2fac, data_few2fac, data_formulae, data_str

_REPO = Path(__file__).resolve().parents[2]

_RSCRIPT = """suppressPackageStartupMessages(library(gamlss))
source('R/gamlss.prepdata/R/data_str.R')
source('R/gamlss.prepdata/R/data_scale.R')
df <- data.frame(a = 1:5, b = c("p", "q", "p", "q", "p"), c = c(1.5, 2.5, 3.5, 4.5, 5.5))
f <- data_formulae(df, "a")
c1 <- data_cha2fac(df)
c2 <- data_few2fac(df, max.levels = 2)
out <- list(
  main = as.character(f$main), inter = as.character(f$inter),
  nm = as.character(f$no_res_main), ni = as.character(f$no_res_inter),
  cha_classes = as.character(sapply(c1, function(x) class(x)[1])),
  cha_levels = levels(c1$b),
  few_classes = as.character(sapply(c2, function(x) class(x)[1])),
  few_levels_b = levels(c2$b))
cat("JSONSTART")
cat(jsonlite::toJSON(out, auto_unbox = FALSE))
"""

_KIND = {
    "int64": "integer",
    "float64": "numeric",
    "category": "factor",
    "object": "character",
}


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {"a": list(range(1, 6)), "b": ["p", "q", "p", "q", "p"], "c": [1.5, 2.5, 3.5, 4.5, 5.5]}
    )


def test_formulae_shapes() -> None:
    formulas = data_formulae(_frame(), "a")
    assert formulas == {
        "main": "a ~ b + c",
        "inter": "a ~ (b + c)^2",
        "no_res_main": "~ b + c",
        "no_res_inter": "~ (b + c)^2",
    }


def test_formulae_accepts_a_column_index_and_validates() -> None:
    assert data_formulae(_frame(), 1)["main"] == "a ~ b + c"
    with pytest.raises(ValueError, match="response should be set"):
        data_formulae(_frame(), "missing")


def test_cha2fac_converts_every_character_column() -> None:
    out = data_cha2fac(_frame(), print=False)
    kinds = [_KIND.get(str(dtype), str(dtype)) for dtype in out.dtypes]
    assert kinds == ["integer", "factor", "numeric"]
    assert list(out["b"].cat.categories) == ["p", "q"]


def test_cha2fac_reports_when_nothing_to_do() -> None:
    frame = pd.DataFrame({"a": [1, 2, 3]})
    out = data_cha2fac(frame, print=False)
    assert out.equals(frame)


def test_few2fac_converts_low_cardinality_columns() -> None:
    out = data_few2fac(_frame(), max_levels=2, print=False)
    kinds = [_KIND.get(str(dtype), str(dtype)) for dtype in out.dtypes]
    assert kinds == ["integer", "factor", "numeric"]
    assert list(out["b"].cat.categories) == ["p", "q"]


def test_few2fac_keeps_high_cardinality_columns() -> None:
    out = data_few2fac(_frame(), max_levels=4, print=False)
    assert pd.api.types.is_float_dtype(out["c"])


def test_data_str_reports_classes_and_suggestions() -> None:
    result = data_str(_frame(), print=False)
    summary = result["summary"]
    assert summary["class"].tolist() == ["integer", "character", "numeric"]
    assert summary["dist.values"].tolist() == [5, 2, 5]
    assert result["character_to_factor"] == ["b"]
    # Integer column a has 5 distinct values: below min.levels, above nothing.
    assert result["integer_to_factor"] == ["a"]
    assert result["numeric_to_factor"] == ["c"]
    assert result["integer_to_numeric"] == []


def test_data_str_integer_to_numeric_threshold() -> None:
    frame = pd.DataFrame({"a": list(range(1, 6))})
    result = data_str(frame, min_values=3, print=False)
    assert result["integer_to_numeric"] == ["a"]


@pytest.mark.slow
def test_matches_live_r() -> None:
    if shutil.which("Rscript") is None:
        pytest.skip("Rscript not available")
    if not (_REPO / "R" / "gamlss.prepdata" / "R" / "data_str.R").exists():
        pytest.skip("vendored gamlss.prepdata not present")
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "str.R"
        path.write_text(_RSCRIPT, encoding="utf-8")
        proc = subprocess.run(
            ["Rscript", str(path)], capture_output=True, text=True, timeout=300, cwd=str(_REPO)
        )
    if "JSONSTART" not in proc.stdout:
        pytest.skip(f"R failed: {proc.stderr[-300:]}")
    ref = json.loads(proc.stdout.split("JSONSTART", 1)[1].strip())

    frame = _frame()
    formulas = data_formulae(frame, "a")
    assert formulas["main"] == " ".join([ref["main"][1], ref["main"][0], ref["main"][2]])
    assert formulas["inter"] == " ".join([ref["inter"][1], ref["inter"][0], ref["inter"][2]])
    assert formulas["no_res_main"] == " ".join(ref["nm"])
    assert formulas["no_res_inter"] == " ".join(ref["ni"])

    converted = data_cha2fac(frame, print=False)
    assert [_KIND.get(str(dtype), str(dtype)) for dtype in converted.dtypes] == list(ref["cha_classes"])
    assert list(converted["b"].cat.categories) == list(ref["cha_levels"])

    few = data_few2fac(frame, max_levels=2, print=False)
    assert [_KIND.get(str(dtype), str(dtype)) for dtype in few.dtypes] == list(ref["few_classes"])
    assert list(few["b"].cat.categories) == list(ref["few_levels_b"])
