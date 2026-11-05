"""R gamlss drop1All() / add1All() single-term tables.

Both are compared with live R on the same NO linear model.  For drop1All the
AIC/LRT/p-values agree to ~1e-13; for add1All R refits with what="All" (the
term is added to mu and sigma) and the two fitting engines differ by ~1e-5 in
the refitted deviance, which is the documented tolerance.
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

import omnilss
from omnilss.stepwise_all import add1All, drop1All

_REPO = Path(__file__).resolve().parents[2]

_RSCRIPT = """suppressPackageStartupMessages(library(gamlss))
library(jsonlite)
a <- commandArgs(trailingOnly = TRUE)
d <- fromJSON(a[1])
df <- data.frame(y = d$y, x1 = d$x1, x2 = d$x2, x3 = d$x3)
m <- gamlss(y ~ x1 + x2, family = NO, data = df, trace = FALSE)
dr <- drop1All(m, test = "Chisq")
ar <- add1All(m, scope = c("x3"), test = "Chisq")
pack <- function(t) list(rownames = rownames(t), colnames = colnames(t),
                         Df = as.numeric(t$Df), AIC = as.numeric(t$AIC),
                         LRT = as.numeric(t$LRT), P = as.numeric(t[["Pr(Chi)"]]))
cat("JSONSTART"); cat(toJSON(list(drop = pack(dr), add = pack(ar)), auto_unbox = FALSE, digits = 17))
"""


def _fixture():
    np.random.seed(0)
    n = 120
    x1 = np.linspace(-1.0, 1.0, n)
    x2 = np.random.normal(size=n)
    x3 = np.random.normal(size=n)
    y = 1.0 + 2.0 * x1 + 0.5 * x2 + np.random.normal(0.0, 0.4, n)
    return {"y": y, "x1": x1, "x2": x2, "x3": x3}


def _model(data):
    return omnilss.gamlss("y ~ x1 + x2", family="NO", data=data)


def test_drop1_all_table_contract() -> None:
    table = drop1All(_model(_fixture()))
    assert isinstance(table, pd.DataFrame)
    assert list(table.index) == ["<none>", "x1", "x2"]
    assert list(table.columns) == ["Df", "AIC", "LRT", "Pr(Chi)"]
    assert np.isnan(table.loc["<none>", "Df"])
    assert table.attrs["heading"][0] == "Single term deletions"


def test_drop1_all_df_and_aic_are_ordered() -> None:
    model = _model(_fixture())
    table = drop1All(model)
    assert table.loc["<none>", "AIC"] == pytest.approx(model.g_dev + 2.0 * model.df_fit)
    # Dropping a real covariate must never improve AIC on this fixture.
    assert table.loc["x1", "AIC"] > table.loc["<none>", "AIC"]
    assert table.loc["x2", "AIC"] > table.loc["<none>", "AIC"]
    assert (table.loc[["x1", "x2"], "Df"] > 0).all()


def test_drop1_all_test_none_has_two_columns() -> None:
    table = drop1All(_model(_fixture()), test="none")
    assert list(table.columns) == ["Df", "AIC"]


def test_drop1_all_sorted_orders_by_aic() -> None:
    table = drop1All(_model(_fixture()), sorted=True)
    values = table["AIC"].to_numpy(dtype=float)
    assert np.all(np.diff(values) >= -1e-12)


def test_add1_all_adds_to_every_parameter() -> None:
    data = _fixture()
    table = add1All(_model(data), ["x3"])
    assert list(table.index) == ["<none>", "x3"]
    assert list(table.columns) == ["Df", "AIC", "LRT", "Pr(Chi)"]
    # R's what="All" adds x3 to mu and sigma, so df grows by two for NO.
    assert table.loc["x3", "Df"] == pytest.approx(2.0)
    assert table.attrs["heading"][0] == "Single term additions for"


def test_add1_all_test_none_and_sorted() -> None:
    model = _model(_fixture())
    assert list(add1All(model, ["x3"], test="none").columns) == ["Df", "AIC"]
    assert len(add1All(model, ["x3"], sorted=True)) == 2


def test_add1_all_requires_a_scope() -> None:
    model = _model(_fixture())
    with pytest.raises(ValueError, match="no terms in scope"):
        add1All(model, None)
    with pytest.raises(ValueError, match="no terms in scope"):
        add1All(model, [])


def test_scope_accepts_formula_strings_and_lists() -> None:
    model = _model(_fixture())
    assert list(add1All(model, "~ x3").index) == ["<none>", "x3"]
    assert list(add1All(model, "x3 + x2").index) == ["<none>", "x3"]  # x2 already in the model


def test_invalid_test_argument() -> None:
    model = _model(_fixture())
    with pytest.raises(ValueError, match="should be one of"):
        drop1All(model, test="F")
    with pytest.raises(ValueError, match="should be one of"):
        add1All(model, ["x3"], test="F")


@pytest.mark.slow
def test_matches_live_r() -> None:
    if shutil.which("Rscript") is None:
        pytest.skip("Rscript not available")
    if not (_REPO / "R" / "gamlss" / "R" / "stepGAICAll-B-Parallel.R").exists():
        pytest.skip("vendored stepGAICAll-B-Parallel.R not present")
    data = _fixture()
    payload = {key: value.tolist() for key, value in data.items()}
    with tempfile.TemporaryDirectory() as tmp:
        data_path = Path(tmp) / "d.json"
        data_path.write_text(json.dumps(payload), encoding="utf-8")
        script_path = Path(tmp) / "s.R"
        script_path.write_text(_RSCRIPT, encoding="utf-8")
        proc = subprocess.run(
            ["Rscript", str(script_path), str(data_path)],
            capture_output=True,
            text=True,
            timeout=300,
            cwd=str(_REPO),
        )
    if "JSONSTART" not in proc.stdout:
        pytest.skip(f"R failed: {proc.stderr[-300:]}")
    ref = json.loads(proc.stdout.split("JSONSTART", 1)[1].strip())

    def numeric(values):
        return np.asarray([np.nan if value == "NA" else value for value in values], dtype=float)

    model = _model(data)
    tables = {"drop": drop1All(model), "add": add1All(model, ["x3"])}
    for name, table in tables.items():
        r = ref[name]
        assert list(table.index) == list(r["rownames"]), name
        assert list(table.columns) == list(r["colnames"]), name
        tolerance = 1e-8 if name == "drop" else 1e-4
        for column, key in (("Df", "Df"), ("AIC", "AIC"), ("LRT", "LRT"), ("Pr(Chi)", "P")):
            np.testing.assert_allclose(
                table[column].to_numpy(dtype=float),
                numeric(r[key]),
                rtol=tolerance,
                atol=tolerance,
                equal_nan=True,
                err_msg=f"{name} {column}",
            )
