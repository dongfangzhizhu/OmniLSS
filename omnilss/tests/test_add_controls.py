"""R gamlss.add control constructors (nn.control / ga.control).

nn.control() is a plain settings list; ga.control() nests mgcv::gam.control().
Both are compared with live R, including the two R quirks: nn.control's size
guard writes to "order" (so the value survives) and ga.control overwrites its
own "control" argument.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest

from omnilss.add_controls import ga_control, gam_control, nn_control

_REPO = Path(__file__).resolve().parents[2]

_RSCRIPT = """suppressPackageStartupMessages(library(gamlss))
suppressPackageStartupMessages(library(mgcv))
library(jsonlite)
for (f in c("nnet_gamlss.R", "ga_new.R")) try(source(file.path("R/gamlss.add/R", f)), silent = TRUE)
out <- list(
  nn = nn.control(),
  nn_guard = suppressWarnings(nn.control(size = -1, rang = 0, decay = -1, maxit = -1,
                                   MaxNWts = -1, abstol = -1, reltol = -1)),
  ga = ga.control(),
  ga_eps = ga.control(epsilon = 1e-5),
  ga_ctrl = ga.control(control = list(maxit = 50)))
cat("JSONSTART"); cat(toJSON(out, auto_unbox = TRUE, digits = 17, null = "null"))
"""


def test_nn_control_defaults() -> None:
    assert nn_control() == {
        "size": 3,
        "linout": True,
        "entropy": False,
        "softmax": False,
        "censored": False,
        "skip": False,
        "rang": 0.7,
        "decay": 0.0,
        "maxit": 100,
        "Hess": False,
        "trace": False,
        "MaxNWts": 1000,
        "abstol": 1e-4,
        "reltol": 1e-8,
    }


def test_nn_control_guards_and_the_size_no_op() -> None:
    with pytest.warns(UserWarning, match="value of rang supplied"):
        guarded = nn_control(rang=0.0)
    assert guarded["rang"] == 0.7
    with pytest.warns(UserWarning, match="value of decay supplied"):
        assert nn_control(decay=-1.0)["decay"] == 0.0
    with pytest.warns(UserWarning, match="value of maxit supplied"):
        assert nn_control(maxit=-1)["maxit"] == 100
    with pytest.warns(UserWarning, match="value of MaxNWts supplied"):
        assert nn_control(MaxNWts=-1)["MaxNWts"] == 1000
    with pytest.warns(UserWarning, match="value of abstol supplied"):
        assert nn_control(abstol=-1.0)["abstol"] == 1e-4
    with pytest.warns(UserWarning, match="value of reltol supplied"):
        assert nn_control(reltol=-1.0)["reltol"] == 1e-8
    # R assigns to "order" here, so a negative size is only warned about.
    with pytest.warns(UserWarning, match="value of size supplied"):
        assert nn_control(size=-1)["size"] == -1


def test_ga_control_structure_and_gam_control_defaults() -> None:
    settings = ga_control()
    assert settings["method"] == "REML"
    assert settings["optimizer"] == ["outer", "newton"]
    assert settings["gamma"] == 1.0
    assert settings["drop.unused.levels"] is True
    inner = settings["control"]
    assert inner["epsilon"] == pytest.approx(1e-7)
    assert inner["maxit"] == 200
    assert inner["rank.tol"] == pytest.approx(np.sqrt(np.finfo(np.float64).eps), rel=1e-15)
    assert inner["nlm"] == {
        "ndigit": 7,
        "gradtol": 1e-6,
        "stepmax": 2.0,
        "steptol": 1e-4,
        "iterlim": 200,
        "check.analyticals": False,
    }
    assert inner["optim"] == {"factr": 1e7}
    assert inner["newton"]["conv.tol"] == pytest.approx(1e-6)


def test_ga_control_kwargs_reach_both_levels() -> None:
    settings = ga_control(epsilon=1e-5)
    assert settings["control"]["epsilon"] == pytest.approx(1e-5)
    assert settings["epsilon"] == pytest.approx(1e-5)


def test_ga_control_ignores_its_control_argument() -> None:
    # R overwrites the formal with gam.control(...) inside the body.
    settings = ga_control(control={"maxit": 50})
    assert settings["control"]["maxit"] == 200


def test_gam_control_merges_the_sub_lists() -> None:
    inner = gam_control(nlm={"iterlim": 50}, newton={"maxHalf": 10}, optim={"factr": 1e6})
    assert inner["nlm"]["iterlim"] == 50
    assert inner["nlm"]["ndigit"] == 7
    assert inner["newton"]["maxHalf"] == 10
    assert inner["optim"]["factr"] == 1e6


@pytest.mark.slow
def test_matches_live_r() -> None:
    if shutil.which("Rscript") is None:
        pytest.skip("Rscript not available")
    if not (_REPO / "R" / "gamlss.add" / "R" / "ga_new.R").exists():
        pytest.skip("vendored gamlss.add not present")
    with tempfile.TemporaryDirectory() as tmp:
        script_path = Path(tmp) / "ctrl.R"
        script_path.write_text(_RSCRIPT, encoding="utf-8")
        proc = subprocess.run(
            ["Rscript", str(script_path)],
            capture_output=True,
            text=True,
            timeout=600,
            cwd=str(_REPO),
        )
    if "JSONSTART" not in proc.stdout:
        pytest.skip(f"R failed: {proc.stderr[-300:]}")
    ref = json.loads(proc.stdout.split("JSONSTART", 1)[1].strip())

    assert set(nn_control()) == set(ref["nn"])
    for key, value in nn_control().items():
        assert value == ref["nn"][key], key

    with pytest.warns(UserWarning):
        guarded = nn_control(size=-1, rang=0, decay=-1, maxit=-1, MaxNWts=-1, abstol=-1, reltol=-1)
    assert guarded == ref["nn_guard"]

    settings = ga_control()
    assert set(settings) == set(ref["ga"])
    assert settings["optimizer"] == list(ref["ga"]["optimizer"])
    assert settings["control"]["epsilon"] == pytest.approx(float(ref["ga"]["control"]["epsilon"]))
    assert settings["control"]["maxit"] == ref["ga"]["control"]["maxit"]
    assert settings["control"]["rank.tol"] == pytest.approx(
        float(ref["ga"]["control"]["rank.tol"]), rel=1e-15
    )
    assert settings["control"]["nlm"]["iterlim"] == ref["ga"]["control"]["nlm"]["iterlim"]
    assert settings["control"]["newton"]["maxHalf"] == ref["ga"]["control"]["newton"]["maxHalf"]

    overridden = ga_control(epsilon=1e-5)
    assert overridden["control"]["epsilon"] == pytest.approx(float(ref["ga_eps"]["control"]["epsilon"]))
    assert overridden["epsilon"] == pytest.approx(float(ref["ga_eps"]["epsilon"]))

    ignored = ga_control(control={"maxit": 50})
    assert ignored["control"]["maxit"] == ref["ga_ctrl"]["control"]["maxit"]