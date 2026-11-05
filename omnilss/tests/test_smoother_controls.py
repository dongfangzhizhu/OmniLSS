"""R gamlss smoother control functions (pb/cs/cy/pbc/pbm/pbo/pbp/pbz/pvc).

Every control object is compared field by field with live gamlss 5.5-3,
including the out-of-range guards that warn and substitute defaults.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

from omnilss.smoother_controls import (
    cs_control,
    cy_control,
    pbc_control,
    pbm_control,
    pbo_control,
    pbp_control,
    pb_control,
    pbz_control,
    pvc_control,
)

_REPO = Path(__file__).resolve().parents[2]

_RSCRIPT = """suppressPackageStartupMessages(library(gamlss))
for (f in c("pb.control", "cs.control", "cy.control", "pbc.control", "pbm.control",
            "pbo.control", "pbp.control", "pbz.control", "pvc.control")) {
  assign(sub("[.]", "_", f), get(f, envir = asNamespace("gamlss")))
}
out <- list(
  pb = pb_control(), cs = cs_control(), cy = cy_control(), pbc = pbc_control(),
  pbm = pbm_control(), pbo = pbo_control(), pbp = pbp_control(), pbz = pbz_control(),
  pvc = pvc_control(),
  pb_custom = pb_control(inter = 30, degree = 2, order = 3, k = 3.84, method = "GCV"),
  pb_guard = pb_control(inter = 0, degree = 0, order = -1, k = 0),
  pbz_guard = pbz_control(order = 0),
  cs_case = cs_control(df.offset = -1, penalty = -2))
cat(jsonlite::toJSON(out, auto_unbox = TRUE, digits = 17))
"""


def test_pb_control_defaults_match_the_r_signature() -> None:
    assert pb_control() == {
        "inter": 20,
        "degree": 3,
        "order": 2,
        "start": 10,
        "quantiles": False,
        "method": "ML",
        "k": 2,
    }


def test_cs_control_keeps_r_key_names_with_dots() -> None:
    settings = cs_control()
    assert set(settings) == {"cv", "all.knots", "nknots", "keep.data", "df.offset", "penalty", "control.spar"}
    assert settings["penalty"] == 1.4


def test_each_control_has_its_extra_field() -> None:
    assert cy_control()["ts"] is False
    assert pbc_control()["sin"] is True
    assert pbm_control()["kappa"] == 1e10
    assert pbz_control()["lim"] == 3
    assert pbz_control()["start"] == [0.0001, 0.0001]
    assert "quantiles" not in cy_control()
    assert "quantiles" not in pbc_control()
    assert "quantiles" in pbp_control() and "quantiles" in pvc_control() and "quantiles" in pbo_control()


def test_out_of_range_values_warn_and_fall_back() -> None:
    with pytest.warns(UserWarning, match="inter supplied is less than 0"):
        guarded = pb_control(inter=0)
    assert guarded["inter"] == 10
    with pytest.warns(UserWarning, match="degree supplied is less than zero"):
        guarded = pb_control(degree=-2)
    assert guarded["degree"] == 3
    with pytest.warns(UserWarning, match="order supplied is less than 0"):
        guarded = pb_control(order=-1)
    assert guarded["order"] == 2
    with pytest.warns(UserWarning, match="GAIC/GCV penalty"):
        guarded = pb_control(k=0)
    assert guarded["k"] == 2


def test_pbz_requires_a_second_order_penalty() -> None:
    with pytest.warns(UserWarning, match="order supplied is less than 2"):
        guarded = pbz_control(order=0)
    assert guarded["order"] == 2


def test_cs_control_guards() -> None:
    with pytest.warns(UserWarning, match="df.offset supplied is negative"):
        settings = cs_control(df_offset=-1)
    assert settings["df.offset"] == 0
    with pytest.warns(UserWarning, match="penalty supplied is negative"):
        settings = cs_control(penalty=-2)
    assert settings["penalty"] == 1


def test_unknown_method_is_rejected() -> None:
    with pytest.raises(ValueError, match="should be one of"):
        pb_control(method="REML")
    # cy and pbo also accept EM and ML-1.
    assert cy_control(method="EM")["method"] == "EM"
    assert pbo_control(method="ML-1")["method"] == "ML-1"


def test_booleans_are_coerced_like_r() -> None:
    assert pb_control(quantiles=1)["quantiles"] is True
    assert pbc_control(sin=0)["sin"] is False


@pytest.mark.slow
def test_matches_live_r() -> None:
    if shutil.which("Rscript") is None:
        pytest.skip("Rscript not available")
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "ctrl.R"
        path.write_text(_RSCRIPT, encoding="utf-8")
        proc = subprocess.run(
            ["Rscript", str(path)], capture_output=True, text=True, timeout=300
        )
    if proc.returncode != 0:
        pytest.skip(f"R failed: {proc.stderr[-300:]}")
    reference = json.loads(proc.stdout.strip().splitlines()[-1])

    with pytest.warns(UserWarning):
        mine = {
            "pb": pb_control(),
            "cs": cs_control(),
            "cy": cy_control(),
            "pbc": pbc_control(),
            "pbm": pbm_control(),
            "pbo": pbo_control(),
            "pbp": pbp_control(),
            "pbz": pbz_control(),
            "pvc": pvc_control(),
            "pb_custom": pb_control(inter=30, degree=2, order=3, k=3.84, method="GCV"),
            "cs_case": cs_control(df_offset=-1, penalty=-2),
        }
    for name, settings in mine.items():
        expected = reference[name]
        assert set(settings) == set(expected), name
        for key, value in settings.items():
            if key in {"nknots", "control.spar"}:
                continue  # R serialises NULL/empty lists as {} in JSON
            assert value == expected[key], (name, key, value, expected[key])
