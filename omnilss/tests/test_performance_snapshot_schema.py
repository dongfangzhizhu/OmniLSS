import json
from pathlib import Path


def test_performance_snapshot_v0_3_0_schema_contract():
    snap_path = Path(__file__).parent / "fixtures" / "performance_snapshot_v0_3_0.json"
    payload = json.loads(snap_path.read_text())

    assert payload.get("version") == "v0.3.0"
    assert isinstance(payload.get("date"), str) and payload["date"]

    cases = payload.get("cases")
    assert isinstance(cases, dict) and cases

    for case_name, case in cases.items():
        assert isinstance(case_name, str) and case_name
        assert isinstance(case.get("family"), str) and case["family"] in {"NO", "PO", "GA"}
        assert isinstance(case.get("n"), int) and case["n"] > 0
        assert isinstance(case.get("formula"), str) and case["formula"]
        assert isinstance(case.get("sigma_formula"), str) and case["sigma_formula"]
        assert isinstance(case.get("max_seconds"), (int, float))
        assert float(case["max_seconds"]) > 0.0
