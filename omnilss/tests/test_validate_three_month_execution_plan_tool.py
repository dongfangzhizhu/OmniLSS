import importlib.util
from pathlib import Path

TOOL_PATH = Path(__file__).resolve().parents[1] / "tools" / "validate_three_month_execution_plan.py"
SPEC = importlib.util.spec_from_file_location("validate_three_month_execution_plan", TOOL_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
SPEC.loader.exec_module(MODULE)
validate_three_month_checklist = MODULE.validate_three_month_checklist


def test_three_month_checklist_is_complete_and_ordered():
    report = validate_three_month_checklist()

    assert report["ok"] is True
    assert report["weeks_observed"] == list(range(1, 13))
    assert report["issues"] == []


def test_three_month_checklist_detects_sequence_gap(tmp_path: Path):
    checklist = tmp_path / "checklist.md"
    checklist.write_text(
        "\n".join(
            [
                "- [x] Week 1 / Task 1: done",
                "- [x] Week 3 / Task 1: done",
            ]
        ),
        encoding="utf-8",
    )

    report = validate_three_month_checklist(checklist)

    assert report["ok"] is False
    assert any(issue["code"] == "WEEK_SEQUENCE_MISMATCH" for issue in report["issues"])
