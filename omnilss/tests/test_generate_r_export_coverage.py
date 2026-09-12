"""Regression tests for the registry inventory command line interface."""
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "tools" / "generate_r_export_coverage.py"


def test_help_does_not_write_output(tmp_path):
    output = tmp_path / "inventory.json"
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--output", str(output), "--help"],
        capture_output=True, text=True, check=True, timeout=120,
    )
    assert "--markdown" in result.stdout
    assert not output.exists()


def test_cli_generates_consistent_inventory(tmp_path):
    output = tmp_path / "nested" / "inventory.json"
    markdown = tmp_path / "summary.md"
    subprocess.run(
        [sys.executable, str(SCRIPT), "--output", str(output), "--markdown", str(markdown)],
        capture_output=True, text=True, check=True, timeout=120,
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["stats"]["total_registered"] == len(payload["families"])
    assert sum(payload["stats"]["status_distribution"].values()) == len(payload["families"])
    text = markdown.read_text(encoding="utf-8")
    assert "does not establish R export coverage" in text
    for name, info in payload["families"].items():
        assert f"| {name} | {info.get('implementation_status', 'missing')} |" in text