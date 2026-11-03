"""Regression tests for the registry inventory command line interface."""
import json
from pathlib import Path
import subprocess
import sys


# D2.160: the inventory generator parses the whole vendored gamlss.dist source
# tree through R, which measured 3m33s on this machine -- the 120 s cap turned a
# slow-but-correct run into a failure whenever the box was busy (it was the only
# remaining red in the last full sweep).  The cap is a hang guard now, not a
# performance assertion.
TOOL_TIMEOUT_S = 900

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "tools" / "generate_r_export_coverage.py"


def test_help_does_not_write_output(tmp_path):
    output = tmp_path / "inventory.json"
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--output", str(output), "--help"],
        capture_output=True, text=True, check=True, timeout=TOOL_TIMEOUT_S,
    )
    assert "--markdown" in result.stdout
    assert not output.exists()


def test_cli_generates_consistent_inventory(tmp_path):
    output = tmp_path / "nested" / "inventory.json"
    markdown = tmp_path / "summary.md"
    subprocess.run(
        [sys.executable, str(SCRIPT), "--output", str(output), "--markdown", str(markdown)],
        capture_output=True, text=True, check=True, timeout=TOOL_TIMEOUT_S,
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["stats"]["total_registered"] == len(payload["families"])
    assert sum(payload["stats"]["status_distribution"].values()) == len(payload["families"])
    text = markdown.read_text(encoding="utf-8")
    assert "does not establish R export coverage" in text
    for name, info in payload["families"].items():
        assert f"| {name} | {info.get('implementation_status', 'missing')} |" in text