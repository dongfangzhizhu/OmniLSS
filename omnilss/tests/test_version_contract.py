import re
from pathlib import Path

import omnilss
from omnilss._version import PACKAGE_VERSION


def test_runtime_and_project_versions_stay_in_sync():
    pyproject = (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text(
        encoding="utf-8"
    )
    project_section = pyproject.split("[project]", 1)[1].split("\n[", 1)[0]
    match = re.search(r'^version\s*=\s*"([^"]+)"', project_section, re.MULTILINE)

    assert match is not None
    assert match.group(1) == PACKAGE_VERSION
    assert omnilss.__version__ == PACKAGE_VERSION
