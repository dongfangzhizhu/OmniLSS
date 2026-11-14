"""P0-1: the offline devcontainer must build from a fresh (clean) clone.

The vendored R reference trees ``R/gamlss`` / ``R/gamlss.dist`` are gitignored
(``.gitignore: R/``) and therefore ABSENT in a fresh checkout.  An earlier
Dockerfile did ``COPY R/gamlss ...`` plus a ``test -f .../DESCRIPTION`` -- both
of which fail ``docker build`` from a clean clone.  P0-1 moves that optional
bundle to post-create (``.codex/setup.sh``, best-effort) and keeps the image's
R packages coming from the online ``install.packages`` + CRAN tarball cache.

These tests are purely static (no docker daemon required in CI) and pin the
invariants so the clean-clone build cannot regress:

* the Dockerfile does NOT ``COPY`` or ``test -f`` anything under ``R/``;
* ``R/`` is gitignored (so the above would be unsafe);
* R packages are installed via ``install.packages`` (not the vendored trees);
* ``.codex/setup.sh`` exists, is tracked, passes ``bash -n``, and is referenced
  by BOTH CPU and GPU ``postCreateCommand``;
* ``setup.sh`` bundles ``R/`` only when present and never fails when it is not.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DC = ROOT / ".devcontainer"
SETUP = ROOT / ".codex" / "setup.sh"


def _read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")


def test_setup_sh_is_tracked_and_syntax_valid():
    assert SETUP.exists(), ".codex/setup.sh is missing (devcontainer postCreate would fail)"
    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", str(SETUP.relative_to(ROOT).as_posix())],
        cwd=str(ROOT), capture_output=True, text=True,
    )
    assert tracked.returncode == 0, ".codex/setup.sh is not tracked by git (absent in a fresh clone)"
    if shutil.which("bash") is not None:
        proc = subprocess.run(
            ["bash", "-n", str(SETUP)], cwd=str(ROOT), capture_output=True, text=True,
        )
        assert proc.returncode == 0, f"setup.sh has a syntax error: {proc.stderr}"


def test_setup_sh_bundles_r_sources_only_when_present():
    text = SETUP.read_text(encoding="utf-8")
    # conditional on both vendored trees existing...
    assert 'R/gamlss"' in text and 'R/gamlss.dist"' in text
    assert "if [[ -d" in text
    # ...with an else branch that is explicitly non-fatal
    assert "not present" in text or "WARNING" in text
    assert "OMNILSS_R_SOURCE" in text
    # no unconditional COPY-like failure: cp is inside the conditional, guarded
    assert "cp -r" in text


def test_both_devcontainers_reference_tracked_setup_sh():
    for name in ("devcontainer.json", "devcontainer.gpu.json"):
        data = json.loads((DC / name).read_text(encoding="utf-8"))
        post = data.get("postCreateCommand", "")
        assert ".codex/setup.sh" in post, f"{name} postCreateCommand does not call setup.sh"
        assert "bash" in post, f"{name} should invoke setup.sh via bash (exec bit not guaranteed)"


def test_dockerfile_does_not_copy_gitignored_r_tree():
    dockerfile = _read(".devcontainer/Dockerfile")
    # The fatal clean-clone patterns must be gone.
    assert "COPY R/" not in dockerfile, "Dockerfile COPYs gitignored R/ -> fails on fresh clone"
    assert not any(
        "test -f" in line and "OMNILSS_R_SOURCE" in line and "DESCRIPTION" in line
        for line in dockerfile.splitlines()
    ), "Dockerfile still hard-asserts a vendored R DESCRIPTION file"
    # But the R_SOURCE dir is still prepared for the optional post-create bundle.
    assert "OMNILSS_R_SOURCE" in dockerfile
    assert 'mkdir -p "${OMNILSS_R_SOURCE}"' in dockerfile


def test_dockerfile_installs_r_packages_from_cran():
    dockerfile = _read(".devcontainer/Dockerfile")
    # authoritative offline source = online install + CRAN tarball cache
    assert "install.packages" in dockerfile
    assert "gamlss.dist" in dockerfile
    assert "OMNILSS_R_CACHE" in dockerfile
    assert "write_PACKAGES" in dockerfile


def test_r_directory_is_gitignored():
    gitignore = _read(".gitignore")
    lines = [l.strip() for l in gitignore.splitlines() if l.strip() and not l.startswith("#")]
    assert "R/" in lines, "R/ is expected to be gitignored (reference-only vendored sources)"


def test_devcontainer_configs_are_valid_json_with_required_env():
    for name in ("devcontainer.json", "devcontainer.gpu.json"):
        data = json.loads((DC / name).read_text(encoding="utf-8"))
        assert "build" in data and data["build"]["dockerfile"] == "Dockerfile"
        env = data.get("remoteEnv", {})
        for key in ("OMNILSS_WHEELHOUSE", "OMNILSS_R_CACHE", "OMNILSS_R_SOURCE", "OMNILSS_OFFLINE"):
            assert key in env, f"{name} remoteEnv missing {key}"
        assert data["remoteEnv"]["OMNILSS_OFFLINE"] == "1"
