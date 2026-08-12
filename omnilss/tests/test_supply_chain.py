"""Phase 5.6: supply-chain hardening.

- omnilss-server/requirements.txt pins every dependency exactly.
- Dockerfile runs as a non-root user and documents base-image digest pinning.
- docker-compose sets resource limits per service.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]


def test_requirements_exactly_pinned() -> None:
    """Every line in requirements.txt is an exact == pin (no ranges/bare names)."""
    text = (REPO / "omnilss-server" / "requirements.txt").read_text(encoding="utf-8")
    lines = [ln.strip() for ln in text.splitlines() if ln.strip() and not ln.startswith("#")]
    assert lines, "requirements.txt is empty"
    for line in lines:
        assert re.fullmatch(r"[A-Za-z0-9_.-]+==[0-9][A-Za-z0-9_.!+]*", line), (
            f"not an exact pin: {line!r}"
        )


def test_dockerfile_runs_as_non_root() -> None:
    """Dockerfile creates and switches to an unprivileged user."""
    text = (REPO / "omnilss-server" / "Dockerfile").read_text(encoding="utf-8")
    assert "USER appuser" in text
    assert "useradd" in text


def test_dockerfile_pins_base_image() -> None:
    """Dockerfile references a pinned base image (digest documented)."""
    text = (REPO / "omnilss-server" / "Dockerfile").read_text(encoding="utf-8")
    assert "python:3.11-slim" in text
    assert "@sha256" in text or "sha256:<digest>" in text


def test_compose_has_resource_limits() -> None:
    """Both compose services declare CPU/memory resource limits."""
    import yaml

    config = yaml.safe_load((REPO / "docker-compose.yml").read_text(encoding="utf-8"))
    for service in ("omnilss-core-grpc", "omnilss-core-http"):
        resources = config["services"][service]["deploy"]["resources"]["limits"]
        assert "cpus" in resources, f"{service} missing cpus limit"
        assert "memory" in resources, f"{service} missing memory limit"
