"""Artifact locking and deterministic replay utilities."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Callable


@dataclass(frozen=True)
class ArtifactLock:
    runtime_config_hash: str
    solver_config_hash: str
    family_version_hash: str


@dataclass(frozen=True)
class ReplayArtifact:
    lock: ArtifactLock
    inputs: dict[str, Any]
    outputs: dict[str, Any]


def _stable_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_artifact_lock(
    runtime_config: dict[str, Any],
    solver_config: dict[str, Any],
    family_version: dict[str, Any],
) -> ArtifactLock:
    return ArtifactLock(
        runtime_config_hash=_stable_hash(runtime_config),
        solver_config_hash=_stable_hash(solver_config),
        family_version_hash=_stable_hash(family_version),
    )


def write_replay_artifact(path: str | Path, artifact: ReplayArtifact) -> None:
    payload = {
        "lock": asdict(artifact.lock),
        "inputs": artifact.inputs,
        "outputs": artifact.outputs,
    }
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def read_replay_artifact(path: str | Path) -> ReplayArtifact:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    lock = ArtifactLock(**payload["lock"])
    return ReplayArtifact(lock=lock, inputs=payload["inputs"], outputs=payload["outputs"])


def deterministic_replay(
    artifact: ReplayArtifact,
    runner: Callable[[dict[str, Any]], dict[str, Any]],
    *,
    atol: float = 1e-12,
) -> bool:
    replay_outputs = runner(dict(artifact.inputs))
    if replay_outputs.keys() != artifact.outputs.keys():
        return False

    for key, expected in artifact.outputs.items():
        actual = replay_outputs[key]
        if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
            if abs(float(expected) - float(actual)) > atol:
                return False
        else:
            if actual != expected:
                return False
    return True
