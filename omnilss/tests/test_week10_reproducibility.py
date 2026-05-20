from __future__ import annotations

from omnilss.validation.reproducibility import (
    ReplayArtifact,
    build_artifact_lock,
    deterministic_replay,
    read_replay_artifact,
    write_replay_artifact,
)


def test_artifact_lock_hashes_are_stable():
    lock1 = build_artifact_lock(
        runtime_config={"dtype": "float64", "seed": 0},
        solver_config={"method": "RS", "tol": 1e-6},
        family_version={"name": "NO", "version": "1"},
    )
    lock2 = build_artifact_lock(
        runtime_config={"seed": 0, "dtype": "float64"},
        solver_config={"tol": 1e-6, "method": "RS"},
        family_version={"version": "1", "name": "NO"},
    )
    assert lock1 == lock2


def test_replay_artifact_roundtrip_and_deterministic_replay(tmp_path):
    lock = build_artifact_lock(
        runtime_config={"dtype": "float64"},
        solver_config={"method": "RS"},
        family_version={"name": "NO", "version": "1"},
    )
    artifact = ReplayArtifact(lock=lock, inputs={"x": 2.0}, outputs={"y": 5.0})
    path = tmp_path / "artifact.json"
    write_replay_artifact(path, artifact)

    loaded = read_replay_artifact(path)
    assert loaded.lock == lock

    assert deterministic_replay(loaded, runner=lambda inp: {"y": inp["x"] + 3.0}) is True
    assert deterministic_replay(loaded, runner=lambda inp: {"y": inp["x"] + 4.0}) is False
