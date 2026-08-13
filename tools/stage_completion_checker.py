"""Stage completion checker with Chinese-commit integrity validation.

Implements OmniLSS DEVELOPMENT_PLAN Phase 0 task 0.5, aligning with
``.kiro/specs/r-gamlss-migration-plan/tasks.md`` 0.5, Requirements 7.1-7.4 and
Design Property 14 ("Phase completion records only verified Chinese commits").

Functionality
-------------
1. **Persistence** — a :class:`CompletionRecordStore` persists per-task
   completion records (verification summary, artifact paths, commit ID) as
   JSON.  A task is only recorded as ``completed`` when every prerequisite
   (implementation, tests, evidence and a valid Chinese task commit) passes;
   otherwise the record is stored as ``incomplete`` with the failing reasons.
2. **Commit validation** — :func:`validate_task_completion` verifies that a
   recorded commit identifier:
     - resolves to an existing commit in the repository (``commit_valid``);
     - has a Chinese subject (``subject_has_chinese``);
     - has a subject that identifies the task, i.e. carries a ``任务X.Y`` /
       ``task X.Y`` marker matching ``task_id`` (``subject_identifies_task``);
     - contains implementation files (``contains_implementation``);
     - contains test files (``contains_tests``);
     - contains generated evidence artifacts (``contains_evidence``).
3. **No auto-completion** — recording a completion re-validates the commit;
   failing prerequisites keep the task incomplete (Requirement 7.3).

The git access is abstracted behind :class:`CommitInspector` so property
tests can exercise the validation logic with in-memory commits, while
:class:`GitCommitInspector` provides the real ``git`` implementation.

Usage
-----
::

    python tools/stage_completion_checker.py check --task 0.5 --commit <sha>
    python tools/stage_completion_checker.py record --task 0.5 --commit <sha> \
        --store completion_records.json --summary "..." --artifact path/to/evidence.json
    python tools/stage_completion_checker.py phase --phase 0 --store completion_records.json
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterable, Mapping, Optional, Protocol

__all__ = [
    "CommitInfo",
    "CommitInspector",
    "CompletionRecord",
    "CompletionRecordStore",
    "FakeCommitInspector",
    "GitCommitInspector",
    "ValidationResult",
    "contains_chinese",
    "extract_task_ids",
    "is_evidence_file",
    "is_implementation_file",
    "is_test_file",
    "validate_task_completion",
    "check_phase_records",
]

STATUS_COMPLETED = "completed"
STATUS_INCOMPLETE = "incomplete"

#: Matches a task marker such as "任务0.5" / "任务8.6" / "task 1.2".
#: The mandatory "任务"/"task" prefix prevents false positives from plain
#: version numbers like "0.4.0" appearing in a subject.
_TASK_ID_RE = re.compile(r"(?:任务|task\s*)(\d+(?:\.\d+)+)", re.IGNORECASE)
_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def contains_chinese(text: str) -> bool:
    """Return True when *text* contains at least one CJK ideograph."""
    return bool(_CJK_RE.search(text or ""))


def extract_task_ids(subject: str) -> list[str]:
    """Return the ordered, de-duplicated task identifiers found in *subject*.

    Only ``任务X.Y`` / ``task X.Y`` markers are recognised so that bare
    version numbers (e.g. ``0.4.0``) are never mistaken for task IDs.
    """
    seen: list[str] = []
    for match in _TASK_ID_RE.findall(subject or ""):
        if match not in seen:
            seen.append(match)
    return seen


def is_implementation_file(path: str) -> bool:
    """True for a non-test ``*.py`` source file."""
    p = path.replace("\\", "/")
    if is_test_file(p):
        return False
    return p.endswith(".py")


def is_test_file(path: str) -> bool:
    """True for a test module (``test_*.py`` / ``*_test.py`` / ``*.py`` under tests/).

    Non-Python files under a ``tests/`` directory (fixtures, golden artifacts)
    are not test modules.
    """
    p = path.replace("\\", "/")
    name = p.rsplit("/", 1)[-1]
    if name.startswith("test_") or name.endswith("_test.py"):
        return True
    return p.endswith(".py") and ("/tests/" in f"/{p}" or p.startswith("tests/"))


def is_evidence_file(path: str) -> bool:
    """True for generated evidence artifacts.

    Evidence artifacts are the R golden fixtures/manifests (``fixtures/``,
    ``golden``, ``regenerated/``, ``manifest``), benchmark results
    (``/results/``, ``/benchmarks/``) and validation/audit reports
    (``/reports/``, ``/validation/``) that back "consistent with R" claims.
    """
    p = path.replace("\\", "/").lower()
    if "/benchmarks/" in p or p.startswith("benchmarks/"):
        return True
    markers = (
        "fixtures/",
        "golden",
        "regenerated/",
        "manifest",
        "/results/",
        "/reports/",
        "/validation/",
    )
    return any(marker in p for marker in markers)


# ---------------------------------------------------------------------------
# Commit model and inspectors
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CommitInfo:
    commit_id: str
    subject: str
    changed_files: tuple[str, ...] = ()


class CommitInspector(Protocol):
    """Resolves a commit identifier to commit metadata, or None if invalid."""

    def resolve(self, commit_id: str) -> Optional[CommitInfo]:  # pragma: no cover
        ...


class GitCommitInspector:
    """Real ``git`` backed inspector."""

    def __init__(self, repo: Optional[str | Path] = None) -> None:
        self._repo = repo

    def _git(self, *args: str) -> subprocess.CompletedProcess:
        cmd = ["git"]
        if self._repo is not None:
            cmd += ["-C", str(self._repo)]
        cmd += list(args)
        return subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )

    def resolve(self, commit_id: str) -> Optional[CommitInfo]:
        full = self._git("rev-parse", "--verify", "--quiet", f"{commit_id}^{{commit}}")
        sha = full.stdout.strip()
        if full.returncode != 0 or not sha:
            return None
        subject = self._git("log", "-1", "--format=%s", sha)
        if subject.returncode != 0:
            return None
        files = self._git("diff-tree", "--no-commit-id", "--name-only", "-r", "--root", sha)
        if files.returncode != 0:
            return None
        changed = tuple(line.strip() for line in files.stdout.splitlines() if line.strip())
        return CommitInfo(commit_id=sha, subject=subject.stdout.strip(), changed_files=changed)


class FakeCommitInspector:
    """In-memory inspector for tests and property-based validation."""

    def __init__(self, commits: Mapping[str, CommitInfo]) -> None:
        self._commits = dict(commits)

    def resolve(self, commit_id: str) -> Optional[CommitInfo]:
        return self._commits.get(commit_id)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
@dataclass
class ValidationResult:
    task_id: str
    commit_id: str
    ok: bool
    checks: dict[str, bool]
    reasons: list[str]
    commit: Optional[CommitInfo] = None


def validate_task_completion(
    task_id: str,
    commit_id: str,
    inspector: CommitInspector,
    *,
    require_implementation: bool = True,
    require_tests: bool = True,
    require_evidence: bool = True,
    required_artifact_paths: Iterable[str] = (),
) -> ValidationResult:
    """Validate that *commit_id* is an acceptable completion commit for *task_id*.

    When ``required_artifact_paths`` is non-empty, evidence presence is judged
    against those explicit artifact paths (every one must be in the commit);
    otherwise evidence files are auto-detected with :func:`is_evidence_file`.
    """
    checks: dict[str, bool] = {}
    reasons: list[str] = []
    commit = inspector.resolve(commit_id)

    checks["commit_valid"] = commit is not None
    if commit is None:
        reasons.append(
            f"commit {commit_id!r} 无法解析（无效 commit ID 或非 git 仓库）"
        )
        for name in (
            "subject_has_chinese",
            "subject_identifies_task",
            "contains_implementation",
            "contains_tests",
            "contains_evidence",
        ):
            checks[name] = False
        return ValidationResult(task_id, commit_id, False, checks, reasons)

    checks["subject_has_chinese"] = contains_chinese(commit.subject)
    if not checks["subject_has_chinese"]:
        reasons.append(f"提交主题 {commit.subject!r} 不含中文")

    detected = extract_task_ids(commit.subject)
    checks["subject_identifies_task"] = task_id in detected
    if not checks["subject_identifies_task"]:
        reasons.append(
            f"提交主题 {commit.subject!r} 未标识任务 {task_id}"
            f"（识别到任务: {detected or '无'}）"
        )

    files = commit.changed_files
    implementations = [f for f in files if is_implementation_file(f)]
    tests = [f for f in files if is_test_file(f)]

    checks["contains_implementation"] = (not require_implementation) or bool(implementations)
    if require_implementation and not implementations:
        reasons.append("提交未包含实现文件（*.py 且非测试）")

    checks["contains_tests"] = (not require_tests) or bool(tests)
    if require_tests and not tests:
        reasons.append("提交未包含测试文件（test_*.py / tests/ 下）")

    explicit = list(required_artifact_paths)
    if explicit:
        normalized_files = {f.replace("\\", "/") for f in files}
        missing = [a for a in explicit if a.replace("\\", "/") not in normalized_files]
        checks["contains_evidence"] = (not require_evidence) or not missing
        if require_evidence and missing:
            reasons.append(f"提交缺少声明的证据 artifact: {missing}")
    else:
        evidence = [f for f in files if is_evidence_file(f)]
        checks["contains_evidence"] = (not require_evidence) or bool(evidence)
        if require_evidence and not evidence:
            reasons.append("提交未包含证据/artifact 文件（golden/fixtures/benchmarks/reports）")

    return ValidationResult(task_id, commit_id, all(checks.values()), checks, reasons, commit)


# ---------------------------------------------------------------------------
# Completion records
# ---------------------------------------------------------------------------
@dataclass
class CompletionRecord:
    task_id: str
    commit_id: str
    subject: str
    status: str
    verification_summary: str = ""
    artifact_paths: tuple[str, ...] = ()
    impl_paths: tuple[str, ...] = ()
    test_paths: tuple[str, ...] = ()
    reasons: tuple[str, ...] = ()
    validated_at: str = ""

    @classmethod
    def from_validation(
        cls,
        result: ValidationResult,
        *,
        verification_summary: str = "",
        artifact_paths: Iterable[str] = (),
    ) -> "CompletionRecord":
        """Build a record from a validation result.

        Only an ``ok`` result is stored as ``completed``; anything else stays
        ``incomplete`` with the failing reasons (Requirement 7.3 - no
        auto-completion without a verified Chinese task commit).
        """
        commit = result.commit
        subject = commit.subject if commit is not None else ""
        impl_paths = (
            tuple(p for p in commit.changed_files if is_implementation_file(p))
            if commit is not None
            else ()
        )
        test_paths = (
            tuple(p for p in commit.changed_files if is_test_file(p))
            if commit is not None
            else ()
        )
        return cls(
            task_id=result.task_id,
            commit_id=result.commit_id,
            subject=subject,
            status=STATUS_COMPLETED if result.ok else STATUS_INCOMPLETE,
            verification_summary=verification_summary,
            artifact_paths=tuple(artifact_paths),
            impl_paths=impl_paths,
            test_paths=test_paths,
            reasons=tuple(result.reasons),
            validated_at=datetime.now().isoformat(timespec="seconds"),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "task_id": self.task_id,
            "commit_id": self.commit_id,
            "subject": self.subject,
            "status": self.status,
            "verification_summary": self.verification_summary,
            "artifact_paths": list(self.artifact_paths),
            "impl_paths": list(self.impl_paths),
            "test_paths": list(self.test_paths),
            "reasons": list(self.reasons),
            "validated_at": self.validated_at,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "CompletionRecord":
        return cls(
            task_id=str(data["task_id"]),
            commit_id=str(data["commit_id"]),
            subject=str(data.get("subject", "")),
            status=str(data["status"]),
            verification_summary=str(data.get("verification_summary", "")),
            artifact_paths=tuple(str(p) for p in data.get("artifact_paths", ())),
            impl_paths=tuple(str(p) for p in data.get("impl_paths", ())),
            test_paths=tuple(str(p) for p in data.get("test_paths", ())),
            reasons=tuple(str(r) for r in data.get("reasons", ())),
            validated_at=str(data.get("validated_at", "")),
        )


@dataclass
class PhaseCheckResult:
    phase_id: str
    records: dict[str, CompletionRecord]
    passed: bool
    missing_tasks: tuple[str, ...] = ()

    @property
    def incomplete_tasks(self) -> list[str]:
        return [tid for tid, rec in self.records.items() if rec.status != STATUS_COMPLETED]


def check_phase_records(
    records: Mapping[str, CompletionRecord],
    phase_id: str,
    expected_task_ids: Iterable[str] | None = None,
) -> PhaseCheckResult:
    """Check the completion records of a phase.

    Without *expected_task_ids* only the *recorded* tasks are verified to be
    completed.  With *expected_task_ids* the check additionally requires that
    every expected task of the phase has a completion record (integrity).
    """
    prefix = f"{phase_id}."
    relevant = {
        tid: rec for tid, rec in records.items() if tid.startswith(prefix)
    }
    missing: tuple[str, ...] = ()
    if expected_task_ids is not None:
        expected = list(expected_task_ids)
        missing = tuple(tid for tid in expected if tid not in relevant)
    all_completed = bool(relevant) and all(
        rec.status == STATUS_COMPLETED for rec in relevant.values()
    )
    passed = all_completed and not missing
    return PhaseCheckResult(
        phase_id=phase_id,
        records=relevant,
        passed=passed,
        missing_tasks=missing,
    )


class CompletionRecordStore:
    """JSON-backed persistence for per-task completion records."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def load(self) -> dict[str, CompletionRecord]:
        if not self.path.exists():
            return {}
        data = json.loads(self.path.read_text(encoding="utf-8"))
        return {
            task_id: CompletionRecord.from_dict(raw)
            for task_id, raw in data.items()
        }

    def save(self, records: Mapping[str, CompletionRecord]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            task_id: rec.to_dict()
            for task_id, rec in sorted(records.items())
        }
        self.path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def record(
        self,
        result: ValidationResult,
        *,
        verification_summary: str = "",
        artifact_paths: Iterable[str] = (),
    ) -> CompletionRecord:
        """Validate-then-persist: re-runs validation via the given result.

        The caller is expected to have produced *result* from
        :func:`validate_task_completion`; the record's status follows
        ``result.ok`` so unverified tasks are never auto-completed.
        """
        records = self.load()
        record = CompletionRecord.from_validation(
            result,
            verification_summary=verification_summary,
            artifact_paths=artifact_paths,
        )
        records[result.task_id] = record
        self.save(records)
        return record


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="OmniLSS stage completion checker (Phase 0 task 0.5).",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    check = subparsers.add_parser("check", help="Validate a task completion commit.")
    check.add_argument("--task", required=True, help="Task id, e.g. 0.5")
    check.add_argument("--commit", required=True, help="Commit id (short or full sha)")
    check.add_argument("--repo", default=None, help="Repository path (default: cwd)")
    check.add_argument("--artifact", action="append", default=[],
                       help="Explicit evidence artifact path in the commit (repeatable)")
    check.add_argument("--no-impl", action="store_true", help="Do not require implementation files")
    check.add_argument("--no-tests", action="store_true", help="Do not require test files")
    check.add_argument("--no-evidence", action="store_true", help="Do not require evidence files")
    check.set_defaults(func=cmd_check)

    record = subparsers.add_parser("record", help="Validate and persist a completion record.")
    record.add_argument("--task", required=True, help="Task id, e.g. 0.5")
    record.add_argument("--commit", required=True, help="Commit id (short or full sha)")
    record.add_argument("--store", required=True, help="Path to the completion records JSON")
    record.add_argument("--repo", default=None, help="Repository path (default: cwd)")
    record.add_argument("--summary", default="", help="Verification summary")
    record.add_argument("--artifact", action="append", default=[],
                        help="Artifact path referenced by the record (e.g. phase-0 R golden fixture)")
    record.set_defaults(func=cmd_record)

    phase = subparsers.add_parser("phase", help="Check all records of a phase.")
    phase.add_argument("--phase", required=True, help="Phase id, e.g. 0")
    phase.add_argument("--store", required=True, help="Path to the completion records JSON")
    phase.add_argument(
        "--tasks",
        default="",
        help="Comma-separated expected task ids of the phase, e.g. 0.1,0.2 (integrity check)",
    )
    phase.set_defaults(func=cmd_phase)

    return parser


def _print_result(result: ValidationResult) -> None:
    print(f"task:       {result.task_id}")
    print(f"commit:     {result.commit_id}")
    print(f"ok:         {result.ok}")
    for name, passed in result.checks.items():
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
    for reason in result.reasons:
        print(f"  reason: {reason}")


def cmd_check(args: argparse.Namespace) -> int:
    inspector = GitCommitInspector(args.repo)
    result = validate_task_completion(
        args.task,
        args.commit,
        inspector,
        require_implementation=not args.no_impl,
        require_tests=not args.no_tests,
        require_evidence=not args.no_evidence,
        required_artifact_paths=args.artifact,
    )
    _print_result(result)
    return 0 if result.ok else 1


def cmd_record(args: argparse.Namespace) -> int:
    inspector = GitCommitInspector(args.repo)
    # Evidence presence is auto-detected; --artifact records the referenced
    # artifacts (e.g. phase-0 fixed R golden fixtures) in the completion
    # record without requiring them to be touched by this commit.
    result = validate_task_completion(
        args.task,
        args.commit,
        inspector,
    )
    _print_result(result)
    store = CompletionRecordStore(args.store)
    store.record(
        result,
        verification_summary=args.summary,
        artifact_paths=args.artifact,
    )
    status = "completed" if result.ok else "incomplete"
    print(f"recorded:   {args.store} -> {result.task_id}: {status}")
    return 0 if result.ok else 1


def cmd_phase(args: argparse.Namespace) -> int:
    store = CompletionRecordStore(args.store)
    expected = (
        [t.strip() for t in args.tasks.split(",") if t.strip()] if args.tasks else None
    )
    result = check_phase_records(store.load(), args.phase, expected_task_ids=expected)
    print(f"phase:      {result.phase_id}")
    print(f"records:    {len(result.records)}")
    for tid, rec in sorted(result.records.items()):
        print(f"  [{rec.status}] {tid} @ {rec.commit_id} {rec.subject}")
    if result.missing_tasks:
        print(f"missing:    {', '.join(result.missing_tasks)}")
    print(f"passed:     {result.passed}")
    return 0 if result.passed else 1


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
