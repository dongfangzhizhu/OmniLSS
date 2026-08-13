"""Phase 0 admission / completion gate (DEVELOPMENT_PLAN task 0.6).

Implements the stage-0 gate defined in ``.kiro/tasks.md`` 0.6:

- run the full Phase 0 fixed-R suite with ``--require-r`` (fail-closed: any
  skip or smoke-only result fails the gate);
- verify the Phase 0 JSON / Markdown artifacts and runtime manifest are
  present, readable and carry the required-R success results and provenance;
- verify every Phase 0 leaf task has a completion record whose commit is
  valid, whose subject is Chinese, and whose implementation / test / evidence
  files exist in the repository;
- any failure keeps Phase 0 incomplete (fail-closed) and blocks Phase 2.

Historical tasks (0.1-0.4, 0.7, 0.8) predate the strict single-commit
discipline of task 0.5, so their records are written with the relaxed
``record`` subcommand (valid Chinese commit + repository-level evidence
presence), while new tasks continue to be validated strictly by
``tools/stage_completion_checker.py``.

Usage
-----
::

    python tools/stage0_gate.py check [--repo .] [--records docs/reports/completion_records.json]
    python tools/stage0_gate.py record --task 0.4 --commit <sha> \
        --impl .github/workflows/ci.yml --test omnilss/tests/test_run_suite_gate.py \
        --artifact .github/workflows/ci.yml --store docs/reports/completion_records.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterable

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.stage_completion_checker import (  # noqa: E402
    STATUS_COMPLETED,
    STATUS_INCOMPLETE,
    CompletionRecord,
    CompletionRecordStore,
    GitCommitInspector,
    contains_chinese,
    is_evidence_file,
    is_implementation_file,
    is_test_file,
)

__all__ = [
    "GateIssue",
    "GateReport",
    "STAGE0_TASKS",
    "run_phase0_suite",
    "verify_completion_records",
    "verify_stage0_artifacts",
]

#: Every Phase 0 leaf task id (blocking Phase 1 by dependency graph).
STAGE0_TASKS = ["0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8"]

#: Provenance fields required in every golden fixture manifest entry.
PROVENANCE_FIELDS = (
    "R_version",
    "gamlss_version",
    "gamlss_dist_version",
    "seed",
    "content_hash",
)

MANIFEST_PATH = "omnilss/tests/fixtures/r_golden/regenerated/manifest.json"
GENERATE_SCRIPT_PATH = "omnilss/tests/fixtures/r_golden/generate_golden.R"
R_REFERENCE_PATH = "benchmarks/r_reference_results.json"
COMPLETION_RECORDS_PATH = "docs/reports/completion_records.json"
EVIDENCE_0_5_PATH = "docs/reports/phase0-task-0.5-evidence.json"


@dataclass
class GateIssue:
    """A single fail-closed finding with a human-readable detail."""

    ok: bool
    detail: str


@dataclass
class GateReport:
    """Aggregated result of one gate stage."""

    name: str
    issues: list[GateIssue] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(issue.ok for issue in self.issues)

    def add(self, ok: bool, detail: str) -> None:
        self.issues.append(GateIssue(ok=ok, detail=detail))


# ---------------------------------------------------------------------------
# Artifact verification
# ---------------------------------------------------------------------------
def _read_json(path: Path) -> tuple[object | None, str]:
    """Return (parsed, error) — error is empty when the JSON is readable."""
    try:
        return json.loads(path.read_text(encoding="utf-8")), ""
    except Exception as exc:  # noqa: BLE001 - gate must report any read failure
        return None, f"{exc.__class__.__name__}: {exc}"


def verify_stage0_artifacts(root: str | Path, report: GateReport | None = None) -> GateReport:
    """Verify the Phase 0 JSON/Markdown artifacts and runtime manifest."""
    root = Path(root)
    report = report or GateReport("artifacts")

    manifest_path = root / MANIFEST_PATH
    if not manifest_path.exists():
        report.add(False, f"golden manifest 缺失: {MANIFEST_PATH}")
    else:
        manifest, err = _read_json(manifest_path)
        if err:
            report.add(False, f"golden manifest 不可读: {err}")
        elif not isinstance(manifest, dict):
            report.add(False, "golden manifest 结构错误: 顶层应为 object")
        else:
            manifest_ok = True
            if not manifest:
                report.add(False, "golden manifest 为空")
                manifest_ok = False
            for name, entry in manifest.items():
                missing = [f for f in PROVENANCE_FIELDS if f not in (entry or {})]
                if missing:
                    report.add(
                        False,
                        f"manifest 条目 {name} 缺 provenance 字段: {missing}",
                    )
                    manifest_ok = False
            if manifest_ok:
                # Every checked-in regenerated fixture must be covered by the manifest.
                regenerated = root / "omnilss/tests/fixtures/r_golden/regenerated"
                fixtures = {
                    p.stem
                    for p in regenerated.glob("*.json")
                    if p.name != "manifest.json"
                }
                uncovered = sorted(fixtures - set(manifest))
                if uncovered:
                    report.add(
                        False,
                        f"checked-in fixture 未被 manifest 覆盖: {uncovered}",
                    )
                else:
                    report.add(True, f"golden manifest provenance 完整（{len(manifest)} 条目）")

    for rel, label in (
        (GENERATE_SCRIPT_PATH, "golden 生成脚本"),
        (R_REFERENCE_PATH, "R 参考结果"),
        (COMPLETION_RECORDS_PATH, "完成记录"),
        (EVIDENCE_0_5_PATH, "任务 0.5 验收证据"),
    ):
        path = root / rel
        if not path.exists():
            report.add(False, f"{label} 缺失: {rel}")
        elif rel == R_REFERENCE_PATH:
            data, err = _read_json(path)
            if err:
                report.add(False, f"{label} 不可读: {err}")
            elif isinstance(data, dict) and data.get("status") == "generated_by_r":
                report.add(True, f"{label} 可读且 status=generated_by_r")
            else:
                report.add(
                    False,
                    f"{label} 不含 required-R 成功结果（status != generated_by_r）",
                )
        elif rel == COMPLETION_RECORDS_PATH:
            data, err = _read_json(path)
            if err:
                report.add(False, f"{label} 不可读: {err}")
            else:
                report.add(True, f"{label} 可读（{len(data) if isinstance(data, dict) else '?'} 条）")
        else:
            report.add(True, f"{label} 存在: {rel}")

    return report


# ---------------------------------------------------------------------------
# Completion-record verification
# ---------------------------------------------------------------------------
def _tracked_files(root: Path) -> set[str]:
    result = subprocess.run(
        ["git", "-C", str(root), "ls-files"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode != 0:
        return set()
    return {line.strip() for line in result.stdout.splitlines() if line.strip()}


def verify_completion_records(
    root: str | Path,
    report: GateReport | None = None,
    *,
    tasks: Iterable[str] = STAGE0_TASKS,
    records_path: str | Path | None = None,
) -> GateReport:
    """Verify every Phase 0 task has a valid completed record with existing files."""
    root = Path(root)
    report = report or GateReport("completion-records")
    store = CompletionRecordStore(records_path or (root / COMPLETION_RECORDS_PATH))
    records = store.load()
    inspector = GitCommitInspector(root)
    tracked = _tracked_files(root)

    for task_id in tasks:
        rec = records.get(task_id)
        if rec is None:
            report.add(False, f"任务 {task_id} 缺少 completion record")
            continue
        if rec.status != STATUS_COMPLETED:
            report.add(False, f"任务 {task_id} 未完成（status={rec.status}）")
            continue
        commit = inspector.resolve(rec.commit_id)
        if commit is None:
            report.add(False, f"任务 {task_id} 的 commit {rec.commit_id} 无效")
            continue
        if not contains_chinese(rec.subject):
            report.add(False, f"任务 {task_id} 的提交主题非中文: {rec.subject!r}")
            continue
        declared = list(rec.impl_paths) + list(rec.test_paths) + list(rec.artifact_paths)
        if not declared:
            report.add(False, f"任务 {task_id} 未声明实现/测试/证据文件")
            continue
        missing = [p for p in declared if p not in tracked]
        if missing:
            report.add(False, f"任务 {task_id} 声明文件不在仓库: {missing}")
            continue
        report.add(
            True,
            f"任务 {task_id} 完成记录有效（commit {rec.commit_id[:8]}，"
            f"实现 {len(rec.impl_paths)} / 测试 {len(rec.test_paths)} / 证据 {len(rec.artifact_paths)}）",
        )
    return report


# ---------------------------------------------------------------------------
# Suite execution
# ---------------------------------------------------------------------------
def run_phase0_suite(
    root: str | Path,
    report: GateReport | None = None,
    *,
    require_r: bool = True,
    runner: str | None = None,
) -> GateReport:
    """Run the full phase0_gate suite, fail-closed when *require_r* is set."""
    root = Path(root)
    report = report or GateReport("suite")
    runner = runner or str(Path(sys.executable))
    cmd = [runner, "omnilss/tests/run_suite.py", "--suite", "phase0_gate"]
    if require_r:
        cmd.append("--require-r")
    try:
        result = subprocess.run(
            cmd,
            cwd=str(root),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=3600,
        )
    except subprocess.TimeoutExpired:
        report.add(False, "phase0_gate suite 执行超时（>3600s）")
        return report
    tail = (result.stdout or "")[-4000:]
    if result.returncode == 0:
        report.add(True, "phase0_gate suite 通过（--require-r，exit=0）")
    else:
        report.add(False, f"phase0_gate suite 失败（exit={result.returncode}）")
        report.add(False, f"suite 输出尾部: {tail}")
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _safe_print(text: str) -> None:
    """Print without crashing on non-encodable characters (Windows GBK console)."""
    try:
        print(text)
    except UnicodeEncodeError:
        enc = sys.stdout.encoding or "utf-8"
        fallback = text.encode(enc, errors="replace").decode(enc, errors="replace")
        print(fallback)


def _print_report(report: GateReport) -> None:
    _safe_print(f"[{report.name}] {'PASS' if report.ok else 'FAIL'}")
    for issue in report.issues:
        _safe_print(f"  [{'PASS' if issue.ok else 'FAIL'}] {issue.detail}")


def cmd_check(args: argparse.Namespace) -> int:
    root = Path(args.repo)
    reports = [
        verify_stage0_artifacts(root),
        verify_completion_records(
            root,
            tasks=args.tasks.split(",") if args.tasks else STAGE0_TASKS,
            records_path=args.records,
        ),
    ]
    suite_report = run_phase0_suite(root, require_r=not args.skip_r)
    reports.append(suite_report)
    for report in reports:
        _print_report(report)
    ok = all(report.ok for report in reports)
    print("")
    print(f"Phase 0 gate: {'PASSED' if ok else 'FAILED (fail-closed, Phase 0 stays incomplete)'}")
    return 0 if ok else 1


def cmd_record(args: argparse.Namespace) -> int:
    """Write a relaxed completion record for a historical Phase 0 task.

    Requires a valid commit whose subject is Chinese; the declared
    implementation / test / evidence paths must exist in the repository
    (verified later by ``check``).
    """
    root = Path(args.repo)
    store = CompletionRecordStore(args.store)
    records = store.load()
    inspector = GitCommitInspector(root)
    commit = inspector.resolve(args.commit)
    reasons: list[str] = []
    if commit is None:
        reasons.append(f"commit {args.commit!r} 无法解析（无效 commit ID 或非 git 仓库）")
    elif not contains_chinese(commit.subject):
        reasons.append(f"提交主题 {commit.subject!r} 不含中文")

    impl = list(args.impl)
    tests = list(args.test)
    artifacts = list(args.artifact)
    if commit is not None:
        files = commit.changed_files
        impl = impl or [f for f in files if is_implementation_file(f)]
        tests = tests or [f for f in files if is_test_file(f)]
        artifacts = artifacts or [f for f in files if is_evidence_file(f)]
    tracked = _tracked_files(root)
    for label, paths in (("实现", impl), ("测试", tests), ("证据", artifacts)):
        missing = [p for p in paths if p not in tracked]
        if missing:
            reasons.append(f"声明的{label}文件不在仓库: {missing}")

    subject = commit.subject if commit is not None else ""
    ok = not reasons
    rec = CompletionRecord(
        task_id=args.task,
        commit_id=args.commit if commit is None else commit.commit_id,
        subject=subject,
        status=STATUS_COMPLETED if ok else STATUS_INCOMPLETE,
        verification_summary=args.summary,
        artifact_paths=tuple(artifacts),
        impl_paths=tuple(impl),
        test_paths=tuple(tests),
        reasons=tuple(reasons),
        validated_at=datetime.now().isoformat(timespec="seconds"),
    )
    records[args.task] = rec
    store.save(records)
    print(f"task:       {args.task}")
    print(f"commit:     {rec.commit_id}")
    print(f"ok:         {rec.status == STATUS_COMPLETED}")
    for reason in reasons:
        print(f"  reason: {reason}")
    print(f"recorded:   {args.store} -> {args.task}: {rec.status}")
    return 0 if ok else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="OmniLSS Phase 0 admission/completion gate (task 0.6).",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    check = subparsers.add_parser("check", help="Run the full fail-closed Phase 0 gate.")
    check.add_argument("--repo", default=".", help="Repository root (default: cwd)")
    check.add_argument("--records", default=None,
                       help="Completion records JSON (default: docs/reports/completion_records.json)")
    check.add_argument("--tasks", default="",
                       help="Comma-separated expected task ids (default: 0.1..0.8)")
    check.add_argument("--skip-r", action="store_true",
                       help="Do NOT run the required-R suite (non-release smoke only)")
    check.set_defaults(func=cmd_check)

    record = subparsers.add_parser("record", help="Record a historical Phase 0 task completion.")
    record.add_argument("--task", required=True, help="Task id, e.g. 0.4")
    record.add_argument("--commit", required=True, help="Completion commit id")
    record.add_argument("--store", required=True, help="Completion records JSON path")
    record.add_argument("--repo", default=".", help="Repository root (default: cwd)")
    record.add_argument("--summary", default="", help="Verification summary")
    record.add_argument("--impl", action="append", default=[], help="Implementation path (repeatable)")
    record.add_argument("--test", action="append", default=[], help="Test path (repeatable)")
    record.add_argument("--artifact", action="append", default=[], help="Evidence path (repeatable)")
    record.set_defaults(func=cmd_record)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
