"""Generate the placeholder-family migration queue checklist (A0.4).

Produces docs/development/placeholder-migration-queue-<date>.md listing all
currently-placeholder families (by code-derived implementation_status) grouped
into the Tier-1 batches A1-A4 from the full-migration roadmap, as a tickable
checklist.

The batch membership list is taken from the roadmap's appendix A batch
suggestions. Families whose status has since changed (e.g. registered in A0.3)
are grouped under their batch but flagged with their current status.
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

from omnilss.family_capabilities import implementation_status_for_family
from omnilss.distribution_registry import _REGISTERED_FAMILIES

# Batch -> list of R family names (from roadmap appendix A).
BATCHES = {
    "A1 偏态 t 与 SEP": [
        "ST1", "ST2", "ST3", "ST3C", "ST4", "ST5", "SST",
        "SEP", "SEP1", "SEP2", "SEP3", "SEP4",
    ],
    "A2 Box-Cox 与有界混合": [
        "BCCGo", "BCPEo", "BCTo", "BEINF0", "BEINF1", "BEOI", "BEZI",
        "LG", "LNO", "LOGITNO", "NET",
    ],
    "A3 离散与零膨胀计数": [
        "DEL", "DPO", "SI", "WARING", "YULE", "NBF", "GPO", "NBII",
        "ZALG", "ZIP", "ZIP2", "ZIPF",
    ],
    "A4 正实数与其余连续": [
        "GB1", "GB2", "GIG", "NO2", "NOF", "PARETO", "PARETO1",
        "PE2", "PIG2", "RGE", "WEI2", "exGAUS", "GAF", "EGB2",
        "LOGSHASH", "LQNO", "SN1", "SN2", "MN3", "MN4", "MN5",
    ],
}

STATUS_CN = {
    "placeholder": "占位",
    "experimental": "experimental",
    "compat_alias": "compat_alias",
    "missing": "missing",
    "implemented": "implemented",
}

# registry names for special cases (exGAUS registered as EXGAUS).
def registry_name(r_name: str) -> str:
    mapping = {"exGAUS": "EXGAUS", "BCCGo": "BCCGO", "BCPEo": "BCPEO", "BCTo": "BCTO"}
    return mapping.get(r_name, r_name)


def main() -> int:
    out = Path("docs/development") / f"placeholder-migration-queue-{date.today().isoformat()}.md"

    lines: list[str] = []
    lines.append("# GAMLSS 分布族 placeholder 迁移队列清单")
    lines.append("")
    lines.append(f"> 生成日期：{date.today().isoformat()}")
    lines.append("> 生成方式：`tools/generate_placeholder_migration_queue.py`")
    lines.append(
        "> 状态来源：`omnilss.family_capabilities` 的代码派生 "
        "`implementation_status`（placeholder = 生产路由被拒，需重写）"
    )
    lines.append("> 批次划分：`docs/development/full-gamlss-migration-audit-and-roadmap-2026-08-22.md` 附录 A")
    lines.append("")
    lines.append("勾选规则：`- [ ]` 表示待迁移；`- [x]` 表示已完成（状态已离开 placeholder）。")
    lines.append("")

    total_placeholder = 0
    for batch, r_names in BATCHES.items():
        lines.append(f"## {batch}")
        lines.append("")
        lines.append("| 勾选 | R family | registry | 当前状态 | 备注 |")
        lines.append("|---|---|---|---|---|")
        for r_name in r_names:
            reg = registry_name(r_name)
            status = "missing"
            if reg in _REGISTERED_FAMILIES:
                status = implementation_status_for_family(reg).value
            checked = "[ ]" if status == "placeholder" else "[x]"
            if status == "placeholder":
                total_placeholder += 1
            note = ""
            if status != "placeholder":
                note = f"状态已离开 placeholder（{STATUS_CN.get(status, status)}）"
            lines.append(f"| {checked} | `{r_name}` | `{reg}` | {STATUS_CN.get(status, status)} | {note} |")
        lines.append("")

    # Unregistered families that are still placeholder-like or missing.
    lines.append("## 附录：尚未在任何批次中的状态")
    lines.append("")
    lines.append(f"当前 placeholder 家族总数（批次内）：{total_placeholder}")
    lines.append("")

    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out} ({total_placeholder} placeholders)")
    return 0


if __name__ == "__main__":
    sys.exit(main())