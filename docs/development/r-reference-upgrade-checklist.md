# R 参考版升级与重标定清单（D2.308/R2）

适用场景：升级 R 本身、gamlss 或 gamlss.dist；或 CRAN 归档被替换导致 lock 失效。
目标：升级是一次可审计的重标定，而不是放宽阈值把红灯压绿。

## 0. 事实源

- 版本、URL、SHA256：docs/development/r-reference-lock.json
- smooth 阈值：docs/development/smooth-parity-baseline.json
- Group B skip：docs/development/group-b-skip-baseline.json
- 历史归因：docs/development/group-b-attribution-2026-11-22.md

任何升级 PR 必须同时更新受影响的事实源，不允许只改测试。

## 1. 准备新参考版

1. 下载新版本 tarball，计算 SHA256。
2. 在 GitHub Release 上传资产（tag 形如 r-reference-YYYY-MM-DD），记录 asset digest。
3. 更新 lock：version、url、sha256、mirror_url。
4. 本机安装到独立库验证：

       python tools/install_r_reference.py --lock docs/development/r-reference-lock.json --dest /tmp/r-reference
       Rscript -e 'install.packages(commandArgs(TRUE), repos=NULL, type="source")' /tmp/r-reference/*.tar.gz
       Rscript -e 'print(packageDescription("gamlss.dist")$Version)'

验收：三个包版本与 lock 一致，SHA256 校验通过。

## 2. 跑全量 Group B（新参考版）

使用 CI 之外的独立分支或 PR，运行：

    python -m pytest tests/test_*r_consistency*.py -rs -q

记录：

- run ID
- passed / failed / skipped
- 每个失败的测试名与首行断言差异

不要在这个阶段修改任何断言。

## 3. 逐族判定

对每个失败族回答两个问题：

1. 新 R 的行为是否真的变了？（用最小 R 脚本直接验证，不经过 Python）
2. 我们的实现是否与旧 R 一致、只是与新 R 不同？

判定表：

| 判定 | 处理 |
|---|---|
| R 行为变化，我方仍按旧语义 | 更新期望值；若需同时支持新旧 R，保留双分支断言 |
| 我方回归（旧 R 也失败） | 修实现，不放宽阈值 |
| 测试本身写错（比较对象/单位错） | 修测试，引用正确参照 |
| 仅浮点末位差异 | 按 R5 规则改为容差，注明 ULP |

在 PR 描述里逐族写出结论，不接受“全部更新期望值”。

## 4. 更新基线

- smooth：设置 OMNILSS_SMOOTH_BASELINE_RECORD 重新记录 observed，再由公式派生阈值：

      OMNILSS_SMOOTH_BASELINE_RECORD=tmp/smooth.jsonl         python -m pytest tests/test_r_consistency_smooth.py -q

  用记录的 observed 更新 docs/development/smooth-parity-baseline.json，保持
  corr_min = max(observed - 0.01, legacy)
  rel_max = max(observed * 1.5, 0.05) 且不超过旧阈值。

- skip：若合法 skip 的原因集合变化，更新 group-b-skip-baseline.json；
  仅数量增长不算理由，必须解释原因。

- 归因报告：追加本次升级的 run ID、版本、逐族结论。

## 5. 验收

- Group B：0 failed；passed >= 300；skip 不超过 baseline + allowed_growth。
- Tiered Collection Gate：success（M0 provenance 已重生成）。
- CI 10/10 job success。
- 没有阈值在无观测依据的情况下被放宽。
- lock、镜像资产、两个 baseline 与归因报告在同一 PR 内更新。

## 6. 回滚

若新参考版无法在合理时间内支持：

1. 回滚 lock 与 baseline 到上一个提交。
2. 保留新 Release tag（资产不删，便于后续重试）。
3. 在归因报告记录放弃原因与新版本的具体阻塞点。

## 7. PR 模板

    参考版：gamlss X / gamlss.dist Y / R Z
    旧 run：<id>  （0 failed / N skipped）
    新 run：<id>  （...）
    逐族结论：<family: R 行为变化 / 我方回归 / 测试错误>
    阈值变化：<无 / 列出 observed 与派生公式>
    lock / baseline / 报告：<已更新>
