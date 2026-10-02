"""D1-E：PARETO / PARETO1 / WEI2 / RGE / NO2 / PE2 / NOF / GAF 的 d/p/q/r 验收测试。

分两类：

1. **无条件运行的不变量**（不依赖 R）：密度非负、CDF 单调且落在 [0,1]、
   ``q`` 与 ``p`` 互为反函数、抽样落在支撑集内、族不再是 placeholder。
2. **R 对照**（``gamlss.dist`` 可用时）：逐点与 R 的 d/p/q 比对。

R 缺失时第 2 类整类跳过，但第 1 类仍全量执行 —— 不允许出现“R 不在场就什么都不验”。
R 侧已知缺陷（见 ``omnilss/dpqr_batch_e.py`` 模块 docstring）在
``test_r_quirk_inventory_is_explicit`` 中显式登记，而不是悄悄容忍。

运行：
    cd omnilss && JAX_ENABLE_X64=true python -m pytest tests/test_dpqr_batch_e.py -q
"""

from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "omnilss" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import jax.numpy as jnp  # noqa: E402
import jax.random as jr  # noqa: E402

from omnilss.dpqr_batch_e import (  # noqa: E402
    R_QUIRKS,
    dGAF,
    dNO2,
    dNOF,
    dPARETO,
    dPARETO1,
    dPE2,
    dRGE,
    dWEI2,
    pGAF,
    pNO2,
    pNOF,
    pPARETO,
    pPARETO1,
    pPE2,
    pRGE,
    pWEI2,
    qGAF,
    qNO2,
    qNOF,
    qPARETO,
    qPARETO1,
    qPE2,
    qRGE,
    qWEI2,
    rGAF,
    rNO2,
    rNOF,
    rPARETO,
    rPARETO1,
    rPE2,
    rRGE,
    rWEI2,
)
from omnilss.family_capabilities import implementation_status_for_family  # noqa: E402

from tests._r_bridge_helper import R_AVAILABLE  # noqa: E402


def _rscript() -> str | None:
    from tests._r_bridge_helper import _resolve_rscript

    return _resolve_rscript()


def _r_values(family: str, kind: str, args, *, lower_tail=True, log_p=False) -> np.ndarray:
    """逐点调用 R 的 ``d<pkg>/p<...>/q<...>`` 并返回 numpy 数组。

    逐点调用（而不是整向量）是为了绕开 gamlss.dist 中部分族向量化不完整的问题，
    与 ``tests/rbus/R_scripts/eval_family.R`` 的做法一致。
    """
    rscript = _rscript()
    vectors = [np.atleast_1d(np.asarray(a, dtype=float)) for a in args]
    n = max(len(v) for v in vectors)
    names = ["x", "mu", "sigma", "nu"][: len(vectors)]
    decls = "\n".join(
        f"{name} <- c({', '.join(repr(float(v[i] if len(v) > 1 else v[0])) for i in range(n))})"
        for name, v in zip(names, vectors)
    )
    call_args = ", ".join(names)
    if kind == "d":
        extra = f", log={'TRUE' if log_p else 'FALSE'}"
    else:
        extra = f", lower.tail={'TRUE' if lower_tail else 'FALSE'}, log.p={'TRUE' if log_p else 'FALSE'}"
    code = (
        "suppressPackageStartupMessages({library(gamlss); library(gamlss.dist)})\n"
        f"{decls}\n"
        f"out <- sapply(seq_len({n}), function(i) as.numeric({kind}{family}({', '.join(f'{nm}[i]' for nm in names)}{extra})))\n"
        'cat(sprintf("%.17g", out), sep="\\n")\n'
    )
    proc = subprocess.run(
        [rscript, "-e", code], capture_output=True, text=True, timeout=300, check=False
    )
    if proc.returncode != 0:
        raise AssertionError(f"Rscript failed: {proc.stderr[-800:]}")
    lines = [line for line in proc.stdout.splitlines() if line.strip()]
    if len(lines) != n:
        raise AssertionError(f"R returned {len(lines)} values, expected {n}: {proc.stdout[:400]}")
    out = []
    for line in lines:
        token = line.strip()
        out.append(float("nan") if token in {"NA", "NaN"} else float(token))
    return np.asarray(out, dtype=float)


#: (family, d, p, q, r, parameter grid, x grid inside the support)
CASES = {
    "PARETO": (
        dPARETO, pPARETO, qPARETO, rPARETO,
        [(0.5,), (1.0,), (2.0,), (5.0,)],
        [1.05, 1.5, 2.0, 3.0, 10.0, 50.0],
        1.0,
    ),
    "PARETO1": (
        dPARETO1, pPARETO1, qPARETO1, rPARETO1,
        [(0.5,), (1.0,), (2.0,), (5.0,)],
        [0.05, 0.5, 1.0, 3.0, 10.0, 50.0],
        0.0,
    ),
    "WEI2": (
        dWEI2, pWEI2, qWEI2, rWEI2,
        [(0.5, 1.0), (1.0, 1.5), (2.0, 0.8)],
        [0.1, 0.5, 1.0, 2.0, 5.0],
        0.0,
    ),
    "RGE": (
        dRGE, pRGE, qRGE, rRGE,
        [(1.0, 0.1, 1.0), (2.0, 0.5, 0.3), (0.0, 1.0, 2.0)],
        [2.0, 3.0, 5.0, 8.0],
        None,
    ),
    "NO2": (
        dNO2, pNO2, qNO2, rNO2,
        [(0.0, 1.0), (1.0, 4.0), (-2.0, 0.25)],
        [-1.0, 0.0, 0.5, 1.0, 2.0],
        None,
    ),
    # --- 批次二 -----------------------------------------------------------
    "PE2": (
        dPE2, pPE2, qPE2, rPE2,
        [(0.0, 1.0, 2.0), (1.0, 0.5, 1.5)],
        [-2.0, -0.5, 0.0, 0.5, 1.0, 2.5],
        None,
    ),
    "NOF": (
        dNOF, pNOF, qNOF, rNOF,
        [(1.0, 1.0, 0.0), (2.0, 0.5, 1.0), (1.5, 1.0, -0.5)],
        [-1.0, 0.0, 0.5, 1.0, 2.5],
        None,
    ),
    "GAF": (
        dGAF, pGAF, qGAF, rGAF,
        [(1.0, 1.0, 2.0), (2.0, 0.5, 1.0), (3.0, 0.8, 3.0)],
        [0.3, 1.0, 2.0, 5.0],
        0.0,
    ),
}

#: R ``gamlss.dist`` 声明的参数个数（``nopar``）。Python 侧的族必须一致，
#: 否则拟合时会按错误的参数个数组装设计矩阵。
EXPECTED_NOPAR = {
    "PARETO": 1,
    "PARETO1": 1,
    "WEI2": 2,
    "RGE": 3,
    "NO2": 2,
    "PE2": 3,
    "NOF": 3,
    "GAF": 3,
}


def _arrays(xs, params):
    x = np.asarray(xs, dtype=float)
    parr = [np.full(len(x), float(v)) for v in params]
    return x, [jnp.asarray(v) for v in parr]


class TestBatchEInvariants(unittest.TestCase):
    """不依赖 R 的不变量：任何环境下都必须通过。"""

    def test_families_are_no_longer_placeholders(self) -> None:
        for name in CASES:
            status = implementation_status_for_family(name)
            self.assertNotEqual(
                getattr(status, "value", status),
                "placeholder",
                f"{name} 仍是 placeholder：d/p/q/r 未真正接线",
            )

    def test_densities_are_finite_and_non_negative(self) -> None:
        for name, (dfun, _p, _q, _r, grid, xs, _low) in CASES.items():
            for params in grid:
                x, parr = _arrays(xs, params)
                values = np.asarray(dfun(jnp.asarray(x), *parr), dtype=float)
                self.assertTrue(np.all(np.isfinite(values)), f"{name} d 出现非有限值: {values}")
                self.assertTrue(np.all(values >= 0.0), f"{name} d 出现负值: {values}")

    def test_cdf_is_monotone_and_bounded(self) -> None:
        for name, (_d, pfun, _q, _r, grid, xs, _low) in CASES.items():
            for params in grid:
                x, parr = _arrays(sorted(xs), params)
                values = np.asarray(pfun(jnp.asarray(x), *parr), dtype=float)
                self.assertTrue(np.all(np.diff(values) >= -1e-9), f"{name} p 非单调: {values}")
                self.assertTrue(np.all(values >= -1e-12) and np.all(values <= 1 + 1e-12), f"{name} p 越界: {values}")

    def test_quantile_inverts_the_cdf(self) -> None:
        probs = [0.05, 0.25, 0.5, 0.75, 0.95]
        for name, (_d, pfun, qfun, _r, grid, _xs, _low) in CASES.items():
            for params in grid:
                p = np.asarray(probs, dtype=float)
                parr = [jnp.asarray(np.full(len(p), float(v))) for v in params]
                qvals = np.asarray(qfun(jnp.asarray(p), *parr), dtype=float)
                back = np.asarray(pfun(jnp.asarray(qvals), *parr), dtype=float)
                np.testing.assert_allclose(back, p, rtol=1e-6, atol=1e-6, err_msg=f"{name} q∘p ≠ id")

    def test_upper_tail_matches_one_minus_lower_tail(self) -> None:
        for name, (_d, pfun, _q, _r, grid, xs, _low) in CASES.items():
            for params in grid:
                x, parr = _arrays(xs, params)
                jx = jnp.asarray(x)
                lower = np.asarray(pfun(jx, *parr), dtype=float)
                upper = np.asarray(pfun(jx, *parr, lower_tail=False), dtype=float)
                np.testing.assert_allclose(lower + upper, 1.0, rtol=0, atol=1e-9, err_msg=f"{name} 上尾不等于 1-下尾")

    def test_log_p_matches_log_of_probability(self) -> None:
        for name, (_d, pfun, _q, _r, grid, xs, _low) in CASES.items():
            for params in grid:
                x, parr = _arrays(xs, params)
                jx = jnp.asarray(x)
                value = np.asarray(pfun(jx, *parr), dtype=float)
                logged = np.asarray(pfun(jx, *parr, log_p=True), dtype=float)
                np.testing.assert_allclose(logged, np.log(value), rtol=1e-9, atol=1e-12, err_msg=f"{name} log_p 不等于 log(p)")

    def test_sampler_stays_inside_the_support(self) -> None:
        key = jr.PRNGKey(20261002)
        for name, (_d, _p, _q, rfun, grid, _xs, low) in CASES.items():
            for params in grid:
                parr = [jnp.asarray(np.array([float(v)])) for v in params]
                draws = np.asarray(rfun(key, 200, *parr), dtype=float)
                self.assertTrue(np.all(np.isfinite(draws)), f"{name} r 产生非有限值")
                if low is not None:
                    self.assertTrue(np.all(draws > low), f"{name} r 越出支撑集下界 {low}")

    def test_parameter_count_matches_r(self) -> None:
        """Python 族的 parameter 个数必须等于 R 的 ``nopar``。

        NOF/GAF 曾经只声明 ``(mu, sigma)``，而 R 的 ``NOF``/``GAF`` 是三参数族
        （``nu`` 决定方差函数 ``sigma^2*|mu|^nu`` / ``sigma^2*mu^nu``）。少一个参数
        会让拟合按错误的维度组装，所以这里锁死。
        """
        from omnilss.distribution_registry import resolve

        for name, nopar in EXPECTED_NOPAR.items():
            self.assertEqual(len(resolve(name).parameters), nopar, f"{name} 参数个数与 R 不一致")

    def test_quantile_honours_log_p(self) -> None:
        """``q(p, log_p=True)`` 必须等于 ``q(exp(p))``。"""
        probs = [0.05, 0.25, 0.5, 0.75, 0.95]
        for name, (_d, _p, qfun, _r, grid, _xs, _low) in CASES.items():
            for params in grid:
                p = np.asarray(probs, dtype=float)
                parr = [jnp.asarray(np.full(len(p), float(v))) for v in params]
                direct = np.asarray(qfun(jnp.asarray(p), *parr), dtype=float)
                logged = np.asarray(qfun(jnp.asarray(np.log(p)), *parr, log_p=True), dtype=float)
                np.testing.assert_allclose(logged, direct, rtol=1e-8, atol=1e-10, err_msg=f"{name} q 未正确处理 log_p")

    def test_r_quirk_inventory_is_explicit(self) -> None:
        """R gamlss.dist 的已知缺陷清单 —— 我们选择不复制，必须显式记录。

        清单来自 ``omnilss.dpqr_batch_e.R_QUIRKS``（与实现同处一个模块，避免漂移）；
        新增同类偏差必须同步更新那里。
        """
        self.assertTrue(R_QUIRKS)
        for name in R_QUIRKS:
            self.assertIn(name, CASES, f"{name} 登记了 R 缺陷但没有进入对照清单")


@unittest.skipUnless(R_AVAILABLE, "R / Rscript not available")
class TestBatchERConsistency(unittest.TestCase):
    """逐点 R 对照（需要 R + gamlss + gamlss.dist）。"""

    def test_density_matches_r(self) -> None:
        for name, (dfun, _p, _q, _r, grid, xs, _low) in CASES.items():
            for params in grid:
                x, parr = _arrays(xs, params)
                got = np.asarray(dfun(jnp.asarray(x), *parr), dtype=float)
                exp = _r_values(name, "d", [x, *[np.asarray(v) for v in parr]])
                np.testing.assert_allclose(got, exp, rtol=1e-8, atol=1e-10, err_msg=f"{name} d params={params}")

    def test_lower_tail_cdf_matches_r(self) -> None:
        for name, (_dfun, pfun, _q, _r, grid, xs, _low) in CASES.items():
            for params in grid:
                x, parr = _arrays(xs, params)
                got = np.asarray(pfun(jnp.asarray(x), *parr), dtype=float)
                exp = _r_values(name, "p", [x, *[np.asarray(v) for v in parr]])
                np.testing.assert_allclose(got, exp, rtol=1e-8, atol=1e-10, err_msg=f"{name} p params={params}")

    def test_upper_tail_cdf_matches_r(self) -> None:
        """上尾对照：PARETO/PARETO1 因 R 缺陷被排除，其余族必须与 R 一致。"""
        excluded = {"PARETO", "PARETO1"}
        for name, (_dfun, pfun, _q, _r, grid, xs, _low) in CASES.items():
            if name in excluded:
                continue
            for params in grid:
                x, parr = _arrays(xs, params)
                got = np.asarray(pfun(jnp.asarray(x), *parr, lower_tail=False), dtype=float)
                exp = _r_values(name, "p", [x, *[np.asarray(v) for v in parr]], lower_tail=False)
                np.testing.assert_allclose(got, exp, rtol=1e-8, atol=1e-10, err_msg=f"{name} upper p params={params}")

    def test_quantile_matches_r_lower_tail(self) -> None:
        probs = [0.01, 0.25, 0.5, 0.75, 0.99]
        for name, (_dfun, _p, qfun, _r, grid, _xs, _low) in CASES.items():
            for params in grid:
                p = np.asarray(probs, dtype=float)
                parr = [jnp.asarray(np.full(len(p), float(v))) for v in params]
                got = np.asarray(qfun(jnp.asarray(p), *parr), dtype=float)
                exp = _r_values(name, "q", [p, *[np.asarray(v) for v in parr]])
                np.testing.assert_allclose(got, exp, rtol=1e-8, atol=1e-10, err_msg=f"{name} q params={params}")

    def test_r_ignores_log_p_for_the_nof_quantile(self) -> None:
        """实测确认 R 的 qNOF 忽略 log.p（不是我们的实现差异）。

        R 的 ``qNOF`` 把 ``qnorm(p, mean, sd, lower.tail)`` 的 ``log.p`` 漏掉了，
        所以 ``log.p=TRUE`` 与 ``log.p=FALSE`` 返回完全相同的值 —— 也就是说 R 根本
        无法在 log 尺度上求分位数（传入 ``log(p)`` 会直接报
        ``p must be between 0 and 1``）。

        我们按数学正确的方式处理 ``log_p``：传入 ``log(p)`` 时应当得到 R 在概率
        尺度下给出的同一个分位数。这两条断言一起把偏差钉死。
        """
        params = (1.5, 1.0, -0.5)
        p = np.asarray([0.25, 0.5, 0.75], dtype=float)
        parr = [np.full(len(p), float(v)) for v in params]
        jpar = [jnp.asarray(v) for v in parr]
        r_plain = _r_values("NOF", "q", [p, *parr], log_p=False)
        r_logged = _r_values("NOF", "q", [p, *parr], log_p=True)
        self.assertTrue(np.all(np.isfinite(r_plain)), f"R qNOF 返回非有限值: {r_plain}")
        np.testing.assert_allclose(
            r_plain, r_logged, err_msg="R qNOF 现在会处理 log.p 了 —— 可以去掉这条豁免"
        )
        ours = np.asarray(qNOF(jnp.asarray(np.log(p)), *jpar, log_p=True), dtype=float)
        np.testing.assert_allclose(ours, r_plain, rtol=1e-8, atol=1e-10, err_msg="我们的 qNOF log_p 分支不正确")

    def test_r_ignores_lower_tail_for_the_pareto_quantiles(self) -> None:
        """实测确认 R 的 qPARETO/qPARETO1 忽略 lower.tail（不是我们的实现差异）。"""
        for name in ("PARETO", "PARETO1"):
            _d, _p, _q, _r, grid, _xs, _low = CASES[name]
            params = grid[1]
            p = np.asarray([0.25, 0.75], dtype=float)
            parr = [np.asarray(np.full(2, float(v))) for v in params]
            lower = _r_values(name, "q", [p, *[np.asarray(v) for v in parr]], lower_tail=True)
            upper = _r_values(name, "q", [p, *[np.asarray(v) for v in parr]], lower_tail=False)
            np.testing.assert_allclose(lower, upper, err_msg=f"R {name} 现在会处理 lower.tail 了 —— 可以去掉这条豁免")


if __name__ == "__main__":
    unittest.main()
