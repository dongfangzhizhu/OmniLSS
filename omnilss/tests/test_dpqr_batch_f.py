"""D1-E 批次三：``LQNO`` / ``MN3`` / ``MN4`` / ``MN5`` 的 d/p/q/r 验收测试。

与 ``test_dpqr_batch_e.py`` 同样的两层结构：

1. **无条件运行的不变量**（不依赖 R）：密度非负、CDF 单调且落在 [0,1]、
   分位数是 CDF 的（广义）反函数、抽样落在支撑集内、族不再是 placeholder。
2. **R 对照**（``gamlss.dist`` 可用时）：逐点与 R 的 d/p/q 比对。

两点需要单独说明：

* ``LQNO`` 曾经被实现成一个三参数 tanh 变换分布，与 R 的两参数
  ``Normal(mu, sqrt(mu*(1+sigma*mu)))`` 毫无关系。这里用 ``EXPECTED_NOPAR``
  把参数个数钉死，防止再漂移。
* ``MN3``/``MN4``/``MN5`` 是离散分类族，CDF 是阶梯函数，所以
  ``q∘p = id`` 不成立；改用广义反函数条件 ``p(q(u)) >= u > p(q(u)-1)``。

运行：
    cd omnilss && JAX_ENABLE_X64=true python -m pytest tests/test_dpqr_batch_f.py -q
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

from omnilss.dpqr_batch_f import (  # noqa: E402
    R_QUIRKS,
    dEGB2,
    dLQNO,
    dMN3,
    dMN4,
    dMN5,
    dexGAUS,
    pEGB2,
    pLQNO,
    pMN3,
    pMN4,
    pMN5,
    pexGAUS,
    qEGB2,
    qLQNO,
    qMN3,
    qMN4,
    qMN5,
    qexGAUS,
    rEGB2,
    rLQNO,
    rMN3,
    rMN4,
    rMN5,
    rexGAUS,
)
from omnilss.family_capabilities import implementation_status_for_family  # noqa: E402

from tests._r_bridge_helper import R_AVAILABLE  # noqa: E402


def _rscript() -> str | None:
    from tests._r_bridge_helper import _resolve_rscript

    return _resolve_rscript()


def _r_values(family: str, kind: str, args, *, lower_tail=True, log_p=False) -> np.ndarray:
    """逐点调用 R 的 ``d<fam>/p<fam>/q<fam>`` 并返回 numpy 数组。"""
    rscript = _rscript()
    vectors = [np.atleast_1d(np.asarray(a, dtype=float)) for a in args]
    n = max(len(v) for v in vectors)
    names = ["x", "mu", "sigma", "nu", "tau"][: len(vectors)]
    decls = "\n".join(
        f"{name} <- c({', '.join(repr(float(v[i] if len(v) > 1 else v[0])) for i in range(n))})"
        for name, v in zip(names, vectors)
    )
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


#: (family, d, p, q, r, parameter grid, x grid)
CONTINUOUS_CASES = {
    "LQNO": (
        dLQNO, pLQNO, qLQNO, rLQNO,
        [(0.5, 0.5), (1.0, 1.0), (2.0, 0.3), (0.3, 2.0), (5.0, 1.5)],
        [-3.0, -1.0, 0.0, 0.5, 1.0, 2.0, 5.0],
    ),
    # --- 批次四 -----------------------------------------------------------
    "exGAUS": (
        dexGAUS, pexGAUS, qexGAUS, rexGAUS,
        [(5.0, 1.0, 1.0), (0.0, 0.5, 2.0), (2.0, 1.5, 0.5), (-2.0, 1.0, 3.0)],
        [-3.0, -1.0, 0.0, 1.0, 2.0, 5.0, 9.0],
    ),
    "EGB2": (
        dEGB2, pEGB2, qEGB2, rEGB2,
        [(0.0, 1.0, 1.0, 0.5), (1.0, 0.5, 2.0, 1.0), (-1.0, 1.5, 0.5, 3.0), (2.0, 0.8, 3.0, 0.2)],
        [-4.0, -1.0, 0.0, 0.5, 1.0, 2.0, 5.0],
    ),
}

#: (family, d, p, q, r, parameter grid, categories, number of categories)
DISCRETE_CASES = {
    "MN3": (
        dMN3, pMN3, qMN3, rMN3,
        [(0.5, 1.0), (2.0, 0.5), (1.0, 1.0), (3.0, 2.0)],
        [1.0, 2.0, 3.0],
        3,
    ),
    "MN4": (
        dMN4, pMN4, qMN4, rMN4,
        [(0.5, 1.0, 2.0), (2.0, 0.5, 1.0), (1.0, 1.0, 1.0)],
        [1.0, 2.0, 3.0, 4.0],
        4,
    ),
    "MN5": (
        dMN5, pMN5, qMN5, rMN5,
        [(0.5, 1.0, 2.0, 0.5), (2.0, 0.5, 1.0, 3.0), (1.0, 1.0, 1.0, 1.0)],
        [1.0, 2.0, 3.0, 4.0, 5.0],
        5,
    ),
}

#: R ``gamlss.dist`` 的 ``nopar``。Python 侧必须一致，否则拟合会按错误的维度组装。
EXPECTED_NOPAR = {"LQNO": 2, "MN3": 2, "MN4": 3, "MN5": 4, "exGAUS": 3, "EGB2": 4}

#: 数值反演分位数的族（R 用 ``uniroot``，tol = double.eps^0.25 ≈ 1.2e-4，是
#: **x 尺度上的绝对容差**）。这些族的 q 对 R 只能按 uniroot 的精度比较；
#: 我们自己的 ``p∘q = id`` 自洽性另用紧容差断言。
NUMERIC_Q = {"exGAUS": (1e-3, 2e-4)}

_PROBS = [0.05, 0.25, 0.5, 0.75, 0.95]


def _arrays(xs, params):
    x = np.asarray(xs, dtype=float)
    parr = [np.full(len(x), float(v)) for v in params]
    return x, [jnp.asarray(v) for v in parr]


def _parr(n, params):
    return [jnp.asarray(np.full(n, float(v))) for v in params]


def _all_cases():
    for name, (d, p, q, r, grid, xs) in CONTINUOUS_CASES.items():
        yield name, d, p, q, r, grid, xs
    for name, (d, p, q, r, grid, xs, _k) in DISCRETE_CASES.items():
        yield name, d, p, q, r, grid, xs


class TestBatchFInvariants(unittest.TestCase):
    """不依赖 R 的不变量：任何环境下都必须通过。"""

    def test_families_are_no_longer_placeholders(self) -> None:
        for name, *_ in _all_cases():
            status = implementation_status_for_family(name)
            self.assertNotEqual(
                getattr(status, "value", status),
                "placeholder",
                f"{name} 仍是 placeholder：d/p/q/r 未真正接线",
            )

    def test_densities_are_finite_and_non_negative(self) -> None:
        for name, dfun, _p, _q, _r, grid, xs in _all_cases():
            for params in grid:
                x, parr = _arrays(xs, params)
                values = np.asarray(dfun(jnp.asarray(x), *parr), dtype=float)
                self.assertTrue(np.all(np.isfinite(values)), f"{name} d 非有限: {values}")
                self.assertTrue(np.all(values >= 0.0), f"{name} d 负值: {values}")

    def test_cdf_is_monotone_and_bounded(self) -> None:
        for name, _d, pfun, _q, _r, grid, xs in _all_cases():
            for params in grid:
                x, parr = _arrays(sorted(xs), params)
                values = np.asarray(pfun(jnp.asarray(x), *parr), dtype=float)
                self.assertTrue(np.all(np.diff(values) >= -1e-9), f"{name} p 非单调: {values}")
                self.assertTrue(np.all(values >= -1e-12) and np.all(values <= 1 + 1e-12), f"{name} p 越界: {values}")

    def test_upper_tail_matches_one_minus_lower_tail(self) -> None:
        for name, _d, pfun, _q, _r, grid, xs in _all_cases():
            for params in grid:
                x, parr = _arrays(xs, params)
                jx = jnp.asarray(x)
                lower = np.asarray(pfun(jx, *parr), dtype=float)
                upper = np.asarray(pfun(jx, *parr, lower_tail=False), dtype=float)
                np.testing.assert_allclose(lower + upper, 1.0, rtol=0, atol=1e-9, err_msg=f"{name} 上尾 ≠ 1-下尾")

    def test_log_p_matches_log_of_probability(self) -> None:
        for name, _d, pfun, _q, _r, grid, xs in _all_cases():
            for params in grid:
                x, parr = _arrays(xs, params)
                jx = jnp.asarray(x)
                value = np.asarray(pfun(jx, *parr), dtype=float)
                logged = np.asarray(pfun(jx, *parr, log_p=True), dtype=float)
                np.testing.assert_allclose(logged, np.log(value), rtol=1e-9, atol=1e-12, err_msg=f"{name} log_p ≠ log(p)")

    def test_quantile_inverts_the_cdf(self) -> None:
        """连续族：``p(q(u)) == u``。"""
        for name, _d, pfun, qfun, _r, grid, _xs in ((n, *v) for n, v in CONTINUOUS_CASES.items()):
            for params in grid:
                u = np.asarray(_PROBS, dtype=float)
                parr = _parr(len(u), params)
                qvals = np.asarray(qfun(jnp.asarray(u), *parr), dtype=float)
                back = np.asarray(pfun(jnp.asarray(qvals), *parr), dtype=float)
                np.testing.assert_allclose(back, u, rtol=1e-6, atol=1e-6, err_msg=f"{name} q∘p ≠ id")

    def test_quantile_is_the_generalised_inverse(self) -> None:
        """离散族：``p(q(u)) >= u > p(q(u) - 1)``（CDF 是阶梯函数）。"""
        for name, _d, pfun, qfun, _r, grid, _xs, k in (
            (n, *v) for n, v in DISCRETE_CASES.items()
        ):
            for params in grid:
                u = np.asarray(_PROBS, dtype=float)
                parr = _parr(len(u), params)
                qv = np.asarray(qfun(jnp.asarray(u), *parr), dtype=float)
                at_q = np.asarray(pfun(jnp.asarray(qv), *parr), dtype=float)
                below = np.asarray(pfun(jnp.asarray(qv - 1.0), *parr), dtype=float)
                self.assertTrue(np.all(at_q >= u - 1e-12), f"{name} p(q(u)) < u: {at_q} vs {u}")
                self.assertTrue(np.all(below < u + 1e-12), f"{name} p(q(u)-1) >= u: {below} vs {u}")
                self.assertTrue(np.all((qv >= 1.0) & (qv <= k)), f"{name} q 越出 1..{k}: {qv}")

    def test_probabilities_sum_to_one(self) -> None:
        for name, dfun, _p, _q, _r, grid, xs, _k in ((n, *v) for n, v in DISCRETE_CASES.items()):
            for params in grid:
                x, parr = _arrays(xs, params)
                total = float(np.sum(np.asarray(dfun(jnp.asarray(x), *parr), dtype=float)))
                self.assertAlmostEqual(total, 1.0, places=10, msg=f"{name} pmf 之和 ≠ 1: {total}")

    def test_sampler_stays_inside_the_support(self) -> None:
        key = jr.PRNGKey(20261003)
        for name, _d, _p, _q, rfun, grid, _xs, k in ((n, *v) for n, v in DISCRETE_CASES.items()):
            for params in grid:
                parr = [jnp.asarray(np.array([float(v)])) for v in params]
                draws = np.asarray(rfun(key, 300, *parr), dtype=float)
                self.assertTrue(np.all(np.isfinite(draws)), f"{name} r 非有限值")
                self.assertTrue(np.all((draws >= 1.0) & (draws <= k)), f"{name} r 越出 1..{k}")
                self.assertTrue(np.all(np.equal(draws, np.round(draws))), f"{name} r 产生了非整数类别")
        for name, _d, _p, _q, rfun, grid, _xs in ((n, *v) for n, v in CONTINUOUS_CASES.items()):
            for params in grid:
                parr = [jnp.asarray(np.array([float(v)])) for v in params]
                draws = np.asarray(rfun(key, 300, *parr), dtype=float)
                self.assertTrue(np.all(np.isfinite(draws)), f"{name} r 非有限值")

    def test_parameter_count_matches_r(self) -> None:
        """Python 族的 parameter 个数必须等于 R 的 ``nopar``。

        ``LQNO`` 曾经声明 ``(mu, sigma, nu)``，而 R 的 ``LQNO`` 只有两个参数；
        少一个/多一个都会让拟合按错误维度组装设计矩阵，所以这里锁死。
        """
        from omnilss.distribution_registry import resolve

        for name, nopar in EXPECTED_NOPAR.items():
            self.assertEqual(len(resolve(name).parameters), nopar, f"{name} 参数个数与 R 不一致")

    def test_quantile_honours_log_p(self) -> None:
        """``q(log(u), log_p=True)`` 必须等于 ``q(u)``（R 的 qMN* 做不到）。"""
        for name, _d, _p, qfun, _r, grid, _xs in _all_cases():
            for params in grid:
                u = np.asarray(_PROBS, dtype=float)
                parr = _parr(len(u), params)
                direct = np.asarray(qfun(jnp.asarray(u), *parr), dtype=float)
                logged = np.asarray(qfun(jnp.asarray(np.log(u)), *parr, log_p=True), dtype=float)
                np.testing.assert_allclose(logged, direct, rtol=1e-8, atol=1e-10, err_msg=f"{name} q 未处理 log_p")

    def test_quantile_honours_lower_tail(self) -> None:
        """``q(u, lower_tail=False)`` 必须等于 ``q(1 - u)``。"""
        for name, _d, _p, qfun, _r, grid, _xs in _all_cases():
            for params in grid:
                u = np.asarray(_PROBS, dtype=float)
                parr = _parr(len(u), params)
                direct = np.asarray(qfun(jnp.asarray(1.0 - u), *parr), dtype=float)
                upper = np.asarray(qfun(jnp.asarray(u), *parr, lower_tail=False), dtype=float)
                np.testing.assert_allclose(upper, direct, rtol=1e-8, atol=1e-10, err_msg=f"{name} q 未处理 lower_tail")

    def test_numeric_quantile_is_self_consistent(self) -> None:
        """数值反演族（R 用 uniroot）：``p∘q`` 必须以紧容差回到概率本身。

        R 的 ``uniroot`` 只能给出 ~1.2e-4 的绝对精度，所以对 R 只能放宽比较；
        但我们自己的二分法收敛到全精度，自洽性必须用紧容差锁住。
        """
        for name, _d, pfun, qfun, _r, grid, _xs in (
            (n, *v) for n, v in CONTINUOUS_CASES.items() if n in NUMERIC_Q
        ):
            for params in grid:
                u = np.asarray(_PROBS, dtype=float)
                parr = _parr(len(u), params)
                qv = np.asarray(qfun(jnp.asarray(u), *parr), dtype=float)
                back = np.asarray(pfun(jnp.asarray(qv), *parr), dtype=float)
                np.testing.assert_allclose(back, u, rtol=1e-9, atol=1e-11, err_msg=f"{name} 数值 q∘p ≠ id")

    def test_r_quirk_inventory_is_explicit(self) -> None:
        """R 已知缺陷清单来自 ``omnilss.dpqr_batch_f.R_QUIRKS``，必须都进对照集合。"""
        self.assertTrue(R_QUIRKS)
        covered = set(CONTINUOUS_CASES) | set(DISCRETE_CASES)
        for name in R_QUIRKS:
            self.assertIn(name, covered, f"{name} 登记了 R 缺陷但没有进入对照清单")


@unittest.skipUnless(R_AVAILABLE, "R / Rscript not available")
class TestBatchFRConsistency(unittest.TestCase):
    """逐点 R 对照（需要 R + gamlss + gamlss.dist）。"""

    def test_density_matches_r(self) -> None:
        for name, dfun, _p, _q, _r, grid, xs in _all_cases():
            for params in grid:
                x, parr = _arrays(xs, params)
                got = np.asarray(dfun(jnp.asarray(x), *parr), dtype=float)
                exp = _r_values(name, "d", [x, *[np.asarray(v) for v in parr]])
                np.testing.assert_allclose(got, exp, rtol=1e-8, atol=1e-10, err_msg=f"{name} d params={params}")

    def test_log_density_matches_r(self) -> None:
        for name, dfun, _p, _q, _r, grid, xs in _all_cases():
            for params in grid:
                x, parr = _arrays(xs, params)
                got = np.asarray(dfun(jnp.asarray(x), *parr, log=True), dtype=float)
                exp = _r_values(name, "d", [x, *[np.asarray(v) for v in parr]], log_p=True)
                np.testing.assert_allclose(got, exp, rtol=1e-8, atol=1e-10, err_msg=f"{name} log d params={params}")

    def test_lower_tail_cdf_matches_r(self) -> None:
        for name, _d, pfun, _q, _r, grid, xs in _all_cases():
            for params in grid:
                x, parr = _arrays(xs, params)
                got = np.asarray(pfun(jnp.asarray(x), *parr), dtype=float)
                exp = _r_values(name, "p", [x, *[np.asarray(v) for v in parr]])
                np.testing.assert_allclose(got, exp, rtol=1e-8, atol=1e-10, err_msg=f"{name} p params={params}")

    def test_upper_tail_cdf_matches_r(self) -> None:
        for name, _d, pfun, _q, _r, grid, xs in _all_cases():
            for params in grid:
                x, parr = _arrays(xs, params)
                got = np.asarray(pfun(jnp.asarray(x), *parr, lower_tail=False), dtype=float)
                exp = _r_values(name, "p", [x, *[np.asarray(v) for v in parr]], lower_tail=False)
                np.testing.assert_allclose(got, exp, rtol=1e-8, atol=1e-10, err_msg=f"{name} 上尾 p params={params}")

    def test_log_cdf_matches_r(self) -> None:
        for name, _d, pfun, _q, _r, grid, xs in _all_cases():
            for params in grid:
                x, parr = _arrays(xs, params)
                got = np.asarray(pfun(jnp.asarray(x), *parr, log_p=True), dtype=float)
                exp = _r_values(name, "p", [x, *[np.asarray(v) for v in parr]], log_p=True)
                np.testing.assert_allclose(got, exp, rtol=1e-8, atol=1e-10, err_msg=f"{name} log p params={params}")

    def test_quantile_matches_r_lower_tail(self) -> None:
        probs = [0.05, 0.25, 0.5, 0.75, 0.95]
        for name, _d, _p, qfun, _r, grid, _xs in _all_cases():
            rtol, atol = NUMERIC_Q.get(name, (1e-8, 1e-10))
            for params in grid:
                u = np.asarray(probs, dtype=float)
                parr = _parr(len(u), params)
                got = np.asarray(qfun(jnp.asarray(u), *parr), dtype=float)
                exp = _r_values(name, "q", [u, *[np.asarray(v) for v in parr]])
                np.testing.assert_allclose(
                    got, exp, rtol=rtol, atol=atol, err_msg=f"{name} q params={params}"
                )

    def test_r_ignores_tail_flags_for_the_mn_quantiles(self) -> None:
        """实测确认 R 的 qMN3/qMN4/qMN5 忽略 ``lower.tail`` 与 ``log.p``。

        R 的实现里 ``lower.tail``/``log.p`` 只是出现在形式参数表上，函数体完全没有
        用到它们，所以四种组合返回完全相同的值。我们按数学正确的方式处理，这条
        测试把"是 R 的缺陷而不是我们的差异"钉死。
        """
        for name, _d, _p, qfun, _r, grid, _xs, _k in ((n, *v) for n, v in DISCRETE_CASES.items()):
            params = grid[1]
            u = np.asarray([0.15, 0.45, 0.85], dtype=float)
            parr = [np.full(len(u), float(v)) for v in params]
            base = _r_values(name, "q", [u, *parr], lower_tail=True, log_p=False)
            for lt, lp in ((False, False), (True, True), (False, True)):
                other = _r_values(name, "q", [u, *parr], lower_tail=lt, log_p=lp)
                np.testing.assert_allclose(
                    base, other, err_msg=f"R {name} 开始处理 lower.tail/log.p 了 —— 可以去掉这条豁免"
                )
            ours = np.asarray(qfun(jnp.asarray(u), *[jnp.asarray(v) for v in parr]), dtype=float)
            np.testing.assert_allclose(ours, base, rtol=1e-8, atol=1e-10, err_msg=f"{name} q 与 R 不一致")

    def test_r_lqno_quantile_honours_log_p(self) -> None:
        """R 的 ``qLQNO`` *会* 正确处理 log.p（``if (log.p) p <- exp(p)``），
        所以这里要求两侧在 log 尺度上完全一致 —— 与 MN 族形成对照。"""
        params = (2.0, 0.3)
        u = np.asarray([0.05, 0.5, 0.95], dtype=float)
        parr = [np.full(len(u), float(v)) for v in params]
        exp = _r_values("LQNO", "q", [np.log(u), *parr], log_p=True)
        got = np.asarray(qLQNO(jnp.asarray(np.log(u)), *[jnp.asarray(v) for v in parr], log_p=True), dtype=float)
        np.testing.assert_allclose(got, exp, rtol=1e-8, atol=1e-10, err_msg="LQNO q 的 log_p 分支与 R 不一致")


if __name__ == "__main__":
    unittest.main()
