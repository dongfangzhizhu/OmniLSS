"""Link metadata gate: every registered family must carry usable link functions.

Background
----------
``FamilyDefinition`` exposes ``links`` / ``link_functions`` / ``link_inverses`` /
``link_derivatives``.  Several families were registered by ``build_ad_family``
without any of them, which silently leaves ``link_functions=None``.  While the
family is still a ``placeholder`` nothing happens — the capability gate refuses
it.  But as soon as the family is promoted (D1 migration), the gate lets it
through and ``rs_fit`` dies deep inside the algorithm with an opaque
``TypeError: 'NoneType' object is not subscriptable`` instead of a family-level
diagnostic.

That is exactly what happened to ``PARETO1`` once its d/p/q/r kernels landed, so
this gate closes the whole class of defect rather than the one instance.

Run:
    cd omnilss && JAX_ENABLE_X64=true python -m pytest tests/test_family_link_metadata_gate.py -q
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "omnilss" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from omnilss.distribution_registry import list_families, resolve  # noqa: E402

#: Link names the factory machinery understands.
KNOWN_LINK_NAMES = {
    "identity",
    "log",
    "logit",
    "probit",
    "cloglog",
    "inverse",
    "sqrt",
    "1/mu^2",
    "1/nu^2",
    "1/tau^2",
    "logshiftto1",
    "logshiftto2",
    "logitshiftto1",
    "own",
}


def _all_families():
    for name in list_families():
        try:
            yield name, resolve(name)
        except Exception as exc:  # noqa: BLE001
            raise AssertionError(f"resolve({name}) failed: {type(exc).__name__}: {exc}") from exc


class TestFamilyLinkMetadataGate(unittest.TestCase):
    """两部分：结构完整性（必须有）+ 行为自洽（链接/逆必须互为反函数）。"""

    def test_every_family_declares_complete_link_metadata(self) -> None:
        broken = []
        for name, family in _all_families():
            params = tuple(family.parameters)
            links = family.links or {}
            missing = [p for p in params if p not in links]
            problems = []
            if family.links is None:
                problems.append("links is None")
            if family.link_functions is None:
                problems.append("link_functions is None")
            if family.link_inverses is None:
                problems.append("link_inverses is None")
            if family.link_derivatives is None:
                problems.append("link_derivatives is None")
            if missing:
                problems.append(f"missing link entries for {missing}")
            if problems:
                broken.append(f"{name}{params}: {'; '.join(problems)}")
        self.assertEqual(
            [],
            broken,
            "以下族缺少 link 元数据；一旦它们不再是 placeholder，rs_fit 会以\n"
            "TypeError: 'NoneType' object is not subscriptable 崩溃：\n"
            + "\n".join(broken),
        )

    def test_link_names_are_recognised(self) -> None:
        unknown = []
        for name, family in _all_families():
            for param, link_name in (family.links or {}).items():
                if link_name not in KNOWN_LINK_NAMES:
                    unknown.append(f"{name}.{param}={link_name!r}")
        self.assertEqual([], unknown, f"未知 link 名称: {unknown}")

    def test_link_inverse_round_trips(self) -> None:
        """``link_inverse(link(mu)) ≈ mu`` 在合法 mu 上必须成立。"""
        failures = []
        eta_probe = jnp.asarray([-1.0, -0.25, 0.0, 0.5, 1.5], dtype=jnp.float64)
        for name, family in _all_families():
            for param, link_name in (family.links or {}).items():
                inverse = (family.link_inverses or {}).get(param)
                link = (family.link_functions or {}).get(param)
                if inverse is None or link is None:
                    continue
                try:
                    mu = np.asarray(inverse(eta_probe), dtype=float)
                    back = np.asarray(link(jnp.asarray(mu)), dtype=float)
                except Exception as exc:  # noqa: BLE001
                    failures.append(f"{name}.{param}({link_name}): {type(exc).__name__}: {exc}")
                    continue
                if not np.all(np.isfinite(mu)):
                    failures.append(f"{name}.{param}({link_name}): inverse produced {mu}")
                    continue
                expected = np.asarray(eta_probe, dtype=float)
                if not np.allclose(back, expected, rtol=1e-6, atol=1e-6):
                    failures.append(f"{name}.{param}({link_name}): {back} != {expected}")
        self.assertEqual([], failures, "\n".join(failures[:20]))

    def test_link_derivatives_are_finite(self) -> None:
        failures = []
        eta_probe = jnp.asarray([-1.0, 0.0, 1.0], dtype=jnp.float64)
        for name, family in _all_families():
            for param, link_name in (family.links or {}).items():
                derivative = (family.link_derivatives or {}).get(param)
                if derivative is None:
                    continue
                try:
                    value = np.asarray(derivative(eta_probe), dtype=float)
                except Exception as exc:  # noqa: BLE001
                    failures.append(f"{name}.{param}({link_name}): {type(exc).__name__}: {exc}")
                    continue
                if not np.all(np.isfinite(value)):
                    failures.append(f"{name}.{param}({link_name}): derivative {value}")
        self.assertEqual([], failures, "\n".join(failures[:20]))


if __name__ == "__main__":
    unittest.main()
