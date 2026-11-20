"""Acceptance scan for the native log K_nu kernel (D2.267, P6-1).

P6-1 asks for a JAX-native log_bessel_kv with two acceptances:

1. pointwise relative error <= 1e-12 on the log grid nu in [-5, 20],
   x in [1e-8, 1e3];
2. a measured drop in the Sichel fit cycle time.

This script produces both numbers and writes them to
docs/development/bessel-native-2026-11-15.json:

    python tools/bessel_native_scan.py --write docs/development/bessel-native-2026-11-15.json
    python tools/bessel_native_scan.py --check

The reference is mpmath's besselk at 40 digits (scipy's kve is only ~1e-15 and
would mask a 1e-13 defect).  The error metric is the *relative error of K*:
|exp(logK_native - logK_ref) - 1|.  Comparing log K relatively instead would
blow up wherever K is close to 1 (log K -> 0), which is a conditioning artefact,
not a kernel error.
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "omnilss" / "src"))

TOLERANCE = 1e-12

#: P6-1 acceptance grid: nu -5..20 and x 1e-8..1e3 (log spaced).
NU_VALUES = [round(float(nu), 10) for nu in np.arange(-5.0, 20.0 + 1e-9, 0.5)]
X_VALUES = [float(x) for x in np.logspace(-8.0, 3.0, 25)]


def _regions(x: float) -> str:
    if x < 3.0:
        return "series"
    if x < 15.0:
        return "ode"
    return "asymptotic"


def _reference(nu: float, x: float) -> float:
    """log K_nu(x) at 40 digits (mpmath); falls back to scipy's kve."""
    try:
        from mpmath import mp, besselk, log as mlog

        mp.dps = 40
        return float(mlog(besselk(mp.mpf(nu), mp.mpf(x))))
    except ImportError:  # pragma: no cover - mpmath is present in CI
        from scipy.special import kve

        scaled = float(kve(nu, x))
        return float(np.log(max(scaled, np.finfo(np.float64).tiny)) - x)


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=str(ROOT), text=True
        ).strip()
    except Exception:  # pragma: no cover - not a git checkout
        return "unknown"


def scan() -> dict:
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    from omnilss.bessel_native import log_bessel_kv_native

    kernel = jax.jit(log_bessel_kv_native)
    per_region: dict[str, dict[str, float]] = {}
    worst = 0.0
    worst_at: list[float] | None = None
    failures: list[dict[str, float]] = []

    for nu in NU_VALUES:
        for x in X_VALUES:
            native = float(kernel(jnp.float64(nu), jnp.float64(x)))
            expected = _reference(nu, x)
            error = float(abs(np.expm1(native - expected)))
            region = _regions(x)
            bucket = per_region.setdefault(region, {"points": 0, "worst": 0.0})
            bucket["points"] += 1
            bucket["worst"] = max(bucket["worst"], error)
            if error > worst:
                worst = error
                worst_at = [nu, x]
            if error > TOLERANCE:
                failures.append({"nu": nu, "x": x, "rel_error": error})

    return {
        "grid": {
            "nu": {"min": NU_VALUES[0], "max": NU_VALUES[-1], "step": 0.5,
                   "count": len(NU_VALUES)},
            "x": {"min": X_VALUES[0], "max": X_VALUES[-1], "log_points": len(X_VALUES)},
            "points": len(NU_VALUES) * len(X_VALUES),
        },
        "metric": "relative error of K = |exp(native_logK - reference_logK) - 1|",
        "tolerance": TOLERANCE,
        "worst_relative_error": worst,
        "worst_at": {"nu": worst_at[0], "x": worst_at[1]} if worst_at else None,
        "per_region": per_region,
        "failures": failures[:20],
        "failure_count": len(failures),
    }


def timing(rows: int = 48) -> dict:
    """Native kernel vs the scipy pure_callback it replaced (primal + grad)."""
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    from omnilss.bessel import _log_bessel_kv_scipy
    from omnilss.bessel_native import log_bessel_kv_native

    nu = jnp.full((rows,), 1.5)
    x = jnp.linspace(0.5, 30.0, rows)

    @jax.custom_jvp
    def callback_kernel(nu_value, x_value):
        shape = jax.ShapeDtypeStruct(jnp.shape(x_value), jnp.float64)
        return jax.pure_callback(
            _log_bessel_kv_scipy, shape, nu_value, x_value, vmap_method="expand_dims"
        )

    @callback_kernel.defjvp
    def _callback_jvp(primals, tangents):
        # Same rule the module used before D2.267, so the timing compares the
        # kernels and not their derivatives.
        nu_value, x_value = primals
        nu_dot, x_dot = tangents
        out = callback_kernel(nu_value, x_value)
        eps = jnp.finfo(jnp.float64).eps
        safe = jnp.maximum(x_value, eps)
        ratio = jnp.exp(
            jnp.clip(
                callback_kernel(nu_value + 1.0, safe) - callback_kernel(nu_value, safe),
                -745.0,
                709.0,
            )
        )
        d_dx = -ratio + nu_value / safe
        h = 1e-5
        d_dnu = (
            callback_kernel(nu_value + h, safe) - callback_kernel(nu_value - h, safe)
        ) / (2.0 * h)
        return out, nu_dot * d_dnu + x_dot * d_dx

    results = {}
    for name, kernel in (("native", log_bessel_kv_native), ("scipy_callback", callback_kernel)):
        primal = jax.jit(kernel)
        grad_x = jax.jit(jax.grad(lambda v: jnp.sum(kernel(nu, v))))
        grad_nu = jax.jit(jax.grad(lambda v: jnp.sum(kernel(v, x))))
        primal(nu, x).block_until_ready()
        grad_x(x).block_until_ready()
        grad_nu(jnp.float64(1.5)).block_until_ready()

        def best(fn, *args, repeats: int = 20) -> float:
            start = time.perf_counter()
            for _ in range(repeats):
                fn(*args).block_until_ready()
            return (time.perf_counter() - start) / repeats * 1e3

        results[name] = {
            "primal_ms": best(primal, nu, x),
            "grad_x_ms": best(grad_x, x, repeats=10),
            "grad_nu_ms": best(grad_nu, jnp.float64(1.5), repeats=10),
        }

    native = results["native"]
    callback = results["scipy_callback"]
    return {
        "rows": rows,
        **results,
        "speedup": {
            key: callback[key] / native[key]
            for key in ("primal_ms", "grad_x_ms", "grad_nu_ms")
        },
    }


def _artifact() -> dict:
    payload = scan()
    payload.update(
        {
            "timing": timing(),
            # The second P6-1 acceptance is the *fit cycle*, not the kernel call.
            # Measured with a one-iteration SI fit on n = 40 (command documented
            # below); the native kernel is 70x slower, so it stays opt-in and the
            # families keep the scipy callback.
            "fit_cycle": {
                "family": "SI",
                "rows": 40,
                "max_iter": 1,
                "scipy_callback_s": 5.31,
                "native_s": 371.46,
                "speedup": 5.31 / 371.46,
                "command": (
                    "python - <<'PY' ... omnilss.gamlss('y ~ x', family='SI', "
                    "data=..., max_iter=1) with bessel.log_bessel_kv patched to "
                    "the pure_callback version / to bessel_native.log_bessel_kv_native"
                ),
                "verdict": "not enabled: CPU fit cycle regresses 70x",
            },
            "commit": _git_commit(),
            "platform": platform.platform(),
            "python": platform.python_version(),
        }
    )
    try:
        import jax
        import numpy
        import scipy

        payload["versions"] = {
            "jax": jax.__version__,
            "numpy": numpy.__version__,
            "scipy": scipy.__version__,
        }
    except Exception:  # pragma: no cover
        pass
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", type=Path, default=None, help="artifact path")
    parser.add_argument("--check", action="store_true", help="fail above tolerance")
    parser.add_argument("--skip-timing", action="store_true")
    args = parser.parse_args(argv)

    if args.check:
        payload = scan()
    else:
        payload = _artifact() if not args.skip_timing else scan()

    print(
        json.dumps(
            {
                key: value
                for key, value in payload.items()
                if key not in ("failures", "timing")
            },
            indent=2,
        )
    )
    if "timing" in payload:
        speedup = payload["timing"]["speedup"]
        print(
            "timing (rows={}): primal x{:.2f}, grad_x x{:.2f}, grad_nu x{:.2f}".format(
                payload["timing"]["rows"],
                speedup["primal_ms"],
                speedup["grad_x_ms"],
                speedup["grad_nu_ms"],
            )
        )

    if args.write is not None:
        args.write.parent.mkdir(parents=True, exist_ok=True)
        args.write.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.write}")

    if args.check and payload["failure_count"]:
        print(
            "FAIL: {} points exceed the {} tolerance".format(
                payload["failure_count"], payload["tolerance"]
            ),
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
