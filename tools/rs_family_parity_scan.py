#!/usr/bin/env python
# SPDX-License-Identifier: GPL-3.0-or-later
"""Batch family parity scan against live R.

Motivation
----------
``converged=True`` does **not** imply parity.  LOGNO reported
``termination_reason="relative_deviance_converged"`` with a deviance of
8.9e12 while R sat at 173.05 (see commit 7aaea8a).  Catching that class of
defect requires comparing against live R family by family, not trusting the
fit's own convergence flag.

Doing this per family by hand is slow, so this tool runs three gates over
every family in one pass and prints a coverage report:

    gate 1  dpqr      d/p/q values vs live R (gamlss.dist)
    gate 2  deriv     score functions vs central finite differences
    gate 3  rsdev     RS fit deviance vs live R gamlss()

Gate 1 and 3 need R; gate 2 is pure Python and always runs.

Design notes
------------
All R work happens in a **single** Rscript invocation
(:mod:`tools.r_parity_batch`).  Spawning Rscript per family costs ~2s, which
makes a 100+ family scan impractical.

Parameter counts and argument names differ per family, so the probe builds
each job from the family's own ``parameters`` tuple rather than hardcoding a
signature.  Discrete families get integer-ish grids; unit-interval families
get (0,1) grids; the rest get a positive grid.

Usage
-----
    python tools/rs_family_parity_scan.py                 # all families
    python tools/rs_family_parity_scan.py --families NO,GA,LOGNO
    python tools/rs_family_parity_scan.py --json out.json # machine-readable
    python tools/rs_family_parity_scan.py --no-r          # gate 2 only
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import warnings
from pathlib import Path
from typing import NamedTuple

import numpy as np

# Grid size for derivative probing.  Kept small: the finite-difference gate
# runs three density evaluations per parameter over the whole grid.
N_GRID = 6

# IRLS cycles allowed per RS fit, applied to *both* sides so the deviance
# comparison stays like-for-like.  A well-spread response sends many families
# all the way to the cycle limit; 50 cycles pushed a full scan past 30 minutes.
N_CYC = 20
# Observations per synthetic fit.  Large enough for a stable deviance, small
# enough that the 146-family scan stays runnable.
N_OBS = 48

ROOT = Path(__file__).resolve().parents[1]
R_SCRIPT = ROOT / "tools" / "r_parity_batch.R"
RSCRIPT_CANDIDATES = [
    r"C:\Program Files\R\R-4.6.0\bin\x64\Rscript.exe",
    r"C:\Program Files\R\R-4.6.0\bin\Rscript.exe",
    r"C:\Program Files\R\R-4.5.3\bin\x64\Rscript.exe",
    r"C:\Program Files\R\R-4.5.3\bin\Rscript.exe",
    "Rscript",
]

# --- probe calibration -------------------------------------------------------
#
# The first version of this scanner carried hand-maintained UNIT_FAMILIES /
# COUNT_FAMILIES whitelists.  They were the single largest source of *false*
# failures: LOGITNO lives on (0, 1) but was probed on 0.3..4.0, and the
# zero-inflated count families (ZAGA, ZIBI, ...) were probed on a continuous
# grid.  A scanner that cries wolf teaches you to ignore it, so the grids are
# now derived from the family object itself, which already carries everything
# needed:
#
#   ``type``   Continuous / Discrete / Mixed  -> bounds the response
#   ``links``  log / logit / identity / ...   -> bounds each parameter
#
# A family that no candidate grid can probe (density never finite and
# positive) is reported as ``skip`` rather than ``fail``: "we could not check
# this" and "this is wrong" are different statements and must not be conflated.

# Parameter sampling range implied by each link.  ``log`` means the parameter
# lives on (0, inf); ``logit`` means (0, 1); ``identity`` means the real line.
_LINK_PARAM_RANGE = {
    "log": (0.5, 3.0),
    "logit": (0.25, 0.75),
    "probit": (0.05, 0.95),
    "cloglog": (0.05, 0.95),
    "identity": (-2.0, 3.0),
    "inverse": (0.5, 3.0),
    "sqrt": (0.25, 4.0),
    "logshiftto2": (2.5, 8.0),
}
_DEFAULT_PARAM_RANGE = (0.5, 3.0)

# Binomial denominator used when a family declares a fixed ``bd``.  R's
# gamlss() takes the binomial response as counts with `bd` as the trial total
# (a two-column cbind(success, failure) response); our d/p/q take the same.
_PROBE_BD = 10.0


def _response_candidates(kind: str, n: int) -> list[np.ndarray]:
    """Response grids to try, most plausible first, for a family ``type``."""
    if kind == "Discrete":
        return [
            np.arange(n, dtype=np.float64),          # 0, 1, 2, ...
            np.arange(n, dtype=np.float64) + 1.0,    # 1, 2, 3, ...
            np.linspace(0.3, 4.0, n),
            np.linspace(0.05, 0.95, n),
        ]
    if kind == "Mixed":
        # Zero-inflated / zero-adjusted: the response mixes a point mass at 0
        # with a positive (continuous or count) part.
        return [
            np.arange(n, dtype=np.float64),
            np.linspace(0.3, 4.0, n),
            np.linspace(0.05, 0.95, n),
        ]
    return [
        np.linspace(0.3, 4.0, n),
        np.linspace(0.05, 0.95, n),
        np.linspace(-3.0, 3.0, n),
        np.linspace(1.5, 8.0, n),
    ]


class Calibration(NamedTuple):
    """A probe point set that keeps the family's density finite and positive."""

    y: np.ndarray
    params: list[np.ndarray]
    bd: float | None
    ok: bool


def calibrate(family: str, n: int = 6) -> Calibration:
    """Pick a response grid and parameter values suited to ``family``.

    Candidates are scored by how many grid points yield a finite, strictly
    positive density.  ``ok`` is False when no candidate reaches ``n - 1``
    points, in which case the caller should skip rather than report a failure.
    """
    _add_src()
    from omnilss.distributions import resolve_family  # noqa: PLC0415

    fam = resolve_family(family)
    params = list(fam.parameters)
    links = fam.links or {}
    parrays = [
        np.linspace(*_LINK_PARAM_RANGE.get(links.get(p, ""), _DEFAULT_PARAM_RANGE), n)
        for p in params
    ]
    fixed = tuple(getattr(fam, "fixed_parameters", ()) or ())
    bd = _PROBE_BD if "bd" in fixed else None

    def score(y: np.ndarray) -> int:
        """Count grid points that look like they are inside the support.

        The test is *relative* to the largest density on the same grid.  A
        plain ``d > 0`` test is not enough: LOGITNO's density evaluates to a
        denormal-positive number above 1 (where it should be zero), so the
        (0.3, 4.0) grid scored a perfect 6 while only its first point was
        really in support.  Requiring a floor of ``1e-6 * max(d)`` separates
        "genuinely in support" from "numerical dust outside it".
        """
        extra = {"bd": bd} if bd is not None else {}
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                dens = np.asarray(fam.d(y, *parrays, **extra), dtype=np.float64)
        except Exception:
            return -1
        good = np.isfinite(dens) & (dens > 0.0)
        if not good.any():
            return 0
        floor = 1e-6 * np.max(dens[good])
        return int(np.sum(good & (dens >= floor)))

    candidates = _response_candidates(getattr(fam, "type", "Continuous"), n)
    if bd is not None:
        # Binomial responses are counts in 0..bd, not proportions.
        candidates.insert(0, np.arange(n, dtype=np.float64) * (bd / max(n - 1, 1)))

    scored = [(score(y), i, y) for i, y in enumerate(candidates)]
    best = max(scored, key=lambda item: (item[0], -item[1]))
    return Calibration(y=best[2], params=parrays, bd=bd, ok=best[0] >= n - 1)


def _call_args(fam, cal: Calibration, first: np.ndarray) -> list:
    """Positional arguments for a d/p/q call, in R's argument order."""
    args = [first]
    if cal.bd is not None:
        args.append(np.full(len(first), cal.bd))
    return args + list(cal.params)


def _fit_payload(family: str, jobs: list[dict], index: dict):
    """Recover the fitting data used for this family's R fit job."""
    fit_idx = (index.get(family) or {}).get("fit")
    if fit_idx is None:
        return None
    return {
        k: [float(v) for v in vals] for k, vals in jobs[fit_idx]["data"].items()
    }


def _rscript_path() -> str | None:
    for candidate in RSCRIPT_CANDIDATES:
        if candidate == "Rscript":
            import shutil

            found = shutil.which("Rscript")
            if found:
                return found
            continue
        if Path(candidate).exists():
            return candidate
    return None


def _add_src() -> None:
    src = ROOT / "omnilss" / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


_add_src()

from omnilss.distributions import resolve_family  # noqa: E402  (needs sys.path)


def discover_families() -> list[str]:
    """Return every family name the Python registry can resolve.

    Scanning ``dir(omnilss.distributions)`` only finds the ~19 names
    re-exported by that module; the rest live in ``distributions_b*.py``.
    Asking ``resolve_family`` for an unknown name makes it list all
    registered families, which is the complete source.
    """
    _add_src()
    import re

    from omnilss.distributions import resolve_family  # noqa: PLC0415

    try:
        resolve_family("__NO_SUCH_FAMILY__")
    except Exception as exc:
        names = re.findall(r"'([A-Z][A-Z0-9_]*)'", str(exc))
        if names:
            return sorted(set(names))
    # Fallback: whatever the module exports.
    mod = _dist_module()
    return sorted(a for a in dir(mod) if a.isupper() and not a.startswith("_"))


def _dist_module():
    import omnilss.distributions as d  # noqa: PLC0415

    return d


def _family_grids(family: str, n: int) -> tuple[np.ndarray, list, float | None]:
    """Backwards-compatible tuple view of :func:`calibrate`."""
    cal = calibrate(family, n)
    return cal.y, cal.params, cal.bd


# ---------------------------------------------------------------------------
# Gate 2: derivatives vs finite differences (pure Python, always runs)
# ---------------------------------------------------------------------------

def gate_derivatives(family: str) -> dict:
    """Compare each parameter's score function against central differences.

    A mismatch here means the RS/IRLS working weights are built from wrong
    derivatives -- which is exactly what drove LOGNO's sigma to 1e-18.
    """
    _add_src()
    from omnilss.distributions import resolve_family  # noqa: PLC0415

    try:
        fam = resolve_family(family)
    except Exception as exc:  # pragma: no cover - defensive
        return {"status": "skip", "detail": f"resolve failed: {exc}"}

    score_fns = getattr(fam, "score_functions", None)
    d_fn = getattr(fam, "d", None)
    if not score_fns or d_fn is None:
        return {"status": "skip", "detail": "no score_functions or density"}

    params = list(fam.parameters)
    try:
        cal = calibrate(family, N_GRID)
    except Exception as exc:  # pragma: no cover - defensive
        return {"status": "skip", "detail": f"calibration failed: {exc}"[:90]}
    if not cal.ok:
        return {"status": "skip", "detail": "no support grid yields a finite density"}

    y = cal.y
    base = list(cal.params)
    extra = {"bd": cal.bd} if cal.bd is not None else {}
    failures: list[str] = []

    for k, pname in enumerate(params):
        fn = score_fns.get(pname)
        if fn is None:
            continue

        def loglik_at(value: np.ndarray) -> np.ndarray:
            """Log-likelihood at parameter k = ``value``.

            ``fam.d`` returns a *density*, so it must be logged before
            differencing -- differencing the density itself yields the
            derivative of the density, not the score.
            """
            args = list(base)
            args[k] = value
            dens = np.asarray(d_fn(y, *args, **extra), dtype=np.float64)
            with np.errstate(divide="ignore", invalid="ignore"):
                return np.log(dens)

        h = 1e-5
        grid = base[k]
        try:
            fd = (loglik_at(grid + h) - loglik_at(grid - h)) / (2.0 * h)
        except Exception as exc:
            # A family whose density raises (e.g. DPO1 used jnp.lgamma, which
            # JAX does not provide) must not abort the whole scan -- report it
            # as a failure for that family and keep going.
            failures.append(f"{pname}: density raised {type(exc).__name__}: {exc}"[:110])
            continue

        try:
            analytic = np.asarray(fn(y, *base, **extra), dtype=np.float64)
        except Exception as exc:
            failures.append(f"{pname}: raised {type(exc).__name__}")
            continue

        finite = np.isfinite(fd) & np.isfinite(analytic)
        if not finite.any():
            failures.append(f"{pname}: no finite comparison points")
            continue
        scale = np.maximum(np.abs(fd[finite]), 1.0)
        rel = np.abs(analytic[finite] - fd[finite]) / scale
        if np.max(rel) > 1e-3:
            failures.append(
                f"{pname}: max rel diff {np.max(rel):.3e} "
                f"(analytic {np.round(analytic[finite][:2], 5)} vs "
                f"fd {np.round(fd[finite][:2], 5)})"
            )

    if failures:
        return {"status": "fail", "detail": "; ".join(failures)}
    return {"status": "pass", "detail": ""}


# ---------------------------------------------------------------------------
# Gates 1 and 3: live R comparison
# ---------------------------------------------------------------------------

def _build_jobs(families: list[str], n: int) -> tuple[list[dict], dict]:
    """Build R jobs for d/p/q plus an RS fit, one entry per family."""
    _add_src()
    rng = np.random.default_rng(20261006)
    jobs: list[dict] = []
    index: dict[str, dict] = {}

    for family in families:
        entry: dict[str, dict] = {}
        params = _family_parameters(family)
        if params is None:
            index[family] = {"dpqr": None}
            continue

        try:
            y, parrays, bd = _family_grids(family, n)
        except Exception:
            index[family] = {"dpqr": None}
            continue

        # Argument names follow gamlss.dist convention.  NOTE: the first
        # argument is named differently per type -- d* takes `x`, p* takes `q`,
        # q* takes `p`.  Getting this wrong yields "unused argument" from R.
        argnames_by_type = {"d": "x", "p": "q", "q": "p"}
        params = list(params)
        extra = ["bd"] if bd is not None else []

        for type_char in ("d", "p", "q"):
            # q* takes a probability as its first argument, so its grid must
            # be in (0, 1); d*/p* take quantiles from the family's support.
            first = np.linspace(0.1, 0.9, n) if type_char == "q" else y
            vals = [first.tolist()]
            if bd is not None:
                vals.append(np.full(n, bd).tolist())
            for parray in parrays:
                vals.append(np.asarray(parray, dtype=np.float64).tolist())
            jobs.append(
                {
                    "kind": "dpqr",
                    "family": family,
                    "type": type_char,
                    "argnames": [argnames_by_type[type_char], *extra, *params],
                    "args": vals,
                }
            )
            entry[type_char] = len(jobs) - 1

        # RS fit job
        fit_data = _fit_data(family, rng, y)
        jobs.append(
            {
                "kind": "fit",
                "family": family,
                "data": fit_data,
                "formula": "y ~ x",
                "sigma_formula": "~1",
                "n_cyc": N_CYC,
            }
        )
        entry["fit"] = len(jobs) - 1
        index[family] = entry

    return jobs, index


def _family_parameters(family: str) -> tuple[str, ...] | None:
    try:
        return tuple(resolve_family(family).parameters)
    except Exception:
        return None


def _fit_data(family: str, rng: np.random.Generator, y_grid: np.ndarray) -> dict[str, list[float]]:
    """Build a fitting sample whose response lies in the family's support.

    The calibrated grid fixes the *range* of a legal response; the sample is
    then drawn from that range in a way that matches the family's declared
    type.  Tiling the grid instead (the first attempt) produced a response with
    only six distinct values, which is degenerate enough to distort the fit --
    SST reported a deviance 80x R's and TF2 21x, and there was no way to tell a
    real defect from a data artifact.
    """
    n = N_OBS
    x = rng.normal(0.0, 1.0, n)
    lo, hi = float(np.min(y_grid)), float(np.max(y_grid))

    try:
        kind = getattr(resolve_family(family), "type", "Continuous")
    except Exception:
        kind = "Continuous"

    if kind == "Discrete":
        # Integers spanning the calibrated range, inclusive of the upper end.
        y = rng.integers(int(np.floor(lo)), int(np.ceil(hi)) + 1, n).astype(float)
    else:
        y = rng.uniform(lo, hi, n)
        if kind == "Mixed":
            # Zero-inflated / zero-adjusted: keep a point mass at zero so the
            # mixture component is actually exercised.
            mask = rng.random(n) < 0.3
            y[mask] = 0.0
    return {"y": y.tolist(), "x": x.tolist()}


def _run_r(jobs: list[dict]) -> list[dict]:
    rscript = _rscript_path()
    if rscript is None or not R_SCRIPT.exists():
        raise RuntimeError("Rscript or tools/r_parity_batch.R unavailable")

    # Pass the job list through a file: a full-family scan produces a payload
    # far beyond the Windows command-line limit (WinError 206).
    import tempfile

    with tempfile.NamedTemporaryFile(
        "w", suffix=".json", delete=False, encoding="utf-8"
    ) as handle:
        json.dump(jobs, handle)
        payload_path = handle.name
    try:
        proc = subprocess.run(
            [rscript, str(R_SCRIPT), payload_path],
            capture_output=True,
            text=True,
            timeout=1800,
        )
    finally:
        Path(payload_path).unlink(missing_ok=True)

    if proc.returncode != 0:
        raise RuntimeError(f"R failed ({proc.returncode}): {proc.stderr[:400]}")
    return json.loads(proc.stdout)


def gate_dpqr(family: str, results: list[dict], index: dict) -> dict:
    entry = index.get(family) or {}
    if entry.get("d") is None:
        return {"status": "skip", "detail": "no parameter metadata"}
    failures: list[str] = []
    missing_in_r = True
    for type_char in ("d", "p", "q"):
        r = results[entry[type_char]]
        if "error" in r:
            # "missing function dXXX" means R has no such family at all
            # (e.g. Python's DPO1 has no R counterpart -- only DPO exists).
            # That is a registry question, not a parity defect, so it skips.
            if "missing function" in r["error"]:
                continue
            failures.append(f"{type_char}: R error {r['error'][:60]}")
            continue
        missing_in_r = False
        r_vals = np.asarray(r["values"], dtype=np.float64)
        py_vals = _python_dpqr(family, type_char, entry.get("__args"))
        if py_vals is None:
            failures.append(f"{type_char}: no Python function")
            continue
        finite = np.isfinite(r_vals) & np.isfinite(py_vals)
        if not finite.any():
            failures.append(f"{type_char}: no finite overlap")
            continue
        scale = np.maximum(np.abs(r_vals[finite]), 1.0)
        rel = np.abs(py_vals[finite] - r_vals[finite]) / scale
        if np.max(rel) > 1e-6:
            failures.append(f"{type_char}: max rel {np.max(rel):.3e}")
    if missing_in_r:
        return {"status": "skip", "detail": "no d/p/q counterpart in R"}
    if failures:
        return {"status": "fail", "detail": "; ".join(failures)}
    return {"status": "pass", "detail": ""}


def _python_dpqr(family: str, type_char: str, args: list | None):
    """Evaluate the Python d/p/q for the same grid used to build the R job."""
    _add_src()
    from omnilss.distributions import resolve_family  # noqa: PLC0415

    try:
        fam = resolve_family(family)
    except Exception:
        return None
    fn = {"d": fam.d, "p": fam.p, "q": fam.q}.get(type_char)
    if fn is None:
        return None
    n = 8
    _, parrays, bd = _family_grids(family, n)
    # Match the grid used for the R job: q* takes probabilities, d*/p* take
    # quantiles from the family's support.
    if type_char == "q":
        first = np.linspace(0.1, 0.9, n)
    else:
        first = _family_grids(family, n)[0]
    call_args = [first]
    if bd is not None:
        call_args.append(np.full(n, bd))
    call_args += [np.asarray(p, dtype=np.float64) for p in parrays]
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return np.asarray(fn(*call_args), dtype=np.float64).ravel()
    except Exception:
        return None


def gate_rs_deviance(family: str, results: list[dict], index: dict, fit_data) -> dict:
    entry = index.get(family) or {}
    if entry.get("fit") is None:
        return {"status": "skip", "detail": "no fit job"}
    r = results[entry["fit"]]
    if "error" in r:
        return {"status": "skip", "detail": f"R fit error: {r['error'][:60]}"}

    _add_src()
    from omnilss import gamlss  # noqa: PLC0415

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = gamlss(
                "y ~ x",
                family=family,
                data=fit_data,
                sigma_formula="~1",
                method="RS",
                max_iter=N_CYC,
            )
    except Exception as exc:
        return {"status": "fail", "detail": f"python raised {type(exc).__name__}: {exc}"[:90]}

    py_dev = float(model.deviance)
    r_dev = float(r["deviance"])
    if not np.isfinite(py_dev):
        return {"status": "fail", "detail": f"python deviance non-finite ({py_dev})"}
    denom = max(abs(r_dev), 1.0)
    rel = abs(py_dev - r_dev) / denom
    if rel > 1e-3:
        return {
            "status": "fail",
            "detail": f"deviance {py_dev:.6g} vs R {r_dev:.6g} (rel {rel:.3e})",
        }
    if not r.get("converged", True):
        return {"status": "warn", "detail": f"R did not converge (py {py_dev:.6g})"}
    return {"status": "pass", "detail": f"{py_dev:.6g} vs R {r_dev:.6g}"}


# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--families", help="comma-separated family names")
    parser.add_argument("--json", help="write machine-readable results here")
    parser.add_argument("--no-r", action="store_true", help="run gate 2 only")
    parser.add_argument("--n", type=int, default=8, help="grid size")
    opts = parser.parse_args(argv)

    families = (
        [f.strip() for f in opts.families.split(",") if f.strip()]
        if opts.families
        else discover_families()
    )

    jobs, index = _build_jobs(families, opts.n)
    results: list[dict] = []
    r_ok = False
    if not opts.no_r:
        try:
            results = _run_r(jobs)
            r_ok = True
        except Exception as exc:
            print(f"[warn] R unavailable, running gate 2 only: {exc}", file=sys.stderr)

    rows = []
    for family in families:
        row = {"family": family}
        for gate_name, gate_fn in (
            ("deriv", lambda f: gate_derivatives(f)),
            ("dpqr", lambda f: gate_dpqr(f, results, index) if r_ok else None),
            ("rsdev", lambda f: gate_rs_deviance(f, results, index, _fit_payload(f, jobs, index)) if r_ok else None),
        ):
            try:
                info = gate_fn(family)
            except Exception as exc:  # never let one family abort the scan
                info = {"status": "fail", "detail": f"scanner raised {type(exc).__name__}: {exc}"[:110]}
            if info is not None:
                row[gate_name] = info
        rows.append(row)

    _report(rows, r_ok)
    if opts.json:
        Path(opts.json).write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(f"\nwrote {opts.json}")
    return 0


def _report(rows: list[dict], r_ok: bool) -> None:
    gates = ["deriv"] + (["dpqr", "rsdev"] if r_ok else [])
    print(f"\n{'family':10s} " + "  ".join(f"{g:8s}" for g in gates))
    print("-" * (10 + 10 * len(gates)))
    for row in rows:
        cells = []
        for g in gates:
            info = row.get(g, {"status": "skip", "detail": ""})
            cells.append(f"{info['status']:8s}")
        print(f"{row['family']:10s} " + "  ".join(cells))

    print("\n=== coverage ===")
    for g in gates:
        counts: dict[str, int] = {}
        for row in rows:
            s = row.get(g, {"status": "skip"})["status"]
            counts[s] = counts.get(s, 0) + 1
        total = sum(counts.values())
        passed = counts.get("pass", 0)
        print(
            f"{g:6s}: {passed}/{total} pass"
            + "".join(f", {k}={v}" for k, v in sorted(counts.items()) if k != "pass")
        )

    print("\n=== failures ===")
    any_fail = False
    for row in rows:
        for g in gates:
            info = row.get(g)
            if info and info["status"] in {"fail", "warn"}:
                any_fail = True
                print(f"  {row['family']:10s} {g:6s} {info['status']:5s} {info['detail'][:100]}")
    if not any_fail:
        print("  (none)")


if __name__ == "__main__":
    raise SystemExit(main())
