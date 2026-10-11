"""D2.30: re-run OUR RS fit for a few families on the *full-family* stream.

The scanner's fit samples come from one shared rng consumed in family order,
so a family's data depends on how many families precede it (D2.18 criterion
1).  Registrations bind to the full-family stream; re-fitting on a sublist
answers a different question.  This script rebuilds the full-family job list
(the stream up to the requested families) and runs only our side of the fit
with the scanner's exact configuration, so a start-value change can be
judged against the registration's own numbers without a 30-minute scan.

Usage: python tools/d230_refit_full_stream.py ZAP ZAPIG ZIPIG
"""

from __future__ import annotations

import json
import os
import sys
import warnings

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "omnilss"))

from tools.rs_family_parity_scan import (  # noqa: E402
    N_CYC,
    _build_jobs,
    discover_families,
)


def main() -> int:
    targets = sys.argv[1:] or ["ZAP", "ZAPIG", "ZIPIG"]
    jobs, index = _build_jobs(discover_families(), 8)

    from omnilss import gamlss  # noqa: PLC0415

    export: dict[str, dict] = {}
    for family in targets:
        fit_idx = (index.get(family) or {}).get("fit")
        if fit_idx is None:
            print(f"{family}: no fit job")
            continue
        data = dict(jobs[fit_idx]["data"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = gamlss(
                "y ~ x",
                family=family,
                data=data,
                sigma_formula="~1",
                method="RS",
                max_iter=N_CYC,
            )
        coefs = {
            k: [float(v) for v in np.atleast_1d(np.asarray(vals))]
            for k, vals in dict(model.coefficients).items()
        }
        export[family] = {
            "data": data,
            "coefficients": coefs,
            "deviance_python": float(model.deviance),
        }
        print(
            f"{family}: deviance {float(model.deviance):.10f} "
            f"converged={getattr(model, 'converged', None)} "
            f"termination={getattr(model, 'termination_reason', None)}"
        )

    out_path = os.path.join(_REPO, "omnilss", "tmp", "d230_ours.json")
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(export, handle)
    print(f"exported -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
