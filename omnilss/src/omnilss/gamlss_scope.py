"""R gamlss core gamlss.scope() migration.

R source: R/gamlss/R/stepGAIC-03-10-13.R::gamlss.scope.

Builds the named list of candidate model scopes (one per predictor) used by the
stepwise model-selection helpers.  For each predictor that R considers a vector
it appends a smooth call; factors (categorical columns) get only the linear
term, matching R's is.vector() test.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _is_vector_column(values: Any) -> bool:
    """Mirror R's is.vector(frame[[name]]) for a data-frame column."""

    try:  # pandas Categorical / factor is NOT a vector in R
        import pandas as pd

        if isinstance(values, pd.Series) and isinstance(
            values.dtype, pd.CategoricalDtype
        ):
            return False
    except Exception:  # pragma: no cover - pandas is optional here
        pass
    return np.asarray(values).ndim == 1


def gamlss_scope(
    frame: Any,
    response: int | str = 1,
    smoother: str = "cs",
    arg: str | None = None,
    form: bool = True,
) -> dict[str, Any]:
    """R's gamlss.scope: candidate formula/term list for each predictor.

    response is a 1-based column index (or a name); smoother names the smooth
    call; arg is an extra argument string such as 'df=5'; form=True returns
    formula strings ('~1+x+cs(x)'), form=False returns raw term lists.
    """

    if hasattr(frame, "columns"):
        names = list(frame.columns)

        def _get(name: str) -> Any:
            return frame[name]

    else:
        names = list(frame.keys())

        def _get(name: str) -> Any:
            return frame[name]

    if isinstance(response, str):
        response_name = response
    else:
        response_name = names[int(response) - 1]

    predictors = [name for name in names if name != response_name]
    result: dict[str, Any] = {}
    for name in predictors:
        terms = ["1", name]
        if _is_vector_column(_get(name)):
            call = f"{smoother}({name}{'' if arg is None else ',' + str(arg)})"
            terms.append(call)
        result[name] = "~" + "+".join(terms) if form else terms
    return result


__all__ = ["gamlss_scope"]
