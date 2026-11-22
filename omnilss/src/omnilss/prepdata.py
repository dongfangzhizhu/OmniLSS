# SPDX-License-Identifier: GPL-3.0-or-later
"""gamlss.prepdata: data diagnostics, associations and maximal correlation
(D2.254).

R source: R/gamlss.prepdata/R/*.R (ACE5.R, data_outliers.R, data_zscores.R,
data_bucket.R, data_leverage.R, data_association.R, cor_perm_test.R,
data_plot.R, data_xyplot.R).

The R package is mostly exploratory plotting built on ggplot2 plus a few
statistical cores.  The port implements the cores and returns structured
results; plotting is optional (matplotlib when present) and pixel parity is out
of scope, as for the other presentation-layer symbols.

Outlier rules follow R exactly:
  z-scores   fit the SHASHo distribution by ML and flag |residual| > value
  quantile   flag values outside median +/- value * mad (gamlss.ggplots y_dots)
  default    value = |qnorm(1 / (10 n))|
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import stats


class ZScores(np.ndarray):
    """ndarray carrying the fitted distribution parameters (R's attribute)."""

    parameters: dict[str, float]

    def __new__(cls, values: Any, parameters: Mapping[str, float] | None = None):
        obj = np.asarray(values, dtype=np.float64).view(cls)
        obj.parameters = dict(parameters or {})
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.parameters = getattr(obj, "parameters", {})


def _default_value(n: int) -> float:
    return float(abs(stats.norm.ppf(1.0 / (10.0 * max(n, 1)))))


def _numeric(series: Any, name: str) -> np.ndarray:
    values = np.asarray(series)
    if values.dtype.kind not in "iuf":
        raise ValueError(f"{name} should be numeric")
    return np.asarray(values, dtype=np.float64)


def _frame(data: Any) -> pd.DataFrame:
    if data is None:
        raise ValueError("data are required")
    if isinstance(data, pd.DataFrame):
        frame = data.copy()
    elif isinstance(data, Mapping):
        frame = pd.DataFrame(dict(data))
    else:
        raise ValueError("the data should be a data.frame")
    if frame.shape[0] <= 1:
        raise ValueError("nothing to do for this data frame")
    return frame


def _numeric_columns(frame: pd.DataFrame) -> list[str]:
    return [
        name for name in frame.columns if np.asarray(frame[name]).dtype.kind in "iuf"
    ]


def _nominal(series: Any) -> bool:
    values = np.asarray(series)
    return values.dtype.kind in "OUS" or isinstance(series.dtype, pd.CategoricalDtype)


def _power_transform(x: np.ndarray) -> tuple[np.ndarray, float]:
    """y_Ptrans: choose the power minimising the Jarque-Bera statistic."""

    if np.any(x < 0):
        return x, 1.0

    def jb(power: float) -> float:
        transformed = np.log(x) if abs(power) < 1e-3 else np.power(x, power)
        return float(stats.jarque_bera(transformed).statistic)

    from scipy.optimize import minimize_scalar

    result = minimize_scalar(jb, bounds=(0.0, 1.5), method="bounded")
    power = float(result.x)
    transformed = np.log(x) if abs(power) < 1e-3 else np.power(x, power)
    return transformed, power


def y_zscores(
    x: Any = None,
    weights: Any = None,
    family: Any = "SHASHo",
    value: float = 3.0,
    plot: bool = False,
    hist: bool = False,
    transform: bool = False,
) -> ZScores:
    """R y_zscores: ML fit of one variable, return its quantile residuals."""

    from . import gamlss

    values = _numeric(x, "x")
    n = values.size
    weights = np.ones(n) if weights is None else np.asarray(weights, dtype=np.float64)
    if transform:
        values, _ = _power_transform(values)
    # RS, not the staged gamlss_ml path: on the SHASHo fixture gamlss_ml stops
    # at a far worse local optimum (deviance 514.65 against R's 326.65), while
    # RS reproduces R's ML fit.  y_zscores needs the ML optimum, not the path.
    model = gamlss(
        "y ~ 1", family=family, data={"y": values}, weights=weights, max_iter=200
    )
    residuals = np.asarray(model.residuals, dtype=np.float64)
    parameters = {}
    for name in model.family.parameters:
        if name in model.fitted_values:
            parameters[name] = float(np.ravel(model.fitted_values[name])[0])
    return ZScores(residuals, parameters)


def y_outliers(
    x: Any = None,
    value: float | None = None,
    family: Any = "SHASHo",
    type: str = "zscores",
    transform: bool = True,
) -> np.ndarray:
    """R y_outliers: 1-based indices of outliers (zscores or MAD rule)."""

    values = _numeric(x, "x")
    n = values.size
    if value is None:
        value = _default_value(n)
    if str(type) not in {"zscores", "quantile"}:
        raise ValueError('type must be "zscores" or "quantile"')
    if transform:
        values, _ = _power_transform(values)
    if str(type) == "zscores":
        scores = np.asarray(y_zscores(values, family=family), dtype=np.float64)
        return np.nonzero(np.abs(scores) > value)[0] + 1
    median = float(np.median(values))
    spread = float(stats.median_abs_deviation(values, scale="normal"))
    outside = np.abs(values - median) > value * spread
    return np.nonzero(outside)[0] + 1


def y_outliers_both(
    x: Any = None,
    value: float | None = None,
    family: Any = "SHASHo",
    method: str = "intersect",
) -> np.ndarray:
    """R y_outliers_both: combine the z-score and MAD rules."""

    if str(method) not in {"intersect", "union"}:
        raise ValueError('method must be "intersect" or "union"')
    values = _numeric(x, "x")
    if value is None:
        value = _default_value(values.size)
    by_scores = y_outliers(values, value=value, family=family, type="zscores")
    by_quantile = y_outliers(values, value=value, family=family, type="quantile")
    if str(method) == "intersect":
        return np.intersect1d(by_scores, by_quantile)
    return np.union1d(by_scores, by_quantile)


def y_outliers_z(
    x: Any = None,
    by: Any = None,
    value: float | None = None,
    family: Any = "SHASHo",
    transform: bool = True,
    loop: bool = False,
) -> Any:
    """R y_outliers_z: z-score outliers, optionally by group or iteratively."""

    values = _numeric(x, "x")
    if transform:
        values, _ = _power_transform(values)
    if by is not None:
        frame = pd.DataFrame({"x": values, "by": np.asarray(by)})
        return {
            level: y_outliers_z(
                group["x"].to_numpy(),
                value=value,
                family=family,
                transform=False,
                loop=loop,
            )
            for level, group in frame.groupby("by", sort=True)
        }
    n = values.size
    if value is None:
        value = _default_value(n)
    if not loop:
        scores = np.asarray(y_zscores(values, family=family), dtype=np.float64)
        return np.nonzero(np.abs(scores) > value)[0] + 1
    # R keeps the full vector and zeroes the weights of flagged points, so the
    # loop's fit and its residual vector both stay length n.
    keep = np.ones(n, dtype=bool)
    while True:
        scores = np.asarray(
            y_zscores(values, weights=keep.astype(float), family=family),
            dtype=np.float64,
        )
        flagged = np.nonzero((np.abs(scores) > value) & keep)[0]
        if flagged.size == 0:
            break
        keep[flagged] = False
    return np.nonzero(~keep)[0] + 1


def y_outliers_by(
    x: Any = None,
    by: Any = None,
    family: Any = "SHASHo",
    type: str = "zscores",
) -> dict:
    """R y_outliers_by: outliers within each level of a grouping factor."""

    if by is None:
        raise ValueError("the factor should be set")
    values = _numeric(x, "x")
    groups = np.asarray(by)
    if groups.size != values.size:
        raise ValueError("the y and factor should have the some length")
    result = {}
    for level in sorted(set(groups.tolist())):
        mask = groups == level
        n = int(mask.sum())
        value = _default_value(n)
        local = y_outliers(values[mask], value=value, family=family, type=type)
        result[str(level)] = np.nonzero(mask)[0][local - 1] + 1
    return result


def data_outliers(
    data: Any = None,
    value: float | None = None,
    min_distinct: int = 50,
    family: Any = "SHASHo",
    type: str = "zscores",
) -> dict:
    """R data_outliers: y_outliers for every numeric column."""

    frame = _frame(data)
    if frame.shape[0] < 20:
        raise ValueError("the size of the data set is too small")
    columns = [
        name
        for name in _numeric_columns(frame)
        if np.unique(np.asarray(frame[name], dtype=float)).size >= min_distinct
    ]
    if not columns:
        raise ValueError("no variable is left after taking out the factors")
    result = {}
    for name in columns:
        values = np.asarray(frame[name], dtype=np.float64)
        local_value = _default_value(values.size) if value is None else value
        result[name] = y_outliers(values, value=local_value, family=family, type=type)
    return result


def data_zscores(
    data: Any = None,
    family: Any = "SHASHo",
    plot: bool = False,
    **kwargs: Any,
) -> dict:
    """R data_zscores: z-scores for every numeric column."""

    frame = _frame(data)
    result = {}
    for name in _numeric_columns(frame):
        values = np.asarray(frame[name], dtype=np.float64)
        finite = np.isfinite(values)
        scores = np.asarray(y_zscores(values[finite], family=family), dtype=np.float64)
        if finite.all():
            result[name] = ZScores(scores, {})
        else:
            padded = np.full(values.size, np.nan)
            padded[finite] = scores
            result[name] = ZScores(padded, {})
    return result


@dataclass(frozen=True)
class PermutationTest:
    ASL: float
    observed: float
    sample: np.ndarray
    fname: str
    xname: str
    yname: str


@dataclass(frozen=True)
class CorrelationBootstrap(PermutationTest):
    mean: float
    sd: float


def _prepare_xy(x: Any, y: Any, data: Any, xname: str | None, yname: str | None):
    if data is not None:
        frame = _frame(data)
        if xname is None or yname is None:
            raise ValueError("xname and yname are required when data is passed")
        return (
            np.asarray(frame[xname], dtype=np.float64),
            np.asarray(frame[yname], dtype=np.float64),
            xname,
            yname,
        )
    xv = np.asarray(x, dtype=np.float64)
    yv = np.asarray(y, dtype=np.float64)
    if xv.size != yv.size:
        raise ValueError("x and y must have the same length")
    return xv, yv, xname or "x", yname or "y"


def cor_perm_test(
    x: Any = None,
    y: Any = None,
    data: Any = None,
    B: int = 1000,
    seed: int = 123,
    tail: str = "one",
    fun: Any = None,
    xname: str | None = None,
    yname: str | None = None,
    **kwargs: Any,
) -> PermutationTest:
    """R cor_perm_test: permutation test of a correlation statistic."""

    xv, yv, xname, yname = _prepare_xy(x, y, data, xname, yname)
    if str(tail) not in {"one", "two"}:
        raise ValueError('tail must be "one" or "two"')
    statistic = (
        fun if fun is not None else (lambda a, b: float(np.corrcoef(a, b)[0, 1]))
    )
    observed = float(statistic(xv, yv))
    rng = np.random.default_rng(seed)
    samples = np.empty(int(B))
    for index in range(int(B)):
        samples[index] = float(statistic(rng.permutation(xv), yv))
    if str(tail) == "one":
        if observed < 0:
            asl = 1.0 - float(np.mean(samples > observed))
        else:
            asl = float(np.mean(samples > observed))
    else:
        asl = float(np.mean(np.abs(samples) > abs(observed)))
    return PermutationTest(
        ASL=asl,
        observed=observed,
        sample=samples,
        fname=getattr(statistic, "__name__", "cor"),
        xname=xname,
        yname=yname,
    )


def cor_boot(
    x: Any = None,
    y: Any = None,
    data: Any = None,
    B: int = 1000,
    seed: int = 123,
    tail: str = "one",
    fun: Any = None,
    xname: str | None = None,
    yname: str | None = None,
    **kwargs: Any,
) -> CorrelationBootstrap:
    """R cor_boot: bootstrap the correlation statistic."""

    xv, yv, xname, yname = _prepare_xy(x, y, data, xname, yname)
    if str(tail) not in {"one", "two"}:
        raise ValueError('tail must be "one" or "two"')
    statistic = (
        fun if fun is not None else (lambda a, b: float(np.corrcoef(a, b)[0, 1]))
    )
    observed = float(statistic(xv, yv))
    rng = np.random.default_rng(seed)
    n = xv.size
    samples = np.empty(int(B))
    for index in range(int(B)):
        indices = rng.integers(0, n, size=n)
        samples[index] = float(statistic(xv[indices], yv[indices]))
    if str(tail) == "one":
        if observed < 0:
            asl = 1.0 - float(np.mean(samples > observed))
        else:
            asl = float(np.mean(samples > observed))
    else:
        asl = float(np.mean(np.abs(samples) > abs(observed)))
    return CorrelationBootstrap(
        ASL=asl,
        observed=observed,
        sample=samples,
        fname=getattr(statistic, "__name__", "cor"),
        xname=xname,
        yname=yname,
        mean=float(np.mean(samples)),
        sd=float(np.std(samples, ddof=1)),
    )


def _loess_smooth(x: np.ndarray, z: np.ndarray, span: float = 0.5) -> np.ndarray:
    """Local linear smoother with tricube weights (loess-style)."""

    order = np.argsort(x)
    xs = x[order]
    zs = z[order]
    n = xs.size
    k = max(int(np.ceil(span * n)), 3)
    fitted = np.empty(n)
    for i in range(n):
        distances = np.abs(xs - xs[i])
        window = np.partition(distances, min(k, n) - 1)[min(k, n) - 1]
        if window <= 0:
            fitted[i] = zs[i]
            continue
        weights = np.clip(1.0 - (distances / window) ** 3, 0.0, None) ** 3
        design = np.column_stack([np.ones(n), xs - xs[i]])
        weighted = design * weights[:, None]
        try:
            coef = np.linalg.lstsq(weighted, weights * zs, rcond=None)[0]
            fitted[i] = coef[0]
        except np.linalg.LinAlgError:  # pragma: no cover - degenerate window
            fitted[i] = float(np.average(zs, weights=weights))
    output = np.empty(n)
    output[order] = fitted
    return output


@dataclass(frozen=True)
class ACEFit:
    x: np.ndarray
    y: np.ndarray
    tx: np.ndarray
    ty: np.ndarray
    mcor: float
    mrsq: float
    r: float
    diff: float
    method: str
    trace: tuple[float, ...] = ()


def ACE(
    x: Any = None,
    y: Any = None,
    weights: Any = None,
    data: Any = None,
    con_crit: float = 0.01,
    fit_method: str = "loess",
    nseg: int = 10,
    max_df: int = 6,
    xname: str | None = None,
    yname: str | None = None,
    max_iter: int = 100,
) -> ACEFit:
    """R ACE: alternating conditional expectations (maximal correlation).

    R offers loess and P-spline smoothers; the port uses the same tricube local
    linear smoother for both (fit_method is recorded, not switched), and returns
    the trace so ACE_iter can expose the iterations.
    """

    if x is None or y is None:
        if data is None:
            raise ValueError("one of two variables is missing")
    xv, yv, xname, yname = _prepare_xy(x, y, data, xname, yname)
    if xv.size < 4:
        raise ValueError("ACE needs at least four observations")
    xc = xv - xv.mean()
    yc = yv - yv.mean()
    r0 = float(np.corrcoef(xc, yc)[0, 1])
    if abs(r0) > 0.99:
        raise ValueError(
            "ACE failed: very high linear correlation in the original variables"
        )
    theta = yc / np.sqrt(np.sum(yc**2))
    e2 = float(np.sum((yc - xc) ** 2))
    trace = [abs(r0)]
    phi = xc.copy()
    for _ in range(int(max_iter)):
        phi_raw = _loess_smooth(xc, theta)
        norm = np.sqrt(np.sum(phi_raw**2))
        if norm == 0:
            break
        phi = phi_raw / norm
        theta_raw = _loess_smooth(yc, phi)
        norm = np.sqrt(np.sum(theta_raw**2))
        if norm == 0:
            break
        theta = theta_raw / norm
        r1 = float(np.corrcoef(theta, phi)[0, 1])
        e21 = float(np.sum((theta - phi) ** 2))
        trace.append(abs(r1))
        if abs(e21 - e2) <= con_crit:
            e2 = e21
            break
        e2 = e21
    mcor = float(abs(np.corrcoef(theta, phi)[0, 1]))
    return ACEFit(
        x=xv,
        y=yv,
        tx=phi,
        ty=theta,
        mcor=mcor,
        mrsq=mcor**2,
        r=r0,
        diff=mcor - abs(r0),
        method=str(fit_method),
        trace=tuple(trace),
    )


def ACE_iter(
    x: Any = None,
    y: Any = None,
    weights: Any = None,
    data: Any = None,
    con_crit: float = 0.001,
    fit_method: str = "loess",
    nseg: int = 10,
    max_df: int = 6,
    **kwargs: Any,
) -> ACEFit:
    """R ACE.iter: ACE with the iteration trace made explicit."""

    return ACE(
        x,
        y,
        weights,
        data,
        con_crit=con_crit,
        fit_method=fit_method,
        nseg=nseg,
        max_df=max_df,
        **kwargs,
    )


def mcor(x: Any = None, y: Any = None, data: Any = None, **kwargs: Any) -> float:
    """R mcor: maximal correlation from ACE."""

    return float(ACE(x, y, data=data, **kwargs).mcor)


def data_mcor(
    data: Any = None,
    digits: int = 3,
    plot: bool = False,
    fit_method: str = "loess",
    **kwargs: Any,
) -> pd.DataFrame:
    """R data_mcor: pairwise maximal correlation matrix of numeric columns."""

    frame = _frame(data)
    columns = _numeric_columns(frame)
    if len(columns) < 2:
        raise ValueError("at least two numeric variables are required")
    matrix = np.eye(len(columns))
    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):
            value = mcor(
                frame[columns[i]].to_numpy(),
                frame[columns[j]].to_numpy(),
                fit_method=fit_method,
                **kwargs,
            )
            matrix[i, j] = matrix[j, i] = value
    return pd.DataFrame(np.round(matrix, int(digits)), index=columns, columns=columns)


def cramer_phi(x: Any, y: Any, bias_correct: bool = True, digits: int = 4) -> float:
    """R cramer_phi: Cramer's V with the bias correction."""

    values_x = np.asarray(x).astype(str)
    values_y = np.asarray(y).astype(str)
    table = pd.crosstab(values_x, values_y)
    observed = table.to_numpy(dtype=float)
    n = float(observed.sum())
    expected = np.outer(observed.sum(axis=1), observed.sum(axis=0)) / n
    chi2 = float(
        np.sum((observed - expected) ** 2 / np.where(expected == 0, 1, expected))
    )
    phi = chi2 / n
    rows = observed.shape[0]
    cols = observed.shape[1]
    if bias_correct:
        phi = max(0.0, phi - ((rows - 1) * (cols - 1) / (n - 1)))
        adjusted_cols = cols - ((cols - 1) ** 2 / (n - 1))
        adjusted_rows = rows - ((rows - 1) ** 2 / (n - 1))
        denominator = min(adjusted_rows - 1, adjusted_cols - 1)
        return (
            float(np.round(np.sqrt(phi / denominator), int(digits)))
            if denominator > 0
            else 0.0
        )
    denominator = min(rows - 1, cols - 1)
    return (
        float(np.round(np.sqrt(phi / denominator), int(digits)))
        if denominator > 0
        else 0.0
    )


def association(
    x: Any = None,
    y: Any = None,
    xname: str = None,
    yname: str = None,
    data: Any = None,
    cor_method: str = "spearman",
    adjust_cramersv_bias: bool = True,
) -> dict:
    """R association: one pairwise association with its type."""

    if data is not None:
        frame = _frame(data)
        xname = xname or "x"
        yname = yname or "y"
        x = frame[xname]
        y = frame[yname]
    else:
        xname = xname or "x"
        yname = yname or "y"
    x_nominal = _nominal(x)
    y_nominal = _nominal(y)
    if x_nominal and y_nominal:
        value = cramer_phi(x, y, bias_correct=adjust_cramersv_bias)
        kind = "cramersV"
    elif not x_nominal and not y_nominal:
        value = (
            float(abs(stats.spearmanr(x, y).statistic))
            if cor_method == "spearman"
            else float(abs(np.corrcoef(x, y)[0, 1]))
        )
        kind = "correlation"
    elif x_nominal and not y_nominal:
        design = pd.get_dummies(
            pd.Series(np.asarray(x).astype(str)), drop_first=True
        ).to_numpy(float)
        design = np.column_stack([np.ones(len(design)), design])
        response = np.asarray(y, dtype=np.float64)
        coef, *_ = np.linalg.lstsq(design, response, rcond=None)
        residual = response - design @ coef
        r2 = 1.0 - float(np.sum(residual**2)) / float(
            np.sum((response - response.mean()) ** 2)
        )
        value = float(np.sqrt(max(r2, 0.0)))
        kind = "anova"
    else:
        design = pd.get_dummies(
            pd.Series(np.asarray(y).astype(str)), drop_first=True
        ).to_numpy(float)
        design = np.column_stack([np.ones(len(design)), design])
        response = np.asarray(x, dtype=np.float64)
        coef, *_ = np.linalg.lstsq(design, response, rcond=None)
        residual = response - design @ coef
        r2 = 1.0 - float(np.sum(residual**2)) / float(
            np.sum((response - response.mean()) ** 2)
        )
        value = float(np.sqrt(max(r2, 0.0)))
        kind = "anova"
    return {"xname": xname, "yname": yname, "assoc": float(value), "type": kind}


def data_association(
    data: Any = None,
    cor_method: str = "spearman",
    adjust_cramersv_bias: bool = True,
    plot: bool = False,
    **kwargs: Any,
) -> pd.DataFrame:
    """R data_association: pairwise association table of all columns."""

    frame = _frame(data)
    rows = []
    columns = list(frame.columns)
    for i, first in enumerate(columns):
        for second in columns[i + 1 :]:
            rows.append(
                association(
                    frame[first],
                    frame[second],
                    xname=first,
                    yname=second,
                    cor_method=cor_method,
                    adjust_cramersv_bias=adjust_cramersv_bias,
                )
            )
    return pd.DataFrame(rows)


def data_bucket(
    data: Any = None, value: float = 3.0, max_levels: int = 20, **kwargs: Any
) -> pd.DataFrame:
    """R data_bucket: moment diagnostics per numeric column.

    R draws a moment (skewness/kurtosis) bucket plot per variable; the port
    returns the same moments plus the MAD-based outlier count.
    """

    frame = _frame(data)
    rows = []
    for name in _numeric_columns(frame):
        values = np.asarray(frame[name], dtype=np.float64)
        if np.unique(values).size < max_levels:
            continue
        median = float(np.median(values))
        spread = float(stats.median_abs_deviation(values, scale="normal"))
        outliers = int(np.sum(np.abs(values - median) > value * spread))
        rows.append(
            {
                "variable": name,
                "skewness": float(stats.skew(values)),
                "kurtosis": float(stats.kurtosis(values, fisher=True)),
                "outliers": outliers,
            }
        )
    if not rows:
        raise ValueError("no variable is left after taking out the factors")
    return pd.DataFrame(rows).set_index("variable")


def data_leverage(
    data: Any = None,
    response: str = None,
    weights: Any = None,
    quantile_value: float = 0.99,
    plot: bool = False,
    **kwargs: Any,
) -> dict:
    """R data_leverage: hat values of the linear model on all other columns."""

    frame = _frame(data)
    if response is None:
        raise ValueError("the response is required")
    if response not in frame:
        raise ValueError("the response should be in data")
    columns = _numeric_columns(frame)
    if response not in columns:
        raise ValueError("the response should be numeric")
    predictors = [name for name in columns if name != response]
    if not predictors:
        raise ValueError("no variable is left after taking out the factors")
    design = np.column_stack(
        [np.ones(frame.shape[0])]
        + [np.asarray(frame[name], float) for name in predictors]
    )
    weights_array = (
        np.ones(frame.shape[0]) if weights is None else np.asarray(weights, float)
    )
    root_w = np.sqrt(weights_array)
    weighted = design * root_w[:, None]
    gram = weighted.T @ weighted
    hat_matrix = weighted @ np.linalg.pinv(gram) @ weighted.T
    leverage = np.diag(hat_matrix)
    threshold = float(np.quantile(leverage, quantile_value))
    flagged = np.nonzero(leverage > threshold)[0] + 1
    return {
        "leverage": leverage,
        "threshold": threshold,
        "outliers": flagged,
        "rank": int(np.linalg.matrix_rank(design)),
    }


def data_plot(
    data: Any = None,
    value: float = 3.0,
    plot_hist: bool = True,
    **kwargs: Any,
) -> dict:
    """R data_plot: per-column plot data (histogram or category counts)."""

    frame = _frame(data)
    result = {}
    for name in frame.columns:
        values = np.asarray(frame[name])
        if values.dtype.kind in "iuf":
            finite = values[np.isfinite(values)]
            counts, edges = np.histogram(
                finite, bins=min(20, max(5, finite.size // 10))
            )
            result[name] = {"kind": "numeric", "edges": edges, "counts": counts}
        else:
            series = pd.Series(values).astype(str)
            result[name] = {
                "kind": "categorical",
                "counts": series.value_counts().to_dict(),
            }
    return result


def data_response(
    data: Any = None,
    response: str = None,
    plot: bool = False,
    **kwargs: Any,
) -> dict:
    """R data_response: response classification and z-scores."""

    frame = _frame(data)
    if response is None:
        raise ValueError("the response is required")
    if response not in frame:
        raise ValueError("the response should be in data")
    values = np.asarray(frame[response])
    if _nominal(values):
        labels = pd.Series(values).astype(str)
        if labels.nunique() > 20:
            raise ValueError("the number of levels are too big")
        kind = "binary" if labels.nunique() == 2 else "multinomial"
        return {
            "class": "nominal",
            "kind": kind,
            "levels": sorted(labels.unique().tolist()),
        }
    numeric = np.asarray(values, dtype=np.float64)
    if np.any(numeric < 0):
        suggestion = "a continuous distribution on the real line could be used"
    elif np.all((numeric > 0) & (numeric < 1)):
        suggestion = "a continuous distribution on (0,1) should be used"
    else:
        suggestion = "a continuous distribution on (0,inf) could be used"
    scores = np.asarray(y_zscores(numeric), dtype=np.float64)
    return {"class": "numeric", "suggestion": suggestion, "zscores": scores}


def data_xyplot(
    data: Any = None, response: str = None, max_levels: int = 10, **kwargs: Any
) -> dict:
    """R data_xyplot: response-versus-predictor pairs for plotting."""

    frame = _frame(data)
    if response is None:
        raise ValueError("the response is required")
    if response not in frame:
        raise ValueError("the response should be in data")
    series = np.asarray(frame[response])
    if (
        np.unique(
            series[np.isfinite(series)] if series.dtype.kind in "iuf" else series
        ).size
        < max_levels
    ):
        raise ValueError("the response do not seems to have many distinct values")
    target = (
        np.asarray(frame[response], dtype=np.float64)
        if series.dtype.kind in "iuf"
        else None
    )
    pairs = {}
    for name in frame.columns:
        if name == response:
            continue
        predictor = np.asarray(frame[name])
        pairs[name] = {"x": predictor, "y": target if target is not None else series}
    return pairs


__all__ = [
    "ACE",
    "ACEFit",
    "ACE_iter",
    "CorrelationBootstrap",
    "PermutationTest",
    "ZScores",
    "association",
    "cor_boot",
    "cor_perm_test",
    "cramer_phi",
    "data_association",
    "data_bucket",
    "data_leverage",
    "data_mcor",
    "data_outliers",
    "data_plot",
    "data_response",
    "data_xyplot",
    "data_zscores",
    "mcor",
    "y_outliers",
    "y_outliers_both",
    "y_outliers_by",
    "y_outliers_z",
    "y_zscores",
]
