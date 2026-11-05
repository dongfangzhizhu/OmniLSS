"""R gamlss.prepdata data-shaping helpers.

R source: R/gamlss.prepdata/R/data_dim.R, data_str.R, data_rm.R.

R relies on data.frame semantics (factor levels, na.omit, table()) and prints
most summaries as a side effect while returning the data invisibly.  The
Python port accepts a pandas DataFrame or a plain mapping of columns, returns
the shaped frame, and returns the summary itself instead of printing it unless
print=True is requested.

Documented deviations:
* data_na_obs returns 0-based row positions; R's which() is 1-based.
* data_dim returns the summary mapping (R returns the data invisibly).
* data_distinct counts factor levels including unused ones, matching
  length(table(x)); other columns count distinct non-missing values.
"""
from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

_BANNER = "*" * 62


def _as_frame(data: Any) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        return data
    if isinstance(data, Mapping):
        return pd.DataFrame(dict(data))
    raise TypeError("the data must be a data.frame (pandas DataFrame or mapping of columns)")


def _print_banner(print: bool) -> None:
    if print:
        print(_BANNER)
        print(_BANNER)


def _na_count(column: pd.Series) -> int:
    return int(column.isna().sum())


def data_dim(data: Any, print: bool = True) -> dict[str, Any]:
    """R data_dim: class, dimensions and missing-value counts."""

    frame = _as_frame(data)
    rows, columns = frame.shape
    complete = frame.dropna()
    missing_rows = rows - complete.shape[0]
    summary = {
        "class": type(data).__name__,
        "nrow": int(rows),
        "ncol": int(columns),
        "rows_with_missing": int(missing_rows),
        "pct_missing_rows": float(missing_rows / rows * 100.0) if rows else 0.0,
        "na_per_column": {name: _na_count(frame[name]) for name in frame.columns},
    }
    if print:
        _print_banner(True)
        print(f"the R class of the data is: {summary['class']}")
        print(f"the dimensions of the data are: {rows} by {columns}")
        print(f"number of observations with missing values: {missing_rows}")
        print(f"% of NA's in the data: {summary['pct_missing_rows']}%")
        _print_banner(True)
    return summary


def data_names(data: Any, print: bool = True) -> list[str]:
    """R data_names: column names, with a note about long names."""

    frame = _as_frame(data)
    names = [str(name) for name in frame.columns]
    if print:
        _print_banner(True)
        print("the names of variables")
        print(names)
        long_names = [name for name in names if len(name) > 10]
        if long_names:
            print("*" * 33)
            print("the following names are more than 10 characters long:", long_names)
    return names


def data_omit(data: Any, print: bool = True) -> pd.DataFrame:
    """R data_omit: drop observations with any missing value."""

    frame = _as_frame(data)
    before = frame.shape
    complete = frame.dropna()
    if print:
        _print_banner(True)
        print(f"the dimensions of the data before omition are: {before[0]} x {before[1]}")
        print(
            "the dimensions of the data saved after omition are: "
            f"{complete.shape[0]} x {complete.shape[1]}"
        )
        print(f"the number of observations omited: {before[0] - complete.shape[0]}")
        _print_banner(True)
    return complete.reset_index(drop=True)


def data_na_vars(data: Any, print: bool = True) -> dict[str, int]:
    """R data_na_vars: number of missing values per column."""

    frame = _as_frame(data)
    counts = {str(name): _na_count(frame[name]) for name in frame.columns}
    if print:
        for name, value in counts.items():
            print(f"{name}: {value}")
    return counts


def data_na_obs(data: Any, print: bool = True) -> dict[str, list[int]]:
    """R data_na_obs: positions of missing values per column (0-based)."""

    frame = _as_frame(data)
    positions = {
        str(name): [int(index) for index in np.flatnonzero(frame[name].isna().to_numpy())]
        for name in frame.columns
    }
    if print:
        for name, value in positions.items():
            print(f"{name}: {value}")
    return positions


def data_distinct(data: Any, get_distinct: bool = False, print: bool = True):
    """R data_distinct: number of distinct values per column."""

    frame = _as_frame(data)
    counts: dict[str, int] = {}
    for name in frame.columns:
        column = frame[name]
        if isinstance(column.dtype, pd.CategoricalDtype):
            counts[str(name)] = int(len(column.cat.categories))
        else:
            counts[str(name)] = int(column.dropna().nunique())
    if any(value == 1 for value in counts.values()):
        import warnings

        warnings.warn(
            "at least one variable has only 1 distinct value, please remove from data",
            UserWarning,
            stacklevel=2,
        )
    if print:
        print(counts)
    return counts if get_distinct else frame


def data_rm1val(data: Any, print: bool = True) -> pd.DataFrame:
    """R data_rm1val: drop columns that hold a single value."""

    frame = _as_frame(data)
    drop: list[str] = []
    for name in frame.columns:
        column = frame[name]
        if isinstance(column.dtype, pd.CategoricalDtype):
            used = int(column.dropna().nunique())
            if used == 1:
                drop.append(name)
        elif column.dropna().nunique() == 1:
            drop.append(name)
    if drop and print:
        print("Removed the variable(s)", drop)
    return frame.drop(columns=drop)


def data_rm(data: Any, vars: str | int | Sequence[str | int], print: bool = True) -> pd.DataFrame:
    """R data_rm: drop the named (or index-selected) variables."""

    frame = _as_frame(data)
    if isinstance(vars, (str, int)):
        selected = [vars]
    else:
        selected = list(vars)
    names: list[str] = []
    for item in selected:
        if isinstance(item, str):
            names.append(item)
        else:
            names.append(frame.columns[int(item) - 1] if int(item) >= 1 else frame.columns[int(item)])
    return frame.drop(columns=[name for name in names if name in frame.columns])


def data_exclude_class(data: Any, class_out: str = "factor") -> pd.DataFrame:
    """R data_exclude_class: drop columns whose class matches class_out."""

    frame = _as_frame(data)
    if class_out == "factor":
        mask = [isinstance(frame[name].dtype, pd.CategoricalDtype) for name in frame.columns]
    elif class_out == "numeric":
        mask = [pd.api.types.is_numeric_dtype(frame[name]) for name in frame.columns]
    elif class_out in {"character", "string", "object"}:
        mask = [pd.api.types.is_object_dtype(frame[name]) for name in frame.columns]
    else:
        mask = [str(frame[name].dtype) == class_out for name in frame.columns]
    keep = [name for name, drop in zip(frame.columns, mask) if not drop]
    return frame[keep]


def data_only_continuous(data: Any) -> pd.DataFrame:
    """R data_only_continuous: keep only numeric and integer columns."""

    frame = _as_frame(data)
    keep = [
        name
        for name in frame.columns
        if pd.api.types.is_numeric_dtype(frame[name])
        and not isinstance(frame[name].dtype, pd.CategoricalDtype)
    ]
    return frame[keep]


def data_fac2num(data: Any, vars: str | Sequence[str]) -> pd.DataFrame:
    """R data_fac2num: replace factor/character columns by their level codes.

    R uses as.numeric(factor), i.e. 1-based level codes.  Missing values stay
    missing.
    """

    frame = _as_frame(data).copy()
    selected = [vars] if isinstance(vars, str) else list(vars)
    for name in selected:
        column = frame[name]
        if isinstance(column.dtype, pd.CategoricalDtype):
            codes = column.cat.codes.to_numpy(dtype=float)
        elif pd.api.types.is_object_dtype(column) or pd.api.types.is_string_dtype(column):
            codes = pd.Categorical(column).codes.astype(float)
        else:
            raise ValueError("At least one of the vars is not character or factor")
        codes[codes < 0] = np.nan
        frame[name] = codes + 1.0
    return data_omit(frame, print=False)


def data_int2num(data: Any, min_values: int = 50, show_str: bool = False, print: bool = True) -> pd.DataFrame:
    """R data_int2num: promote large-cardinality integer columns to numeric."""

    frame = _as_frame(data).copy()
    integer_columns = [
        name
        for name in frame.columns
        if pd.api.types.is_integer_dtype(frame[name]) and not isinstance(frame[name].dtype, pd.CategoricalDtype)
    ]
    if not integer_columns:
        if print:
            print("no integer vector is found")
        return frame
    converted = 0
    for name in integer_columns:
        if int(frame[name].dropna().nunique()) > min_values:
            frame[name] = frame[name].astype(float)
            converted += 1
    if show_str and print:
        print(frame.dtypes)
    if print:
        noun = "vector" if converted == 1 else "vectors"
        print(
            f"{converted} integer {noun} with more number of values than {min_values} "
            "was transformed to numeric"
        )
    return frame


def data_rmNAvars(data: Any) -> pd.DataFrame:
    """R data_rmNAvars: drop columns that contain no observed value."""

    frame = _as_frame(data)
    keep = [name for name in frame.columns if frame[name].notna().any()]
    return frame[keep]


def data_rename(data: Any, oldnames: str | Sequence[str], newnames: str | Sequence[str]) -> pd.DataFrame:
    """R data_rename: rename columns, keeping the position."""

    frame = _as_frame(data).copy()
    olds = [oldnames] if isinstance(oldnames, str) else list(oldnames)
    news = [newnames] if isinstance(newnames, str) else list(newnames)
    if not all(isinstance(name, str) for name in olds):
        raise ValueError("oldnames should be a character")
    if not all(isinstance(name, str) for name in news):
        raise ValueError("oldnames should be a character")
    if len(olds) != len(news):
        raise ValueError("oldnames and newnames should have the same length")
    mapping = {old: new for old, new in zip(olds, news) if old in frame.columns}
    return frame.rename(columns=mapping)


def data_select(data: Any, vars: str | int | Sequence[str | int]) -> pd.DataFrame:
    """R data_select: keep only the named (or index-selected) variables."""

    frame = _as_frame(data)
    selected = [vars] if isinstance(vars, (str, int)) else list(vars)
    names: list[str] = []
    for item in selected:
        names.append(item if isinstance(item, str) else frame.columns[int(item) - 1])
    return frame[[name for name in names if name in frame.columns]]


def data_shorter_names(data: Any, max: int = 5, newnames: Sequence[str] | None = None, print: bool = True) -> pd.DataFrame:
    """R data_shorter_names: truncate long column names to max characters."""

    frame = _as_frame(data).copy()
    names = [str(name) for name in frame.columns]
    if newnames is not None:
        if len(newnames) != len(names):
            raise ValueError("the newdata should have the same length than the old one")
        frame.columns = list(newnames)
        names = [str(name) for name in newnames]
    shortened = [name if len(name) < max else name[:max] for name in names]
    frame.columns = shortened
    if print:
        print(shortened)
    return frame


def data_cut(data: Any, percentage: float | None = None, seed: int = 123, print: bool = True) -> pd.DataFrame:
    """R data_cut: take a random subsample of the rows.

    R's default percentage depends on the sample size (100% below 50k rows,
    50% below 100k, 20% below 1M, 10% above).  The random stream differs from
    R's sample(); pass seed for reproducibility.
    """

    frame = _as_frame(data)
    nobs = frame.shape[0]
    if percentage is None:
        if nobs < 50000:
            percentage = 1.0
        elif nobs < 100000:
            percentage = 0.5
        elif nobs < 1000000:
            percentage = 0.2
        else:
            percentage = 0.1
    if percentage == 1:
        out = frame
    else:
        rng = np.random.default_rng(seed)
        size = int(percentage * nobs)
        index = rng.choice(nobs, size=size, replace=False)
        out = frame.iloc[np.sort(index)]
    if print:
        print(f"{percentage * 100}% of data are saved,")
        print(f"that is, {int(np.floor(percentage * nobs))} observations.")
    return out.reset_index(drop=True)


def data_part(
    data: Any,
    partition: int = 2,
    probs: Sequence[float] | None = None,
    setseed: int = 123,
    print: bool = True,
) -> pd.DataFrame:
    """R data_part: append a random train/test/valid partition column."""

    frame = _as_frame(data).copy()
    if partition <= 1:
        raise ValueError("data partition should be greater that one")
    if partition >= 21:
        raise ValueError("data partition should be less or equal that 20")

    if partition == 2:
        weights = list(probs) if probs is not None else [0.6, 0.4]
        labels = ["train", "test"]
        if abs(sum(weights) - 1.0) > 1e-10:
            raise ValueError("probs should add up to 1")
        if len(weights) != 2:
            raise ValueError("the length of probs should be 2")
        if print:
            print("data partition into two sets")
    elif partition == 3:
        weights = list(probs) if probs is not None else [0.6, 0.2, 0.2]
        labels = ["train", "valid", "test"]
        if abs(sum(weights) - 1.0) > 1e-10:
            raise ValueError("probs should add up to 1")
        if len(weights) != 3:
            raise ValueError("the length of probs should be  3")
        if print:
            print("data partition into three sets")
    else:
        weights = [1.0 / partition] * partition
        labels = [str(index + 1) for index in range(partition)]
        if print:
            print(f"{partition}-fold data partition")

    rng = np.random.default_rng(setseed)
    draws = rng.choice(len(labels), size=frame.shape[0], p=np.asarray(weights, dtype=float))
    frame["partition"] = pd.Categorical(
        [labels[index] for index in draws], categories=labels
    )
    return frame


def data_part_list(
    data: Any,
    partition: int = 2,
    probs: Sequence[float] | None = None,
    setseed: int = 123,
    print: bool = True,
) -> dict[str, pd.DataFrame]:
    """R data_part_list: split the data into the partition subsets."""

    marked = data_part(data, partition=partition, probs=probs, setseed=setseed, print=print)
    labels = list(marked["partition"].cat.categories)
    subsets = {
        label: marked.loc[marked["partition"] == label]
        .drop(columns=["partition"])
        .reset_index(drop=True)
        for label in labels
    }
    if partition >= 4:
        return {str(index + 1): subsets[label] for index, label in enumerate(labels)}
    return subsets


def _fold_assignment(nobs: int, k: int, seed: int) -> np.ndarray:
    """R's sample(rep_len(1:K, length.out = n), replace = FALSE)."""

    rng = np.random.default_rng(seed)
    folds = np.resize(np.arange(1, k + 1), nobs)
    return folds[rng.permutation(nobs)]


def data_Kfold_weights(data: Any, K: int = 10, setseed: int = 123) -> np.ndarray:
    """R data_Kfold_weights: n x K matrix, 1 where the row is outside the fold.

    Column k holds 1 for the rows used to fit when fold k is held out, matching
    t(sapply(folds, "!=", 1:K)) * 1 in R.
    """

    frame = _as_frame(data)
    folds = _fold_assignment(frame.shape[0], K, setseed)
    return (folds[:, None] != np.arange(1, K + 1)[None, :]).astype(float)


def data_Kfold_index(data: Any, K: int = 10, setseed: int = 123) -> np.ndarray:
    """R data_Kfold_index: n x K matrix with the out-of-fold row numbers."""

    frame = _as_frame(data)
    weights = data_Kfold_weights(frame, K=K, setseed=setseed)
    return weights * np.arange(1, frame.shape[0] + 1)[:, None]


def data_boot_index(data: Any, B: int = 10, setseed: int = 123) -> dict[str, list[np.ndarray]]:
    """R data_boot_index: bootstrap in-bag (IB) and out-of-bag (OOB) indices."""

    frame = _as_frame(data)
    nobs = frame.shape[0]
    rng = np.random.default_rng(setseed)
    draws = rng.integers(1, nobs + 1, size=(B, nobs))
    ib = [np.sort(row) for row in draws]
    oob = [np.setdiff1d(np.arange(1, nobs + 1), row) for row in ib]
    return {"IB": ib, "OOB": oob}


def data_boot_weights(data: Any, B: int = 10, setseed: int = 123) -> np.ndarray:
    """R data_boot_weights: n x B bootstrap frequency matrix."""

    frame = _as_frame(data)
    nobs = frame.shape[0]
    indices = data_boot_index(frame, B=B, setseed=setseed)["IB"]
    weights = np.zeros((nobs, B), dtype=float)
    for column, row in enumerate(indices):
        counts = np.bincount(row.astype(int), minlength=nobs + 1)
        weights[:, column] = counts[1 : nobs + 1]
    return weights


__all__ = [
    "data_boot_index",
    "data_boot_weights",
    "data_cut",
    "data_Kfold_index",
    "data_Kfold_weights",
    "data_part",
    "data_part_list",
    "data_rename",
    "data_select",
    "data_shorter_names",
    "data_dim",
    "data_distinct",
    "data_exclude_class",
    "data_fac2num",
    "data_int2num",
    "data_names",
    "data_na_obs",
    "data_na_vars",
    "data_omit",
    "data_only_continuous",
    "data_rm",
    "data_rm1val",
    "data_rmNAvars",
]
