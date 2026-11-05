"""R gamlss internal distribution groupings and small utilities.

R source: R/gamlss/R/chooseDistParallel.R (.realline/.realplus/.real0to1/
.realAll/.counts/.binom), R/gamlss/R/gamlss-5.R (.gamlss.bi.list/
.gamlss.multin.list/.gamlss.sm.list) and R/gamlss/R/extra.R (.hat.WX).

These are data, not behaviour: R stores the family names grouped by the support
of the response and by parameter count, and chooseDist()/gamlss() consult them
to decide which families are admissible.  The tuples below reproduce R's
vectors element for element (verified against live gamlss 5.5-3).

Naming: R's dot-prefixed names become module-level names with a leading
underscore (the coverage tooling normalises '.' to '_'), and each also has a
public upper-case spelling for readability.
"""
from __future__ import annotations

from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Support groups (chooseDistParallel.R)
# ---------------------------------------------------------------------------

# Families on the whole real line.
REAL_LINE: tuple[str, ...] = (
    "NO",
    "GU",
    "RG",
    "LO",
    "NET",
    "TF",
    "TF2",
    "PE",
    "PE2",
    "SN1",
    "SN2",
    "exGAUS",
    "SHASH",
    "SHASHo",
    "SHASHo2",
    "EGB2",
    "JSU",
    "JSUo",
    "SEP1",
    "SEP2",
    "SEP3",
    "SEP4",
    "ST1",
    "ST2",
    "ST3",
    "ST4",
    "ST5",
    "SST",
    "GT",
)

# Families on the positive half line.
REAL_PLUS: tuple[str, ...] = (
    "EXP",
    "GA",
    "IG",
    "LOGNO",
    "LOGNO2",
    "WEI",
    "WEI2",
    "WEI3",
    "IGAMMA",
    "PARETO2",
    "PARETO2o",
    "GP",
    "BCCG",
    "BCCGo",
    "exGAUS",
    "GG",
    "GIG",
    "LNO",
    "BCTo",
    "BCT",
    "BCPEo",
    "BCPE",
    "GB2",
)

# Families on the unit interval.
REAL_0_TO_1: tuple[str, ...] = (
    "BE",
    "BEo",
    "BEINF0",
    "BEINF1",
    "LOGITNO",
    "SIMPLEX",
    "BEOI",
    "BEZI",
    "BEINF",
    "GB1",
)

# R's .realAll is union(.realline, .realplus): duplicates collapse in order.
REAL_ALL: tuple[str, ...] = tuple(dict.fromkeys(REAL_LINE + REAL_PLUS))

# Counting families.
COUNTS: tuple[str, ...] = (
    "PO",
    "GEOM",
    "GEOMo",
    "LG",
    "YULE",
    "ZIPF",
    "WARING",
    "GPO",
    "DPO",
    "BNB",
    "NBF",
    "NBI",
    "NBII",
    "PIG",
    "ZIP",
    "ZIP2",
    "ZAP",
    "ZALG",
    "DEL",
    "ZAZIPF",
    "SI",
    "SICHEL",
    "ZANBI",
    "ZAPIG",
    "ZINBI",
    "ZIPIG",
    "ZINBF",
    "ZABNB",
    "ZASICHEL",
    # R's vector lists ZINBF twice; kept for element-wise parity.
    "ZINBF",
    "ZIBNB",
    "ZISICHEL",
)

# Binomial families.
BINOMIAL: tuple[str, ...] = ("BI", "BB", "DBI", "ZIBI", "ZABI", "ZIBB", "ZABB")

# ---------------------------------------------------------------------------
# Family lists consulted by gamlss() itself (gamlss-5.R)
# ---------------------------------------------------------------------------

# Families with a binomial denominator.
BI_LIST: tuple[str, ...] = (
    "BI",
    "Binomial",
    "BB",
    "Beta Binomial",
    "ZIBI",
    "ZIBB",
    "ZABI",
    "ZABB",
    "DBI",
    "BItr",
    "BBtr",
    "ZIBItr",
    "ZIBBtr",
    "ZABItr",
    "ZABBtr",
    "DBItr",
)

# Multinomial families.
MULTIN_LIST: tuple[str, ...] = ("MULTIN", "MN3", "MN4", "MN5")

# Smoother aliases the formula parser recognises in R.
SM_LIST: tuple[str, ...] = (
    "cs",
    "scs",
    "ps",
    "pb",
    "cy",
    "tp",
    "pvc",
    "pbm",
    "pbj",
    "pbo",
    "pbz",
    "pbc",
    "pbts",
    "pbp",
    "pbf",
    "pcat",
    "pba",
    "pbq",
    "gmrf",
    "mrfa",
    "mrf",
    "so",
    "SOP",
    "krig",
    "lo",
    "random",
    "re",
    "re4",
    "fp",
    "pp",
    "nl",
    "ri",
    "boost",
    "fk",
    "own",
    "test",
    "test0",
    "test1",
    "arma",
    "rw",
    "ar",
    "seas",
    "srw",
    "sar",
    "la",
    "plag",
    "tr",
    "ga",
    "ba",
    "mm",
    "nn",
    "sv",
    "ma",
    "pcr",
    "pc",
    "h2o",
    "pa",
    "gnet",
    "lrs",
    "rf1",
    "rf2",
    "gbo",
    "gsel",
)

# ---------------------------------------------------------------------------
# Leverage helper (extra.R)
# ---------------------------------------------------------------------------


def _hat_WX(w: Any, x: Any) -> np.ndarray:
    """R .hat.WX: diagonal of the hat matrix of a weighted regression.

    R builds X = cbind(1, x) for a vector x (a matrix is used as is), scales the
    rows by sqrt(w) and returns hat(qr(WX)).  The diagonal of the hat matrix is
    invariant to the QR parameterisation, so we compute it directly.
    """

    values = np.asarray(x, dtype=float)
    if values.ndim == 1:
        design = np.column_stack([np.ones(values.shape[0]), values])
    elif values.ndim == 2:
        design = values
    else:
        raise ValueError("x must be a vector or a matrix")

    weights = np.asarray(w, dtype=float).ravel()
    if design.shape[0] != weights.shape[0]:
        raise ValueError("'w' and 'x' are not having the same length")
    if not np.all(np.isfinite(np.sqrt(weights))):
        import warnings

        warnings.warn("diagonal weights has non-finite entries", UserWarning, stacklevel=2)

    weighted = design * np.sqrt(weights)[:, None]
    gram = weighted.T @ weighted
    leverage = np.sum((weighted @ np.linalg.pinv(gram)) * weighted, axis=1)
    return leverage


# R-name aliases: '.' normalises to '_'.
_realline = REAL_LINE
_realplus = REAL_PLUS
_real0to1 = REAL_0_TO_1
_realAll = REAL_ALL
_counts = COUNTS
_binom = BINOMIAL
_gamlss_bi_list = BI_LIST
_gamlss_multin_list = MULTIN_LIST
_gamlss_sm_list = SM_LIST
_hat_WX_alias = _hat_WX

__all__ = [
    "BINOMIAL",
    "BI_LIST",
    "COUNTS",
    "MULTIN_LIST",
    "REAL_ALL",
    "REAL_LINE",
    "REAL_PLUS",
    "REAL_0_TO_1",
    "SM_LIST",
    "_binom",
    "_counts",
    "_gamlss_bi_list",
    "_gamlss_multin_list",
    "_gamlss_sm_list",
    "_hat_WX",
    "_real0to1",
    "_realAll",
    "_realline",
    "_realplus",
]
