"""Permissible link names per family parameter, generated from live R.

R source: gamlss.dist show.link() in R/gamlss.dist/R/make-link-gamlss.R.
Regenerate by dumping show.link(family) for every exported family (see
docs/development/full-project-audit-2026-11-05.md for the audit context).

This is data, not behaviour: it is the link.List each family declares for
its parameters, exactly what R show.link() returns. Keys are R family
constructor names; lookup in omnilss.gamlss_family.show_link is
case-insensitive with an exact-match preference.
"""

from __future__ import annotations

FAMILY_LINK_OPTIONS: dict[str, dict[str, list[str]]] = {
    "BB": {
        "mu": ["logit", "probit", "cloglog", "cauchit", "log", "own"],
        "sigma": ["inverse", "log", "identity", "sqrt", "own"],
    },
    "BCCG": {
        "mu": ["inverse", "log", "identity", "own"],
        "sigma": ["inverse", "log", "identity", "own"],
        "nu": ["inverse", "log", "identity", "own"],
    },
    "BCCGo": {
        "mu": ["inverse", "log", "identity", "own"],
        "sigma": ["inverse", "log", "identity", "own"],
        "nu": ["inverse", "log", "identity", "own"],
    },
    "BCCGuntr": {
        "mu": ["1/mu^2", "log", "identity"],
        "sigma": ["inverse", "log", "identity"],
        "nu": ["1/nu^2", "log", "identity"],
    },
    "BCPE": {
        "mu": ["inverse", "log", "identity", "own"],
        "sigma": ["inverse", "log", "identity", "own"],
        "nu": ["inverse", "log", "identity", "own"],
        "tau": ["logshiftto1", "log", "identity", "own"],
    },
    "BCPEo": {
        "mu": ["inverse", "log", "identity", "own"],
        "sigma": ["inverse", "log", "identity", "own"],
        "nu": ["inverse", "log", "identity", "own"],
        "tau": ["logshiftto1", "log", "identity", "own"],
    },
    "BCPEuntr": {
        "mu": ["1/mu^2", "log", "identity"],
        "sigma": ["inverse", "log", "identity"],
        "nu": ["1/nu^2", "log", "identity"],
        "tau": ["1/tau^2", "log", "identity"],
    },
    "BCT": {
        "mu": ["inverse", "log", "identity", "own"],
        "sigma": ["inverse", "log", "identity", "own"],
        "nu": ["inverse", "log", "identity", "own"],
        "tau": ["inverse", "log", "identity", "own"],
    },
    "BCTo": {
        "mu": ["inverse", "log", "identity", "own"],
        "sigma": ["inverse", "log", "identity", "own"],
        "nu": ["inverse", "log", "identity", "own"],
        "tau": ["inverse", "log", "identity", "own"],
    },
    "BCTuntr": {
        "mu": ["1/mu^2", "log", "identity"],
        "sigma": ["inverse", "log", "identity"],
        "nu": ["1/nu^2", "log", "identity"],
        "tau": ["1/tau^2", "log", "identity"],
    },
    "BE": {
        "mu": ["logit", "probit", "cloglog", "cauchit", "log", "own"],
        "sigma": ["logit", "probit", "cloglog", "cauchit", "log", "own"],
    },
    "BEINF": {
        "mu": ["logit", "probit", "cloglog", "cauchit", "log", "own"],
        "sigma": ["logit", "probit", "cloglog", "cauchit", "log", "own"],
        "nu": ["inverse", "log", "identity", "own"],
        "tau": ["inverse", "log", "identity", "own"],
    },
    "BEINF0": {
        "mu": ["logit", "probit", "cloglog", "cauchit", "log", "own"],
        "sigma": ["logit", "probit", "cloglog", "cauchit", "log", "own"],
        "nu": ["inverse", "log", "identity", "own"],
    },
    "BEINF1": {
        "mu": ["logit", "probit", "cloglog", "cauchit", "log", "own"],
        "sigma": ["logit", "probit", "cloglog", "cauchit", "log", "own"],
        "nu": ["inverse", "log", "identity", "own"],
    },
    "BEOI": {
        "mu": ["logit", "probit", "cloglog", "log", "own"],
        "sigma": ["inverse", "log", "identity"],
        "nu": ["logit", "probit", "cloglog", "log", "own"],
    },
    "BEZI": {
        "mu": ["logit", "probit", "cloglog", "log", "own"],
        "sigma": ["inverse", "log", "identity"],
        "nu": ["logit", "probit", "cloglog", "log", "own"],
    },
    "BEo": {
        "mu": ["inverse", "log", "identity", "own"],
        "sigma": ["inverse", "log", "identity", "own"],
    },
    "BI": {"mu": ["logit", "probit", "cloglog", "cauchit", "log", "own"]},
    "BNB": {
        "mu": ["1/mu^2", "log", "identity"],
        "sigma": ["inverse", "log", "identity"],
        "nu": ["1/nu^2", "log", "identity"],
    },
    "DBI": {
        "mu": ["logit", "probit", "cloglog", "cauchit", "log", "own"],
        "sigma": ["inverse", "log", "identity", "sqrt"],
    },
    "DBURR12": {
        "mu": ["1/mu^2", "log", "identity"],
        "sigma": ["inverse", "log", "identity"],
        "nu": ["1/nu^2", "log", "identity"],
    },
    "DEL": {
        "mu": ["1/mu^2", "log", "identity"],
        "sigma": ["inverse", "log", "identity"],
        "nu": ["logit", "probit", "cloglog", "cauchit", "log"],
    },
    "DPO": {
        "mu": ["inverse", "log", "identity", "sqrt"],
        "sigma": ["inverse", "log", "identity", "sqrt"],
    },
    "EGB2": {
        "mu": ["inverse", "log", "identity", "own"],
        "sigma": ["inverse", "log", "identity", "own"],
        "nu": ["1/nu^2", "log", "identity", "own"],
        "tau": ["1/tau^2", "log", "identity", "own"],
    },
    "EXP": {"mu": ["inverse", "log", "sqrt", "identity"]},
    "GA": {
        "mu": ["inverse", "log", "identity", "own"],
        "sigma": ["inverse", "log", "identity", "own"],
    },
    "GAF": {
        "mu": ["1/mu^2", "log", "identity"],
        "sigma": ["inverse", "log", "identity"],
        "nu": ["1/mu^2", "log", "identity"],
    },
    "GB1": {
        "mu": ["logit", "probit", "cloglog", "log", "own"],
        "sigma": ["logit", "probit", "cloglog", "log", "own"],
        "nu": ["inverse", "log", "identity", "own"],
        "tau": ["inverse", "log", "identity", "own"],
    },
    "GB2": {
        "mu": ["1/mu^2", "log", "identity"],
        "sigma": ["inverse", "log", "identity"],
        "nu": ["1/nu^2", "log", "identity"],
        "tau": ["1/tau^2", "log", "identity"],
    },
    "GEOM": {"mu": ["log", "probit", "cloglog", "cauchit", "log", "own"]},
    "GEOMo": {"mu": ["logit", "probit", "cloglog", "cauchit", "log", "own"]},
    "GG": {
        "mu": ["1/mu^2", "log", "identity"],
        "sigma": ["inverse", "log", "identity"],
        "nu": ["1/nu^2", "log", "identity"],
    },
    "GIG": {
        "mu": ["1/mu^2", "log", "identity", "own"],
        "sigma": ["inverse", "log", "identity", "own"],
        "nu": ["inverse", "log", "identity", "own"],
    },
    "GP": {
        "mu": ["1/mu^2", "log", "identity", "own"],
        "sigma": ["1/sigma^2", "log", "identity", "own"],
    },
    "GPO": {
        "mu": ["inverse", "log", "identity", "sqrt"],
        "sigma": ["inverse", "log", "identity", "sqrt"],
    },
    "GT": {
        "mu": ["inverse", "log", "identity", "own"],
        "sigma": ["inverse", "log", "identity", "own"],
        "nu": ["inverse", "log", "identity", "own"],
        "tau": ["inverse", "log", "identity", "own"],
    },
    "GU": {
        "mu": ["inverse", "log", "identity", "own"],
        "sigma": ["inverse", "log", "identity", "own"],
    },
    "IG": {
        "mu": ["1/mu^2", "inverse", "log", "identity", "own"],
        "sigma": ["inverse", "log", "identity", "own"],
    },
    "IGAMMA": {
        "mu": ["inverse", "log", "identity", "own"],
        "sigma": ["inverse", "log", "identity", "own"],
    },
    "JSU": {
        "mu": ["inverse", "log", "identity", "own"],
        "sigma": ["inverse", "log", "identity", "own"],
        "nu": ["inverse", "log", "identity", "own"],
        "tau": ["inverse", "log", "identity", "own"],
    },
    "JSUo": {
        "mu": ["inverse", "log", "identity", "own"],
        "sigma": ["inverse", "log", "identity", "own"],
        "nu": ["inverse", "log", "identity", "own"],
        "tau": ["inverse", "log", "identity", "own"],
    },
    "LG": {"mu": ["logit", "probit", "cloglog", "cauchit", "log", "own"]},
    "LO": {
        "mu": ["inverse", "log", "identity", "own"],
        "sigma": ["inverse", "log", "identity", "own"],
    },
    "LOGITNO": {
        "mu": ["logit", "probit", "cloglog", "cauchit", "log", "own"],
        "sigma": ["inverse", "log", "identity"],
    },
    "LOGNO": {
        "mu": ["inverse", "log", "identity", "own"],
        "sigma": ["inverse", "log", "identity", "own"],
    },
    "LOGNO2": {
        "mu": ["inverse", "log", "identity"],
        "sigma": ["inverse", "log", "identity"],
    },
    "LQNO": {"mu": ["log", "identity"], "sigma": ["log", "identity"]},
    "MN3": {"mu": ["l", "o", "g"], "sigma": ["l", "o", "g"]},
    "MN4": {
        "mu": ["1/mu^2", "log", "identity"],
        "sigma": ["1/sigma^2", "log", "identity"],
        "nu": ["1/nu^2", "log", "identity"],
    },
    "MN5": {
        "mu": ["1/mu^2", "log", "identity"],
        "sigma": ["1/sigma^2", "log", "identity"],
        "nu": ["1/nu^2", "log", "identity"],
        "tau": ["1/tau^2", "log", "identity"],
    },
    "NBF": {
        "mu": ["inverse", "log", "identity"],
        "sigma": ["inverse", "log", "identity"],
        "nu": ["1/mu^2", "log", "identity"],
    },
    "NBI": {
        "mu": ["inverse", "log", "identity", "sqrt"],
        "sigma": ["inverse", "log", "identity", "sqrt"],
    },
    "NBII": {
        "mu": ["inverse", "log", "identity", "sqrt"],
        "sigma": ["inverse", "log", "identity", "sqrt"],
    },
    "NET": {
        "mu": ["inverse", "log", "identity", "own"],
        "sigma": ["inverse", "log", "identity", "own"],
        "nu": ["i", "d", "e", "n", "t", "i", "t", "y"],
        "tau": ["i", "d", "e", "n", "t", "i", "t", "y"],
    },
    "NO": {
        "mu": ["inverse", "log", "identity", "own"],
        "sigma": ["inverse", "log", "identity", "own"],
    },
    "NO2": {
        "mu": ["inverse", "log", "identity", "own"],
        "sigma": ["inverse", "log", "identity", "own"],
    },
    "NOF": {
        "mu": ["1/mu^2", "log", "identity"],
        "sigma": ["inverse", "log", "identity"],
        "nu": ["1/mu^2", "log", "identity"],
    },
    "PARETO": {"mu": ["inverse", "log", "sqrt", "identity"]},
    "PARETO1": {"mu": ["inverse", "log", "sqrt", "identity"]},
    "PARETO1o": {
        "mu": ["inverse", "log", "identity", "own"],
        "sigma": ["inverse", "log", "identity", "own"],
    },
    "PARETO2": {
        "mu": ["inverse", "log", "identity", "own"],
        "sigma": ["inverse", "log", "identity", "own"],
    },
    "PARETO2o": {
        "mu": ["inverse", "log", "identity", "own"],
        "sigma": ["inverse", "log", "identity", "own"],
    },
    "PE": {
        "mu": ["inverse", "log", "identity", "own"],
        "sigma": ["inverse", "log", "identity", "own"],
        "nu": ["logshiftto1", "log", "identity", "own"],
    },
    "PE2": {
        "mu": ["inverse", "log", "identity", "own"],
        "sigma": ["inverse", "log", "identity", "own"],
        "nu": ["logshiftto1", "log", "identity", "own"],
    },
    "PIG": {
        "mu": ["inverse", "log", "identity", "sqrt"],
        "sigma": ["inverse", "log", "identity", "sqrt"],
    },
    "PIG2": {
        "mu": ["inverse", "log", "identity", "sqrt"],
        "sigma": ["inverse", "log", "identity", "sqrt"],
    },
    "PO": {"mu": ["inverse", "log", "sqrt", "identity"]},
    "RG": {
        "mu": ["inverse", "log", "identity", "own"],
        "sigma": ["inverse", "log", "identity", "own"],
    },
    "RGE": {
        "mu": ["1/mu^2", "log", "identity"],
        "sigma": ["inverse", "log", "identity"],
        "nu": ["inverse", "log", "identity"],
    },
    "SEP": {
        "mu": ["1/mu^2", "log", "identity"],
        "sigma": ["inverse", "log", "identity"],
        "nu": ["1/nu^2", "log", "identity"],
        "tau": ["1/tau^2", "log", "identity"],
    },
    "SEP1": {
        "mu": ["inverse", "log", "identity", "own"],
        "sigma": ["inverse", "log", "identity", "own"],
        "nu": ["inverse", "log", "identity", "own"],
        "tau": ["inverse", "log", "identity", "own"],
    },
    "SEP2": {
        "mu": ["inverse", "log", "identity", "own"],
        "sigma": ["inverse", "log", "identity", "own"],
        "nu": ["inverse", "log", "identity", "own"],
        "tau": ["inverse", "log", "identity", "own"],
    },
    "SEP3": {
        "mu": ["inverse", "log", "identity", "own"],
        "sigma": ["inverse", "log", "identity", "own"],
        "nu": ["inverse", "log", "identity", "own"],
        "tau": ["inverse", "log", "identity", "own"],
    },
    "SEP4": {
        "mu": ["inverse", "log", "identity", "own"],
        "sigma": ["inverse", "log", "identity", "own"],
        "nu": ["inverse", "log", "identity", "own"],
        "tau": ["inverse", "log", "identity", "own"],
    },
    "SHASH": {
        "mu": ["inverse", "log", "identity", "own"],
        "sigma": ["inverse", "log", "identity", "own"],
        "nu": ["inverse", "log", "identity", "own"],
        "tau": ["inverse", "log", "identity", "own"],
    },
    "SHASHo": {
        "mu": ["inverse", "log", "identity", "own"],
        "sigma": ["inverse", "log", "identity", "own"],
        "nu": ["inverse", "log", "identity", "own"],
        "tau": ["inverse", "log", "identity", "own"],
    },
    "SHASHo2": {
        "mu": ["inverse", "log", "identity", "own"],
        "sigma": ["inverse", "log", "identity", "own"],
        "nu": ["inverse", "log", "identity", "own"],
        "tau": ["inverse", "log", "identity", "own"],
    },
    "SI": {
        "mu": ["1/mu^2", "log", "identity"],
        "sigma": ["inverse", "log", "identity"],
        "nu": ["1/nu^2", "log", "identity"],
    },
    "SICHEL": {
        "mu": ["1/mu^2", "log", "identity"],
        "sigma": ["inverse", "log", "identity"],
        "nu": ["1/nu^2", "log", "identity"],
    },
    "SIMPLEX": {
        "mu": ["logit", "probit", "cloglog", "cauchit", "log", "own"],
        "sigma": ["inverse", "log", "identity", "sqrt", "own"],
    },
    "SN1": {
        "mu": ["inverse", "log", "identity", "own"],
        "sigma": ["inverse", "log", "identity", "own"],
        "nu": ["inverse", "log", "identity", "own"],
    },
    "SN2": {
        "mu": ["inverse", "log", "identity", "own"],
        "sigma": ["inverse", "log", "identity", "own"],
        "nu": ["inverse", "log", "identity", "own"],
    },
    "SST": {
        "mu": ["inverse", "log", "identity", "own"],
        "sigma": ["inverse", "log", "identity", "own"],
        "nu": ["inverse", "log", "identity", "own"],
        "tau": ["logshiftto2", "log", "identity", "own"],
    },
    "ST1": {
        "mu": ["inverse", "log", "identity", "own"],
        "sigma": ["inverse", "log", "identity", "own"],
        "nu": ["inverse", "log", "identity", "own"],
        "tau": ["inverse", "log", "identity", "own"],
    },
    "ST2": {
        "mu": ["inverse", "log", "identity", "own"],
        "sigma": ["inverse", "log", "identity", "own"],
        "nu": ["inverse", "log", "identity", "own"],
        "tau": ["inverse", "log", "identity", "own"],
    },
    "ST3": {
        "mu": ["inverse", "log", "identity", "own"],
        "sigma": ["inverse", "log", "identity", "own"],
        "nu": ["inverse", "log", "identity", "own"],
        "tau": ["inverse", "log", "identity", "own"],
    },
    "ST3C": {
        "mu": ["inverse", "log", "identity", "own"],
        "sigma": ["inverse", "log", "identity", "own"],
        "nu": ["inverse", "log", "identity", "own"],
        "tau": ["inverse", "log", "identity", "own"],
    },
    "ST4": {
        "mu": ["inverse", "log", "identity", "own"],
        "sigma": ["inverse", "log", "identity", "own"],
        "nu": ["inverse", "log", "identity", "own"],
        "tau": ["inverse", "log", "identity", "own"],
    },
    "ST5": {
        "mu": ["inverse", "log", "identity", "own"],
        "sigma": ["inverse", "log", "identity", "own"],
        "nu": ["1/nu^2", "log", "identity", "own"],
        "tau": ["inverse", "log", "identity", "own"],
    },
    "TF": {
        "mu": ["inverse", "log", "identity", "own"],
        "sigma": ["inverse", "log", "identity", "own"],
        "nu": ["inverse", "log", "identity", "own"],
    },
    "TF2": {
        "mu": ["inverse", "log", "identity", "own"],
        "sigma": ["inverse", "log", "identity", "own"],
        "nu": ["logshiftto2", "inverse", "log", "identity", "own"],
    },
    "WARING": {"mu": ["l", "o", "g"], "sigma": ["l", "o", "g"]},
    "WEI": {
        "mu": ["inverse", "log", "identity", "own"],
        "sigma": ["inverse", "log", "identity", "own"],
    },
    "WEI2": {
        "mu": ["inverse", "log", "identity", "own"],
        "sigma": ["inverse", "log", "identity", "own"],
    },
    "WEI3": {
        "mu": ["inverse", "log", "identity"],
        "sigma": ["inverse", "log", "identity"],
    },
    "YULE": {"mu": ["l", "o", "g"]},
    "ZABB": {
        "mu": ["logit", "probit", "cloglog", "cauchit", "log", "own"],
        "sigma": ["inverse", "log", "identity", "sqrt", "own"],
        "nu": ["logit", "probit", "cloglog", "cauchit", "log", "own"],
    },
    "ZABI": {
        "mu": ["logit", "probit", "cloglog", "cauchit", "log", "own"],
        "sigma": ["logit", "probit", "cloglog", "cauchit", "log", "own"],
    },
    "ZABNB": {
        "mu": ["1/mu^2", "log", "identity"],
        "sigma": ["inverse", "log", "identity"],
        "nu": ["1/nu^2", "log", "identity"],
        "tau": ["logit", "probit", "cloglog", "log", "own"],
    },
    "ZAGA": {
        "mu": ["inverse", "log", "identity", "own"],
        "sigma": ["inverse", "log", "identity", "own"],
        "nu": ["logit", "probit", "cloglog", "cauchit", "log", "own"],
    },
    "ZAIG": {
        "mu": ["inverse", "log", "identity", "own"],
        "sigma": ["inverse", "log", "identity", "own"],
        "nu": ["logit", "probit", "cloglog", "cauchit", "log", "own"],
    },
    "ZALG": {
        "mu": ["logit", "probit", "cloglog", "cauchit", "log", "own"],
        "sigma": ["logit", "probit", "cloglog", "cauchit", "log", "own"],
    },
    "ZANBI": {
        "mu": ["inverse", "log", "identity"],
        "sigma": ["inverse", "log", "identity"],
        "nu": ["logit", "probit", "cloglog", "cauchit", "log", "own"],
    },
    "ZAP": {
        "mu": ["1/mu^2", "log", "identity"],
        "sigma": ["logit", "probit", "cloglog", "cauchit", "log", "own"],
    },
    "ZAPIG": {
        "mu": ["inverse", "log", "identity"],
        "sigma": ["inverse", "log", "identity"],
        "nu": ["logit", "probit", "cloglog", "cauchit", "log", "own"],
    },
    "ZASICHEL": {
        "mu": ["1/mu^2", "log", "identity"],
        "sigma": ["inverse", "log", "identity"],
        "nu": ["1/nu^2", "log", "identity"],
        "tau": ["logit", "probit", "cloglog", "log", "own"],
    },
    "ZAZIPF": {
        "mu": ["inverse", "log", "sqrt", "identity"],
        "sigma": ["logit", "probit", "cloglog", "cauchit", "log", "own"],
    },
    "ZIBB": {
        "mu": ["logit", "probit", "cloglog", "cauchit", "log", "own"],
        "sigma": ["inverse", "log", "identity", "sqrt", "own"],
        "nu": ["logit", "probit", "cloglog", "cauchit", "log", "own"],
    },
    "ZIBI": {
        "mu": ["logit", "probit", "cloglog", "cauchit", "log", "own"],
        "sigma": ["logit", "probit", "cloglog", "cauchit", "log", "own"],
    },
    "ZIBNB": {
        "mu": ["1/mu^2", "log", "identity"],
        "sigma": ["inverse", "log", "identity"],
        "nu": ["1/nu^2", "log", "identity"],
        "tau": ["logit", "probit", "cloglog", "log", "own"],
    },
    "ZINBF": {
        "mu": ["1/mu^2", "log", "identity"],
        "sigma": ["inverse", "log", "identity"],
        "nu": ["1/nu^2", "log", "identity"],
        "tau": ["logit", "probit", "cloglog", "log", "own"],
    },
    "ZINBI": {
        "mu": ["inverse", "log", "identity"],
        "sigma": ["inverse", "log", "identity"],
        "nu": ["logit", "probit", "cloglog", "log", "own"],
    },
    "ZIP": {
        "mu": ["1/mu^2", "log", "identity"],
        "sigma": ["logit", "probit", "cloglog", "cauchit", "log", "own"],
    },
    "ZIP2": {
        "mu": ["1/mu^2", "log", "identity"],
        "sigma": ["logit", "probit", "cloglog", "log"],
    },
    "ZIPF": {"mu": ["inverse", "log", "sqrt", "identity"]},
    "ZIPIG": {
        "mu": ["inverse", "log", "identity"],
        "sigma": ["inverse", "log", "identity"],
        "nu": ["logit", "probit", "cloglog", "log", "own"],
    },
    "ZISICHEL": {
        "mu": ["1/mu^2", "log", "identity"],
        "sigma": ["inverse", "log", "identity"],
        "nu": ["1/nu^2", "log", "identity"],
        "tau": ["logit", "probit", "cloglog", "log", "own"],
    },
    "exGAUS": {
        "mu": ["inverse", "log", "identity", "own"],
        "sigma": ["inverse", "log", "identity", "own"],
        "nu": ["logshifted", "log", "identity", "own"],
    },
}

__all__ = ["FAMILY_LINK_OPTIONS"]
