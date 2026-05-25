from __future__ import annotations

from omnilss.distribution_registry import resolve


def test_phase1_compat_registry_names_are_resolvable() -> None:
    names = [
        "TF2", "ST1", "ST2", "ST3", "ST4", "DBI", "DBURR12",
        "BCCGUNTR", "BCPEUNTR", "BCTUNTR",
        "COUNT_1_31", "BINOM_2_33", "CONTRPLUS_2_11", "CONTR01_4_33",
        "DPO1", "PARETO1", "PARETO1O", "SEP", "BCCGO", "LOGSHASHO", "MULTIN",
        "LG", "ZIPF", "ZALG", "SHASHO", "IGA", "PIG2", "PE2", "LOGITNO", "DELAPORT",
        "GENERALISEDPOISSON", "GP", "ZASICHEL", "ZISICHEL", "ZAPIG", "ZIPIG",
        "ZABB", "ZIBB", "ZAZIPF", "ZABNB", "ZIBNB", "ZINBF",
        "GIG", "GB1", "WEI2", "WEI3", "NOF", "GAF", "RGE",
        "SEP1", "SEP2", "SEP3", "SEP4", "SST", "ST3C", "EGB2", "LQNO", "NBF",
        "LOGSHASH", "GPO", "ZABI", "ZANBI",
    ]
    for n in names:
        fam = resolve(n)
        assert fam.name.upper() == n


def test_sep_aliases_use_distinct_family_classes() -> None:
    assert resolve("SEP2").__class__.__name__ == "SEP2Family"
    assert resolve("SEP3").__class__.__name__ == "SEP3Family"
    assert resolve("SEP4").__class__.__name__ == "SEP4Family"


def test_tf2_uses_distinct_compat_family_class() -> None:
    assert resolve("TF2").__class__.__name__ == "TF2Family"


def test_st_aliases_use_distinct_family_classes() -> None:
    assert resolve("ST1").__class__.__name__ == "ST1Family"
    assert resolve("ST2").__class__.__name__ == "ST2Family"
    assert resolve("ST3").__class__.__name__ == "ST3Family"
    assert resolve("ST4").__class__.__name__ == "ST4Family"


def test_dbi_dburr12_and_boxcox_compat_use_distinct_classes() -> None:
    assert resolve("DBI").__class__.__name__ == "DBIFamily"
    assert resolve("DBURR12").__class__.__name__ == "DBURR12Family"
    assert resolve("BCCGUNTR").__class__.__name__ == "BCCGuntrFamily"
    assert resolve("BCPEUNTR").__class__.__name__ == "BCPEuntrFamily"
    assert resolve("BCTUNTR").__class__.__name__ == "BCTuntrFamily"


def test_multin_compat_shell_preserves_name_and_class() -> None:
    fam = resolve("MULTIN")
    assert fam.name == "MULTIN"
    assert fam.__class__.__name__ == "MULTINFamily"


def test_gp_alias_uses_distinct_compat_class() -> None:
    assert resolve("GP").__class__.__name__ == "GPFamily"


def test_generalisedpoisson_alias_uses_distinct_compat_class() -> None:
    assert resolve("GENERALISEDPOISSON").__class__.__name__ == "GENERALISEDPOISSONFamily"


def test_o_suffix_aliases_use_distinct_compat_classes() -> None:
    assert resolve("JSUO").__class__.__name__ == "JSUOFamily"
    assert resolve("SHASHO").__class__.__name__ == "SHASHOFamily"
    assert resolve("SHASHO2").__class__.__name__ == "SHASHO2Family"
    assert resolve("BCPEO").__class__.__name__ == "BCPEOFamily"
    assert resolve("BCTO").__class__.__name__ == "BCTOFamily"
    assert resolve("PARETO1O").__class__.__name__ == "PARETO1OFamily"
    assert resolve("BCCGO").__class__.__name__ == "BCCGOFamily"
    assert resolve("LOGSHASHO").__class__.__name__ == "LOGSHASHOFamily"


def test_zibi_zinbi_aliases_use_distinct_compat_classes() -> None:
    assert resolve("ZIBI").__class__.__name__ == "ZIBIFamily"
    assert resolve("ZINBI").__class__.__name__ == "ZINBIFamily"


def test_zaga_zaig_zap_aliases_use_distinct_compat_classes() -> None:
    assert resolve("ZAGA").__class__.__name__ == "ZAGAFamily"
    assert resolve("ZAIG").__class__.__name__ == "ZAIGFamily"
    assert resolve("ZAP").__class__.__name__ == "ZAPFamily"


def test_zabi_zanbi_aliases_use_distinct_compat_classes() -> None:
    assert resolve("ZABI").__class__.__name__ == "ZABIFamily"
    assert resolve("ZANBI").__class__.__name__ == "ZANBIFamily"


def test_zasichel_zisichel_zapig_zipig_aliases_use_distinct_compat_classes() -> None:
    assert resolve("ZASICHEL").__class__.__name__ == "ZASICHELFamily"
    assert resolve("ZISICHEL").__class__.__name__ == "ZISICHELFamily"
    assert resolve("ZAPIG").__class__.__name__ == "ZAPIGFamily"
    assert resolve("ZIPIG").__class__.__name__ == "ZIPIGFamily"


def test_zabb_zibb_zazipf_zabnb_zibnb_aliases_use_distinct_compat_classes() -> None:
    assert resolve("ZABB").__class__.__name__ == "ZABBFamily"
    assert resolve("ZIBB").__class__.__name__ == "ZIBBFamily"
    assert resolve("ZAZIPF").__class__.__name__ == "ZAZIPFFamily"
    assert resolve("ZABNB").__class__.__name__ == "ZABNBFamily"
    assert resolve("ZIBNB").__class__.__name__ == "ZIBNBFamily"
    assert resolve("ZINBF").__class__.__name__ == "ZINBFFamily"
