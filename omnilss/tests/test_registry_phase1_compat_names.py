from __future__ import annotations

from omnilss.distribution_registry import resolve


def test_phase1_compat_registry_names_are_resolvable() -> None:
    names = [
        "TF2", "ST1", "ST2", "ST3", "ST4", "DBI", "DBURR12",
        "BCCGUNTR", "BCPEUNTR", "BCTUNTR",
        "COUNT_1_31", "BINOM_2_33", "CONTRPLUS_2_11", "CONTR01_4_33",
    ]
    for n in names:
        fam = resolve(n)
        assert fam.name == n
