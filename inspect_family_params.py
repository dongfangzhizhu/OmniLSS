"""Inspect family parameters to generate correct R consistency tests."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "omnilss" / "src"))

from omnilss.distribution_registry import list_families, resolve

# Families to inspect
priority_families = [
    "ZAPIG", "ZIPIG", "ZASICHEL", "ZISICHEL", "ZAZIPF",
    "ZABB", "ZIBB", "ZABNB", "ZIBNB",
    "GAF", "NOF", "RGE", "WEI2", "WEI3", "GB1",
    "PARETO1", "PARETO1O", "PIG2", "PE2",
    "EGB2", "LQNO", "SST", "ST5",
    "SEP", "SEP1", "SEP2", "SEP3", "SEP4",
    "LOGSHASH", "LOGSHASHO", "SHASHO", "SHASHO2", "JSUO",
    "LG", "ZIPF", "ZALG", "GPO", "DELAPORT", "DPO1",
    "GP", "GENERALISEDPOISSON",
    "LOGITNO",
]

print("Family Parameter Inspection")
print("=" * 80)

for family_name in priority_families:
    try:
        fam = resolve(family_name)
        params = fam.parameters
        print(f"{family_name:20s} -> {params}")
    except Exception as e:
        print(f"{family_name:20s} -> ERROR: {e}")

print("\n" + "=" * 80)
print("Summary by parameter count:")
print("=" * 80)

param_groups = {}
for family_name in priority_families:
    try:
        fam = resolve(family_name)
        params = fam.parameters
        n_params = len(params)
        if n_params not in param_groups:
            param_groups[n_params] = []
        param_groups[n_params].append((family_name, params))
    except Exception as e:
        pass

for n_params in sorted(param_groups.keys()):
    print(f"\n{n_params} parameters:")
    for family_name, params in param_groups[n_params]:
        print(f"  {family_name:20s} -> {params}")
