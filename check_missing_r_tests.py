"""Check which families are missing R consistency tests."""
import sys
import os
from pathlib import Path

# Add omnilss to path
sys.path.insert(0, str(Path(__file__).parent / "omnilss" / "src"))

from omnilss.distribution_registry import list_families

# Get all registered families
all_families = set(list_families())

# Find all R consistency test files
test_dir = Path(__file__).parent / "omnilss" / "tests"
r_consistency_files = list(test_dir.glob("test_r_consistency_*.py"))

# Extract family names from test files
tested_families = set()
for test_file in r_consistency_files:
    # Read file and look for family names
    content = test_file.read_text(encoding='utf-8')
    for family in all_families:
        if family in content:
            tested_families.add(family)

# Find missing families
missing_families = all_families - tested_families

print(f"Total families: {len(all_families)}")
print(f"Tested families: {len(tested_families)}")
print(f"Missing R consistency tests: {len(missing_families)}")
print("\nMissing families:")
for family in sorted(missing_families):
    print(f"  - {family}")

print("\nTested families:")
for family in sorted(tested_families):
    print(f"  ✓ {family}")
