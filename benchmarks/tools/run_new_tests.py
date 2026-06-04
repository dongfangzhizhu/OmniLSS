"""Run newly generated R consistency tests and generate a report."""
import subprocess
import sys
from pathlib import Path

# All newly generated test families
NEW_TEST_FAMILIES = [
    # 1-parameter (3)
    "pareto1", "lg", "zipf",
    # 2-parameter (13)
    "zazipf", "gaf", "nof", "wei2", "wei3", "pareto1o", "pig2",
    "zalg", "gpo", "dpo1", "gp", "generalisedpoisson", "logitno",
    # 3-parameter (8)
    "zapig", "zipig", "zabb", "zibb", "rge", "pe2", "lqno", "delaport",
    # 4-parameter (18)
    "zasichel", "zisichel", "zabnb", "zibnb", "gb1", "egb2", "sst", "st5",
    "sep", "sep1", "sep2", "sep3", "sep4",
    "logshash", "logshasho", "shasho", "shasho2", "jsuo",
]

def run_test(family: str, test_dir: Path) -> dict:
    """Run a single test and return results."""
    test_file = test_dir / f"test_r_consistency_{family}.py"
    
    if not test_file.exists():
        return {
            "family": family,
            "status": "missing",
            "message": "Test file not found"
        }
    
    try:
        # Run pytest with minimal output
        result = subprocess.run(
            [sys.executable, "-m", "pytest", str(test_file), "-v", "--tb=line"],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=test_dir.parent
        )
        
        # Parse output
        if result.returncode == 0:
            if "passed" in result.stdout:
                return {
                    "family": family,
                    "status": "passed",
                    "message": "All tests passed"
                }
            else:
                return {
                    "family": family,
                    "status": "skipped",
                    "message": "Tests skipped (R not available?)"
                }
        else:
            # Extract error message
            error_lines = []
            for line in result.stdout.split('\n'):
                if "Error:" in line or "FAILED" in line or "RuntimeError" in line:
                    error_lines.append(line.strip())
            
            error_msg = "; ".join(error_lines[:3]) if error_lines else "Test failed"
            return {
                "family": family,
                "status": "failed",
                "message": error_msg[:200]  # Limit message length
            }
    
    except subprocess.TimeoutExpired:
        return {
            "family": family,
            "status": "timeout",
            "message": "Test timed out after 30 seconds"
        }
    except Exception as e:
        return {
            "family": family,
            "status": "error",
            "message": str(e)[:200]
        }

def main():
    test_dir = Path(__file__).parent / "omnilss" / "tests"
    
    print("Running newly generated R consistency tests...")
    print("=" * 80)
    print(f"Total tests to run: {len(NEW_TEST_FAMILIES)}")
    print("=" * 80)
    print()
    
    results = []
    for i, family in enumerate(NEW_TEST_FAMILIES, 1):
        print(f"[{i}/{len(NEW_TEST_FAMILIES)}] Testing {family.upper()}...", end=" ", flush=True)
        result = run_test(family, test_dir)
        results.append(result)
        
        status_symbol = {
            "passed": "[OK]",
            "failed": "[FAIL]",
            "skipped": "[SKIP]",
            "timeout": "[TIME]",
            "error": "[ERR]",
            "missing": "[MISS]"
        }.get(result["status"], "[?]")
        
        print(f"{status_symbol}")
    
    # Generate summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    status_counts = {}
    for result in results:
        status = result["status"]
        status_counts[status] = status_counts.get(status, 0) + 1
    
    print(f"Total: {len(results)}")
    for status, count in sorted(status_counts.items()):
        print(f"  {status.capitalize()}: {count}")
    
    # Show failed tests
    failed_results = [r for r in results if r["status"] in ["failed", "error", "timeout"]]
    if failed_results:
        print("\n" + "=" * 80)
        print("FAILED TESTS")
        print("=" * 80)
        for result in failed_results:
            print(f"\n{result['family'].upper()}:")
            print(f"  Status: {result['status']}")
            print(f"  Message: {result['message']}")
    
    # Save detailed report
    report_file = Path(__file__).parent / "test_report.txt"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("R Consistency Test Report\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Total tests: {len(results)}\n")
        for status, count in sorted(status_counts.items()):
            f.write(f"{status.capitalize()}: {count}\n")
        f.write("\n" + "=" * 80 + "\n")
        f.write("DETAILED RESULTS\n")
        f.write("=" * 80 + "\n\n")
        for result in results:
            f.write(f"{result['family'].upper()}: {result['status']}\n")
            if result['message']:
                f.write(f"  {result['message']}\n")
            f.write("\n")
    
    print(f"\nDetailed report saved to: {report_file}")
    
    # Return exit code based on results
    if status_counts.get("failed", 0) > 0 or status_counts.get("error", 0) > 0:
        return 1
    return 0

if __name__ == "__main__":
    sys.exit(main())
