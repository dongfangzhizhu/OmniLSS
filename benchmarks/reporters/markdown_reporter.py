"""Markdown report generator."""

from __future__ import annotations

from pathlib import Path
from datetime import datetime
from typing import Any

import numpy as np


def generate_markdown_report(
    results: list[Any],
    output_file: Path | str,
    title: str = "Omni GAMLSS Performance Report",
) -> None:
    """Generate a Markdown performance report.
    
    Parameters
    ----------
    results : list
        List of ComparisonResult objects
    output_file : Path or str
        Output file path
    title : str
        Report title
    """
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, "w", encoding="utf-8") as f:
        # Header
        f.write(f"# {title}\n\n")
        f.write(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("---\n\n")
        
        # Executive Summary
        f.write("## Executive Summary\n\n")
        
        successful = [r for r in results if r.python_success and r.r_success]
        python_only = [r for r in results if r.python_success and not r.r_success]
        r_only = [r for r in results if not r.python_success and r.r_success]
        both_failed = [r for r in results if not r.python_success and not r.r_success]
        
        f.write(f"- **Total Benchmarks**: {len(results)}\n")
        f.write(f"- **Both Succeeded**: {len(successful)} ({len(successful) / len(results) * 100:.1f}%)\n")
        f.write(f"- **Python Only**: {len(python_only)}\n")
        f.write(f"- **R Only**: {len(r_only)}\n")
        f.write(f"- **Both Failed**: {len(both_failed)}\n\n")
        
        if successful:
            speedups = [r.speedup for r in successful]
            f.write("### Performance Overview\n\n")
            f.write(f"- **Mean Speedup**: {np.mean(speedups):.2f}x\n")
            f.write(f"- **Median Speedup**: {np.median(speedups):.2f}x\n")
            f.write(f"- **Min Speedup**: {np.min(speedups):.2f}x\n")
            f.write(f"- **Max Speedup**: {np.max(speedups):.2f}x\n\n")
            
            faster = len([r for r in successful if r.speedup > 1.0])
            similar = len([r for r in successful if 0.8 <= r.speedup <= 1.0])
            slower = len([r for r in successful if r.speedup < 0.8])
            
            f.write(f"- **Python Faster**: {faster} ({faster / len(successful) * 100:.1f}%)\n")
            f.write(f"- **Similar Performance**: {similar} ({similar / len(successful) * 100:.1f}%)\n")
            f.write(f"- **Python Slower**: {slower} ({slower / len(successful) * 100:.1f}%)\n\n")
        
        f.write("---\n\n")
        
        # Detailed Results
        f.write("## Detailed Results\n\n")
        
        if successful:
            f.write("### Successful Benchmarks\n\n")
            f.write("| Distribution | Data Size | Model | Python Time (s) | R Time (s) | Speedup | Deviance Diff |\n")
            f.write("|--------------|-----------|-------|-----------------|------------|---------|---------------|\n")
            
            for r in sorted(successful, key=lambda x: x.speedup, reverse=True):
                dev_diff = f"{r.deviance_diff:.6f}" if r.deviance_diff is not None else "N/A"
                f.write(f"| {r.distribution} | {r.data_size} | {r.model_config} | "
                       f"{r.python_time:.4f} | {r.r_time:.4f} | {r.speedup:.2f}x | {dev_diff} |\n")
            
            f.write("\n")
        
        # Performance by Distribution
        if successful:
            f.write("### Performance by Distribution\n\n")
            
            # Group by distribution
            by_dist: dict[str, list] = {}
            for r in successful:
                if r.distribution not in by_dist:
                    by_dist[r.distribution] = []
                by_dist[r.distribution].append(r)
            
            f.write("| Distribution | Count | Mean Speedup | Median Speedup | Min | Max |\n")
            f.write("|--------------|-------|--------------|----------------|-----|-----|\n")
            
            for dist in sorted(by_dist.keys()):
                results_for_dist = by_dist[dist]
                speedups = [r.speedup for r in results_for_dist]
                f.write(f"| {dist} | {len(results_for_dist)} | "
                       f"{np.mean(speedups):.2f}x | {np.median(speedups):.2f}x | "
                       f"{np.min(speedups):.2f}x | {np.max(speedups):.2f}x |\n")
            
            f.write("\n")
        
        # Performance by Data Size
        if successful:
            f.write("### Performance by Data Size\n\n")
            
            # Group by data size
            by_size: dict[int, list] = {}
            for r in successful:
                if r.data_size not in by_size:
                    by_size[r.data_size] = []
                by_size[r.data_size].append(r)
            
            f.write("| Data Size | Count | Mean Speedup | Median Speedup | Mean Python Time (s) | Mean R Time (s) |\n")
            f.write("|-----------|-------|--------------|----------------|----------------------|-----------------|\n")
            
            for size in sorted(by_size.keys()):
                results_for_size = by_size[size]
                speedups = [r.speedup for r in results_for_size]
                py_times = [r.python_time for r in results_for_size]
                r_times = [r.r_time for r in results_for_size]
                f.write(f"| {size} | {len(results_for_size)} | "
                       f"{np.mean(speedups):.2f}x | {np.median(speedups):.2f}x | "
                       f"{np.mean(py_times):.4f} | {np.mean(r_times):.4f} |\n")
            
            f.write("\n")
        
        # Problem Cases
        if python_only or r_only or both_failed:
            f.write("### Problem Cases\n\n")
            
            if python_only:
                f.write("#### Python Succeeded, R Failed\n\n")
                f.write("| Distribution | Data Size | Model |\n")
                f.write("|--------------|-----------|-------|\n")
                for r in python_only:
                    f.write(f"| {r.distribution} | {r.data_size} | {r.model_config} |\n")
                f.write("\n")
            
            if r_only:
                f.write("#### R Succeeded, Python Failed\n\n")
                f.write("| Distribution | Data Size | Model |\n")
                f.write("|--------------|-----------|-------|\n")
                for r in r_only:
                    f.write(f"| {r.distribution} | {r.data_size} | {r.model_config} |\n")
                f.write("\n")
            
            if both_failed:
                f.write("#### Both Failed\n\n")
                f.write("| Distribution | Data Size | Model |\n")
                f.write("|--------------|-----------|-------|\n")
                for r in both_failed:
                    f.write(f"| {r.distribution} | {r.data_size} | {r.model_config} |\n")
                f.write("\n")
        
        # Recommendations
        f.write("---\n\n")
        f.write("## Recommendations\n\n")
        
        if successful:
            slow_cases = [r for r in successful if r.speedup < 0.5]
            if slow_cases:
                f.write("### Performance Issues\n\n")
                f.write("The following cases show significant performance degradation (speedup < 0.5x):\n\n")
                for r in sorted(slow_cases, key=lambda x: x.speedup):
                    f.write(f"- **{r.distribution}** ({r.data_size} obs): {r.speedup:.2f}x speedup\n")
                f.write("\n")
                f.write("**Action Items**:\n")
                f.write("1. Profile these specific cases to identify bottlenecks\n")
                f.write("2. Compare implementation details with R code\n")
                f.write("3. Consider JAX-specific optimizations (JIT, vectorization)\n\n")
            
            high_dev_diff = [r for r in successful if r.deviance_rel_diff and r.deviance_rel_diff > 0.01]
            if high_dev_diff:
                f.write("### Numerical Accuracy Issues\n\n")
                f.write("The following cases show significant deviance differences (>1%):\n\n")
                for r in sorted(high_dev_diff, key=lambda x: x.deviance_rel_diff or 0, reverse=True):
                    f.write(f"- **{r.distribution}** ({r.data_size} obs): "
                           f"{r.deviance_rel_diff * 100:.2f}% relative difference\n")
                f.write("\n")
                f.write("**Action Items**:\n")
                f.write("1. Verify implementation correctness\n")
                f.write("2. Check numerical stability\n")
                f.write("3. Compare convergence criteria\n\n")
        
        # Footer
        f.write("---\n\n")
        f.write("*Report generated by Omni GAMLSS Performance Testing Framework*\n")
    
    print(f"Markdown report saved to: {output_file}")
