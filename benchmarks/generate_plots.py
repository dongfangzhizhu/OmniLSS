"""Generate performance visualization plots."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from performance.config import RESULTS_DIR


def load_results(results_file: Path) -> dict[str, Any]:
    """Load results from JSON file."""
    with open(results_file, "r") as f:
        return json.load(f)


def plot_speedup_distribution(results: list[dict], output_file: Path) -> None:
    """Plot distribution of speedup values."""
    speedups = [r["speedup"] for r in results if r["python_success"] and r["r_success"]]
    
    if not speedups:
        print("No successful results to plot")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram
    ax1.hist(speedups, bins=30, edgecolor='black', alpha=0.7)
    ax1.axvline(1.0, color='red', linestyle='--', linewidth=2, label='Equal performance')
    ax1.axvline(np.median(speedups), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(speedups):.2f}x')
    ax1.set_xlabel('Speedup (R time / Python time)', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('Distribution of Speedup Values', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Box plot
    ax2.boxplot(speedups, vert=True)
    ax2.axhline(1.0, color='red', linestyle='--', linewidth=2, label='Equal performance')
    ax2.set_ylabel('Speedup (R time / Python time)', fontsize=12)
    ax2.set_title('Speedup Box Plot', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved speedup distribution plot to {output_file}")


def plot_speedup_by_distribution(results: list[dict], output_file: Path) -> None:
    """Plot speedup by distribution family."""
    successful = [r for r in results if r["python_success"] and r["r_success"]]
    
    if not successful:
        print("No successful results to plot")
        return
    
    # Group by distribution
    by_dist: dict[str, list[float]] = {}
    for r in successful:
        dist = r["distribution"]
        if dist not in by_dist:
            by_dist[dist] = []
        by_dist[dist].append(r["speedup"])
    
    # Sort by median speedup
    dist_names = sorted(by_dist.keys(), key=lambda d: np.median(by_dist[d]), reverse=True)
    speedup_data = [by_dist[d] for d in dist_names]
    
    fig, ax = plt.subplots(figsize=(14, max(8, len(dist_names) * 0.3)))
    
    bp = ax.boxplot(speedup_data, vert=False, labels=dist_names, patch_artist=True)
    
    # Color boxes based on performance
    for patch, dist in zip(bp['boxes'], dist_names):
        median_speedup = np.median(by_dist[dist])
        if median_speedup > 1.2:
            patch.set_facecolor('lightgreen')
        elif median_speedup > 0.8:
            patch.set_facecolor('lightyellow')
        else:
            patch.set_facecolor('lightcoral')
    
    ax.axvline(1.0, color='red', linestyle='--', linewidth=2, label='Equal performance')
    ax.set_xlabel('Speedup (R time / Python time)', fontsize=12)
    ax.set_ylabel('Distribution', fontsize=12)
    ax.set_title('Speedup by Distribution Family', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved speedup by distribution plot to {output_file}")


def plot_scalability(results: list[dict], output_file: Path) -> None:
    """Plot performance scalability with data size."""
    successful = [r for r in results if r["python_success"] and r["r_success"]]
    
    if not successful:
        print("No successful results to plot")
        return
    
    # Group by data size
    by_size: dict[int, tuple[list[float], list[float]]] = {}
    for r in successful:
        size = r["data_size"]
        if size not in by_size:
            by_size[size] = ([], [])
        by_size[size][0].append(r["python_time"])
        by_size[size][1].append(r["r_time"])
    
    sizes = sorted(by_size.keys())
    python_times_mean = [np.mean(by_size[s][0]) for s in sizes]
    python_times_std = [np.std(by_size[s][0]) for s in sizes]
    r_times_mean = [np.mean(by_size[s][1]) for s in sizes]
    r_times_std = [np.std(by_size[s][1]) for s in sizes]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Absolute times
    ax1.errorbar(sizes, python_times_mean, yerr=python_times_std, 
                 marker='o', label='Python/JAX', linewidth=2, capsize=5)
    ax1.errorbar(sizes, r_times_mean, yerr=r_times_std, 
                 marker='s', label='R', linewidth=2, capsize=5)
    ax1.set_xlabel('Data Size (observations)', fontsize=12)
    ax1.set_ylabel('Time (seconds)', fontsize=12)
    ax1.set_title('Scalability: Absolute Times', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    
    # Speedup vs size
    speedups = [np.mean(by_size[s][1]) / np.mean(by_size[s][0]) for s in sizes]
    ax2.plot(sizes, speedups, marker='o', linewidth=2, markersize=8)
    ax2.axhline(1.0, color='red', linestyle='--', linewidth=2, label='Equal performance')
    ax2.set_xlabel('Data Size (observations)', fontsize=12)
    ax2.set_ylabel('Speedup (R time / Python time)', fontsize=12)
    ax2.set_title('Scalability: Speedup vs Data Size', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved scalability plot to {output_file}")


def plot_performance_heatmap(results: list[dict], output_file: Path) -> None:
    """Plot heatmap of speedup by distribution and data size."""
    successful = [r for r in results if r["python_success"] and r["r_success"]]
    
    if not successful:
        print("No successful results to plot")
        return
    
    # Group by distribution and size
    by_dist_size: dict[tuple[str, int], list[float]] = {}
    for r in successful:
        key = (r["distribution"], r["data_size"])
        if key not in by_dist_size:
            by_dist_size[key] = []
        by_dist_size[key].append(r["speedup"])
    
    # Get unique distributions and sizes
    distributions = sorted(set(r["distribution"] for r in successful))
    sizes = sorted(set(r["data_size"] for r in successful))
    
    # Create matrix
    matrix = np.full((len(distributions), len(sizes)), np.nan)
    for i, dist in enumerate(distributions):
        for j, size in enumerate(sizes):
            key = (dist, size)
            if key in by_dist_size:
                matrix[i, j] = np.median(by_dist_size[key])
    
    fig, ax = plt.subplots(figsize=(12, max(8, len(distributions) * 0.4)))
    
    im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=0.5, vmax=2.0)
    
    # Set ticks
    ax.set_xticks(range(len(sizes)))
    ax.set_yticks(range(len(distributions)))
    ax.set_xticklabels(sizes)
    ax.set_yticklabels(distributions)
    
    # Rotate x labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Speedup (R time / Python time)', rotation=270, labelpad=20, fontsize=12)
    
    # Add text annotations
    for i in range(len(distributions)):
        for j in range(len(sizes)):
            if not np.isnan(matrix[i, j]):
                text = ax.text(j, i, f'{matrix[i, j]:.2f}',
                             ha="center", va="center", color="black", fontsize=8)
    
    ax.set_xlabel('Data Size (observations)', fontsize=12)
    ax.set_ylabel('Distribution', fontsize=12)
    ax.set_title('Performance Heatmap: Speedup by Distribution and Data Size', 
                fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved performance heatmap to {output_file}")


def main():
    """Generate all plots."""
    if len(sys.argv) < 2:
        print("Usage: python generate_plots.py <results_file.json>")
        print()
        print("Available results files:")
        for f in sorted((RESULTS_DIR / "raw").glob("*.json")):
            print(f"  {f}")
        return 1
    
    results_file = Path(sys.argv[1])
    if not results_file.exists():
        print(f"Error: File not found: {results_file}")
        return 1
    
    print(f"Loading results from {results_file}...")
    data = load_results(results_file)
    results = data["results"]
    
    print(f"Loaded {len(results)} results")
    print()
    
    # Create output directory
    timestamp = data["timestamp"]
    output_dir = RESULTS_DIR / "plots" / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Generating plots...")
    print()
    
    # Generate plots
    plot_speedup_distribution(results, output_dir / "speedup_distribution.png")
    plot_speedup_by_distribution(results, output_dir / "speedup_by_distribution.png")
    plot_scalability(results, output_dir / "scalability.png")
    plot_performance_heatmap(results, output_dir / "performance_heatmap.png")
    
    print()
    print(f"All plots saved to {output_dir}")
    print()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
