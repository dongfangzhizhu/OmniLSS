#!/bin/bash
# GPU benchmark script — runs inside WSL with CUDA JAX environment.
# Usage: wsl bash benchmarks/_gpu_benchmark.sh
set -e

source /mnt/d/AI/my/githubprojs/OmniLSS/debianjaxgpu/bin/activate

REPO=/mnt/d/AI/my/githubprojs/OmniLSS
export PYTHONPATH="$REPO/omnilss/src"

cd "$REPO"

python benchmarks/_gpu_benchmark_runner.py \
    --n-values 1000 5000 10000 50000 100000 200000 500000 \
    --n-reps 5 \
    --families NO GA PO BI WEI TF \
    --out-dir docs/benchmarks \
    2>&1
