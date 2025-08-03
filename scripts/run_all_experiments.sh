#!/usr/bin/env bash
# scripts/run_all_experiments.sh

echo "Running all experiments..."

echo "=== Running MTL No Gradient Balancing ==="
./scripts/run_ts_mtl_no_grad_bal.sh

echo "=== Running MTL with PCGrad ==="
./scripts/run_ts_mtl_pc_grad.sh

echo "=== Running MTL with CAGrad ==="
./scripts/run_ts_mtl_ca_grad.sh

echo "=== Running TSDiff (Baseline) ==="
./scripts/run_ts_diff.sh

echo "=== Running TSDiff with PCGrad ==="
./scripts/run_ts_diff_pc_grad.sh

echo "=== Running TSDiff with CAGrad ==="
./scripts/run_ts_diff_ca_grad.sh

echo "=== Running TSDiff with Gradient Balancing ==="
./scripts/run_ts_diff_grad_bal.sh

echo "All experiments completed!"
