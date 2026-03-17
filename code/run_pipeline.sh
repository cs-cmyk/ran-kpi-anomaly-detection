#!/usr/bin/env bash
# run_pipeline.sh — Smoke test: run the full 01→02→03→04 pipeline
# Run from repository root directory
set -euo pipefail

echo "=== Step 1/4: Synthetic Data Generation ==="
python3 01_synthetic_data.py

echo "=== Step 2/4: Feature Engineering ==="
python3 02_feature_engineering.py

echo "=== Step 3/4: Model Training ==="
python3 03_model_training.py

echo "=== Step 4/4: Evaluation ==="
python3 04_evaluation.py

echo "=== Pipeline complete. Results in data/ and results/ ==="
echo "=== To run production pattern demos: python3 05_production_patterns.py ==="
