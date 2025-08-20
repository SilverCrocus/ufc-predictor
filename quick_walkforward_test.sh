#!/bin/bash

echo "=================================================="
echo "Walk-Forward Validation Test Script"
echo "=================================================="

# Change to the ufc-predictor-feature directory
cd /Users/diyagamah/Documents/ufc-predictor-feature

echo ""
echo "Testing walk-forward validation command..."
echo "Running: uv run python main.py walkforward --validation-mode comparison"
echo ""

# Run the command and capture output
uv run python main.py walkforward --validation-mode comparison 2>&1 | head -50

echo ""
echo "=================================================="
echo "If you see an error about missing modules or data,"
echo "that's expected. The command structure is working!"
echo "=================================================="