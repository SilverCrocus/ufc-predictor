# Temporal Split Implementation Guide

## Overview
Your UFC predictor now supports proper temporal (time-based) splitting to avoid data leakage and get realistic performance metrics.

## The Problem with Random Splits
Previously, your model used random train/test splits which caused:
- **Data leakage**: Model trained on 2024 fights to predict 2022 fights
- **Inflated metrics**: 75-85% accuracy that wouldn't hold in production
- **Unrealistic backtests**: Performance that can't be replicated in live betting

## The Solution: Temporal Splits

### Two Distinct Workflows

#### 1. EVALUATION Mode (Default)
Get honest performance metrics during development:
```bash
# Default: Uses temporal split
python3 main.py pipeline --tune

# Explicitly use temporal split
python3 main.py pipeline --tune --temporal-split

# Old behavior (not recommended)
python3 main.py pipeline --tune --random-split
```

**What happens:**
- Training data: First 80% of fights chronologically
- Test data: Last 20% of fights chronologically
- Result: Realistic accuracy metrics (expect 60-70%)

#### 2. PRODUCTION Mode
Train on ALL data for maximum predictive power:
```bash
# Production mode: trains on 100% of data
python3 main.py pipeline --tune --production
```

**What happens:**
- Training data: ALL available fights
- Test data: None (no holdout)
- Result: Best model for predicting future fights

## Usage Examples

### Getting Realistic Metrics (Development)
```bash
# Evaluate model performance with temporal split
python3 main.py pipeline --tune

# Output will show:
# ⏰ Using TEMPORAL SPLIT for realistic evaluation
# Temporal split:
#   Training: 2800 fights (up to 2023-12-31)
#   Testing:  700 fights (from 2024-01-01)
#   Temporal gap: 1 days
# Model accuracy: 65% (realistic!)
```

### Training for Production
```bash
# Train final model on all data
python3 main.py pipeline --tune --production

# Output will show:
# 🚀 PRODUCTION MODE: Training on ALL available data
# Training on all 3500 samples
# No test set - model uses maximum data
```

### Making Predictions
```bash
# After training in production mode
python3 main.py predict --fighter1 "Jon Jones" --fighter2 "Stipe Miocic"

# Uses model trained on ALL historical data
```

## Expected Performance Changes

| Metric | Random Split (Inflated) | Temporal Split (Realistic) | Change |
|--------|------------------------|---------------------------|---------|
| Accuracy | 75-85% | 60-70% | -15% to -20% |
| AUC-ROC | 0.85-0.90 | 0.70-0.80 | -0.10 to -0.15 |
| Betting ROI | Overestimated | Conservative | More realistic |

## Important Notes

1. **Lower metrics are GOOD** - they're realistic and prevent overconfidence
2. **Always use temporal split for evaluation** - it mimics real-world conditions
3. **Use production mode for final predictions** - maximizes predictive power
4. **The gap is automatic** - system calculates temporal gap between train/test

## Command Reference

| Command | Purpose | When to Use |
|---------|---------|-------------|
| `pipeline --tune` | Evaluate with temporal split | Development/testing |
| `pipeline --tune --production` | Train on all data | Before making predictions |
| `pipeline --tune --random-split` | Old behavior | Never (only for comparison) |

## Workflow Summary

1. **Development Phase**: Use temporal split to get realistic metrics
2. **Understand Performance**: Accept that 65% accuracy is realistic
3. **Production Training**: Retrain with `--production` flag
4. **Make Predictions**: Use production model for upcoming fights

## Technical Details

The implementation:
- Parses fight dates from the 'Date' column
- Sorts fights chronologically
- Splits at 80% point in time
- Maintains temporal ordering (no shuffle)
- Calculates temporal gap between sets

## Next Steps

1. **Retrain with temporal split** to see realistic performance:
   ```bash
   python3 main.py pipeline --tune
   ```

2. **Adjust betting strategy** based on realistic 60-70% accuracy

3. **Use production mode** for actual predictions:
   ```bash
   python3 main.py pipeline --tune --production
   ```

Remember: The "drop" in accuracy is not a problem - it's the truth! Your model will now perform consistently between testing and production.