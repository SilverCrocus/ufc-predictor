# UFC Predictor: Probability Calibration Guide

## Overview

This guide explains how to use probability calibration in the UFC Predictor to improve betting performance and model reliability based on 2024 research findings.

## Why Calibration Matters for UFC Betting

### Key Research Finding (2024)
**Models optimized for calibration rather than accuracy generated 69.86% higher average returns** in sports betting scenarios. This is critical for UFC betting where:

- Overconfident probabilities lead to overbetting and losses
- Well-calibrated probabilities are essential for Kelly criterion
- Professional bettors use conservative sizing due to calibration uncertainty

### Expected Calibration Error (ECE)
ECE measures how much predicted probabilities deviate from actual outcomes:
- **ECE < 0.05**: Well-calibrated model
- **ECE > 0.10**: Poorly calibrated, needs correction

## Implementation

### 1. Training with Calibration

```python
from ufc_predictor.models.model_training import train_complete_pipeline

# Train models with Platt scaling calibration
trainer = train_complete_pipeline(
    X, y,
    enable_calibration=True,
    calibration_method='platt'  # Recommended for UFC data
)
```

### 2. Prediction with Calibration

```python
from ufc_predictor.core.prediction import UFCPredictor

# Initialize predictor with calibration enabled
predictor = UFCPredictor(
    model_path="model.joblib",
    use_calibration=True
)

# Predictions automatically use calibrated probabilities
result = predictor.predict_fight_symmetrical("Fighter A", "Fighter B")
```

### 3. Configuration

```python
from configs.calibration_config import get_calibration_config

# Get betting-optimized calibration settings
config = get_calibration_config(
    model_type='xgboost',
    for_betting=True
)
```

## Calibration Methods

### Platt Scaling (Recommended)
- **Best for**: UFC data with ~21K samples
- **Advantages**: Robust, less prone to overfitting, handles imbalanced classes
- **Use when**: Default choice for most scenarios

```python
trainer.set_calibration_method('platt')
```

### Isotonic Regression (Advanced)
- **Best for**: Large datasets, non-sigmoid calibration curves
- **Advantages**: Can correct any monotonic distortion
- **Use when**: You have >50K samples and complex calibration issues

```python
trainer.set_calibration_method('isotonic')
```

## Betting Integration

### Kelly Criterion with Calibration
```python
def kelly_bet_size(calibrated_prob, odds, bankroll=1000):
    """Calculate conservative Kelly bet size."""
    ev = (calibrated_prob * odds) - 1
    if ev <= 0:
        return 0
    
    kelly_fraction = (calibrated_prob * odds - 1) / (odds - 1)
    conservative_kelly = kelly_fraction * 0.25  # 25% of Kelly (professional standard)
    
    max_bet = bankroll * 0.05  # Never bet more than 5%
    return min(bankroll * conservative_kelly, max_bet)
```

### Expected Value Calculation
```python
def calculate_expected_value(calibrated_prob, decimal_odds):
    """Calculate expected value with calibrated probability."""
    return (calibrated_prob * decimal_odds) - 1

# Only bet when EV > 5%
if calculate_expected_value(calibrated_prob, odds) > 0.05:
    bet_size = kelly_bet_size(calibrated_prob, odds)
```

## Monitoring Calibration

### 1. Calibration Summary
```python
# Get calibration status for all models
cal_summary = trainer.get_calibration_summary()
print(cal_summary)
```

### 2. Expected Calibration Error
```python
from ufc_predictor.evaluation.calibration import UFCProbabilityCalibrator

calibrator = UFCProbabilityCalibrator()
ece = calibrator._calculate_ece(y_true, y_prob)
print(f"ECE: {ece:.4f}")
```

### 3. Reliability Diagram
```python
from ufc_predictor.evaluation.calibration import plot_reliability_diagram

reliability_data = plot_reliability_diagram(y_true, y_prob)
# Plot with your preferred visualization library
```

## Best Practices

### 1. Model Training
- Always enable calibration for betting applications
- Use Platt scaling as default method
- Validate calibration on out-of-fold data
- Monitor ECE during model development

### 2. Betting Applications
- Use conservative Kelly criterion (25% of optimal)
- Never bet more than 5% of bankroll on single fight
- Require minimum 5% expected value
- Monitor calibration drift over time

### 3. Production Deployment
- Save calibrators with trained models
- Load calibrators automatically in prediction pipeline
- Log calibration metrics for monitoring
- Periodically retrain calibrators

## Configuration Files

### calibration_config.py
Contains research-based settings:
```python
CALIBRATION_SETTINGS = {
    'method': 'platt',
    'enabled': True,
    'ece_threshold': 0.05,
    'validation_split': 0.2
}

BETTING_CALIBRATION = {
    'ece_threshold': 0.03,
    'kelly_conservative_factor': 0.25,
    'max_bet_percentage': 0.05
}
```

## Testing

### Run Calibration Tests
```bash
python scripts/test_calibration.py
```

This script:
- Tests Platt vs Isotonic methods
- Simulates betting impact
- Validates configuration settings
- Provides performance recommendations

## Common Issues

### 1. Overconfident Predictions
**Symptom**: ECE > 0.10, poor betting performance
**Solution**: Enable Platt scaling, reduce Kelly factor to 10%

### 2. Underconfident Predictions
**Symptom**: Model won't find profitable bets
**Solution**: Check if calibration is too aggressive, consider isotonic method

### 3. Calibration Not Loading
**Symptom**: "Could not load calibrator" messages
**Solution**: Ensure calibrators are saved with models, check file paths

## Advanced Features

### Segment-Specific Calibration
```python
# Calibrate by weight class and fight type (future feature)
calibrator = UFCProbabilityCalibrator(
    segment_cols=['weight_class', 'title_fight'],
    min_samples_for_segment=100
)
```

### Confidence Intervals
```python
# Bootstrap confidence intervals for betting decisions
from ufc_predictor.utils.confidence_intervals import bootstrap_prediction

ci_lower, ci_upper = bootstrap_prediction(
    model, X, n_bootstrap=1000, confidence_level=0.95
)
```

## References

- Walsh, C. J., & Joshi, A. (2024). Machine learning for sports betting: Should model selection be based on accuracy or calibration? *Machine Learning with Applications*.
- Platt, J. (1999). Probabilistic outputs for support vector machines.
- Zadrozny, B., & Elkan, C. (2002). Transforming classifier scores into accurate multiclass probability estimates.

## Support

For calibration-specific issues:
1. Check ECE values with `trainer.get_calibration_summary()`
2. Run test script: `python scripts/test_calibration.py`
3. Review configuration in `configs/calibration_config.py`
4. Monitor betting performance with calibrated vs uncalibrated probabilities