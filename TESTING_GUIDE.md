# Testing Guide for Enhanced UFC Predictor Implementation

## Quick Test Commands

### 1. Test All New Components
```bash
# Run comprehensive test suite
python3 test_enhanced_implementation.py
```

### 2. Test Individual Components

#### Test Feature Generation
```bash
# Test new feature engineering modules
python3 -c "
from src.ufc_predictor.features.context_features import ContextFeatureGenerator
from src.ufc_predictor.features.matchup_features import MatchupFeatureGenerator
print('✓ Features modules loaded successfully')
"
```

#### Test Temporal Splitting
```bash
# Test walk-forward cross-validation
python3 -c "
from src.ufc_predictor.evaluation.temporal_split import TemporalWalkForwardSplitter
import pandas as pd
splitter = TemporalWalkForwardSplitter(train_years=2, test_months=6)
print('✓ Temporal splitter initialized')
"
```

#### Test Calibration
```bash
# Test probability calibration
python3 -c "
from src.ufc_predictor.evaluation.calibration import UFCProbabilityCalibrator
calibrator = UFCProbabilityCalibrator(method='isotonic')
print('✓ Calibrator initialized')
"
```

### 3. Test Enhanced Pipeline with Sample Data

```bash
# Create sample data and test the pipeline
python3 -c "
import pandas as pd
import numpy as np
from run_enhanced_pipeline import EnhancedUFCPipeline

# Initialize pipeline
pipeline = EnhancedUFCPipeline(
    'configs/features.yaml',
    'configs/backtest.yaml'
)

# Create minimal test data
test_data = pd.DataFrame({
    'date': pd.date_range('2023-01-01', periods=20),
    'fighter_a': ['Fighter_' + str(i%5) for i in range(20)],
    'fighter_b': ['Fighter_' + str((i+3)%5) for i in range(20)],
    'winner': np.random.choice([0, 1], 20),
    'venue': 'Las Vegas',
    'division': 'Lightweight',
    'rounds': 3
})

# Generate features
enhanced = pipeline.generate_enhanced_features(test_data, test_data)
print(f'✓ Generated {len(enhanced.columns)} total features')
"
```

### 4. Test with Your Existing Data

```bash
# Test with real UFC data (evaluation mode only)
python3 run_enhanced_pipeline.py \
    --data data/ufc_fights.csv \
    --mode evaluation

# Test with odds data (full backtest)
python3 run_enhanced_pipeline.py \
    --data data/ufc_fights_with_odds.csv \
    --mode full
```

### 5. Test Integration with Existing Pipeline

```bash
# Test that new features work with existing model training
python3 main.py --mode pipeline --tune \
    --use-enhanced-features
```

### 6. Test Betting Components

```bash
# Test Kelly staking
python3 -c "
from src.ufc_predictor.betting.staking import KellyStaking
kelly = KellyStaking(kelly_fraction=0.25, use_pessimistic=True)
stake = kelly.calculate_kelly_stake(prob=0.65, odds=2.1, bankroll=1000)
print(f'Kelly stake: \${stake.bet_amount:.2f} ({stake.bet_percentage:.1%})')
"

# Test parlay correlation
python3 -c "
from src.ufc_predictor.betting.correlation import ParlayCorrelationEstimator
estimator = ParlayCorrelationEstimator()
legs = [
    {'fighter': 'A', 'probability': 0.6, 'odds': 1.8, 'event': 'UFC 300'},
    {'fighter': 'B', 'probability': 0.7, 'odds': 1.5, 'event': 'UFC 300'}
]
analysis = estimator.analyze_parlay(legs)
print(f'Parlay EV: {analysis.expected_value:.2%}')
"
```

### 7. Test Monitoring Dashboards

```bash
# Generate test visualizations
python3 -c "
from src.ufc_predictor.monitoring.dashboards import PerformanceDashboard
import numpy as np

dashboard = PerformanceDashboard(output_dir='test_artifacts')
y_true = np.random.choice([0, 1], 100)
y_prob = np.random.beta(2, 2, 100)

metrics = dashboard.create_calibration_plots(y_true, y_prob)
print(f'✓ Calibration plots created. ECE: {metrics[\"ece\"]:.3f}')
"
```

## Sequential Testing Steps

### Step 1: Verify Installation
```bash
# Check all modules are importable
python3 -c "
import src.ufc_predictor.evaluation.temporal_split
import src.ufc_predictor.features.context_features
import src.ufc_predictor.models.stacking
import src.ufc_predictor.betting.simulator
import src.ufc_predictor.monitoring.dashboards
print('✓ All modules imported successfully')
"
```

### Step 2: Run Component Tests
```bash
# Run the comprehensive test suite
python3 test_enhanced_implementation.py
```

### Step 3: Test with Sample Data
```bash
# Create and test with synthetic data
python3 run_enhanced_pipeline.py \
    --data test_data.csv \
    --mode evaluation \
    --features-config configs/features.yaml \
    --backtest-config configs/backtest.yaml
```

### Step 4: Test with Real Data (if available)
```bash
# Test with your actual UFC data
python3 run_enhanced_pipeline.py \
    --data data/ufc_fights.csv \
    --mode full
```

## Expected Outputs

When tests run successfully, you should see:

1. **Import Tests**: All modules load without errors
2. **Feature Generation**: New features added to dataframe
3. **Temporal Split**: Multiple folds created with proper train/test separation
4. **Calibration**: Probabilities adjusted with ECE < 0.05
5. **Betting Components**: Kelly stakes calculated, parlays analyzed
6. **Monitoring**: Plots generated in `artifacts/monitoring/`
7. **Integration**: Full pipeline runs without errors

## Troubleshooting

### If imports fail:
```bash
# Check Python path
python3 -c "import sys; print(sys.path)"

# Add project to path if needed
export PYTHONPATH="${PYTHONPATH}:/Users/diyagamah/Documents/ufc-predictor"
```

### If configs are missing:
```bash
# Create default configs
python3 -c "
from test_enhanced_implementation import create_default_configs
create_default_configs()
"
```

### If data format issues:
```bash
# Check required columns
python3 -c "
import pandas as pd
df = pd.read_csv('your_data.csv')
required = ['date', 'fighter_a', 'fighter_b', 'winner']
missing = [c for c in required if c not in df.columns]
print(f'Missing columns: {missing}')
"
```

## Performance Validation

After successful testing, validate improvements:

```bash
# Compare baseline vs enhanced
python3 -c "
# Run baseline model
baseline_metrics = run_baseline_model()

# Run enhanced model
enhanced_metrics = run_enhanced_model()

# Calculate improvements
improvement = {
    'brier': (baseline['brier'] - enhanced['brier']) / baseline['brier'],
    'ece': (baseline['ece'] - enhanced['ece']) / baseline['ece'],
    'roi': enhanced['roi'] - baseline['roi']
}
print(f'Improvements: {improvement}')
"
```

## Next Steps

Once all tests pass:

1. **Full Training**: Run complete model training with new features
2. **Hyperparameter Tuning**: Optimize new model parameters
3. **Production Testing**: Test on live upcoming fights
4. **Monitor Performance**: Track metrics over time

## Quick Commands Summary

```bash
# Test everything
python3 test_enhanced_implementation.py

# Run enhanced pipeline
python3 run_enhanced_pipeline.py --data your_data.csv --mode full

# Generate monitoring reports
python3 -c "from run_enhanced_pipeline import EnhancedUFCPipeline; pipeline = EnhancedUFCPipeline('configs/features.yaml', 'configs/backtest.yaml'); pipeline.generate_reports(eval_results, backtest_results)"
```