# UFC Bet Tracking System - Complete Guide

## Overview

The UFC Bet Tracking System provides comprehensive tracking and performance analysis for your UFC betting predictions. It seamlessly integrates with your existing `UFC_Predictions_Clean.ipynb` notebook to automatically record predictions, track betting outcomes, and analyze performance over time.

## 🎯 Key Features

### 1. **Comprehensive Data Tracking**
- **39 columns** covering every aspect of prediction and betting performance
- **Automatic recording** of predictions from your notebook
- **Manual result updates** after fights complete
- **Performance calculation** with statistical rigor

### 2. **Model Validation**
- **Prediction accuracy** tracking (winner and method)
- **Calibration analysis** to validate probability estimates
- **Confidence level** performance breakdown
- **Expected value accuracy** validation

### 3. **Portfolio Performance**
- **ROI tracking** with risk-adjusted metrics
- **Bankroll management** analysis
- **Kelly criterion effectiveness** measurement
- **Sharpe ratio** and drawdown analysis

### 4. **Risk Management**
- **Stake size analysis** by risk level
- **Expected value thresholds** performance
- **Portfolio correlation** tracking for multi-bets
- **Warning systems** for excessive risk

## 📊 CSV Schema (39 Columns)

### Event Information (5 columns)
- `event_id` - Unique event identifier
- `event_name` - UFC event name  
- `event_date` - Official fight date
- `analysis_timestamp` - When predictions were made
- `card_sequence` - Fight order on card

### Fight Details (7 columns)
- `fight_id` - Unique fight identifier
- `fighter_name` - Predicted fighter name
- `opponent_name` - Opponent fighter name
- `fight_order` - Position on card
- `weight_class` - Fight division
- `title_fight` - Championship fight flag
- `actual_result` - PENDING/COMPLETED status

### Prediction Data (8 columns)
- `model_version` - Model version used
- `predicted_winner` - Model's pick
- `predicted_probability` - Win probability
- `predicted_method` - Finish method prediction
- `method_probabilities` - All method probabilities (JSON)
- `confidence_level` - HIGH/MEDIUM/LOW/VERY_LOW
- `symmetrical_avg` - Symmetrical prediction flag
- `prediction_notes` - Additional notes

### Betting Information (10 columns)
- `bet_placed` - Whether bet was actually placed
- `bookmaker` - Sportsbook used
- `bet_type` - MONEYLINE/PROP/etc
- `stake_amount` - Amount wagered
- `decimal_odds` - Decimal odds received
- `american_odds` - American odds format
- `implied_probability` - Market implied probability
- `expected_value` - Calculated EV
- `kelly_fraction` - Kelly criterion sizing
- `bet_timestamp` - When bet was placed

### Risk Management (4 columns)
- `bankroll_at_bet` - Bankroll when bet placed
- `stake_percentage` - Bet size as % of bankroll
- `risk_level` - LOW/MEDIUM/HIGH/VERY_HIGH
- `portfolio_correlation` - Correlation penalty

### Results Tracking (5 columns)
- `actual_winner` - Who actually won
- `actual_method` - How fight ended
- `actual_round` - Which round it ended
- `result_updated` - When results were recorded
- `payout_received` - Amount won

### Performance Metrics (5 columns)
- `prediction_correct` - Prediction accuracy flag
- `method_correct` - Method prediction accuracy
- `profit_loss` - Net profit/loss on bet
- `roi_percentage` - Return on investment
- `running_bankroll` - Cumulative bankroll

## 🔄 Data Workflow

### Stage 1: Prediction Generation
```
UFC_Predictions_Clean.ipynb
├── Generate card predictions
├── Run TABProfitabilityAnalyzer  
├── Get betting opportunities
└── Auto-trigger bet tracking
```

### Stage 2: Bet Recording
```
track_card_predictions()
├── Extract prediction data
├── Match with betting opportunities
├── Calculate risk metrics
├── Append to ufc_bet_tracking.csv
└── Return event_id for future reference
```

### Stage 3: Result Updates
```
update_fight_results(event_id, results)
├── Match fighters to outcomes
├── Calculate performance metrics
├── Update profit/loss
├── Recalculate running bankroll
└── Mark results as COMPLETED
```

### Stage 4: Performance Analysis
```
get_performance_summary()
├── Model validation metrics
├── Portfolio performance analysis
├── Risk assessment
├── Calibration analysis
└── Generate recommendations
```

## 🛠️ Installation & Setup

### 1. **Install the System**
```bash
# Files are already created in your project:
# - src/bet_tracking.py (main system)
# - notebook_bet_tracking_integration.py (notebook code)
# - demo_bet_tracking.py (demonstration)
```

### 2. **Test the System**
```bash
# Run the demo to test functionality
python demo_bet_tracking.py --demo

# View CSV structure
python demo_bet_tracking.py --structure

# Check tracking status
python demo_bet_tracking.py --summary
```

### 3. **Integrate with Notebook**
Add the code cells from `notebook_bet_tracking_integration.py` to your `UFC_Predictions_Clean.ipynb`:

1. **Import cell** (after existing imports)
2. **Enhanced prediction recording** (replace betting summary section)
3. **Result update interface** (new cell)
4. **Performance dashboard** (new cell)
5. **Export utilities** (new cell)
6. **Quick actions** (new cell)

## 📱 Usage Examples

### Recording Predictions (Automatic)
```python
# In your notebook, this happens automatically:
from src.bet_tracking import track_card_predictions

event_id = track_card_predictions(
    card_results=tomorrow_results,
    profitability_results=profitability_results,
    event_name="UFC Fight Night - Whittaker vs de Ridder",
    bankroll=21.38
)
```

### Updating Results (Manual)
```python
# After fights complete:
from src.bet_tracking import update_fight_results

results = {
    "Robert Whittaker": {"winner": True, "method": "Decision", "round": "3"},
    "Reinier de Ridder": {"winner": False, "method": "Decision", "round": "3"},
    "Petr Yan": {"winner": True, "method": "TKO", "round": "2"},
    "Marcus McGhee": {"winner": False, "method": "TKO", "round": "2"}
}

success = update_fight_results(event_id, results)
```

### Performance Analysis
```python
# Get comprehensive performance metrics:
from src.bet_tracking import get_performance_summary

performance = get_performance_summary()

# Access specific metrics:
roi = performance['portfolio_performance']['total_roi_percentage']
accuracy = performance['model_performance']['prediction_accuracy']
win_rate = performance['portfolio_performance']['win_rate']
```

### Export and Reporting
```python
# Generate reports:
tracker = BetTracker()
report_file = tracker.generate_performance_report()

# Export data:
tracker.export_to_csv("my_betting_data.csv")

# Quick summary:
tracker.print_summary()
```

## 📈 Performance Metrics

### Model Validation
- **Prediction Accuracy**: Overall win/loss prediction rate
- **Method Accuracy**: Finish method prediction rate  
- **Calibration Error**: How well probabilities match reality
- **Confidence Analysis**: Performance by confidence level

### Portfolio Performance
- **Total ROI**: Overall return on investment
- **Win Rate**: Percentage of winning bets
- **Bankroll Growth**: Total bankroll progression
- **Sharpe Ratio**: Risk-adjusted return metric
- **Expected Value Accuracy**: How well EV predictions hold

### Risk Analysis
- **Stake Analysis**: Bet sizing consistency and safety
- **Risk Level Performance**: Results by risk classification
- **EV Threshold Analysis**: Performance by EV minimum
- **Correlation Impact**: Multi-bet correlation effects

## 🗂️ File Organization

```
performance_tracking/
├── ufc_bet_tracking.csv              # Master database
├── ufc_bet_tracking_backup.csv       # Daily backup
├── performance_reports/
│   ├── performance_report_YYYYMMDD_HHMMSS.md
│   ├── monthly_performance_2025-07.csv
│   └── model_validation_2025-07.csv
└── archives/
    ├── completed_events/
    │   └── ufc_fight_night_whittaker_de_ridder.csv
    └── model_versions/
        └── v2.0_performance.csv
```

## 🔐 Data Integrity

### Backup System
- **Daily backups** created automatically
- **Atomic writes** using temporary files
- **Data validation** on every write
- **Duplicate prevention** with unique IDs

### Validation Checks
- **Row consistency** validation
- **Data type checking** 
- **Range validation** for numerical fields
- **Event locking** to prevent duplicates

## 🎯 Best Practices

### Recording Predictions
1. **Always use the notebook integration** for consistency
2. **Record immediately** after generating predictions
3. **Include event names** for better organization
4. **Verify bankroll amounts** before recording

### Updating Results
1. **Update results promptly** after fights
2. **Use exact fighter names** as recorded
3. **Include method and round** for complete analysis
4. **Double-check results** before submitting

### Performance Analysis
1. **Review performance monthly** for trends
2. **Export data regularly** for backup
3. **Monitor calibration** to validate model
4. **Adjust strategy** based on risk analysis

### Risk Management
1. **Never exceed 25%** of bankroll per event
2. **Limit individual bets** to 10% of bankroll
3. **Monitor stake consistency** 
4. **Use Kelly sizing** recommendations

## 🔧 Advanced Usage

### Custom Analysis
```python
# Load data for custom analysis:
import pandas as pd
df = pd.read_csv("performance_tracking/ufc_bet_tracking.csv")

# Filter to specific periods:
recent = df[df['analysis_timestamp'] >= '2025-07-01']

# Analyze by confidence level:
high_conf = df[df['confidence_level'] == 'HIGH']
accuracy = (high_conf['prediction_correct'] == True).mean()

# ROI by risk level:
roi_by_risk = df.groupby('risk_level')['roi_percentage'].mean()
```

### Integration with External Tools
```python
# Export for external analysis:
tracker = BetTracker()
tracker.export_to_csv("betting_data_for_analysis.csv")

# Import to Excel/Google Sheets for visualization
# Use CSV data in R/Python for advanced statistics
# Connect to BI tools for dashboard creation
```

## 🚨 Troubleshooting

### Common Issues

**"No tracking file found"**
- Run demo first: `python demo_bet_tracking.py --demo`
- Check file permissions in performance_tracking/

**"Event not found for update"**
- Use exact event_id from recording
- Check pending events: `tracker.get_pending_events()`

**"Fighter names don't match"**
- Use exact names as recorded in predictions
- Check fighter names in CSV file

**"Performance analysis fails"**
- Ensure results have been updated
- Check for completed bets in data

### Getting Help
1. **Run the demo** to verify system works
2. **Check file structure** with `--structure` flag
3. **View system status** with `--summary` flag
4. **Review integration code** in notebook file

## 🎉 Success Metrics

### Target Performance Goals
- **Prediction Accuracy**: 55%+ on value bets
- **ROI**: 10%+ per event 
- **Bankroll Growth**: 20%+ monthly
- **Risk Management**: <25% exposure per event

### Warning Thresholds
- **Win Rate**: <50% indicates model issues
- **Calibration Error**: >0.1 indicates poor probability estimates
- **Max Stake**: >15% indicates excessive risk
- **Consecutive Losses**: >5 indicates strategy review needed

## 📚 Next Steps

1. **Integrate with your notebook** using provided code
2. **Run predictions** and watch automatic tracking
3. **Place recommended bets** following the analysis
4. **Update results** after fights complete
5. **Review performance** regularly and adjust strategy
6. **Export data** for additional analysis as needed

The bet tracking system is designed to grow with your betting success, providing increasingly sophisticated analysis as your data history builds. Start with the basic integration and explore advanced features as you become comfortable with the system.

---
*Generated by UFC Enhanced Card Analysis System v2.0*