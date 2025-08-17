# 🤝 Handover Note - UFC Predictor Project

## Current Date: 2025-08-17

## Project Overview
UFC Predictor is a machine learning system for predicting UFC fight outcomes and optimizing betting strategies. The user has been experiencing -84% ROI and needed critical fixes to their betting system.

## Critical Context
**IMPORTANT**: The user had a -84% ROI problem caused by:
1. **20% bankroll bets** (catastrophically high)
2. **Forced parlays** with 0% win rate
3. **No calibration** leading to overconfident predictions
4. **Only 12 bets** in sample (statistically insufficient for strong conclusions)

## Recent Major Changes (This Session)

### 1. Conservative Betting System Implementation
- Created `src/ufc_predictor/betting/conservative_calibration.py`
- Implemented 2% max bet rule (down from 20%)
- Disabled parlays by default
- Added mild temperature scaling (T=1.1) for conservative calibration
- Raised minimum edge requirement to 3%

### 2. Prediction Tracking System
- Created `src/ufc_predictor/betting/prediction_tracker.py`
- Tracks ALL predictions (not just bets) for true model accuracy
- Records both raw and calibrated probabilities
- Generates comprehensive accuracy reports
- Uses `model_predictions_tracker.csv` for storage

### 3. Updated Notebooks
- Created `enhanced_ufc_betting_v2.ipynb` with all conservative fixes
- Includes integrated tracking system
- Conservative calibration built-in
- Comprehensive performance analysis functions

## Key Files and Their Purpose

### Core Betting System
- `src/ufc_predictor/betting/conservative_calibration.py` - Main conservative betting logic
- `src/ufc_predictor/betting/prediction_tracker.py` - Prediction tracking system
- `improved_betting_strategy.py` - Reference implementation with examples

### Notebooks
- `notebooks/production/enhanced_ufc_betting_v2.ipynb` - PRIMARY notebook (use this!)
- `notebooks/production/enhanced_ufc_betting_calibrated.ipynb` - Alternative with full calibration

### Documentation
- `IMPLEMENTATION_GUIDE.md` - Step-by-step implementation instructions
- `CLAUDE.md` - Project-specific instructions (DO NOT DELETE)
- `betting_records.csv` - Historical betting data (12 bets)
- `model_predictions_tracker.csv` - New tracking file for all predictions

## User Preferences and Important Notes

### Git/GitHub
- **CRITICAL**: Always commit as the user only (no co-authors)
- Never add "Co-Authored-By" in commits
- User wants sole credit on all commits

### Betting Strategy Status
- User is skeptical about temperature scaling due to small sample (12 bets)
- Implemented MILD calibration (T=1.1) not aggressive (T=0.604)
- User wants to track model predictions separately from betting performance
- User's notebook: `notebooks/production/enhanced_ufc_betting.ipynb` (now updated to v2)

### Current Configuration
```python
bankroll = 17.0  # Current bankroll
MAX_BET = 0.02   # 2% maximum (was 20%)
MIN_EDGE = 0.03  # 3% minimum edge
TEMPERATURE = 1.1 # Conservative calibration
PARLAYS = False  # Effectively disabled
```

## Statistical Context
- **Sample Size**: 12 bets (need 100+ for statistical significance)
- **Win Rate**: 25% actual vs 34.5% predicted
- **Singles**: 33% win rate (has potential)
- **Parlays**: 0% win rate (mathematically expected)

## Next Steps for User
1. Use `enhanced_ufc_betting_v2.ipynb` for all future predictions
2. Track ALL predictions (not just bets) for model accuracy
3. After 30+ predictions, reassess calibration
4. NEVER exceed 2% bet size
5. Avoid parlays completely

## Common Commands

### Running Analysis
```bash
python3 src/ufc_predictor/betting/conservative_calibration.py
python3 improved_betting_strategy.py
jupyter notebook notebooks/production/enhanced_ufc_betting_v2.ipynb
```

### Model Training
```bash
python3 main.py pipeline --tune
python3 optimize_latest_model.py
```

## Warnings and Critical Rules
1. **NEVER** allow bets over 2% of bankroll
2. **AVOID** parlays - they have -100% historical ROI
3. **USE** conservative calibration until more data available
4. **TRACK** everything for future analysis
5. **REMEMBER** user wants sole git credit (no co-authors)

## Technical Debt and Future Improvements
- Need 100+ bets for proper statistical calibration
- Consider A/B testing raw vs calibrated predictions
- Implement automatic calibration updates after 30+ bets
- Add real-time odds scraping integration

## Session Summary
Transformed a -84% ROI system into a conservative, mathematically sound approach. Key insight: execution matters more than prediction accuracy. Even with perfect predictions, 20% bets and forced parlays guarantee failure.

## Contact with User
User is engaged and understands the math. They suggested tracking model predictions separately (implemented). They're appropriately skeptical about calibration given small sample size (addressed with conservative T=1.1).

---

**For next Claude**: The user now has a much safer system. Focus on helping them track predictions and analyze performance over time. Be conservative with any changes until more data is available.