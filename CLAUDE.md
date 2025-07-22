# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

UFC Predictor is a machine learning system for predicting UFC fight outcomes and betting profitability. It combines data scraping, feature engineering, model training, and real-time betting analysis.

## Core Architecture

### Main Components
- **main.py**: Command-line interface for the complete ML pipeline (training, prediction, evaluation)
- **src/**: Core Python modules for feature engineering, training, and prediction
- **model/**: Trained models and processed datasets with versioned storage
- **webscraper/**: Data collection tools for UFC stats and betting odds
- **config/**: Configuration files and model parameters

### Key Workflows
1. **Data Pipeline**: Raw UFC data → Feature engineering → Model training → Predictions
2. **Profitability Analysis**: Predictions → Live odds scraping → Expected value calculation → Betting recommendations
3. **Multi-bet Analysis**: Single bet opportunities → Combination analysis → Risk-adjusted recommendations

## Common Commands

### Model Training and Prediction
```bash
# Complete pipeline (data processing + training + evaluation)
python3 main.py --mode pipeline --tune

# Train models only (requires processed data)
python3 main.py --mode train --tune

# Single fight prediction
python3 main.py --mode predict --fighter1 "Fighter Name" --fighter2 "Opponent Name"

# Multiple fight predictions (full card)
python3 main.py --mode card --fights "Fighter1 vs Fighter2" "Fighter3 vs Fighter4"

# View training results
python3 main.py --mode results
```

### Profitability Analysis
```bash
# Quick analysis launcher (interactive menu)
./quick_analysis.sh

# Full profitability analysis with live odds
python3 run_profitability_analysis.py --sample

# Fast analysis with cached odds
python3 run_profitability_analysis.py --sample --no-live-odds

# Custom bankroll analysis
python3 run_profitability_analysis.py --sample --bankroll 500
```

### Jupyter Notebooks
```bash
# Feature engineering and model development
jupyter notebook model/feature_engineering.ipynb

# UFC predictions with profitability integration
jupyter notebook model/ufc_predictions.ipynb

# Method prediction analysis
jupyter notebook model/ufc_method_prediction.ipynb
```

### Testing and Validation
```bash
# Test specific profitability scenarios
python3 test_tab_australia_profitability.py
python3 test_ufc317_profitability.py

# Live odds scraping tests
python3 test_live_fightodds_profitability.py
```

## Data Flow Architecture

### Model Training Flow
1. **Raw Data**: `data/scrape_*/ufc_fighters_raw_*.csv` and `data/scrape_*/ufc_fights_*.csv`
2. **Feature Engineering**: `src/feature_engineering.py` creates differential features
3. **Model Training**: `src/model_training.py` with GridSearch hyperparameter tuning
4. **Versioned Storage**: `model/training_YYYY-MM-DD_HH-MM/` contains all artifacts
5. **Standard Locations**: Models copied to `model/` root for easy access

### Prediction Flow
1. **Model Loading**: Auto-detects latest trained models or uses specified paths
2. **Feature Generation**: Creates same differential features as training
3. **Symmetrical Prediction**: Averages predictions from both fighter perspectives
4. **Output**: Winner probabilities + method predictions (Decision/KO/TKO/Submission)

### Profitability Flow
1. **Prediction Input**: Fighter win probabilities from model or manual entry
2. **Odds Scraping**: Live TAB Australia odds via selenium/stealth scraping
3. **Fuzzy Matching**: Matches fighter names between predictions and odds
4. **EV Calculation**: Expected Value = (Probability × Odds) - 1
5. **Kelly Sizing**: Optimal bet size based on bankroll and edge
6. **Multi-bet Analysis**: Identifies profitable parlay combinations

## Key Configuration

### Model Settings (`config/model_config.py`)
- **CORRECTED_FIGHTERS_DATA**: Latest processed fighter stats
- **RF_TUNED_MODEL_PATH**: Tuned Random Forest model location
- **RANDOM_STATE**: 42 (for reproducible results)
- **RF_PARAM_GRID**: Hyperparameter search space for GridSearchCV

### Profitability Settings
- **MAX_BET_PERCENTAGE**: 5% of bankroll for single bets, 2% for multi-bets
- **MIN_EXPECTED_VALUE**: 5% minimum EV threshold for recommendations
- **CORRELATION_PENALTY**: 8% penalty for same-event multi-bet correlations

## Development Patterns

### Model Versioning
All training runs create timestamped directories with complete model artifacts, metadata, and performance metrics. The system automatically uses the latest models while maintaining historical versions.

### Symmetrical Predictions
Fight predictions consider both fighter perspectives (A vs B and B vs A) then average the results to eliminate positional bias from blue/red corner effects.

### Multi-bet Strategy
The system identifies profitable single bets first, then analyzes 2-4 leg combinations with proper compound probability calculations and risk assessment.

### Error Handling
Robust fallback mechanisms exist throughout:
- Latest scraped data → hardcoded paths
- Live odds scraping → cached odds
- Latest models → standard model locations
- Fuzzy name matching for fighter identification

## Development Tips

### For Model Development
- Use Jupyter notebooks for interactive development and visualization
- Run `--mode pipeline --tune` for complete model retraining with hyperparameter optimization
- Check `--mode results` to review detailed training metrics

### For Production Betting
- Use `./quick_analysis.sh` for fast analysis
- Monitor live odds changes with `--sample` flag (live scraping)
- Validate results with `--no-live-odds` for consistency checks

### For Adding Features
- Add new features in `src/feature_engineering.py`
- Update differential feature creation for fighter comparisons  
- Retrain models with `--mode pipeline` to incorporate new features