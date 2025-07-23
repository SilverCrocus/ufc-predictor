# UFC Predictor Command Reference 📚

A comprehensive reference of all available commands, options, and usage patterns.

## 🎯 Main Pipeline Commands

### Core Training Pipeline
```bash
# Complete pipeline: data processing + model training + evaluation
uv run python3 main.py --mode pipeline --tune

# Pipeline without hyperparameter tuning (faster)
uv run python3 main.py --mode pipeline

# Train models only (requires existing processed data)  
uv run python3 main.py --mode train --tune
uv run python3 main.py --mode train
```

### Model Results and Analysis
```bash
# View detailed training results
uv run python3 main.py --mode results

# View results from specific training run
uv run python3 main.py --mode results --metadata path/to/metadata.json
```

## 🥊 Prediction Commands

### Single Fight Predictions
```bash
# Predict specific fighter matchup
uv run python3 main.py --mode predict --fighter1 "Jon Jones" --fighter2 "Stipe Miocic"

# Interactive prediction mode (prompts for fighter names)
uv run python3 main.py --mode predict

# Prediction with custom model paths
uv run python3 main.py --mode predict --fighter1 "A" --fighter2 "B" --model-path path/to/model.joblib
```

### Multiple Fight Predictions
```bash
# Predict full UFC card
uv run python3 main.py --mode card --fights "Fighter1 vs Fighter2" "Fighter3 vs Fighter4"

# Multiple fights with custom formatting
uv run python3 main.py --mode card --fights "Jon Jones vs. Stipe Miocic" "Islam Makhachev vs. Charles Oliveira"
```

## 🕷️ Data Scraping Commands

### Optimized Scraper (Recommended)
```bash
# Full optimized scrape (concurrent processing)
uv run python3 webscraper/optimized_scraping.py

# Run with custom configuration (modify config in file)
uv run python3 webscraper/optimized_scraping.py
```

### Original Scraper (Backup)
```bash
# Sequential scraping (slower but reliable)
uv run python3 webscraper/scraping.py
```

### Scraper Performance Testing
```bash
# Quick performance test (20 fighters)
uv run python3 test_scraper_performance.py --mode quick

# Full performance comparison test
uv run python3 test_scraper_performance.py --mode compare --limit 100

# Test with specific fighter limit
uv run python3 test_scraper_performance.py --mode quick --limit 50

# Save test results to JSON
uv run python3 test_scraper_performance.py --mode compare --save
```

## 💰 Profitability Analysis Commands

### Interactive Analysis
```bash
# Quick analysis launcher (interactive menu)
./quick_analysis.sh

# Make script executable if needed
chmod +x quick_analysis.sh
```

### Comprehensive Profitability Analysis
```bash
# Full profitability analysis with live odds
uv run python3 run_profitability_analysis.py --sample

# Analysis with cached odds (faster)
uv run python3 run_profitability_analysis.py --sample --no-live-odds

# Custom bankroll analysis
uv run python3 run_profitability_analysis.py --sample --bankroll 500
uv run python3 run_profitability_analysis.py --sample --bankroll 1000

# Specific event analysis
uv run python3 run_profitability_analysis.py --sample --event "UFC 317"
```

### TAB Australia Profitability
```bash
# TAB-specific profitability testing
uv run python3 test_tab_australia_profitability.py

# UFC event-specific TAB analysis  
uv run python3 test_ufc317_profitability.py

# Live FightOdds profitability
uv run python3 test_live_fightodds_profitability.py
```

## 🧪 Testing and Validation Commands

### System Testing
```bash
# Test current system setup
uv run python3 test_current_setup.py

# Test enhanced ML system
uv run python3 test_enhanced_system.py

# Validate scraper optimizations
uv run python3 validate_optimization.py
```

### Performance Validation
```bash
# Quick optimization demo
uv run python3 validate_optimization.py

# Performance comparison with different settings
uv run python3 test_scraper_performance.py --mode compare --limit 200
```

## 🚀 Enhanced System Commands

### Demo and Showcase
```bash
# Comprehensive enhanced system demo
uv run python3 enhanced_system_demo.py

# Show enhanced capabilities
uv run python3 show_enhanced_capabilities.py

# Simple demonstration
uv run python3 simple_demo.py

# Quick start demo
uv run python3 quick_start.py
```

### ELO System
```bash
# ELO system demonstration
uv run python3 examples/elo_system_demo.py

# Bootstrap enhanced system
uv run python3 bootstrap_enhanced_system.py
```

### Model Comparison
```bash
# Compare different model approaches
uv run python3 compare_models.py

# Demo enhanced models
uv run python3 demo_enhanced_models.py
```

## 📊 Jupyter Notebook Commands

### Start Jupyter Server
```bash
# Start Jupyter Lab
jupyter lab

# Start classic Jupyter notebook
jupyter notebook

# Start with specific notebook
jupyter notebook model/ufc_predictions.ipynb
```

### Key Notebooks
- `model/ufc_predictions.ipynb` - Main predictions and analysis
- `model/feature_engineering.ipynb` - Feature development and exploration  
- `model/ufc_method_prediction.ipynb` - Method prediction analysis

## 🔧 Development and Debugging Commands

### Installation and Setup
```bash
# Install dependencies
uv sync                          # UV package manager
pip install -r requirements.txt  # Standard pip

# Install development dependencies  
uv run python3 install_dependencies.py

# Run tests
uv run python3 run_tests.py
pytest tests/                    # If pytest available
```

### Debugging and Development
```bash
# Debug pipeline issues
uv run python3 debug_pipeline.py

# Create method prediction model
uv run python3 create_method_model.py

# Convert fights data format
uv run python3 convert_fights_data.py

# Simple model training
uv run python3 simple_train.py
```

## ⚙️ Command Line Arguments and Options

### Main.py Arguments
```bash
--mode pipeline|train|predict|card|results  # Operation mode
--tune                                       # Enable hyperparameter tuning
--fighter1 "Fighter Name"                    # First fighter for prediction
--fighter2 "Fighter Name"                    # Second fighter for prediction  
--fights "F1 vs F2" "F3 vs F4"             # Multiple fights for card mode
--metadata path/to/metadata.json            # Specific metadata file for results
--model-path path/to/model.joblib           # Custom model path
```

### Profitability Analysis Arguments
```bash
--sample                     # Use sample predictions for analysis
--no-live-odds              # Use cached odds instead of live scraping
--bankroll AMOUNT           # Custom bankroll amount (default: varies)
--event "Event Name"        # Specific UFC event analysis
```

### Scraper Test Arguments
```bash
--mode quick|full|compare   # Test mode
--limit NUMBER              # Limit number of fighters for testing
--save                      # Save results to JSON file
```

## 🚦 Command Execution Order

### Full System Setup (First Time)
```bash
1. uv sync
2. uv run python3 webscraper/optimized_scraping.py
3. uv run python3 main.py --mode pipeline --tune
4. uv run python3 main.py --mode results
```

### Daily Prediction Workflow
```bash
1. uv run python3 main.py --mode predict --fighter1 "A" --fighter2 "B"
2. uv run python3 run_profitability_analysis.py --sample
```

### Weekly Model Updates  
```bash
1. uv run python3 webscraper/optimized_scraping.py
2. uv run python3 main.py --mode train --tune
3. uv run python3 main.py --mode results
```

### Monthly Full Refresh
```bash
1. uv run python3 webscraper/optimized_scraping.py
2. uv run python3 main.py --mode pipeline --tune
3. uv run python3 test_scraper_performance.py --mode compare
4. uv run python3 main.py --mode results
```

## ⚡ Quick Reference Cards

### Essential Commands
```bash
# Train models with fresh data
uv run python3 main.py --mode pipeline --tune

# Make prediction
uv run python3 main.py --mode predict --fighter1 "A" --fighter2 "B"

# View results  
uv run python3 main.py --mode results

# Scrape fresh data
uv run python3 webscraper/optimized_scraping.py
```

### Troubleshooting Commands
```bash
# Fix dependencies
uv sync

# Retrain models
uv run python3 main.py --mode pipeline

# Test scraper
uv run python3 test_scraper_performance.py --mode quick

# Check system
uv run python3 test_current_setup.py
```

### Performance Commands
```bash
# Fast scraping
uv run python3 webscraper/optimized_scraping.py

# Quick model training (no tuning)
uv run python3 main.py --mode train

# Performance comparison
uv run python3 test_scraper_performance.py --mode compare
```

## 🎯 Command Success Indicators

### Successful Scraping
```
✅ Discovered 4378 fighter URLs total  
✅ Batch X complete: 200 fighters, 1000 fights
🎉 OPTIMIZED UFC SCRAPING COMPLETE!
```

### Successful Training
```
✅ Loaded latest data: 4364 fighters and 21531 fights
Random Forest Results: Accuracy: 0.7219 (72.19%)
🎉 Training completed successfully!
```

### Successful Predictions
```
🥊 FIGHT PREDICTION: Fighter A vs. Fighter B
Winner: Fighter A (67.3% confidence)  
Method: KO/TKO (45.2% probability)
```

Use this reference to quickly find and execute the right commands for your UFC prediction tasks!