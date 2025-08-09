# Getting Started with UFC Predictor 🎯

A complete step-by-step guide to running the UFC prediction system from scratch to making profitable betting predictions.

## 📋 Prerequisites

### System Requirements
- **Python 3.9 or higher** 
- **8GB+ RAM** (for large dataset processing)
- **2GB free disk space** (for models and data)
- **Internet connection** (for data scraping)

### Choose Your Package Manager
**Option A: UV (Recommended)**
```bash
# Install UV if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Verify installation
uv --version
```

**Option B: Standard pip**
```bash
# Ensure you have Python 3.9+
python3 --version
```

## 🚀 Step 1: Environment Setup

### Clone and Setup
```bash
# Navigate to your projects folder
cd ~/Documents  # or wherever you want the project

# Enter the project directory
cd ufc-predictor

# Install all dependencies
uv sync
# OR with pip: pip install -r requirements.txt

# Verify installation worked
uv run python3 --version
```

## 📊 Step 2: Data Collection (5-35 minutes)

### Option A: Optimized Scraper (Recommended - 95% faster)
```bash
# Run the optimized concurrent scraper
uv run python3 webscraper/optimized_scraping.py
```

**What this does:**
- Scrapes 4,000+ fighters and 20,000+ fights from UFC Stats
- Uses concurrent processing (8 simultaneous requests)
- Saves data to `data/scrape_YYYY-MM-DD_HH-MM/`
- **Time**: 5-10 minutes (vs 90+ minutes with original)

### Option B: Original Scraper (Backup)
```bash
# If you prefer the original sequential scraper
uv run python3 webscraper/scraping.py
```

### Verify Data Collection
```bash
# Check what data was scraped
ls -la data/scrape_*/
# Should see: ufc_fighters_raw_*.csv, ufc_fights_*.csv
```

## 🤖 Step 3: Train Machine Learning Models (30-60 minutes)

### Full Pipeline with Hyperparameter Tuning
```bash
# Process scraped data and train optimized models
uv run python3 main.py --mode pipeline --tune
```

**What this command does:**
1. **Loads** your fresh scraped data automatically
2. **Engineers** 70+ differential features comparing fighters  
3. **Trains** 3 winner prediction models (Logistic, RF, Tuned RF)
4. **Trains** method prediction models (KO/TKO, Decision, Submission)
5. **Optimizes** hyperparameters with GridSearch
6. **Saves** models to timestamped directories + standard locations

### Alternative: Quick Training (5-10 minutes)
```bash
# Train without hyperparameter tuning (faster but less accurate)
uv run python3 main.py --mode pipeline
```

### Verify Training Worked
```bash
# View detailed training results and model performance
uv run python3 main.py --mode results
```

**Expected Output:**
```
Winner Model Accuracy: 72%+ 
Method Model Accuracy: 75%+
Models saved to: model/training_YYYY-MM-DD_HH-MM/
```

## 🎯 Step 4: Make Fight Predictions

### Single Fight Prediction
```bash
# Predict a specific fight
uv run python3 main.py --mode predict --fighter1 "Jon Jones" --fighter2 "Stipe Miocic"
```

**Example Output:**
```
🥊 FIGHT PREDICTION: Jon Jones vs. Stipe Miocic
Winner: Jon Jones (67.3% confidence)
Method: KO/TKO (45.2% probability)
```

### Multiple Fight Predictions (Full Card)
```bash
# Predict multiple fights at once
uv run python3 main.py --mode card --fights "Fighter1 vs Fighter2" "Fighter3 vs Fighter4" "Fighter5 vs Fighter6"
```

### Interactive Prediction Mode
```bash
# Get prompts to enter fighter names
uv run python3 main.py --mode predict
```

## 💰 Step 5: Profitability Analysis (Optional)

### Quick Betting Analysis
```bash
# Run interactive profitability analysis
./quick_analysis.sh
```

### Full Profitability Analysis
```bash
# Comprehensive betting analysis with live odds
uv run python3 run_profitability_analysis.py --sample

# Custom bankroll analysis
uv run python3 run_profitability_analysis.py --sample --bankroll 1000
```

**What this provides:**
- Expected value calculations
- Kelly Criterion bet sizing
- Multi-bet parlay analysis
- Risk-adjusted recommendations

## 🔄 Regular Usage Workflow

### Daily/Weekly Updates
```bash
# 1. Scrape latest fight data (if new fights added)
uv run python3 webscraper/optimized_scraping.py

# 2. Retrain models with new data (monthly recommended)
uv run python3 main.py --mode train --tune

# 3. Make predictions for upcoming fights
uv run python3 main.py --mode predict --fighter1 "A" --fighter2 "B"
```

### Quick Predictions (Using Existing Models)
```bash
# Skip scraping/training, just predict with current models
uv run python3 main.py --mode predict --fighter1 "Fighter A" --fighter2 "Fighter B"
```

## 📈 Performance Testing & Validation

### Test Scraper Performance
```bash
# Compare old vs new scraper performance
uv run python3 test_scraper_performance.py --mode compare --limit 50

# Quick scraper test
uv run python3 test_scraper_performance.py --mode quick
```

### Validate Optimizations
```bash
# Demonstrate optimization improvements
uv run python3 validate_optimization.py
```

## 🛠️ Advanced Features

### Jupyter Notebook Analysis
```bash
# Start Jupyter for interactive analysis
jupyter notebook

# Open these notebooks:
# - model/ufc_predictions.ipynb (main analysis)
# - model/feature_engineering.ipynb (feature exploration)
```

### Enhanced ML System
```bash
# Demonstrate advanced features (ELO, ensemble methods)
uv run python3 enhanced_system_demo.py

# Run comprehensive system tests
uv run python3 test_enhanced_system.py
```

## 🔧 Troubleshooting Common Issues

### "Module not found" errors
```bash
# Reinstall dependencies
uv sync --reinstall
# OR: pip install -r requirements.txt --force-reinstall
```

### "No models found" errors
```bash
# Retrain models
uv run python3 main.py --mode pipeline --tune
```

### Scraping timeouts/errors
```bash
# Use conservative scraper settings or try original scraper
uv run python3 webscraper/scraping.py
```

### Low prediction accuracy
```bash
# Ensure you've trained with fresh, comprehensive data
uv run python3 main.py --mode pipeline --tune
uv run python3 main.py --mode results  # Check accuracy scores
```

## 📊 Understanding Results

### Model Accuracy Expectations
- **Winner Prediction**: 70-75% accuracy (good for sports betting)
- **Method Prediction**: 75%+ accuracy for Decision/KO/TKO classification
- **Confidence Scores**: Higher confidence = more reliable predictions

### Data Quality Indicators
- **Fighter Count**: Should see 4,000+ fighters after scraping
- **Fight Count**: Should see 20,000+ fights after scraping
- **Feature Count**: Should see 70+ features after engineering

### Training Time Expectations
- **Scraping**: 5-35 minutes depending on internet/settings
- **Feature Engineering**: 2-5 minutes  
- **Model Training**: 30-60 minutes with hyperparameter tuning
- **Predictions**: Near-instantaneous once trained

## ⚡ Quick Commands Reference

```bash
# Complete workflow from scratch
uv run python3 webscraper/optimized_scraping.py
uv run python3 main.py --mode pipeline --tune
uv run python3 main.py --mode results

# Daily prediction workflow  
uv run python3 main.py --mode predict --fighter1 "A" --fighter2 "B"
uv run python3 run_profitability_analysis.py --sample

# Maintenance/updates
uv run python3 webscraper/optimized_scraping.py  # New data
uv run python3 main.py --mode train --tune       # Update models
uv run python3 main.py --mode results            # Check performance

# Testing and validation
uv run python3 test_scraper_performance.py --mode quick
uv run python3 validate_optimization.py
```

## 🎯 Success Checklist

After following this guide, you should have:

- ✅ **Fresh UFC data** (4,000+ fighters, 20,000+ fights)
- ✅ **Trained ML models** (70%+ accuracy)  
- ✅ **Working predictions** for any fighter matchup
- ✅ **Profitability analysis** for betting opportunities
- ✅ **Performance optimizations** (95% faster scraping)

## 📞 Getting Help

1. **Check logs**: Look in `model/training_*/training_metadata_*.json`
2. **Review documentation**: See `CLAUDE.md` for technical details
3. **Test components**: Use the testing scripts to isolate issues
4. **Start fresh**: Delete `model/` and `data/` directories and re-run pipeline

**You're now ready to predict UFC fights and analyze betting opportunities!** 🥊