# UFC Predictor 🥊

A comprehensive machine learning system for predicting UFC fight outcomes and betting profitability analysis. Uses advanced feature engineering, ensemble methods, and real-time odds scraping.

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- [UV](https://docs.astral.sh/uv/) package manager (recommended) or pip

### 1. Setup Environment
```bash
# Clone and enter directory
cd ufc-predictor

# Install dependencies (UV recommended)
uv sync
# OR with pip:
# pip install -r requirements.txt
```

### 2. Complete Workflow (Start Here)
```bash
# Step 1: Scrape fresh UFC data (5-35 minutes)
uv run python3 webscraper/optimized_scraping.py

# Step 2: Train ML models with fresh data (30-60 minutes)
uv run python3 main.py --mode pipeline --tune

# Step 3: View training results
uv run python3 main.py --mode results

# Step 4: Make predictions
uv run python3 main.py --mode predict --fighter1 "Jon Jones" --fighter2 "Stipe Miocic"
```

## 📊 Main Features

### 🤖 Machine Learning
- **Winner Prediction**: Random Forest with hyperparameter tuning (72%+ accuracy)
- **Method Prediction**: Multi-class classification (KO/TKO, Decision, Submission)
- **Advanced Features**: 70+ differential features comparing fighters
- **Model Versioning**: Timestamped training runs with full metadata

### 🕷️ Data Collection
- **Optimized Scraper**: 95% faster than original (concurrent processing)
- **UFC Stats Integration**: Comprehensive fighter statistics and fight history
- **Live Odds Scraping**: Real-time betting odds from multiple sources
- **Data Validation**: Robust error handling and retry mechanisms

### 💰 Betting Analysis
- **Profitability Calculator**: Expected value and Kelly Criterion sizing
- **Multi-bet Analysis**: Optimal parlay combinations
- **Live Odds Integration**: TAB Australia and other bookmaker odds
- **Risk Management**: Correlation penalties and bankroll management

## 🎯 Common Use Cases

### Training New Models
```bash
# Full pipeline with latest data
uv run python3 main.py --mode pipeline --tune

# Quick retraining (uses existing processed data)
uv run python3 main.py --mode train --tune
```

### Making Predictions
```bash
# Single fight
uv run python3 main.py --mode predict --fighter1 "Fighter A" --fighter2 "Fighter B"

# Full card predictions
uv run python3 main.py --mode card --fights "Fighter1 vs Fighter2" "Fighter3 vs Fighter4"

# Interactive mode
uv run python3 main.py --mode predict
```

### Profitability Analysis
```bash
# Quick analysis with interactive menu
./quick_analysis.sh

# Full profitability analysis with live odds
uv run python3 run_profitability_analysis.py --sample

# Custom bankroll analysis
uv run python3 run_profitability_analysis.py --sample --bankroll 500
```

### Data Management
```bash
# Performance test of scraper
uv run python3 test_scraper_performance.py --mode compare

# Validate scraper optimizations
uv run python3 validate_optimization.py
```

## 📁 Project Structure

```
ufc-predictor/
├── main.py                 # Main CLI interface
├── webscraper/            
│   ├── optimized_scraping.py   # 95% faster concurrent scraper
│   └── scraping.py            # Original scraper
├── src/                   # Core ML modules
│   ├── feature_engineering.py # Feature creation and processing
│   ├── model_training.py     # ML model training
│   ├── prediction.py         # Fight prediction logic
│   └── enhanced_*.py         # Advanced ML components
├── model/                 # Trained models and data
├── data/                  # Raw and processed datasets
├── config/                # Configuration files
└── examples/              # Demo scripts and tutorials
```

## 🛠️ Advanced Usage

### Development and Analysis
```bash
# Jupyter notebooks for interactive analysis
jupyter notebook model/ufc_predictions.ipynb
jupyter notebook model/feature_engineering.ipynb

# Enhanced system demonstration
uv run python3 enhanced_system_demo.py
```

### Performance Testing
```bash
# Quick scraper test (20 fighters)
uv run python3 test_scraper_performance.py --mode quick

# Full performance comparison
uv run python3 test_scraper_performance.py --mode compare --limit 100
```

### Configuration
Key settings in `config/model_config.py`:
- Model parameters and hyperparameter grids
- Data file paths and processing options
- Feature engineering configurations

## 📈 Performance

### Scraper Optimization
- **Original**: 90-120 minutes for full scrape
- **Optimized**: 5-10 minutes (95% improvement)
- **Concurrent requests**: 8 simultaneous connections
- **Smart delays**: Respectful but efficient server interaction

### Model Accuracy
- **Winner Prediction**: 72%+ accuracy with tuned Random Forest
- **Method Prediction**: 75%+ accuracy across KO/TKO, Decision, Submission
- **Feature Engineering**: 70+ differential features comparing fighters
- **Validation**: Robust train/test splits with comprehensive metrics

## 🔧 Troubleshooting

### Common Issues
```bash
# Missing dependencies
uv sync  # or pip install -r requirements.txt

# Model not found errors
uv run python3 main.py --mode pipeline  # Retrain models

# Scraping issues
uv run python3 webscraper/optimized_scraping.py  # Use optimized scraper
```

### Getting Help
1. Check `CLAUDE.md` for detailed technical documentation
2. Run `python3 main.py --help` for command options
3. Check logs in timestamped directories under `model/training_*/`

## 📋 Dependencies

### Core Requirements
- **Data Processing**: pandas, numpy, scikit-learn
- **Web Scraping**: requests, beautifulsoup4, selenium, aiohttp
- **Machine Learning**: xgboost, lightgbm, joblib
- **Visualization**: matplotlib, seaborn
- **Utilities**: PyYAML, jupyter

See `requirements.txt` for complete list with versions.

## 🤝 Contributing

1. Use the optimized scraper for data collection
2. Follow the established pipeline for model training
3. Maintain backward compatibility with existing interfaces
4. Add tests for new features
5. Update documentation for significant changes

## 📄 License

This project is for educational and research purposes. Please use responsibly and in accordance with terms of service of data sources.

---

**Quick Commands Summary:**
```bash
# Complete workflow
uv run python3 webscraper/optimized_scraping.py
uv run python3 main.py --mode pipeline --tune
uv run python3 main.py --mode predict --fighter1 "A" --fighter2 "B"
uv run python3 run_profitability_analysis.py --sample
```