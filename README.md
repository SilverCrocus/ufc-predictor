# UFC Predictor 🥊

A comprehensive machine learning system for predicting UFC fight outcomes and betting profitability analysis. Features unified architecture with clean interfaces for prediction, betting analysis, and odds scraping.

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- [uv](https://docs.astral.sh/uv/) package manager

### Installation
```bash
# Install uv if needed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone repository
git clone https://github.com/yourusername/ufc-predictor.git
cd ufc-predictor

# Install dependencies with uv
uv sync

# Install dev dependencies (optional - for testing/linting)
uv sync --dev
```

## 📋 Complete Command Reference (with uv)

### Core Commands

#### 🥊 Fight Predictions
```bash
# Single fight prediction
uv run python main.py predict --fighter1 "Jon Jones" --fighter2 "Stipe Miocic"

# Multiple fights (full card)
uv run python main.py card --fights "Holloway vs Topuria" "Pantoja vs Kara-France"

# Show model information
uv run python main.py info
```

#### 💰 Betting Analysis
```bash
# Analyze betting opportunities with live odds
uv run python main.py betting --bankroll 1000 --source tab

# Use cached odds for faster analysis
uv run python main.py betting --bankroll 500 --source cached

# Export analysis to file
uv run python main.py betting --bankroll 1000 --export analysis.json
```

#### 🔄 Model Training Pipeline
```bash
# Step 1: Scrape fresh UFC data (5-10 minutes with optimized scraper)
uv run python webscraper/optimized_scraping.py

# Step 2: Full pipeline - process data + train models (30-60 minutes)
uv run python scripts/main.py --mode pipeline --tune

# Alternative: Train only (if data already processed)
uv run python scripts/main.py --mode train --tune

# Step 3: View training results
uv run python scripts/main.py --mode results
```

### Profitability Analysis Scripts

```bash
# Quick interactive analysis (recommended for beginners)
./quick_analysis.sh

# Full profitability analysis with live TAB odds
uv run python scripts/run_profitability_analysis.py --sample

# Fast analysis with cached odds
uv run python scripts/run_profitability_analysis.py --sample --no-live-odds

# Custom bankroll and parameters
uv run python scripts/run_profitability_analysis.py --bankroll 500 --min-ev 0.08
```

### Testing & Validation

```bash
# Run all tests
uv run pytest

# Run specific test modules
uv run pytest tests/test_feature_engineering.py
uv run pytest tests/integration/

# Test profitability scenarios
uv run python tests/integration/test_tab_australia_profitability.py
uv run python tests/integration/test_ufc317_profitability.py

# Test ensemble predictions
uv run python tests/integration/test_production_ensemble.py
```

### Development Tools

```bash
# Format code with black
uv run black src tests scripts

# Lint with ruff
uv run ruff check src tests

# Type checking (if mypy installed)
uv run mypy src

# Run Jupyter notebooks
uv run jupyter notebook notebooks/production/ufc_predictions.ipynb
uv run jupyter notebook notebooks/development/feature_engineering.ipynb
```

## 📊 Key Workflows

### 1. Fresh Start - Complete Setup
```bash
# Full workflow from scratch
uv sync                                          # Install dependencies
uv run python webscraper/optimized_scraping.py  # Scrape data (5-10 min)
uv run python scripts/main.py --mode pipeline --tune  # Train models (30-60 min)
uv run python main.py info                      # Verify models loaded
```

### 2. Daily Predictions
```bash
# Morning routine for upcoming fights
uv run python main.py card --fights "Fighter1 vs Fighter2" "Fighter3 vs Fighter4"
uv run python main.py betting --bankroll 1000 --source tab
```

### 3. Model Retraining
```bash
# Weekly/monthly model update
uv run python webscraper/optimized_scraping.py  # Get latest data
uv run python scripts/main.py --mode pipeline --tune  # Retrain
uv run python scripts/main.py --mode results    # Check performance
```

### 4. Research & Analysis
```bash
# Interactive exploration
uv run jupyter notebook notebooks/production/profitable_betting_analysis.ipynb
uv run jupyter notebook notebooks/exploratory/UFC_Enhanced_Card_Analysis.ipynb
```

## 🏗️ Project Structure

```
ufc-predictor/
├── main.py                    # Simplified main entry point
├── scripts/
│   ├── main.py               # Original full-featured CLI
│   └── run_profitability_analysis.py
├── src/ufc_predictor/
│   ├── core/
│   │   ├── unified_predictor.py    # Consolidated prediction logic
│   │   └── prediction.py           # Legacy prediction
│   ├── betting/
│   │   ├── unified_analyzer.py     # Consolidated betting analysis
│   │   └── tab_profitability.py    # TAB-specific analysis
│   ├── scrapers/
│   │   ├── unified_scraper.py      # Consolidated odds scraping
│   │   └── tab_australia_scraper.py
│   ├── data/
│   │   └── feature_engineering.py  # Feature creation
│   ├── models/
│   │   └── model_training.py       # Model training pipeline
│   └── utils/                      # Utilities (ELO, validation, etc.)
├── webscraper/
│   ├── optimized_scraping.py       # Fast concurrent scraper
│   └── scraping.py                 # Original scraper
├── model/                          # Trained models & datasets
├── data/                           # Raw & processed data
├── notebooks/                      # Jupyter notebooks
├── tests/                          # Test suite
├── configs/                        # Configuration files
└── pyproject.toml                 # uv project configuration
```

## ⚙️ Configuration

### Environment Variables
Create a `.env` file (optional):
```bash
# Betting parameters
DEFAULT_BANKROLL=1000
MIN_EV_THRESHOLD=0.05
MAX_BET_PERCENTAGE=0.05

# Scraping
CONCURRENT_REQUESTS=8
REQUEST_DELAY=0.5
```

### Model Configuration
Edit `configs/model_config.py`:
```python
# Key settings
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# Hyperparameter grids for tuning
RF_PARAM_GRID = {...}
XGB_PARAM_GRID = {...}
```

## 📈 Performance Metrics

### Scraping Performance
- **Optimized Scraper**: 5-10 minutes for full dataset
- **Original Scraper**: 90-120 minutes
- **Improvement**: 95% faster with concurrent processing

### Model Performance
- **Winner Prediction**: 72-75% accuracy
- **Method Prediction**: 70% accuracy (Decision/KO/Submission)
- **Betting ROI**: 5-15% expected value on identified opportunities

### System Requirements
- **Memory**: 4GB RAM minimum, 8GB recommended
- **Storage**: 2GB for models and data
- **Processing**: Multi-core CPU recommended for training

## 🐛 Troubleshooting

### Common Issues & Solutions

```bash
# Issue: ImportError or module not found
uv sync  # Reinstall dependencies

# Issue: No models found
uv run python scripts/main.py --mode pipeline --tune  # Train models

# Issue: Scraping timeout or failures
uv run python webscraper/optimized_scraping.py --limit 50  # Test with fewer fighters

# Issue: Out of memory during training
# Edit configs/model_config.py to reduce n_estimators or max_depth

# Issue: uv command not found
curl -LsSf https://astral.sh/uv/install.sh | sh  # Reinstall uv
source ~/.bashrc  # or ~/.zshrc
```

### Logs & Debugging
```bash
# Check training logs
ls -la model/training_*/

# View latest log
cat logs/autonomous_pipeline.log

# Enable debug mode (in scripts)
export DEBUG=1
uv run python main.py predict --fighter1 "test" --fighter2 "test"
```

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Make changes and test:
   ```bash
   uv run pytest tests/
   uv run black src/
   uv run ruff check src/
   ```
4. Commit (`git commit -m 'Add amazing feature'`)
5. Push (`git push origin feature/amazing-feature`)
6. Open Pull Request

## 📚 Additional Resources

- **Technical Documentation**: See [CLAUDE.md](CLAUDE.md) for AI assistance context
- **Command Reference**: [docs/COMMAND_REFERENCE.md](docs/COMMAND_REFERENCE.md)
- **Getting Started Guide**: [docs/GETTING_STARTED.md](docs/GETTING_STARTED.md)
- **ELO System Guide**: [docs/UFC_ELO_SYSTEM_GUIDE.md](docs/UFC_ELO_SYSTEM_GUIDE.md)

## ⚖️ Disclaimer

This project is for educational and research purposes only. Sports betting involves risk, and you should never bet more than you can afford to lose. Please gamble responsibly and in accordance with your local laws and regulations.

## 📄 License

MIT License - See LICENSE file for details

---

**Quick Reference Card:**
```bash
# Essential Commands with uv
uv sync                                          # Setup
uv run python main.py predict --fighter1 "A" --fighter2 "B"  # Predict
uv run python main.py betting --bankroll 1000   # Analyze bets
uv run python scripts/main.py --mode pipeline   # Train models
uv run pytest                                    # Run tests
```