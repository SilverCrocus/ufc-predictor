# 💰 UFC Profitability Analysis Guide

This guide shows you the **best ways to run profitability analysis** with your UFC prediction models and live TAB Australia odds.

## 🎯 **Quick Start Options**

### 1. 📓 **Notebook Flow (Recommended for Development)**

**Best for**: Interactive analysis, visualization, model development

```bash
# Open the UFC predictions notebook
jupyter notebook model/ufc_predictions.ipynb

# Run all cells up to cell 17 (card predictions)
# Then run the enhanced profitability cell (18) for instant analysis
```

**Features:**
- ✅ **Auto-extracts predictions** from your card results
- 🔄 **Live odds scraping** with fuzzy name matching
- 📊 **Detailed visualization** and debugging
- 🎯 **Integrated workflow** from predictions to betting instructions

---

### 2. ⚡ **Terminal Flow (Recommended for Speed)**

**Best for**: Quick analysis, automation, production use

```bash
# Quick launcher with menu options
./quick_analysis.sh

# OR direct execution
python3 run_profitability_analysis.py --sample

# Custom bankroll
python3 run_profitability_analysis.py --sample --bankroll 500

# Fast mode (cached odds)
python3 run_profitability_analysis.py --sample --no-live-odds
```

---

### 3. 🔄 **Hybrid Flow (Recommended for Production)**

**Best for**: Model development + rapid deployment

```bash
# 1. Generate predictions in notebook
jupyter notebook model/ufc_predictions.ipynb

# 2. Save predictions to file (add this to notebook):
# import json
# with open('latest_predictions.json', 'w') as f:
#     json.dump(my_predictions, f)

# 3. Run terminal analysis with saved predictions
python3 run_profitability_analysis.py --predictions "$(cat latest_predictions.json)"
```

---

## 🚀 **Execution Examples**

### Terminal Examples

```bash
# 1. Full live analysis with default $1000 bankroll
python3 run_profitability_analysis.py --sample

# 2. Quick test with cached odds
python3 run_profitability_analysis.py --sample --no-live-odds

# 3. Custom predictions
python3 run_profitability_analysis.py --predictions "Ilia Topuria:0.69,Charles Oliveira:0.31"

# 4. Interactive mode
python3 run_profitability_analysis.py

# 5. Custom bankroll
python3 run_profitability_analysis.py --sample --bankroll 2000
```

### Notebook Examples

```python
# Enhanced notebook cell automatically:
# 1. Detects if you ran card predictions
# 2. Extracts fighter probabilities 
# 3. Scrapes live TAB Australia odds
# 4. Matches fighters using fuzzy matching
# 5. Calculates expected value and optimal bets
# 6. Provides step-by-step betting instructions
```

---

## 📊 **Performance Comparison**

| Method | Speed | Accuracy | Features | Best For |
|--------|-------|----------|----------|----------|
| **Notebook** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Development |
| **Terminal** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | Production |
| **Hybrid** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | Best of both |

---

## 🔧 **Configuration Options**

### Bankroll Settings
```python
BANKROLL = 1000  # Your available betting capital (AUD)
```

### Odds Source Settings
```python
USE_LIVE_ODDS = True   # Scrape fresh odds (slow but accurate)
USE_LIVE_ODDS = False  # Use cached odds (fast but may be outdated)
```

### Betting Strategy Settings
```python
# In src/tab_profitability.py - TABProfitabilityAnalyzer
MAX_BET_PERCENTAGE = 0.05  # Max 5% of bankroll per bet
MIN_EXPECTED_VALUE = 0.05  # Minimum 5% EV to recommend bet
```

---

## 🏆 **Best Practices**

### 1. **Development Workflow**
```bash
# Use notebook for model development and testing
jupyter notebook model/ufc_predictions.ipynb

# Switch to terminal for production betting
python3 run_profitability_analysis.py --sample
```

### 2. **Live Event Workflow**
```bash
# Morning: Full analysis with live odds
./quick_analysis.sh  # Choose option 1

# Just before betting: Quick update
python3 run_profitability_analysis.py --sample --no-live-odds
```

### 3. **Automation Workflow**
```bash
# Schedule this script to run every hour
*/60 * * * * cd /path/to/ufc-predictor && python3 run_profitability_analysis.py --sample > latest_analysis.txt
```

---

## 🔍 **Troubleshooting**

### Common Issues

**No profitable opportunities found:**
```bash
# Check if odds are available
python3 run_profitability_analysis.py --sample --no-live-odds

# Debug fighter name matching
# The analysis will automatically show matching info
```

**Scraping errors:**
```bash
# Use cached odds as fallback
python3 run_profitability_analysis.py --sample --no-live-odds

# Check network connection and TAB Australia availability
```

**Import errors:**
```bash
# Install dependencies
pip install -r requirements.txt

# Check Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
```

---

## 📈 **Expected Outputs**

### Successful Analysis
```
🇦🇺 TAB AUSTRALIA PROFITABILITY ANALYSIS
==================================================
💰 Charles Oliveira: 29.5% EV, $50.00 bet, $14.75 profit
💰 Kai Kara-France: 33.1% EV, $50.00 bet, $16.54 profit

📊 SUMMARY
------------------------------
💰 Profitable opportunities: 2
💵 Total expected profit: $31.29 AUD
💳 Bankroll: $1000.00 AUD

📋 BETTING INSTRUCTIONS:
1. 🔍 Log into TAB Australia
2. 🥊 Navigate to UFC section
3. 💰 Place the following bets:
   • Charles Oliveira: $50.00 at 4.25 odds
   • Kai Kara-France: $50.00 at 2.95 odds
```

---

## 🎯 **Recommendations**

### For Beginners
1. **Start with notebook** - Run `model/ufc_predictions.ipynb` 
2. **Use sample predictions** to understand the workflow
3. **Start with small bankroll** ($100-200) for testing

### For Advanced Users  
1. **Use terminal script** for speed and automation
2. **Implement live odds scraping** for maximum accuracy
3. **Create custom prediction pipelines** feeding into the profitability analyzer

### For Production
1. **Hybrid approach** - Develop in notebook, deploy via terminal
2. **Schedule automated analysis** with cron jobs
3. **Log all results** for model performance tracking 