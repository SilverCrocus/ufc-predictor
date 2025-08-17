# 🚀 UFC Predictor - Quick Start Commands

## Step-by-Step Commands

### 1️⃣ Scrape Latest Fighter Data
```bash
cd /Users/diyagamah/Documents/ufc-predictor
uv run src/ufc_predictor/scrapers/fast_scraping.py
```

### 2️⃣ Train & Optimize Models (32 features)
```bash
uv run main.py pipeline --tune
```

### 3️⃣ Open Betting Notebook
```bash
jupyter notebook notebooks/production/enhanced_ufc_betting_v2.ipynb
```

---

## 🎯 All-in-One Command
Run everything in sequence:
```bash
uv run src/ufc_predictor/scrapers/fast_scraping.py && uv run main.py pipeline --tune && jupyter notebook notebooks/production/enhanced_ufc_betting_v2.ipynb
```

---

## 📝 Notes
- `pipeline --tune` automatically creates the 32-feature optimized model
- The notebook has 2% max bet limits and conservative calibration built-in
- All predictions are automatically tracked in `model_predictions_tracker.csv`