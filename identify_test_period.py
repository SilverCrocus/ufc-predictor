#!/usr/bin/env python3
"""
Identify Your Model's Test Period
==================================

Quick script to identify the exact date range of your temporal test set.
This tells you which historical odds to fetch for backtesting.
"""

import pandas as pd
from pathlib import Path
from datetime import datetime
import re
import sys

# Try both paths
project_root = Path(__file__).parent
alt_root = project_root.parent

# Find fights data
fights_paths = [
    project_root / "src" / "ufc_predictor" / "data" / "ufc_fights.csv",
    alt_root / "src" / "ufc_predictor" / "data" / "ufc_fights.csv",
    project_root / "data" / "ufc_fights.csv",
    alt_root / "data" / "ufc_fights.csv"
]

fights_path = None
for path in fights_paths:
    if path.exists():
        fights_path = path
        break

if not fights_path:
    print("❌ Could not find ufc_fights.csv")
    sys.exit(1)

print(f"Loading fights from: {fights_path}")
fights_df = pd.read_csv(fights_path)

# Parse dates (matching your pipeline logic)
dates = []
for idx, row in fights_df.iterrows():
    date = None
    
    # Try Date column first
    if 'Date' in fights_df.columns and pd.notna(row.get('Date')):
        try:
            date = pd.to_datetime(row['Date'])
        except:
            pass
    
    # Try extracting from Event column
    if date is None and 'Event' in fights_df.columns:
        event = row.get('Event', '')
        match = re.search(r'([A-Z][a-z]+)\. (\d{1,2}), (\d{4})', event)
        if match:
            month_str, day, year = match.groups()
            months = {
                'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
                'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
            }
            month = months.get(month_str[:3], 1)
            try:
                date = datetime(int(year), month, int(day))
            except:
                pass
    
    dates.append(date)

fights_df['parsed_date'] = dates

# Remove invalid dates and sort
valid_dates = fights_df['parsed_date'].notna()
fights_with_dates = fights_df[valid_dates].sort_values('parsed_date')

print(f"\nTotal fights with valid dates: {len(fights_with_dates)}")

# Calculate 80/20 split (matching your training pipeline)
split_idx = int(len(fights_with_dates) * 0.8)

# Get training and test sets
train_fights = fights_with_dates.iloc[:split_idx]
test_fights = fights_with_dates.iloc[split_idx:]

print("\n" + "=" * 70)
print("YOUR MODEL'S TEMPORAL SPLIT")
print("=" * 70)

print(f"\n📊 TRAINING SET (80% - Model HAS seen these)")
print("-" * 50)
print(f"Period: {train_fights['parsed_date'].min().date()} to {train_fights['parsed_date'].max().date()}")
print(f"Number of fights: {len(train_fights)}")
print(f"Duration: {(train_fights['parsed_date'].max() - train_fights['parsed_date'].min()).days} days")

print(f"\n🎯 TEST SET (20% - Model has NOT seen these)")
print("-" * 50)
print(f"Period: {test_fights['parsed_date'].min().date()} to {test_fights['parsed_date'].max().date()}")
print(f"Number of fights: {len(test_fights)}")
print(f"Duration: {(test_fights['parsed_date'].max() - test_fights['parsed_date'].min()).days} days")

# Show some sample test fights
print(f"\nSample test fights (first 5):")
for idx, (_, fight) in enumerate(test_fights.head().iterrows()):
    print(f"  {idx+1}. {fight.get('Fighter', 'Unknown')} vs {fight.get('Opponent', 'Unknown')} ({fight['parsed_date'].date()})")

print("\n" + "=" * 70)
print("ACTION REQUIRED")
print("=" * 70)

test_start = test_fights['parsed_date'].min()
test_end = test_fights['parsed_date'].max()

print(f"\n1️⃣ Fetch historical odds for your test period:")
print(f"\n   python3 fetch_all_historical_odds.py \\")
print(f"     --start-date {test_start.date()} \\")
print(f"     --end-date {test_end.date()}")

print(f"\n2️⃣ Then run temporal backtest:")
print(f"\n   python3 backtest_temporal_model.py")

print(f"\n3️⃣ This will give you REALISTIC ROI estimates because:")
print(f"   • Your model has NEVER seen these {len(test_fights)} fights")
print(f"   • The odds are from the actual time period")
print(f"   • No look-ahead bias or data leakage")

print("\n" + "=" * 70)
print("ESTIMATED API USAGE")
print("=" * 70)
days = (test_end - test_start).days
print(f"Date range: {days} days")
print(f"Estimated API credits: {days * 10} - {days * 30} credits")
print(f"Your available credits: 20,000 - More than enough!")
print()