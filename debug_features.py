#!/usr/bin/env python3
"""
Debug script to check what features are being generated.
"""

import pandas as pd
from run_enhanced_pipeline import EnhancedUFCPipeline

# Initialize pipeline
pipeline = EnhancedUFCPipeline('configs/features.yaml', 'configs/backtest.yaml')

# Load adapted data
df = pd.read_csv('data/adapted_ufc_data_20250809_162935.csv')
print(f"\nOriginal data shape: {df.shape}")
print(f"Original columns: {list(df.columns)}")

# Generate enhanced features on a small sample
sample_df = df.head(100).copy()
enhanced_df = pipeline.generate_enhanced_features(sample_df, sample_df)

print(f"\nEnhanced data shape: {enhanced_df.shape}")
print(f"Enhanced columns: {list(enhanced_df.columns)}")

# Check data types
print("\nColumn types:")
for col in enhanced_df.columns:
    print(f"  {col}: {enhanced_df[col].dtype}")

# Check which columns would be selected as features
exclude_cols = {'fighter_a', 'fighter_b', 'winner', 'date', 'event', 
               'method', 'round', 'time', 'venue', 'division', 'event_id', 
               'fight_number', 'fighter_url', 'opponent_url'}

feature_cols = [col for col in enhanced_df.columns 
                if col not in exclude_cols
                and enhanced_df[col].dtype in ['float64', 'int64', 'float32', 'int32']]

print(f"\nSelected feature columns ({len(feature_cols)}):")
for col in feature_cols:
    print(f"  {col}")

# Also check columns ending with '_feature'
feature_suffix_cols = [col for col in enhanced_df.columns if col.endswith('_feature')]
print(f"\nColumns ending with '_feature' ({len(feature_suffix_cols)}):")
for col in feature_suffix_cols:
    print(f"  {col}")