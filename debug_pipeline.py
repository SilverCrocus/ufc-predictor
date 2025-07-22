#!/usr/bin/env python3
"""
Debug version of pipeline to identify NaN issues.
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src import engineer_features_final, create_differential_features, merge_fight_data, prepare_modeling_data
from config.model_config import *

def debug_pipeline():
    """Run pipeline with debug output to identify NaN sources."""
    
    # 1. Load data
    print("=== Loading Data ===")
    fighters_raw_df = pd.read_csv('data/ufc_fighters_raw.csv')
    fights_df = pd.read_csv('data/ufc_fights.csv')
    print(f"Loaded: {len(fighters_raw_df)} fighters, {len(fights_df)} fights")
    
    # 2. Engineer features
    print("\n=== Engineering Features ===")
    fighters_engineered_df = engineer_features_final(fighters_raw_df)
    print(f"Engineered features shape: {fighters_engineered_df.shape}")
    print(f"NaN values in fighters: {fighters_engineered_df.isnull().sum().sum()}")
    
    # 3. Merge fight data
    print("\n=== Merging Fight Data ===")
    fight_dataset = merge_fight_data(fights_df, fighters_engineered_df)
    print(f"Merged dataset shape: {fight_dataset.shape}")
    print(f"NaN values in merged: {fight_dataset.isnull().sum().sum()}")
    
    # 4. Create differential features
    print("\n=== Creating Differential Features ===")
    fight_dataset_with_diffs = create_differential_features(fight_dataset)
    print(f"With diffs shape: {fight_dataset_with_diffs.shape}")
    nan_counts = fight_dataset_with_diffs.isnull().sum()
    print(f"NaN values in diffs: {nan_counts.sum()}")
    
    # Show columns with NaN values
    nan_columns = nan_counts[nan_counts > 0]
    if len(nan_columns) > 0:
        print(f"Columns with NaN values: {dict(nan_columns)}")
    
    # 5. Prepare modeling data
    print("\n=== Preparing Modeling Data ===")
    try:
        X_winner, y_winner = prepare_modeling_data(fight_dataset_with_diffs)
        print(f"Final dataset shape: {X_winner.shape}")
        final_nan_counts = X_winner.isnull().sum()
        print(f"NaN values in final: {final_nan_counts.sum()}")
        
        if final_nan_counts.sum() > 0:
            nan_cols_final = final_nan_counts[final_nan_counts > 0]
            print(f"Final NaN columns: {dict(nan_cols_final)}")
            
            # Show data types of problematic columns
            print(f"Data types of NaN columns:")
            for col in nan_cols_final.index:
                print(f"  {col}: {X_winner[col].dtype}")
        
    except Exception as e:
        print(f"Error in prepare_modeling_data: {e}")

if __name__ == "__main__":
    debug_pipeline()