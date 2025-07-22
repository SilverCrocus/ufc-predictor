#!/usr/bin/env python3
"""
Simple training script to create working models quickly.
"""

import pandas as pd
import numpy as np
import joblib
import json
import sys
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src import engineer_features_final, create_differential_features, merge_fight_data, prepare_modeling_data
from config.model_config import *

def simple_train():
    """Simple training without hyperparameter tuning."""
    
    print("=== Simple Model Training ===")
    
    # 1. Load and process data
    fighters_raw_df = pd.read_csv('data/ufc_fighters_raw.csv')
    fights_df = pd.read_csv('data/ufc_fights.csv')
    print(f"Loaded: {len(fighters_raw_df)} fighters, {len(fights_df)} fights")
    
    # 2. Engineer features
    fighters_engineered_df = engineer_features_final(fighters_raw_df)
    
    # 3. Merge and prepare data
    fight_dataset = merge_fight_data(fights_df, fighters_engineered_df)
    fight_dataset_with_diffs = create_differential_features(fight_dataset)
    X, y = prepare_modeling_data(fight_dataset_with_diffs)
    
    print(f"Dataset shape: {X.shape}, Target distribution: {y.value_counts().to_dict()}")
    
    # 4. Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
    
    # 5. Train Random Forest
    print("Training Random Forest...")
    rf_model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    
    # 6. Evaluate
    y_pred = rf_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {accuracy:.3f}")
    
    # 7. Save model
    model_path = "model/ufc_random_forest_model_tuned.joblib"
    joblib.dump(rf_model, model_path)
    print(f"Model saved to: {model_path}")
    
    # 8. Save column names
    columns_path = "model/winner_model_columns.json"
    with open(columns_path, 'w') as f:
        json.dump(list(X.columns), f)
    print(f"Columns saved to: {columns_path}")
    
    # 9. Save corrected fighters data for predictions
    fighters_path = "model/ufc_fighters_engineered_corrected.csv"
    fighters_engineered_df.to_csv(fighters_path, index=False)
    print(f"Fighters data saved to: {fighters_path}")
    
    print("✅ Simple training completed successfully!")
    return rf_model

if __name__ == "__main__":
    simple_train()