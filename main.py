#!/usr/bin/env python3
"""
Main script for the UFC Predictor project.

This script provides a complete pipeline for:
1. Loading and engineering features from raw UFC data
2. Training machine learning models
3. Making fight predictions

Usage:
    python main.py --help
"""

import argparse
import sys
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src import (
    engineer_features_final, 
    create_differential_features,
    merge_fight_data,
    prepare_modeling_data,
    train_complete_pipeline,
    create_predictor
)
from config.model_config import *
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="UFC Fight Predictor")
    parser.add_argument("--mode", choices=["pipeline", "predict", "train"], 
                       default="predict", help="Mode to run")
    parser.add_argument("--fighter1", type=str, help="First fighter name")
    parser.add_argument("--fighter2", type=str, help="Second fighter name")
    parser.add_argument("--model-path", type=str, 
                       default=str(RF_TUNED_MODEL_PATH),
                       help="Path to trained model")
    parser.add_argument("--tune", action="store_true", 
                       help="Tune model hyperparameters during training")
    
    args = parser.parse_args()
    
    if args.mode == "pipeline":
        run_complete_pipeline(tune_hyperparameters=args.tune)
    elif args.mode == "train":
        train_models(tune_hyperparameters=args.tune)
    elif args.mode == "predict":
        if args.fighter1 and args.fighter2:
            make_prediction(args.fighter1, args.fighter2, args.model_path)
        else:
            interactive_prediction(args.model_path)


def run_complete_pipeline(tune_hyperparameters: bool = True):
    """Run the complete data processing and model training pipeline."""
    print("=== UFC Predictor: Complete Pipeline ===")
    
    # 1. Load raw data
    print("\n1. Loading raw data...")
    try:
        fighters_raw_df = pd.read_csv(RAW_FIGHTERS_DATA)
        fights_df = pd.read_csv(FIGHTS_DATA)
        print(f"Loaded {len(fighters_raw_df)} fighters and {len(fights_df)} fights")
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        return
    
    # 2. Engineer features
    print("\n2. Engineering features...")
    fighters_engineered_df = engineer_features_final(fighters_raw_df)
    
    # 3. Merge fight data
    print("\n3. Merging fight data with fighter statistics...")
    fight_dataset = merge_fight_data(fights_df, fighters_engineered_df)
    
    # 4. Create differential features
    print("\n4. Creating differential features...")
    fight_dataset_with_diffs = create_differential_features(fight_dataset)
    
    # 5. Prepare modeling data
    print("\n5. Preparing data for modeling...")
    X, y = prepare_modeling_data(fight_dataset_with_diffs)
    print(f"Dataset shape: {X.shape}, Target distribution: {y.value_counts().to_dict()}")
    
    # 6. Train models
    print("\n6. Training models...")
    trainer = train_complete_pipeline(X, y, tune_hyperparameters=tune_hyperparameters)
    
    # 7. Save models
    print("\n7. Saving models...")
    if tune_hyperparameters:
        trainer.save_model('random_forest_tuned', str(RF_TUNED_MODEL_PATH), X.columns.tolist())
    trainer.save_model('random_forest', str(RF_MODEL_PATH), X.columns.tolist())
    
    # 8. Save processed data
    print("\n8. Saving processed data...")
    fighters_engineered_df.to_csv(CORRECTED_FIGHTERS_DATA, index=False)
    fight_dataset_with_diffs.to_csv(FIGHT_DATASET_WITH_DIFFS, index=False)
    
    print("\n=== Pipeline Complete ===")


def train_models(tune_hyperparameters: bool = True):
    """Train models using existing processed data."""
    print("=== Training Models ===")
    
    try:
        # Load processed data
        fight_dataset_with_diffs = pd.read_csv(FIGHT_DATASET_WITH_DIFFS)
        print(f"Loaded dataset with {len(fight_dataset_with_diffs)} fights")
        
        # Prepare modeling data
        X, y = prepare_modeling_data(fight_dataset_with_diffs)
        print(f"Dataset shape: {X.shape}")
        
        # Train models
        trainer = train_complete_pipeline(X, y, tune_hyperparameters=tune_hyperparameters)
        
        # Save models
        if tune_hyperparameters:
            trainer.save_model('random_forest_tuned', str(RF_TUNED_MODEL_PATH), X.columns.tolist())
        trainer.save_model('random_forest', str(RF_MODEL_PATH), X.columns.tolist())
        
        print("\n=== Training Complete ===")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Run with --mode pipeline first to generate processed data.")


def make_prediction(fighter1: str, fighter2: str, model_path: str):
    """Make a prediction for two fighters."""
    print(f"=== Predicting: {fighter1} vs {fighter2} ===")
    
    try:
        # Load predictor
        predictor = create_predictor(
            model_path=model_path,
            columns_path=str(MODEL_COLUMNS_PATH),
            fighters_data_path=str(CORRECTED_FIGHTERS_DATA)
        )
        
        # Make prediction
        result = predictor.predict_fight_symmetrical(fighter1, fighter2)
        
        if 'error' in result:
            print(f"Error: {result['error']}")
        else:
            print(f"\nPredicted Winner: {result['predicted_winner']}")
            fighter_a_name = result['fighter_a']
            fighter_b_name = result['fighter_b']
            prob_key_a = f"{fighter_a_name}_win_probability"
            prob_key_b = f"{fighter_b_name}_win_probability"
            print(f"{fighter_a_name}: {result[prob_key_a]}")
            print(f"{fighter_b_name}: {result[prob_key_b]}")
            print(f"Confidence: {result['confidence']}")
            
    except Exception as e:
        print(f"Error making prediction: {e}")


def interactive_prediction(model_path: str):
    """Interactive prediction mode."""
    print("=== UFC Predictor: Interactive Mode ===")
    
    try:
        # Load predictor
        predictor = create_predictor(
            model_path=model_path,
            columns_path=str(MODEL_COLUMNS_PATH),
            fighters_data_path=str(CORRECTED_FIGHTERS_DATA)
        )
        
        print("Type 'quit' to exit")
        
        while True:
            fighter1 = input("\nEnter first fighter name: ").strip()
            if fighter1.lower() == 'quit':
                break
                
            fighter2 = input("Enter second fighter name: ").strip()
            if fighter2.lower() == 'quit':
                break
            
            # Make prediction
            result = predictor.predict_fight_symmetrical(fighter1, fighter2)
            
            if 'error' in result:
                print(f"Error: {result['error']}")
                
                # Show similar fighter names
                similar1 = predictor.get_available_fighters(fighter1)[:5]
                similar2 = predictor.get_available_fighters(fighter2)[:5]
                
                if similar1:
                    print(f"Similar to '{fighter1}': {', '.join(similar1)}")
                if similar2:
                    print(f"Similar to '{fighter2}': {', '.join(similar2)}")
            else:
                print(f"\nPredicted Winner: {result['predicted_winner']}")
                fighter_a_name = result['fighter_a']
                fighter_b_name = result['fighter_b']
                prob_key_a = f"{fighter_a_name}_win_probability"
                prob_key_b = f"{fighter_b_name}_win_probability"
                print(f"{fighter_a_name}: {result[prob_key_a]}")
                print(f"{fighter_b_name}: {result[prob_key_b]}")
                print(f"Confidence: {result['confidence']}")
                
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()