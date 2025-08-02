#!/usr/bin/env python3
"""
Main script for the UFC Predictor project.

This script provides a complete pipeline for:
1. Loading and engineering features from raw UFC data
2. Training machine learning models (winner + method prediction)
3. Making fight predictions with both outcome and method

Usage:
    python main.py --help
"""

import argparse
import sys
import json
import joblib
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from ufc_predictor.data.feature_engineering import (
    engineer_features_final, 
    create_differential_features,
    merge_fight_data,
    prepare_modeling_data
)
from ufc_predictor.models.model_training import (
    train_complete_pipeline
)
from ufc_predictor.core.prediction import (
    create_predictor
)
from configs.model_config import *
import pandas as pd


# --- Versioning and Model Management ---
def setup_model_versioning():
    """Create versioned directory structure for models."""
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M')
    models_dir = Path("model") / f"training_{timestamp}"
    models_dir.mkdir(exist_ok=True)
    return models_dir, timestamp

def get_latest_scraped_data():
    """Find and return paths to the latest scraped data files."""
    data_dir = Path('data')
    
    if not data_dir.exists():
        raise FileNotFoundError("Data directory not found. Run scraping first.")
    
    # Find all scrape directories
    scrape_dirs = [d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith('scrape_')]
    
    if not scrape_dirs:
        raise FileNotFoundError("No scraped data directories found. Run scraping first.")
    
    # Get the most recent scrape directory
    latest_scrape_dir = max(scrape_dirs, key=lambda x: x.stat().st_mtime)
    
    # Look for the required files in the latest directory
    fighters_files = list(latest_scrape_dir.glob('ufc_fighters_raw_*.csv'))
    fights_files = list(latest_scrape_dir.glob('ufc_fights_*.csv'))
    
    if not fighters_files:
        raise FileNotFoundError(f"No fighters data found in {latest_scrape_dir}")
    if not fights_files:
        raise FileNotFoundError(f"No fights data found in {latest_scrape_dir}")
    
    # Get the most recent files (in case there are multiple)
    latest_fighters_file = max(fighters_files, key=lambda x: x.stat().st_mtime)
    latest_fights_file = max(fights_files, key=lambda x: x.stat().st_mtime)
    
    print(f"📁 Using latest scraped data from: {latest_scrape_dir.name}")
    print(f"   Fighters file: {latest_fighters_file.name}")
    print(f"   Fights file: {latest_fights_file.name}")
    
    return str(latest_fighters_file), str(latest_fights_file), str(latest_scrape_dir)

def get_latest_trained_models():
    """Find and return paths to the latest trained models."""
    model_dir = Path('model')
    
    # Find all training directories
    training_dirs = [d for d in model_dir.iterdir() if d.is_dir() and d.name.startswith('training_')]
    
    if not training_dirs:
        # Fallback to standard model locations
        print("⚠️  No training directories found, using standard model locations...")
        winner_model_path = 'model/ufc_random_forest_model_tuned.joblib'
        method_model_path = 'model/ufc_multiclass_model.joblib'
        winner_cols_path = 'model/winner_model_columns.json'
        method_cols_path = 'model/method_model_columns.json'
        fighters_data_path = str(CORRECTED_FIGHTERS_DATA)
        
        # Check if files exist
        if not Path(winner_model_path).exists():
            winner_model_path = 'model/ufc_random_forest_model.joblib'
        
        return winner_model_path, method_model_path, winner_cols_path, method_cols_path, fighters_data_path
    
    # Get the most recent training directory
    latest_training_dir = max(training_dirs, key=lambda x: x.stat().st_mtime)
    
    # Look for model files in the latest directory
    winner_models = list(latest_training_dir.glob('ufc_winner_model_*.joblib'))
    winner_tuned_models = list(latest_training_dir.glob('ufc_winner_model_tuned_*.joblib'))
    method_models = list(latest_training_dir.glob('ufc_method_model_*.joblib'))
    winner_cols_files = list(latest_training_dir.glob('winner_model_columns_*.json'))
    method_cols_files = list(latest_training_dir.glob('method_model_columns_*.json'))
    fighters_data_files = list(latest_training_dir.glob('ufc_fighters_engineered_*.csv'))
    
    # Prefer tuned models if available
    if winner_tuned_models:
        winner_model_path = str(max(winner_tuned_models, key=lambda x: x.stat().st_mtime))
    elif winner_models:
        winner_model_path = str(max(winner_models, key=lambda x: x.stat().st_mtime))
    else:
        winner_model_path = 'model/ufc_random_forest_model_tuned.joblib'
    
    method_model_path = str(max(method_models, key=lambda x: x.stat().st_mtime)) if method_models else 'model/ufc_multiclass_model.joblib'
    winner_cols_path = str(max(winner_cols_files, key=lambda x: x.stat().st_mtime)) if winner_cols_files else 'model/winner_model_columns.json'
    method_cols_path = str(max(method_cols_files, key=lambda x: x.stat().st_mtime)) if method_cols_files else 'model/method_model_columns.json'
    fighters_data_path = str(max(fighters_data_files, key=lambda x: x.stat().st_mtime)) if fighters_data_files else str(CORRECTED_FIGHTERS_DATA)
    
    print(f"📁 Using latest trained models from: {latest_training_dir.name}")
    print(f"   Winner model: {Path(winner_model_path).name}")
    print(f"   Method model: {Path(method_model_path).name}")
    
    return winner_model_path, method_model_path, winner_cols_path, method_cols_path, fighters_data_path

def clean_method_column(df):
    """Clean the Method column for multiclass training (from notebook)."""
    def clean_method(method):
        if pd.isna(method):
            return None
        method_upper = str(method).upper()
        if method_upper.startswith('KO/TKO'):
            return 'KO/TKO'
        if method_upper.startswith('SUB'):
            return 'Submission'
        if 'DEC' in method_upper or method_upper == 'DECISION':
            return 'Decision'
        return None
    
    df['Method_Cleaned'] = df['Method'].apply(clean_method)
    return df.dropna(subset=['Method_Cleaned'])

def main():
    parser = argparse.ArgumentParser(description="UFC Fight Predictor - Winner & Method Prediction")
    parser.add_argument("--mode", choices=["pipeline", "predict", "train", "card", "results"], 
                       default="predict", help="Mode to run")
    parser.add_argument("--fighter1", type=str, help="First fighter name")
    parser.add_argument("--fighter2", type=str, help="Second fighter name")
    parser.add_argument("--fights", type=str, nargs='+', 
                       help="Multiple fights in format 'Fighter1 vs Fighter2'")
    parser.add_argument("--model-path", type=str, 
                       default=str(RF_TUNED_MODEL_PATH),
                       help="Path to trained winner model")
    parser.add_argument("--method-model-path", type=str,
                       default="model/ufc_multiclass_model.joblib",
                       help="Path to trained method model")
    parser.add_argument("--tune", action="store_true", 
                       help="Tune model hyperparameters during training")
    parser.add_argument("--metadata", type=str,
                       help="Path to training metadata file for results display")
    
    args = parser.parse_args()
    
    if args.mode == "pipeline":
        run_complete_pipeline(tune_hyperparameters=args.tune)
    elif args.mode == "train":
        train_models(tune_hyperparameters=args.tune)
    elif args.mode == "predict":
        if args.fighter1 and args.fighter2:
            make_prediction(args.fighter1, args.fighter2, args.model_path, args.method_model_path)
        else:
            interactive_prediction(args.model_path, args.method_model_path)
    elif args.mode == "card":
        if args.fights:
            predict_card(args.fights, args.model_path, args.method_model_path)
        else:
            print("Error: --fights required for card mode")
            print("Example: --fights 'Fighter1 vs Fighter2' 'Fighter3 vs Fighter4'")
    elif args.mode == "results":
        if args.metadata:
            display_training_results(args.metadata)
        else:
            # Find the most recent metadata file
            model_dir = Path('model')
            metadata_files = list(model_dir.glob('*/training_metadata_*.json'))
            if metadata_files:
                latest_metadata = max(metadata_files, key=lambda x: x.stat().st_mtime)
                print(f"Using latest metadata file: {latest_metadata}")
                display_training_results(str(latest_metadata))
            else:
                print("No training metadata files found. Run training first with --mode pipeline or --mode train")


def run_complete_pipeline(tune_hyperparameters: bool = True):
    """Run the complete data processing and model training pipeline."""
    print("=== UFC Predictor: Complete Pipeline (Winner + Method) ===")
    
    # Setup versioning
    models_dir, timestamp = setup_model_versioning()
    print(f"Models will be saved to: {models_dir}")
    
    # 1. Load raw data (automatically find latest)
    print("\n1. Loading latest scraped data...")
    try:
        # Try to get latest scraped data first
        try:
            latest_fighters_path, latest_fights_path, scrape_dir = get_latest_scraped_data()
            fighters_raw_df = pd.read_csv(latest_fighters_path)
            fights_df = pd.read_csv(latest_fights_path)
            print(f"✅ Loaded latest data: {len(fighters_raw_df)} fighters and {len(fights_df)} fights")
        except FileNotFoundError as scrape_error:
            # Fallback to hardcoded paths if no versioned data exists
            print(f"⚠️  Latest scraped data not found: {scrape_error}")
            print("   Falling back to hardcoded data paths...")
            fighters_raw_df = pd.read_csv(RAW_FIGHTERS_DATA)
            fights_df = pd.read_csv(FIGHTS_DATA)
            print(f"✅ Loaded fallback data: {len(fighters_raw_df)} fighters and {len(fights_df)} fights")
    except FileNotFoundError as e:
        print(f"❌ Error loading data: {e}")
        print("   Please run scraping first or ensure data files exist.")
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
    
    # 5. Prepare modeling data for WINNER prediction
    print("\n5. Preparing data for winner modeling...")
    X_winner, y_winner = prepare_modeling_data(fight_dataset_with_diffs)
    print(f"Winner dataset shape: {X_winner.shape}, Target distribution: {y_winner.value_counts().to_dict()}")
    
    # 6. Prepare modeling data for METHOD prediction
    print("\n6. Preparing data for method modeling...")
    fight_dataset_clean_methods = clean_method_column(fight_dataset_with_diffs)
    
    # Remove identifier columns for method prediction
    cols_to_drop_method = [
        'Outcome', 'Fighter', 'Opponent', 'Event', 'Method', 'Time', 'Method_Cleaned',
        'fighter_url', 'opponent_url', 'blue_fighter_url', 'red_fighter_url',
        'blue_Name', 'red_Name'
    ]
    X_method = fight_dataset_clean_methods.drop(columns=[col for col in cols_to_drop_method if col in fight_dataset_clean_methods.columns])
    y_method = fight_dataset_clean_methods['Method_Cleaned']
    
    # Fill missing values
    X_method.fillna(X_method.median(), inplace=True)
    print(f"Method dataset shape: {X_method.shape}, Target distribution: {y_method.value_counts().to_dict()}")
    
    # 7. Train WINNER models
    print("\n7. Training winner prediction models...")
    trainer = train_complete_pipeline(X_winner, y_winner, tune_hyperparameters=tune_hyperparameters)
    
    # 8. Train METHOD models with GridSearch
    print("\n8. Training method prediction models...")
    X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(
        X_method, y_method, test_size=0.2, random_state=42, stratify=y_method
    )
    
    if tune_hyperparameters:
        print("   Performing GridSearch for method prediction model...")
        method_param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        rf_method_base = RandomForestClassifier(random_state=42, n_jobs=-1)
        method_grid_search = GridSearchCV(
            rf_method_base, method_param_grid, 
            cv=5, scoring='accuracy', n_jobs=-1, verbose=1
        )
        method_grid_search.fit(X_train_m, y_train_m)
        
        rf_method_model = method_grid_search.best_estimator_
        print(f"   Best method model parameters: {method_grid_search.best_params_}")
        print(f"   Best CV score: {method_grid_search.best_score_:.4f}")
        
        # Store grid search results for metadata
        method_grid_results = {
            'best_params': method_grid_search.best_params_,
            'best_cv_score': float(method_grid_search.best_score_),
            'cv_results': {
                'mean_test_scores': method_grid_search.cv_results_['mean_test_score'].tolist(),
                'params': [str(p) for p in method_grid_search.cv_results_['params']]
            }
        }
    else:
        print("   Training method model with default parameters...")
        rf_method_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf_method_model.fit(X_train_m, y_train_m)
        method_grid_results = None
    
    # Evaluate method model
    y_pred_m = rf_method_model.predict(X_test_m)
    method_accuracy = accuracy_score(y_test_m, y_pred_m)
    print(f"   Method Model Test Accuracy: {method_accuracy:.4f} ({method_accuracy*100:.2f}%)")
    
    # Detailed method model evaluation
    print(f"\n   Method Model Classification Report:")
    method_report = classification_report(y_test_m, y_pred_m, output_dict=True)
    print(classification_report(y_test_m, y_pred_m))
    
    # Method confusion matrix
    method_cm = confusion_matrix(y_test_m, y_pred_m, labels=['Decision', 'KO/TKO', 'Submission'])
    print(f"\n   Method Model Confusion Matrix:")
    print("   Predicted ->  Decision  KO/TKO  Submission")
    for i, actual_class in enumerate(['Decision', 'KO/TKO', 'Submission']):
        print(f"   {actual_class:<12} {method_cm[i][0]:>8} {method_cm[i][1]:>7} {method_cm[i][2]:>11}")
    
    # 9. Save WINNER models
    print("\n9. Saving winner models...")
    winner_model_path = models_dir / f'ufc_winner_model_{timestamp}.joblib'
    winner_model_tuned_path = models_dir / f'ufc_winner_model_tuned_{timestamp}.joblib'
    winner_columns_path = models_dir / f'winner_model_columns_{timestamp}.json'
    
    if tune_hyperparameters:
        trainer.save_model('random_forest_tuned', str(winner_model_tuned_path), X_winner.columns.tolist())
        # Also save to standard location
        joblib.dump(trainer.models['random_forest_tuned'], 'model/ufc_random_forest_model_tuned.joblib')
    
    trainer.save_model('random_forest', str(winner_model_path), X_winner.columns.tolist())
    
    # Save winner model columns
    with open(winner_columns_path, 'w') as f:
        json.dump(X_winner.columns.tolist(), f)
    with open('model/winner_model_columns.json', 'w') as f:
        json.dump(X_winner.columns.tolist(), f)
    
    # 10. Save METHOD models
    print("\n10. Saving method models...")
    method_model_path = models_dir / f'ufc_method_model_{timestamp}.joblib'
    method_columns_path = models_dir / f'method_model_columns_{timestamp}.json'
    
    joblib.dump(rf_method_model, method_model_path)
    joblib.dump(rf_method_model, 'model/ufc_multiclass_model.joblib')  # Standard location
    
    # Save method model columns
    with open(method_columns_path, 'w') as f:
        json.dump(X_method.columns.tolist(), f)
    with open('model/method_model_columns.json', 'w') as f:
        json.dump(X_method.columns.tolist(), f)
    
    # 11. Save processed data
    print("\n11. Saving processed data...")
    fighters_data_path = models_dir / f'ufc_fighters_engineered_{timestamp}.csv'
    fight_data_path = models_dir / f'ufc_fight_dataset_with_diffs_{timestamp}.csv'
    
    fighters_engineered_df.to_csv(fighters_data_path, index=False)
    fighters_engineered_df.to_csv(CORRECTED_FIGHTERS_DATA, index=False)  # Standard location
    
    fight_dataset_with_diffs.to_csv(fight_data_path, index=False)
    fight_dataset_with_diffs.to_csv(FIGHT_DATASET_WITH_DIFFS, index=False)  # Standard location
    
    # 12. Get detailed winner model scores
    winner_scores = {}
    winner_scores['random_forest'] = {
        'accuracy': float(trainer.get_model_score('random_forest')),
        'model_type': 'RandomForest (default params)'
    }
    
    if tune_hyperparameters:
        winner_scores['random_forest_tuned'] = {
            'accuracy': float(trainer.get_model_score('random_forest_tuned')),
            'model_type': 'RandomForest (GridSearch tuned)'
        }
        # Get GridSearch results for winner model if available
        if hasattr(trainer, 'grid_search_results'):
            winner_scores['grid_search_results'] = trainer.grid_search_results
    
    # 13. Save comprehensive training metadata
    metadata = {
        'training_timestamp': timestamp,
        'winner_models': winner_scores,
        'method_model': {
            'accuracy': float(method_accuracy),
            'classification_report': method_report,
            'confusion_matrix': method_cm.tolist(),
            'model_type': 'RandomForest (GridSearch tuned)' if tune_hyperparameters else 'RandomForest (default params)'
        },
        'datasets': {
            'winner_dataset_shape': list(X_winner.shape),
            'method_dataset_shape': list(X_method.shape),
            'winner_target_distribution': y_winner.value_counts().to_dict(),
            'method_target_distribution': y_method.value_counts().to_dict()
        },
        'hyperparameter_tuning': tune_hyperparameters,
        'files_created': {
            'winner_model': str(winner_model_path.name),
            'method_model': str(method_model_path.name),
            'winner_columns': str(winner_columns_path.name),
            'method_columns': str(method_columns_path.name),
            'fighters_data': str(fighters_data_path.name),
            'fight_data': str(fight_data_path.name)
        }
    }
    
    # Add method GridSearch results if available
    if method_grid_results:
        metadata['method_model']['grid_search_results'] = method_grid_results
    
    if tune_hyperparameters:
        metadata['files_created']['winner_model_tuned'] = str(winner_model_tuned_path.name)
    
    metadata_path = models_dir / f'training_metadata_{timestamp}.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"TRAINING PIPELINE COMPLETE")
    print(f"{'='*60}")
    print(f"📊 COMPREHENSIVE TRAINING SUMMARY:")
    print(f"\n🏆 WINNER MODEL RESULTS:")
    for model_name, scores in winner_scores.items():
        if 'accuracy' in scores:
            print(f"   {model_name}: {scores['accuracy']:.4f} ({scores['accuracy']*100:.2f}%)")
            print(f"      Type: {scores['model_type']}")
    
    print(f"\n⚔️  METHOD MODEL RESULTS:")
    print(f"   Test Accuracy: {method_accuracy:.4f} ({method_accuracy*100:.2f}%)")
    print(f"   Model Type: {metadata['method_model']['model_type']}")
    
    if tune_hyperparameters and method_grid_results:
        print(f"   Best CV Score: {method_grid_results['best_cv_score']:.4f}")
        print(f"   Best Parameters: {method_grid_results['best_params']}")
    
    print(f"\n📁 FILES SAVED:")
    print(f"   Training Directory: {models_dir}")
    print(f"   Metadata File: {metadata_path}")
    print(f"   Total Files Created: {len(metadata['files_created'])}")
    
    print(f"\n📈 DATASET INFO:")
    print(f"   Winner Training Data: {X_winner.shape[0]:,} fights, {X_winner.shape[1]} features")
    print(f"   Method Training Data: {X_method.shape[0]:,} fights, {X_method.shape[1]} features")
    
    print(f"\n✅ Models are ready for prediction!")


def display_training_results(metadata_path: str):
    """Display comprehensive training results from metadata file."""
    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        print(f"\n{'='*60}")
        print(f"DETAILED TRAINING RESULTS")
        print(f"{'='*60}")
        print(f"Training Session: {metadata['training_timestamp']}")
        print(f"Hyperparameter Tuning: {'Enabled' if metadata['hyperparameter_tuning'] else 'Disabled'}")
        
        # Winner model results
        print(f"\n🏆 WINNER MODEL PERFORMANCE:")
        for model_name, scores in metadata['winner_models'].items():
            if 'accuracy' in scores:
                print(f"   {model_name}:")
                print(f"      Accuracy: {scores['accuracy']:.4f} ({scores['accuracy']*100:.2f}%)")
                print(f"      Type: {scores['model_type']}")
        
        # Method model results
        method_data = metadata['method_model']
        print(f"\n⚔️  METHOD MODEL PERFORMANCE:")
        print(f"   Test Accuracy: {method_data['accuracy']:.4f} ({method_data['accuracy']*100:.2f}%)")
        print(f"   Model Type: {method_data['model_type']}")
        
        if 'grid_search_results' in method_data:
            gs_results = method_data['grid_search_results']
            print(f"   Best CV Score: {gs_results['best_cv_score']:.4f}")
            print(f"   Best Parameters: {gs_results['best_params']}")
        
        # Classification report for method model
        if 'classification_report' in method_data:
            print(f"\n📊 METHOD MODEL CLASSIFICATION REPORT:")
            class_report = method_data['classification_report']
            for class_name in ['Decision', 'KO/TKO', 'Submission']:
                if class_name in class_report:
                    metrics = class_report[class_name]
                    print(f"   {class_name}:")
                    print(f"      Precision: {metrics['precision']:.3f}")
                    print(f"      Recall: {metrics['recall']:.3f}")
                    print(f"      F1-Score: {metrics['f1-score']:.3f}")
                    print(f"      Support: {metrics['support']}")
        
        # Dataset information
        datasets = metadata['datasets']
        print(f"\n📈 DATASET INFORMATION:")
        print(f"   Winner Model Dataset: {datasets['winner_dataset_shape'][0]:,} fights x {datasets['winner_dataset_shape'][1]} features")
        print(f"   Method Model Dataset: {datasets['method_dataset_shape'][0]:,} fights x {datasets['method_dataset_shape'][1]} features")
        
        print(f"\n   Winner Target Distribution:")
        for outcome, count in datasets['winner_target_distribution'].items():
            print(f"      {outcome}: {count:,}")
        
        print(f"\n   Method Target Distribution:")
        for method, count in datasets['method_target_distribution'].items():
            print(f"      {method}: {count:,}")
        
    except Exception as e:
        print(f"Error reading training results: {e}")

def train_models(tune_hyperparameters: bool = True):
    """Train models using existing processed data with detailed results."""
    print("=== Training Models (Standalone Mode) ===")
    
    # Setup versioning
    models_dir, timestamp = setup_model_versioning()
    print(f"Models will be saved to: {models_dir}")
    
    try:
        # Try to load processed data from latest training session first
        try:
            model_dir = Path('model')
            training_dirs = [d for d in model_dir.iterdir() if d.is_dir() and d.name.startswith('training_')]
            if training_dirs:
                latest_training_dir = max(training_dirs, key=lambda x: x.stat().st_mtime)
                fight_data_files = list(latest_training_dir.glob('ufc_fight_dataset_with_diffs_*.csv'))
                fighter_data_files = list(latest_training_dir.glob('ufc_fighters_engineered_*.csv'))
                
                if fight_data_files and fighter_data_files:
                    latest_fight_data = max(fight_data_files, key=lambda x: x.stat().st_mtime)
                    latest_fighter_data = max(fighter_data_files, key=lambda x: x.stat().st_mtime)
                    
                    fight_dataset_with_diffs = pd.read_csv(latest_fight_data)
                    fighters_df = pd.read_csv(latest_fighter_data)
                    print(f"✅ Loaded latest processed data: {len(fight_dataset_with_diffs)} fights")
                    print(f"   From training session: {latest_training_dir.name}")
                else:
                    raise FileNotFoundError("Processed data files not found in latest training directory")
            else:
                raise FileNotFoundError("No training directories found")
                
        except FileNotFoundError:
            # Fallback to standard locations
            print("⚠️  Latest processed data not found, using standard locations...")
            fight_dataset_with_diffs = pd.read_csv(FIGHT_DATASET_WITH_DIFFS)
            fighters_df = pd.read_csv(CORRECTED_FIGHTERS_DATA)
            print(f"✅ Loaded fallback data: {len(fight_dataset_with_diffs)} fights")
        
        # Prepare winner modeling data
        X_winner, y_winner = prepare_modeling_data(fight_dataset_with_diffs)
        print(f"Winner dataset shape: {X_winner.shape}")
        
        # Prepare method modeling data
        fight_dataset_clean_methods = clean_method_column(fight_dataset_with_diffs)
        cols_to_drop_method = [
            'Outcome', 'Fighter', 'Opponent', 'Event', 'Method', 'Time', 'Method_Cleaned',
            'fighter_url', 'opponent_url', 'blue_fighter_url', 'red_fighter_url',
            'blue_Name', 'red_Name'
        ]
        X_method = fight_dataset_clean_methods.drop(columns=[col for col in cols_to_drop_method if col in fight_dataset_clean_methods.columns])
        y_method = fight_dataset_clean_methods['Method_Cleaned']
        X_method.fillna(X_method.median(), inplace=True)
        print(f"Method dataset shape: {X_method.shape}")
        
        # Train winner models
        print("\nTraining winner models...")
        trainer = train_complete_pipeline(X_winner, y_winner, tune_hyperparameters=tune_hyperparameters)
        
        # Train method models (same logic as in pipeline)
        print("\nTraining method models...")
        X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(
            X_method, y_method, test_size=0.2, random_state=42, stratify=y_method
        )
        
        if tune_hyperparameters:
            print("   Performing GridSearch for method model...")
            method_param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            
            rf_method_base = RandomForestClassifier(random_state=42, n_jobs=-1)
            method_grid_search = GridSearchCV(
                rf_method_base, method_param_grid, 
                cv=5, scoring='accuracy', n_jobs=-1, verbose=1
            )
            method_grid_search.fit(X_train_m, y_train_m)
            rf_method_model = method_grid_search.best_estimator_
            print(f"   Best parameters: {method_grid_search.best_params_}")
        else:
            rf_method_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            rf_method_model.fit(X_train_m, y_train_m)
        
        # Evaluate and save models
        y_pred_m = rf_method_model.predict(X_test_m)
        method_accuracy = accuracy_score(y_test_m, y_pred_m)
        
        # Save models
        winner_model_path = models_dir / f'ufc_winner_model_{timestamp}.joblib'
        method_model_path = models_dir / f'ufc_method_model_{timestamp}.joblib'
        
        if tune_hyperparameters:
            trainer.save_model('random_forest_tuned', str(RF_TUNED_MODEL_PATH), X_winner.columns.tolist())
        trainer.save_model('random_forest', str(RF_MODEL_PATH), X_winner.columns.tolist())
        
        joblib.dump(rf_method_model, method_model_path)
        joblib.dump(rf_method_model, 'model/ufc_multiclass_model.joblib')
        
        # Save column mappings
        with open('model/winner_model_columns.json', 'w') as f:
            json.dump(X_winner.columns.tolist(), f)
        with open('model/method_model_columns.json', 'w') as f:
            json.dump(X_method.columns.tolist(), f)
        
        print(f"\n✅ Training Complete!")
        print(f"   Winner Model Accuracy: {trainer.get_model_score('random_forest'):.4f}")
        print(f"   Method Model Accuracy: {method_accuracy:.4f}")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Run with --mode pipeline first to generate processed data.")


# --- Enhanced Prediction Functions (from notebook) ---
def _get_full_prediction_from_perspective(fighter1_name, fighter2_name, all_fighters_data, win_cols, meth_cols, win_model, meth_model):
    """Helper function to get raw prediction probabilities from a single perspective."""
    fighter1_stats = all_fighters_data[all_fighters_data['Name'] == fighter1_name]
    fighter2_stats = all_fighters_data[all_fighters_data['Name'] == fighter2_name]
    if fighter1_stats.empty or fighter2_stats.empty:
        missing = fighter1_name if fighter1_stats.empty else fighter2_name
        return {"error": f"Fighter '{missing}' not found."}, None

    fighter1_stats, fighter2_stats = fighter1_stats.iloc[0], fighter2_stats.iloc[0]
    blue_stats, red_stats = fighter1_stats.add_prefix('blue_'), fighter2_stats.add_prefix('red_')
    diff_features = {}
    for blue_col in blue_stats.index:
        if blue_col.startswith('blue_') and 'url' not in blue_col and 'Name' not in blue_col:
            red_col, base_name = blue_col.replace('blue_', 'red_'), blue_col.replace('blue_', '')
            if red_col in red_stats.index:
                diff_col_name = base_name.lower().replace(' ', '_').replace('.', '') + '_diff'
                diff_features[diff_col_name] = blue_stats[blue_col] - red_stats[red_col]
    
    single_fight_data = {**blue_stats, **red_stats, **diff_features}
    prediction_df_base = pd.DataFrame([single_fight_data])

    X_winner = prediction_df_base.reindex(columns=win_cols, fill_value=0)
    winner_probs = win_model.predict_proba(X_winner)[0]

    X_method = prediction_df_base.reindex(columns=meth_cols, fill_value=0)
    method_probs = meth_model.predict_proba(X_method)[0]
    
    return winner_probs, method_probs

def predict_fight_symmetrical(fighter_a, fighter_b, all_fighters_data, win_cols, meth_cols, win_model, meth_model):
    """Calculates a final, symmetrical prediction for both winner and method."""
    # Prediction 1: A is in the blue corner
    winner_probs1, method_probs1 = _get_full_prediction_from_perspective(fighter_a, fighter_b, all_fighters_data, win_cols, meth_cols, win_model, meth_model)
    if "error" in winner_probs1: return winner_probs1

    # Prediction 2: B is in the blue corner
    winner_probs2, method_probs2 = _get_full_prediction_from_perspective(fighter_b, fighter_a, all_fighters_data, win_cols, meth_cols, win_model, meth_model)
    if "error" in winner_probs2: return winner_probs2

    # Average the winner probabilities
    prob_a_wins_as_blue = winner_probs1[1]  # P(A wins | A is blue)
    prob_a_wins_as_red = 1 - winner_probs2[1] # P(A wins | A is red) = 1 - P(B wins | B is blue)
    final_prob_a_wins = (prob_a_wins_as_blue + prob_a_wins_as_red) / 2
    final_prob_b_wins = 1 - final_prob_a_wins
    
    # Average the method probabilities
    avg_method_probs = (method_probs1 + method_probs2) / 2
    method_classes = meth_model.classes_
    predicted_method = method_classes[np.argmax(avg_method_probs)]

    # Format the final result
    result = {
        "matchup": f"{fighter_a} vs. {fighter_b}",
        "predicted_winner": fighter_a if final_prob_a_wins > final_prob_b_wins else fighter_b,
        "winner_confidence": f"{max(final_prob_a_wins, final_prob_b_wins)*100:.2f}%",
        "win_probabilities": {
            fighter_a: f"{final_prob_a_wins*100:.2f}%",
            fighter_b: f"{final_prob_b_wins*100:.2f}%",
        },
        "predicted_method": predicted_method,
        "method_probabilities": {
            method_classes[i]: f"{avg_method_probs[i]*100:.2f}%" for i in range(len(method_classes))
        }
    }
    return result

def make_prediction(fighter1: str, fighter2: str, winner_model_path: str = None, method_model_path: str = None):
    """Make a prediction for two fighters with both winner and method."""
    print(f"=== Predicting: {fighter1} vs {fighter2} ===")
    
    try:
        # Auto-detect latest models if paths not provided
        if not winner_model_path or not method_model_path:
            print("🔍 Auto-detecting latest trained models...")
            auto_winner_path, auto_method_path, auto_winner_cols, auto_method_cols, auto_fighters_data = get_latest_trained_models()
            
            winner_model_path = winner_model_path or auto_winner_path
            method_model_path = method_model_path or auto_method_path
            winner_cols_path = auto_winner_cols
            method_cols_path = auto_method_cols
            fighters_data_path = auto_fighters_data
        else:
            # Use standard column files and fighter data
            winner_cols_path = 'model/winner_model_columns.json'
            method_cols_path = 'model/method_model_columns.json'
            fighters_data_path = str(CORRECTED_FIGHTERS_DATA)
        
        # Load models and data
        winner_model = joblib.load(winner_model_path)
        method_model = joblib.load(method_model_path)
        
        with open(winner_cols_path, 'r') as f:
            winner_cols = json.load(f)
        with open(method_cols_path, 'r') as f:
            method_cols = json.load(f)
            
        fighters_df = pd.read_csv(fighters_data_path)
        
        # Make prediction
        result = predict_fight_symmetrical(
            fighter1, fighter2, fighters_df, 
            winner_cols, method_cols, winner_model, method_model
        )
        
        if 'error' in result:
            print(f"Error: {result['error']}")
        else:
            print(f"\n🏆 Predicted Winner: {result['predicted_winner']} ({result['winner_confidence']})")
            print(f"⚔️  Predicted Method: {result['predicted_method']}")
            print(f"\n📊 Win Probabilities:")
            for fighter, prob in result['win_probabilities'].items():
                print(f"   {fighter}: {prob}")
            print(f"\n🥊 Method Probabilities:")
            for method, prob in result['method_probabilities'].items():
                print(f"   {method}: {prob}")
            
    except Exception as e:
        print(f"Error making prediction: {e}")


def interactive_prediction(winner_model_path: str = None, method_model_path: str = None):
    """Interactive prediction mode with both winner and method."""
    print("=== UFC Predictor: Interactive Mode ===")
    
    try:
        # Auto-detect latest models if paths not provided
        if not winner_model_path or not method_model_path:
            print("🔍 Auto-detecting latest trained models...")
            auto_winner_path, auto_method_path, auto_winner_cols, auto_method_cols, auto_fighters_data = get_latest_trained_models()
            
            winner_model_path = winner_model_path or auto_winner_path
            method_model_path = method_model_path or auto_method_path
            winner_cols_path = auto_winner_cols
            method_cols_path = auto_method_cols
            fighters_data_path = auto_fighters_data
        else:
            # Use standard column files and fighter data
            winner_cols_path = 'model/winner_model_columns.json'
            method_cols_path = 'model/method_model_columns.json'
            fighters_data_path = str(CORRECTED_FIGHTERS_DATA)
            
        # Load models and data
        winner_model = joblib.load(winner_model_path)
        method_model = joblib.load(method_model_path)
        
        with open(winner_cols_path, 'r') as f:
            winner_cols = json.load(f)
        with open(method_cols_path, 'r') as f:
            method_cols = json.load(f)
            
        fighters_df = pd.read_csv(fighters_data_path)
        
        print("Type 'quit' to exit")
        print("Available fighters can be found in the dataset")
        
        while True:
            fighter1 = input("\nEnter first fighter name: ").strip()
            if fighter1.lower() == 'quit':
                break
                
            fighter2 = input("Enter second fighter name: ").strip()
            if fighter2.lower() == 'quit':
                break
            
            # Make prediction
            result = predict_fight_symmetrical(
                fighter1, fighter2, fighters_df, 
                winner_cols, method_cols, winner_model, method_model
            )
            
            if 'error' in result:
                print(f"Error: {result['error']}")
                
                # Show similar fighter names
                available_fighters = fighters_df['Name'].tolist()
                similar1 = [f for f in available_fighters if fighter1.lower() in f.lower()][:5]
                similar2 = [f for f in available_fighters if fighter2.lower() in f.lower()][:5]
                
                if similar1:
                    print(f"Similar to '{fighter1}': {', '.join(similar1)}")
                if similar2:
                    print(f"Similar to '{fighter2}': {', '.join(similar2)}")
            else:
                print(f"\n🏆 Predicted Winner: {result['predicted_winner']} ({result['winner_confidence']})")
                print(f"⚔️  Predicted Method: {result['predicted_method']}")
                print(f"\n📊 Win Probabilities:")
                for fighter, prob in result['win_probabilities'].items():
                    print(f"   {fighter}: {prob}")
                print(f"\n🥊 Method Probabilities:")
                for method, prob in result['method_probabilities'].items():
                    print(f"   {method}: {prob}")
                
    except Exception as e:
        print(f"Error: {e}")

def predict_card(fights_list: list, winner_model_path: str = None, method_model_path: str = None):
    """Predict outcomes for multiple fights (full card mode)."""
    print("=== UFC Predictor: Full Card Mode ===")
    
    try:
        # Auto-detect latest models if paths not provided
        if not winner_model_path or not method_model_path:
            print("🔍 Auto-detecting latest trained models...")
            auto_winner_path, auto_method_path, auto_winner_cols, auto_method_cols, auto_fighters_data = get_latest_trained_models()
            
            winner_model_path = winner_model_path or auto_winner_path
            method_model_path = method_model_path or auto_method_path
            winner_cols_path = auto_winner_cols
            method_cols_path = auto_method_cols
            fighters_data_path = auto_fighters_data
        else:
            # Use standard column files and fighter data
            winner_cols_path = 'model/winner_model_columns.json'
            method_cols_path = 'model/method_model_columns.json'
            fighters_data_path = str(CORRECTED_FIGHTERS_DATA)
            
        # Load models and data
        winner_model = joblib.load(winner_model_path)
        method_model = joblib.load(method_model_path)
        
        with open(winner_cols_path, 'r') as f:
            winner_cols = json.load(f)
        with open(method_cols_path, 'r') as f:
            method_cols = json.load(f)
            
        fighters_df = pd.read_csv(fighters_data_path)
        
        results = {
            "total_fights": len(fights_list),
            "processed_fights": 0,
            "failed_fights": 0,
            "predictions": [],
            "method_distribution": {"Decision": 0, "KO/TKO": 0, "Submission": 0}
        }
        
        print(f"\nProcessing {len(fights_list)} fights...\n")
        
        for i, fight_string in enumerate(fights_list, 1):
            if " vs " in fight_string:
                fighter_a, fighter_b = fight_string.split(" vs ")
                fighter_a, fighter_b = fighter_a.strip(), fighter_b.strip()
            elif " vs. " in fight_string:
                fighter_a, fighter_b = fight_string.split(" vs. ")
                fighter_a, fighter_b = fighter_a.strip(), fighter_b.strip()
            else:
                print(f"❌ Fight {i}: Invalid format '{fight_string}'")
                results["failed_fights"] += 1
                continue
            
            result = predict_fight_symmetrical(
                fighter_a, fighter_b, fighters_df, 
                winner_cols, method_cols, winner_model, method_model
            )
            
            if 'error' in result:
                print(f"❌ Fight {i}: {fight_string} - {result['error']}")
                results["failed_fights"] += 1
            else:
                result["fight_number"] = i
                results["predictions"].append(result)
                results["processed_fights"] += 1
                results["method_distribution"][result["predicted_method"]] += 1
                
                print(f"✅ Fight {i}: {result['predicted_winner']} defeats {fighter_a if result['predicted_winner'] == fighter_b else fighter_b}")
                print(f"   Method: {result['predicted_method']} ({result['winner_confidence']} confidence)")
        
        # Summary
        print(f"\n{'='*60}")
        print(f"CARD PREDICTION SUMMARY")
        print(f"{'='*60}")
        print(f"Total Fights: {results['total_fights']}")
        print(f"Successfully Predicted: {results['processed_fights']}")
        print(f"Failed Predictions: {results['failed_fights']}")
        
        if results['processed_fights'] > 0:
            print(f"\nMethod Distribution:")
            for method, count in results['method_distribution'].items():
                percentage = (count / results['processed_fights']) * 100
                print(f"  {method}: {count} fights ({percentage:.1f}%)")
        
    except Exception as e:
        print(f"Error predicting card: {e}")


if __name__ == "__main__":
    main()