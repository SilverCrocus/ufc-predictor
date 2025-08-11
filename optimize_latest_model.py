#!/usr/bin/env python3
"""
Quick script to optimize the latest trained model.
Run this after training to create an optimized version automatically.
"""

import sys
import os
import json
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.ufc_predictor.models.feature_selection import UFCFeatureSelector
from src.ufc_predictor.models.model_training import UFCModelTrainer


def optimize_latest_model(n_features: int = 32):
    """Optimize the latest trained model."""
    
    print("🔧 OPTIMIZING LATEST TRAINED MODEL")
    print("=" * 60)
    
    # Find latest training directory
    model_dir = Path('model')
    training_dirs = sorted([d for d in model_dir.iterdir() 
                          if d.is_dir() and d.name.startswith('training_')])
    
    if not training_dirs:
        print("❌ No training directories found!")
        print("   Run 'python main.py pipeline' first to train models.")
        return
    
    latest_dir = training_dirs[-1]
    print(f"📁 Using latest training: {latest_dir.name}")
    
    # Load the tuned model
    tuned_model_files = list(latest_dir.glob('*winner_model_tuned*.joblib'))
    if not tuned_model_files:
        print("❌ No tuned model found in latest training!")
        return
    
    model_path = tuned_model_files[0]
    print(f"📊 Loading model: {model_path.name}")
    model = joblib.load(model_path)
    
    # Load the training data
    data_files = list(latest_dir.glob('*fight_dataset*.csv'))
    if not data_files:
        # Try to load from standard location
        data_path = Path('model/ufc_fight_dataset_with_diffs.csv')
        if not data_path.exists():
            print("❌ Training data not found!")
            return
    else:
        data_path = data_files[0]
    
    print(f"📊 Loading data: {data_path.name}")
    data = pd.read_csv(data_path)
    
    # Prepare features and target
    # Identify target column (could be Winner or Outcome)
    target_col = None
    for col in ['Winner', 'Outcome', 'winner', 'outcome']:
        if col in data.columns:
            target_col = col
            break
    
    if target_col is None:
        print("❌ No target column found in data!")
        print(f"   Available columns: {list(data.columns[:10])}")
        return
    
    # Create binary target (1 for Blue wins, 0 for Red wins)
    if target_col == 'Outcome':
        # Outcome column has 'Win' or 'Loss' - need to convert
        y = (data[target_col] == 'Win').astype(int)
    else:
        y = data[target_col]
    
    # Identify feature columns - exclude non-numeric and metadata columns
    exclude_cols = [target_col, 'Method', 'Round', 'Title_fight', 'Fighter', 'Opponent', 
                   'Event', 'Time', 'fighter_url', 'opponent_url']
    
    # Also exclude any column with 'url' in it
    exclude_cols.extend([col for col in data.columns if 'url' in col.lower()])
    
    feature_cols = [col for col in data.columns if col not in exclude_cols]
    
    # Filter to only include diff columns and numeric blue_/red_ features
    feature_cols = [col for col in feature_cols 
                   if ('diff' in col.lower() or col.startswith(('blue_', 'red_'))) 
                   and not 'url' in col.lower()]
    
    X = data[feature_cols]
    
    # Make sure all columns are numeric
    X = X.select_dtypes(include=[np.number])
    
    # Split data (using last 20% as test)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"📊 Data shape: {X.shape}")
    print(f"   Training: {X_train.shape}")
    print(f"   Test: {X_test.shape}")
    
    # Run optimization
    print(f"\n⚡ Optimizing to {n_features} features...")
    
    # Initialize feature selector with importance-based method
    # Note: This will use the model's feature importances internally
    selector = UFCFeatureSelector(
        method='importance_based',
        n_features=n_features
    )
    
    # Fit the selector to find best features (it will train its own RF internally)
    selector.fit(X_train, y_train)
    selected_features = selector.selected_features
    
    print(f"✅ Selected {len(selected_features)} features")
    
    # Train optimized model with selected features
    X_train_opt = X_train[selected_features]
    X_test_opt = X_test[selected_features]
    
    # Train new model with selected features
    trainer = UFCModelTrainer()
    optimized_model = trainer.train_random_forest(X_train_opt, y_train)
    
    # Evaluate optimized model
    y_pred = optimized_model.predict(X_test_opt)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Calculate AUC if we have probability predictions
    if hasattr(optimized_model, 'predict_proba'):
        try:
            y_proba = optimized_model.predict_proba(X_test_opt)
            if y_proba.ndim > 1 and y_proba.shape[1] > 1:
                y_proba = y_proba[:, 1]
            else:
                y_proba = y_proba.ravel()
            auc = roc_auc_score(y_test, y_proba)
        except Exception as e:
            print(f"⚠️  Could not calculate AUC: {e}")
            auc = 0.0
    else:
        auc = 0.0
    
    f1 = f1_score(y_test, y_pred)
    
    # Save optimized model and selector
    optimized_dir = Path('model/optimized')
    optimized_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp_suffix = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save with timestamp
    joblib.dump(optimized_model, 
               optimized_dir / f'ufc_model_optimized_{timestamp_suffix}.joblib')
    selector.save(str(optimized_dir / f'feature_selector_{timestamp_suffix}.json'))
    
    # Also save as 'latest' for easy access
    joblib.dump(optimized_model, 
               optimized_dir / 'ufc_model_optimized_latest.joblib')
    selector.save(str(optimized_dir / 'feature_selector_latest.json'))
    
    # Save metrics
    metrics = {
        'accuracy': accuracy,
        'auc': auc,
        'f1': f1,
        'n_features': len(selected_features)
    }
    
    with open(optimized_dir / f'metrics_{timestamp_suffix}.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    optimization_results = {'metrics': metrics}
    
    # Load original model accuracy from metadata
    metadata_files = list(latest_dir.glob('*metadata*.json'))
    if metadata_files:
        with open(metadata_files[0], 'r') as f:
            metadata = json.load(f)
        original_accuracy = metadata.get('winner_models', {}).get('random_forest_tuned', {}).get('accuracy', 0)
    else:
        original_accuracy = 0
    
    # Display results
    print("\n" + "=" * 60)
    print("✅ OPTIMIZATION COMPLETE!")
    print("=" * 60)
    print(f"📊 Original model ({X.shape[1]} features): {original_accuracy:.2%}")
    print(f"⚡ Optimized model ({n_features} features): {optimization_results['metrics']['accuracy']:.2%}")
    print(f"🚀 Speed improvement: {X.shape[1] / n_features:.1f}x faster")
    print(f"\n📁 Optimized model saved to:")
    print(f"   model/optimized/ufc_model_optimized_latest.joblib")
    print(f"   model/optimized/feature_selector_latest.json")
    print(f"\n🎯 Ready for production use in betting notebook!")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Optimize latest UFC model')
    parser.add_argument('--n-features', type=int, default=32,
                       help='Number of features to select (default: 32)')
    
    args = parser.parse_args()
    
    optimize_latest_model(n_features=args.n_features)