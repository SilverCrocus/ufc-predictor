#!/usr/bin/env python3
"""
Train and validate an optimized UFC prediction model with feature selection.
Compares performance between full and reduced feature sets.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import logging
import joblib
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, classification_report
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, 'src')

from ufc_predictor.models.feature_selection import UFCFeatureSelector, create_optimized_feature_selector

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def load_data():
    """Load the UFC fight dataset."""
    # Try different paths
    paths = [
        'model/ufc_fight_dataset_with_diffs.csv',
        'data/processed/ufc_fight_dataset_with_diffs.csv',
        'data/ufc_fight_dataset_with_diffs.csv'
    ]
    
    for path in paths:
        if Path(path).exists():
            logger.info(f"Loading data from: {path}")
            return pd.read_csv(path)
    
    # Try loading from latest training directory
    model_dir = Path('model')
    training_dirs = [d for d in model_dir.iterdir() 
                    if d.is_dir() and d.name.startswith('training_')]
    
    if training_dirs:
        latest_dir = max(training_dirs, key=lambda x: x.stat().st_mtime)
        data_files = list(latest_dir.glob('ufc_fight_dataset_with_diffs*.csv'))
        if data_files:
            logger.info(f"Loading data from: {data_files[0]}")
            return pd.read_csv(data_files[0])
    
    raise FileNotFoundError("No dataset found")


def prepare_data(df):
    """Prepare features and target from dataset."""
    # Get all numeric columns first
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    
    # Columns to exclude from features (but keep Round as it might be useful)
    exclude_cols = ['Date', 'date', 'Winner', 'winner', 'Outcome', 'outcome',
                   'blue_fighter', 'red_fighter', 'Fighter', 'Opponent',
                   'loser_fighter', 'Event', 'event', 'Method', 'method', 
                   'Time', 'time', 'fighter_a', 'fighter_b',
                   'fighter_url', 'opponent_url', 'blue_fighter_url', 'red_fighter_url',
                   'blue_Name', 'red_Name']
    
    # Get feature columns - all numeric columns not in exclude list
    feature_cols = [col for col in numeric_cols if col not in exclude_cols]
    
    if not feature_cols:
        raise ValueError(f"No feature columns found. Numeric cols: {numeric_cols[:10]}")
    
    X = df[feature_cols]
    
    # Get target - convert Outcome to binary
    if 'Winner' in df.columns:
        y = df['Winner']
    elif 'Outcome' in df.columns:
        # win = 1 (fighter won), loss = 0 (fighter lost), exclude nc/draw
        y = (df['Outcome'] == 'win').astype(int)
        # Filter out nc (no contest) and draw
        valid_mask = df['Outcome'].isin(['win', 'loss'])
        X = X[valid_mask]
        y = y[valid_mask]
        df = df[valid_mask]  # Keep df in sync for other uses
    else:
        raise ValueError("No target column found")
    
    return X, y, feature_cols


def train_full_model(X_train, y_train, X_test, y_test, tune=False):
    """Train model with all features."""
    logger.info("\n" + "="*50)
    logger.info("TRAINING WITH ALL FEATURES")
    logger.info("="*50)
    logger.info(f"Features: {X_train.shape[1]}")
    logger.info(f"Training samples: {X_train.shape[0]}")
    
    if tune:
        logger.info("Performing hyperparameter tuning...")
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }
        
        model = RandomForestClassifier(random_state=42)
        grid_search = GridSearchCV(
            model, param_grid, cv=5, 
            scoring='accuracy', n_jobs=-1, verbose=1
        )
        grid_search.fit(X_train, y_train)
        model = grid_search.best_estimator_
        logger.info(f"Best params: {grid_search.best_params_}")
    else:
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_split=2,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    f1 = f1_score(y_test, y_pred)
    
    logger.info(f"\nResults:")
    logger.info(f"  Accuracy: {accuracy:.4f}")
    logger.info(f"  AUC: {auc:.4f}")
    logger.info(f"  F1: {f1:.4f}")
    
    return model, {
        'accuracy': accuracy,
        'auc': auc,
        'f1': f1,
        'n_features': X_train.shape[1]
    }


def train_optimized_model(X_train, y_train, X_test, y_test, selector, tune=False):
    """Train model with selected features."""
    logger.info("\n" + "="*50)
    logger.info("TRAINING WITH SELECTED FEATURES")
    logger.info("="*50)
    
    # Transform data
    X_train_selected = selector.transform(X_train)
    X_test_selected = selector.transform(X_test)
    
    logger.info(f"Features: {X_train_selected.shape[1]} (reduced from {X_train.shape[1]})")
    logger.info(f"Training samples: {X_train_selected.shape[0]}")
    logger.info(f"Feature reduction: {(1 - X_train_selected.shape[1]/X_train.shape[1])*100:.1f}%")
    
    if tune:
        logger.info("Performing hyperparameter tuning...")
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }
        
        model = RandomForestClassifier(random_state=42)
        grid_search = GridSearchCV(
            model, param_grid, cv=5, 
            scoring='accuracy', n_jobs=-1, verbose=1
        )
        grid_search.fit(X_train_selected, y_train)
        model = grid_search.best_estimator_
        logger.info(f"Best params: {grid_search.best_params_}")
    else:
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_split=2,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train_selected, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test_selected)
    y_proba = model.predict_proba(X_test_selected)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    f1 = f1_score(y_test, y_pred)
    
    logger.info(f"\nResults:")
    logger.info(f"  Accuracy: {accuracy:.4f}")
    logger.info(f"  AUC: {auc:.4f}")
    logger.info(f"  F1: {f1:.4f}")
    
    return model, {
        'accuracy': accuracy,
        'auc': auc,
        'f1': f1,
        'n_features': X_train_selected.shape[1]
    }


def save_optimized_model(model, selector, metrics, output_dir='model/optimized'):
    """Save the optimized model and selector."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save model
    model_path = output_path / f'ufc_model_optimized_{timestamp}.joblib'
    joblib.dump(model, model_path)
    logger.info(f"Model saved to: {model_path}")
    
    # Save selector
    selector_path = output_path / f'feature_selector_{timestamp}.json'
    selector.save(selector_path)
    logger.info(f"Selector saved to: {selector_path}")
    
    # Save metrics
    metrics_path = output_path / f'metrics_{timestamp}.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Metrics saved to: {metrics_path}")
    
    # Create a "latest" symlink or copy
    latest_model = output_path / 'ufc_model_optimized_latest.joblib'
    latest_selector = output_path / 'feature_selector_latest.json'
    
    # Copy files as "latest"
    import shutil
    shutil.copy(model_path, latest_model)
    shutil.copy(selector_path, latest_selector)
    
    return model_path, selector_path


def main():
    """Main training pipeline."""
    print("\n" + "="*70)
    print("OPTIMIZED MODEL TRAINING WITH FEATURE SELECTION")
    print("="*70)
    
    # Configuration
    n_features = 32  # Top features to select
    test_size = 0.2
    tune = False  # Set to True for hyperparameter tuning
    
    # Load data
    print("\n1. Loading data...")
    df = load_data()
    print(f"✓ Loaded {len(df)} samples")
    
    # Prepare features
    print("\n2. Preparing features...")
    X, y, feature_cols = prepare_data(df)
    print(f"✓ Prepared {len(feature_cols)} features")
    
    # Split data
    print("\n3. Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    print(f"✓ Train: {len(X_train)} samples")
    print(f"✓ Test: {len(X_test)} samples")
    
    # Create feature selector
    print(f"\n4. Creating feature selector (top {n_features} features)...")
    selector = UFCFeatureSelector(
        method='importance_based',
        n_features=n_features,
        importance_file='artifacts/feature_importance/feature_importance.csv'
    )
    
    # Fit selector on training data only (prevent data leakage)
    selector.fit(X_train, y_train)
    print(f"✓ Selected {len(selector.selected_features)} features")
    
    # Show selected features
    print("\n📋 SELECTED FEATURES:")
    importance_summary = selector.get_feature_importance_summary()
    if not importance_summary.empty:
        for i, row in importance_summary.head(10).iterrows():
            if 'importance_pct' in row:
                print(f"   {row['feature']}: {row['importance_pct']:.2f}%")
            else:
                print(f"   {row['feature']}")
    
    # Train full model
    print("\n5. Training models for comparison...")
    model_full, metrics_full = train_full_model(X_train, y_train, X_test, y_test, tune=tune)
    
    # Train optimized model
    model_optimized, metrics_optimized = train_optimized_model(
        X_train, y_train, X_test, y_test, selector, tune=tune
    )
    
    # Compare results
    print("\n" + "="*70)
    print("PERFORMANCE COMPARISON")
    print("="*70)
    
    print(f"\n{'Metric':<15} {'All Features':<15} {'Selected':<15} {'Difference':<15}")
    print("-" * 60)
    
    metrics = ['accuracy', 'auc', 'f1', 'n_features']
    for metric in metrics:
        all_val = metrics_full[metric]
        sel_val = metrics_optimized[metric]
        
        if metric == 'n_features':
            diff = all_val - sel_val
            print(f"{metric:<15} {all_val:<15d} {sel_val:<15d} {diff:>10d} ({(1-sel_val/all_val)*100:.1f}% reduction)")
        else:
            diff = sel_val - all_val
            print(f"{metric:<15} {all_val:<15.4f} {sel_val:<15.4f} {diff:>+10.4f} ({diff*100:+.2f}%)")
    
    # Analysis
    accuracy_loss = metrics_full['accuracy'] - metrics_optimized['accuracy']
    feature_reduction = (1 - metrics_optimized['n_features'] / metrics_full['n_features']) * 100
    
    print("\n" + "="*70)
    print("ANALYSIS SUMMARY")
    print("="*70)
    
    print(f"\n✅ Feature Reduction: {feature_reduction:.1f}%")
    print(f"✅ Accuracy Impact: {accuracy_loss*100:.2f}% loss")
    print(f"✅ Training Speed: ~{metrics_full['n_features']/metrics_optimized['n_features']:.1f}x faster")
    print(f"✅ Memory Usage: {feature_reduction:.1f}% less")
    
    if accuracy_loss < 0.01:  # Less than 1% accuracy loss
        print("\n🎯 RECOMMENDATION: Use optimized model")
        print("   Minimal accuracy loss with significant efficiency gains")
    elif accuracy_loss < 0.02:  # Less than 2% accuracy loss
        print("\n⚖️  RECOMMENDATION: Consider optimized model")
        print("   Small accuracy trade-off for efficiency")
    else:
        print("\n⚠️  RECOMMENDATION: Use full model")
        print("   Accuracy loss may be too significant")
    
    # Save optimized model
    if accuracy_loss < 0.02:  # Save if acceptable performance
        print("\n6. Saving optimized model...")
        model_path, selector_path = save_optimized_model(
            model_optimized, selector, metrics_optimized
        )
        print(f"\n✅ Optimized model saved successfully!")
        print(f"   Use for predictions: model/optimized/ufc_model_optimized_latest.joblib")
        print(f"   Feature selector: model/optimized/feature_selector_latest.json")
    
    # Final recommendations
    print("\n" + "="*70)
    print("NEXT STEPS")
    print("="*70)
    print("\n1. Test optimized model on upcoming fights")
    print("2. Run temporal backtest with optimized features")
    print("3. Monitor performance in production")
    print("4. Consider further optimization if needed")
    
    return selector, model_optimized, metrics_optimized


if __name__ == "__main__":
    main()