#!/usr/bin/env python3
"""
Stacking Ensemble Demonstration for UFC Predictions
=================================================

This script demonstrates the complete stacking ensemble implementation,
showing how to train base models, create out-of-fold predictions,
train meta-learners, and generate production-ready predictions with
confidence intervals.

Usage:
    python examples/stacking_ensemble_demo.py [--real-data]

Features demonstrated:
1. Base model training (RF, XGBoost, Logistic Regression)
2. Out-of-fold prediction generation with temporal validation
3. Meta-learner selection and optimization
4. Stacking ensemble predictions with confidence intervals
5. Performance comparison with base models
6. Model persistence and loading
7. Production deployment considerations
"""

import sys
import argparse
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.ufc_predictor.models.model_training import UFCModelTrainer, train_complete_pipeline
from src.ufc_predictor.models.stacking_ensemble import (
    ProductionStackingManager, 
    create_stacking_config,
    StackingResult
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_synthetic_ufc_data(n_samples: int = 1000, n_features: int = 30) -> tuple:
    """
    Create synthetic UFC-like data for demonstration
    
    Returns:
        Tuple of (X, y, fighter_pairs, feature_names)
    """
    logger.info(f"Creating synthetic UFC data: {n_samples} samples, {n_features} features")
    
    # Create base classification data
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=int(n_features * 0.7),
        n_redundant=int(n_features * 0.2),
        n_clusters_per_class=1,
        flip_y=0.05,  # Small amount of noise
        class_sep=0.8,
        random_state=42
    )
    
    # Create UFC-like feature names
    feature_names = []
    categories = [
        'striking', 'grappling', 'physical', 'experience', 
        'recent_form', 'style_matchup', 'cardio', 'mental'
    ]
    
    for i in range(n_features):
        category = categories[i % len(categories)]
        feature_names.append(f'{category}_{i//len(categories)+1}_diff')
    
    X = pd.DataFrame(X, columns=feature_names)
    y = pd.Series(y, name='blue_wins')
    
    # Create fighter pairs
    fighter_pairs = []
    for i in range(len(X)):
        red_fighter = f"Red_Fighter_{i % 100}"
        blue_fighter = f"Blue_Fighter_{(i + 50) % 100}"
        fighter_pairs.append((red_fighter, blue_fighter))
    
    return X, y, fighter_pairs, feature_names


def load_real_ufc_data() -> tuple:
    """
    Load real UFC data if available
    
    Returns:
        Tuple of (X, y, fighter_pairs, feature_names) or None if not available
    """
    try:
        # Try to load from standard locations
        data_paths = [
            project_root / "model" / "ufc_fight_dataset_with_diffs.csv",
            project_root / "data" / "ufc_fight_dataset_with_diffs.csv"
        ]
        
        for data_path in data_paths:
            if data_path.exists():
                logger.info(f"Loading real UFC data from {data_path}")
                df = pd.read_csv(data_path)
                
                # Extract features and target
                exclude_cols = [
                    'Outcome', 'Fighter', 'Opponent', 'Event', 'Method', 'Time',
                    'fighter_url', 'opponent_url', 'blue_fighter_url', 'red_fighter_url',
                    'blue_Name', 'red_Name', 'blue_is_winner'
                ]
                
                feature_cols = [col for col in df.columns if col not in exclude_cols]
                X = df[feature_cols]
                y = df['blue_is_winner'] if 'blue_is_winner' in df.columns else df['Outcome']
                
                # Create fighter pairs
                if 'blue_Name' in df.columns and 'red_Name' in df.columns:
                    fighter_pairs = list(zip(df['red_Name'], df['blue_Name']))
                else:
                    fighter_pairs = [(f"Fighter_A_{i}", f"Fighter_B_{i}") for i in range(len(X))]
                
                return X, y, fighter_pairs, feature_cols
        
        logger.warning("Real UFC data not found, using synthetic data")
        return None
        
    except Exception as e:
        logger.error(f"Error loading real UFC data: {e}")
        return None


def demonstrate_base_model_training(X_train: pd.DataFrame, y_train: pd.Series,
                                  X_test: pd.DataFrame, y_test: pd.Series) -> UFCModelTrainer:
    """Demonstrate base model training"""
    
    print("\n" + "="*80)
    print("📚 PHASE 1: BASE MODEL TRAINING")
    print("="*80)
    
    trainer = UFCModelTrainer()
    
    # Train individual models
    print("\n1. Training Logistic Regression...")
    trainer.train_logistic_regression(X_train, y_train)
    lr_results = trainer.evaluate_model('logistic_regression', X_test, y_test, show_plots=False)
    
    print("\n2. Training Random Forest...")
    trainer.train_random_forest(X_train, y_train, n_estimators=100)
    rf_results = trainer.evaluate_model('random_forest', X_test, y_test, show_plots=False)
    
    print("\n3. Training XGBoost...")
    trainer.train_xgboost(X_train, y_train)
    xgb_results = trainer.evaluate_model('xgboost', X_test, y_test, show_plots=False)
    
    # Hyperparameter tuning
    print("\n4. Hyperparameter Tuning...")
    trainer.tune_random_forest(X_train, y_train, n_iter=20, cv=3)
    rf_tuned_results = trainer.evaluate_model('random_forest_tuned', X_test, y_test, show_plots=False)
    
    trainer.tune_xgboost(X_train, y_train, n_iter=20, cv=3)
    xgb_tuned_results = trainer.evaluate_model('xgboost_tuned', X_test, y_test, show_plots=False)
    
    # Store test data
    trainer.X_test = X_test
    trainer.y_test = y_test
    trainer.feature_columns = X_train.columns.tolist()
    
    print("\n📊 Base Model Performance Summary:")
    print("-" * 50)
    print(f"Logistic Regression: {lr_results['accuracy']:.4f}")
    print(f"Random Forest:       {rf_results['accuracy']:.4f}")
    print(f"XGBoost:            {xgb_results['accuracy']:.4f}")
    print(f"RF Tuned:           {rf_tuned_results['accuracy']:.4f}")
    print(f"XGBoost Tuned:      {xgb_tuned_results['accuracy']:.4f}")
    
    return trainer


def demonstrate_stacking_ensemble(trainer: UFCModelTrainer, 
                                X_train: pd.DataFrame, y_train: pd.Series,
                                X_test: pd.DataFrame, y_test: pd.Series,
                                fighter_pairs_test: list) -> dict:
    """Demonstrate stacking ensemble training and prediction"""
    
    print("\n" + "="*80)
    print("🎯 PHASE 2: STACKING ENSEMBLE")
    print("="*80)
    
    # Enable and configure stacking
    print("\n1. Configuring Stacking Ensemble...")
    
    # Adjust CV splits based on dataset size
    cv_splits = min(5, len(X_train) // 20)  # At least 20 samples per fold
    cv_splits = max(3, cv_splits)  # At least 3 splits
    
    print(f"   Using {cv_splits} CV splits for {len(X_train)} training samples")
    
    trainer.enable_stacking_ensemble(
        cv_splits=cv_splits,
        temporal_validation=True,
        enable_optimization=True
    )
    
    # Train stacking ensemble
    print("\n2. Training Stacking Ensemble...")
    print("   - Generating out-of-fold predictions...")
    print("   - Selecting optimal meta-learner...")
    print("   - Training meta-learner...")
    
    trainer.train_stacking_ensemble(X_train, y_train)
    
    # Evaluate stacking performance
    print("\n3. Evaluating Stacking Performance...")
    stacking_scores = trainer.evaluate_stacking_ensemble(X_test, y_test)
    
    # Get detailed predictions
    print("\n4. Generating Detailed Predictions...")
    prediction_results = trainer.predict_with_stacking(X_test, fighter_pairs_test)
    
    return {
        'trainer': trainer,
        'scores': stacking_scores,
        'predictions': prediction_results
    }


def demonstrate_confidence_intervals(stacking_results: list):
    """Demonstrate confidence interval analysis"""
    
    print("\n" + "="*80)
    print("📊 PHASE 3: CONFIDENCE INTERVAL ANALYSIS")
    print("="*80)
    
    # Extract confidence intervals
    probabilities = [r.stacked_probability for r in stacking_results]
    ci_lower = [r.confidence_interval[0] for r in stacking_results]
    ci_upper = [r.confidence_interval[1] for r in stacking_results]
    uncertainties = [r.confidence_interval[1] - r.confidence_interval[0] for r in stacking_results]
    
    print(f"\n📈 Confidence Interval Statistics:")
    print("-" * 40)
    print(f"Mean prediction:     {np.mean(probabilities):.3f}")
    print(f"Mean CI width:       {np.mean(uncertainties):.3f}")
    print(f"Max CI width:        {np.max(uncertainties):.3f}")
    print(f"Min CI width:        {np.min(uncertainties):.3f}")
    
    # High and low confidence predictions
    high_conf_idx = np.argsort(uncertainties)[:5]
    low_conf_idx = np.argsort(uncertainties)[-5:]
    
    print(f"\n🎯 Highest Confidence Predictions:")
    for i in high_conf_idx:
        result = stacking_results[i]
        print(f"  Prediction: {result.stacked_probability:.3f}, "
              f"CI: [{result.confidence_interval[0]:.3f}, {result.confidence_interval[1]:.3f}], "
              f"Width: {uncertainties[i]:.3f}")
    
    print(f"\n❓ Lowest Confidence Predictions:")
    for i in low_conf_idx:
        result = stacking_results[i]
        print(f"  Prediction: {result.stacked_probability:.3f}, "
              f"CI: [{result.confidence_interval[0]:.3f}, {result.confidence_interval[1]:.3f}], "
              f"Width: {uncertainties[i]:.3f}")


def demonstrate_model_persistence(trainer: UFCModelTrainer):
    """Demonstrate model saving and loading"""
    
    print("\n" + "="*80)
    print("💾 PHASE 4: MODEL PERSISTENCE")
    print("="*80)
    
    # Create temporary directory for demo
    temp_dir = Path("/tmp/ufc_stacking_demo")
    temp_dir.mkdir(exist_ok=True)
    
    # Save stacking ensemble
    stacking_path = temp_dir / "stacking_ensemble.joblib"
    print(f"\n1. Saving stacking ensemble to {stacking_path}...")
    trainer.save_stacking_ensemble(str(stacking_path))
    
    # Save base models
    base_model_paths = {}
    for model_name, model in trainer.models.items():
        model_path = temp_dir / f"{model_name}.joblib"
        trainer.save_model(model_name, str(model_path))
        base_model_paths[model_name] = str(model_path)
    
    print(f"2. Base models saved: {list(base_model_paths.keys())}")
    
    # Create new trainer and load everything
    print(f"\n3. Creating new trainer and loading models...")
    new_trainer = UFCModelTrainer()
    
    # Load base models
    for model_name, model_path in base_model_paths.items():
        model = UFCModelTrainer.load_model(model_path)
        new_trainer.models[model_name] = model
    
    # Load stacking ensemble
    new_trainer.load_stacking_ensemble(str(stacking_path))
    
    print(f"✅ All models loaded successfully!")
    print(f"   Stacking enabled: {new_trainer.enable_stacking}")
    print(f"   Base models: {list(new_trainer.models.keys())}")
    
    # Cleanup
    import shutil
    shutil.rmtree(temp_dir)
    print(f"🧹 Cleanup completed")
    
    return new_trainer


def create_performance_comparison_chart(base_scores: dict, stacking_scores: dict):
    """Create performance comparison visualization"""
    
    print("\n" + "="*80)
    print("📊 PERFORMANCE VISUALIZATION")
    print("="*80)
    
    # Extract metrics
    models = []
    accuracies = []
    aucs = []
    
    # Base model scores
    for model_name in ['logistic_regression', 'random_forest', 'xgboost']:
        if f'{model_name}_accuracy' in stacking_scores:
            models.append(model_name.replace('_', ' ').title())
            accuracies.append(stacking_scores[f'{model_name}_accuracy'])
            aucs.append(stacking_scores.get(f'{model_name}_auc', 0.5))
    
    # Add stacking results
    models.append('Stacking Ensemble')
    accuracies.append(stacking_scores['stacking_accuracy'])
    aucs.append(stacking_scores['stacking_auc'])
    
    # Create comparison DataFrame
    comparison_df = pd.DataFrame({
        'Model': models,
        'Accuracy': accuracies,
        'AUC': aucs
    })
    
    print("\n📋 Performance Comparison Table:")
    print(comparison_df.to_string(index=False, float_format='%.4f'))
    
    # Create visualization if matplotlib available
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Accuracy comparison
        bars1 = ax1.bar(models, accuracies, color=['skyblue', 'lightgreen', 'orange', 'red'])
        ax1.set_title('Model Accuracy Comparison')
        ax1.set_ylabel('Accuracy')
        ax1.set_ylim(0, 1)
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, acc in zip(bars1, accuracies):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{acc:.3f}', ha='center', va='bottom')
        
        # AUC comparison
        bars2 = ax2.bar(models, aucs, color=['skyblue', 'lightgreen', 'orange', 'red'])
        ax2.set_title('Model AUC Comparison')
        ax2.set_ylabel('AUC')
        ax2.set_ylim(0, 1)
        ax2.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, auc in zip(bars2, aucs):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{auc:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        chart_path = "/tmp/stacking_performance_comparison.png"
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        print(f"\n📊 Performance chart saved to: {chart_path}")
        
        # Don't show plot in non-interactive environments
        # plt.show()
        plt.close()
        
    except Exception as e:
        logger.warning(f"Could not create visualization: {e}")


def main():
    """Main demonstration function"""
    
    parser = argparse.ArgumentParser(description='UFC Stacking Ensemble Demonstration')
    parser.add_argument('--real-data', action='store_true', 
                       help='Use real UFC data if available')
    parser.add_argument('--n-samples', type=int, default=1000,
                       help='Number of synthetic samples to generate')
    
    args = parser.parse_args()
    
    print("🥊 UFC STACKING ENSEMBLE DEMONSTRATION")
    print("=" * 80)
    print("This demo showcases advanced stacking ensemble methods for UFC predictions")
    print("=" * 80)
    
    # Load or create data
    if args.real_data:
        data_result = load_real_ufc_data()
        if data_result:
            X, y, fighter_pairs, feature_names = data_result
        else:
            X, y, fighter_pairs, feature_names = create_synthetic_ufc_data(args.n_samples)
    else:
        X, y, fighter_pairs, feature_names = create_synthetic_ufc_data(args.n_samples)
    
    print(f"\n📊 Dataset Information:")
    print(f"   Samples: {len(X)}")
    print(f"   Features: {len(X.columns)}")
    print(f"   Class distribution: {y.value_counts().to_dict()}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    train_pairs = fighter_pairs[:len(X_train)]
    test_pairs = fighter_pairs[len(X_train):]
    
    print(f"   Training samples: {len(X_train)}")
    print(f"   Test samples: {len(X_test)}")
    
    try:
        # Phase 1: Base model training
        trainer = demonstrate_base_model_training(X_train, y_train, X_test, y_test)
        
        # Phase 2: Stacking ensemble
        stacking_demo = demonstrate_stacking_ensemble(
            trainer, X_train, y_train, X_test, y_test, test_pairs
        )
        
        # Phase 3: Confidence interval analysis
        stacking_results = stacking_demo['predictions']['stacking_results']
        demonstrate_confidence_intervals(stacking_results)
        
        # Phase 4: Model persistence
        demonstrate_model_persistence(trainer)
        
        # Phase 5: Performance visualization
        create_performance_comparison_chart({}, stacking_demo['scores'])
        
        # Final summary
        print("\n" + "="*80)
        print("✅ DEMONSTRATION COMPLETE")
        print("="*80)
        
        scores = stacking_demo['scores']
        print(f"🏆 Final Results:")
        print(f"   Meta-learner: {scores['meta_learner']}")
        print(f"   Stacking Test Accuracy: {scores['stacking_accuracy']:.4f}")
        print(f"   Stacking Test AUC: {scores['stacking_auc']:.4f}")
        print(f"   OOF Validation AUC: {scores['oof_validation_auc']:.4f}")
        print(f"   Predictions generated: {len(stacking_results)}")
        
        # Best individual model comparison
        best_base_acc = max([v for k, v in scores.items() if k.endswith('_accuracy') and 'stacking' not in k])
        improvement = scores['stacking_accuracy'] - best_base_acc
        print(f"   Improvement over best base model: {improvement:.4f} ({improvement*100:+.2f}%)")
        
        print(f"\n🚀 Stacking ensemble ready for production deployment!")
        
    except Exception as e:
        logger.error(f"Demonstration failed: {e}")
        raise


if __name__ == "__main__":
    main()