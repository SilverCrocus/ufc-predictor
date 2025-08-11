#!/usr/bin/env python3
"""
Complete Training Pipeline with Automatic Optimization
Trains models and automatically creates optimized versions for production use.
"""

import sys
import os
import json
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Tuple
import logging
from sklearn.model_selection import train_test_split

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.ufc_predictor.models.model_training import UFCModelTrainer
from src.ufc_predictor.models.feature_selection import UFCFeatureSelector
from src.ufc_predictor.data.feature_engineering import prepare_modeling_data

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class CompletePipeline:
    """Complete training pipeline with automatic optimization."""
    
    def __init__(self, base_dir: Path = None):
        """Initialize the pipeline."""
        self.base_dir = base_dir or project_root
        self.model_dir = self.base_dir / 'model'
        self.data_dir = self.base_dir / 'data'
        self.optimized_dir = self.model_dir / 'optimized'
        
        # Create directories if they don't exist
        self.optimized_dir.mkdir(parents=True, exist_ok=True)
        
        # Timestamp for this training run
        self.timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M')
        self.training_dir = self.model_dir / f'training_{self.timestamp}'
        self.training_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Pipeline initialized. Training directory: {self.training_dir}")
    
    def load_and_prepare_data(self) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """Load and prepare data for training."""
        logger.info("Loading and preparing data...")
        
        # Try to load existing processed data first
        processed_path = self.model_dir / 'ufc_fight_dataset_with_diffs.csv'
        if not processed_path.exists():
            # Try alternative paths
            alt_paths = [
                self.model_dir / 'ufc_fight_dataset_with_diffs_corrected.csv',
                self.data_dir / 'ufc_fight_dataset_with_diffs.csv'
            ]
            for path in alt_paths:
                if path.exists():
                    processed_path = path
                    break
        
        if not processed_path.exists():
            # Load raw data and process it
            logger.info("Processing raw fight data...")
            fights_path = self.data_dir / 'ufc_fights.csv'
            if not fights_path.exists():
                raise FileNotFoundError(f"No fight data found at {fights_path}")
            
            fights_df = pd.read_csv(fights_path)
            X, y = prepare_modeling_data(fights_df)
        else:
            # Load processed data
            logger.info(f"Loading processed data from {processed_path}")
            data = pd.read_csv(processed_path)
            
            # Separate features and target
            target_col = 'Winner'
            if target_col not in data.columns:
                target_col = 'winner'  # Try lowercase
            
            feature_cols = [col for col in data.columns 
                          if col not in [target_col, 'Method', 'Round', 'Title_fight']]
            
            X = data[feature_cols]
            y = data[target_col]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Save processed data for reference
        train_data = pd.concat([X_train, y_train], axis=1)
        train_data.to_csv(self.training_dir / f'training_data_{self.timestamp}.csv', index=False)
        
        logger.info(f"Data loaded: {len(X_train)} training samples, {len(X_test)} test samples")
        logger.info(f"Features: {X_train.shape[1]}")
        
        return X_train, X_test, y_train, y_test
    
    def train_models(self, X_train, X_test, y_train, y_test, tune: bool = True) -> Dict:
        """Train all models."""
        logger.info("Training models...")
        
        # Initialize trainer
        trainer = UFCModelTrainer()
        
        # Train Random Forest
        logger.info("Training Random Forest...")
        rf_model = trainer.train_random_forest(X_train, y_train)
        rf_score = trainer.evaluate_model('random_forest', X_test, y_test, show_plots=False)
        
        # Tune Random Forest if requested
        if tune:
            logger.info("Tuning Random Forest hyperparameters...")
            rf_tuned = trainer.tune_random_forest(X_train, y_train)
            rf_tuned_score = trainer.evaluate_model('random_forest_tuned', X_test, y_test, show_plots=False)
        else:
            rf_tuned = rf_model
            rf_tuned_score = rf_score
        
        # Save models
        logger.info("Saving models...")
        
        # Save standard model
        joblib.dump(rf_model, self.training_dir / f'ufc_winner_model_{self.timestamp}.joblib')
        
        # Save tuned model
        joblib.dump(rf_tuned, self.training_dir / f'ufc_winner_model_tuned_{self.timestamp}.joblib')
        
        # Save feature columns
        with open(self.training_dir / f'winner_model_columns_{self.timestamp}.json', 'w') as f:
            json.dump(list(X_train.columns), f)
        
        # Save metadata
        metadata = {
            'training_timestamp': self.timestamp,
            'n_features_original': X_train.shape[1],
            'n_training_samples': len(X_train),
            'n_test_samples': len(X_test),
            'models': {
                'random_forest': {
                    'accuracy': rf_score['accuracy'],
                    'type': 'RandomForest (default params)'
                },
                'random_forest_tuned': {
                    'accuracy': rf_tuned_score['accuracy'],
                    'type': 'RandomForest (GridSearch tuned)'
                }
            }
        }
        
        with open(self.training_dir / f'training_metadata_{self.timestamp}.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Models saved to {self.training_dir}")
        
        return {
            'rf_model': rf_model,
            'rf_tuned': rf_tuned,
            'rf_score': rf_score,
            'rf_tuned_score': rf_tuned_score,
            'metadata': metadata
        }
    
    def optimize_model(self, model, X_train, y_train, X_test, y_test, 
                      n_features: int = 32) -> Dict:
        """Create optimized version of the model with feature selection."""
        logger.info(f"Creating optimized model with {n_features} features...")
        
        # Initialize feature selector
        selector = UFCFeatureSelector()
        
        # Select best features
        selected_features = selector.select_features(
            X_train, y_train, model, 
            n_features=n_features
        )
        
        logger.info(f"Selected {len(selected_features)} features")
        
        # Train optimized model with selected features
        X_train_opt = X_train[selected_features]
        X_test_opt = X_test[selected_features]
        
        # Train new model with selected features
        optimized_trainer = UFCModelTrainer()
        optimized_model = optimized_trainer.train_random_forest(X_train_opt, y_train)
        
        # Evaluate optimized model
        y_pred = optimized_model.predict(X_test_opt)
        from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
        
        accuracy = accuracy_score(y_test, y_pred)
        
        # Calculate AUC if we have probability predictions
        if hasattr(optimized_model, 'predict_proba'):
            y_proba = optimized_model.predict_proba(X_test_opt)[:, 1]
            auc = roc_auc_score(y_test, y_proba)
        else:
            auc = 0.0
        
        f1 = f1_score(y_test, y_pred)
        
        logger.info(f"Optimized model performance:")
        logger.info(f"  - Accuracy: {accuracy:.4f}")
        logger.info(f"  - AUC: {auc:.4f}")
        logger.info(f"  - F1: {f1:.4f}")
        
        # Save optimized model and selector
        timestamp_suffix = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save with timestamp
        joblib.dump(optimized_model, 
                   self.optimized_dir / f'ufc_model_optimized_{timestamp_suffix}.joblib')
        selector.save(str(self.optimized_dir / f'feature_selector_{timestamp_suffix}.json'))
        
        # Also save as 'latest' for easy access
        joblib.dump(optimized_model, 
                   self.optimized_dir / 'ufc_model_optimized_latest.joblib')
        selector.save(str(self.optimized_dir / 'feature_selector_latest.json'))
        
        # Save metrics
        metrics = {
            'accuracy': accuracy,
            'auc': auc,
            'f1': f1,
            'n_features': len(selected_features)
        }
        
        with open(self.optimized_dir / f'metrics_{timestamp_suffix}.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        
        logger.info(f"Optimized model saved to {self.optimized_dir}")
        
        return {
            'model': optimized_model,
            'selector': selector,
            'features': selected_features,
            'metrics': metrics
        }
    
    def run_complete_pipeline(self, tune: bool = True, optimize: bool = True, 
                             n_features: int = 32) -> Dict:
        """Run the complete training and optimization pipeline."""
        logger.info("="*60)
        logger.info("COMPLETE UFC TRAINING PIPELINE WITH AUTO-OPTIMIZATION")
        logger.info("="*60)
        
        try:
            # Step 1: Load and prepare data
            logger.info("\n📊 STEP 1: Data Preparation")
            X_train, X_test, y_train, y_test = self.load_and_prepare_data()
            
            # Step 2: Train models
            logger.info("\n🎯 STEP 2: Model Training")
            training_results = self.train_models(X_train, X_test, y_train, y_test, tune=tune)
            
            # Step 3: Optimize best model (automatic!)
            if optimize:
                logger.info("\n⚡ STEP 3: Automatic Model Optimization")
                
                # Use the tuned model if available, otherwise use standard
                best_model = training_results.get('rf_tuned', training_results['rf_model'])
                
                optimization_results = self.optimize_model(
                    best_model, X_train, y_train, X_test, y_test,
                    n_features=n_features
                )
                
                # Compare performance
                logger.info("\n📈 Performance Comparison:")
                logger.info(f"  Original ({X_train.shape[1]} features): "
                          f"{training_results['rf_tuned_score']['accuracy']:.4f}")
                logger.info(f"  Optimized ({n_features} features): "
                          f"{optimization_results['metrics']['accuracy']:.4f}")
                
                speed_improvement = X_train.shape[1] / n_features
                logger.info(f"  Speed improvement: {speed_improvement:.1f}x faster")
            else:
                optimization_results = None
            
            # Step 4: Create summary report
            logger.info("\n📋 STEP 4: Pipeline Summary")
            
            summary = {
                'timestamp': self.timestamp,
                'training_dir': str(self.training_dir),
                'data': {
                    'n_training': len(X_train),
                    'n_test': len(X_test),
                    'n_features_original': X_train.shape[1]
                },
                'models': {
                    'standard': {
                        'accuracy': training_results['rf_score']['accuracy'],
                        'location': str(self.training_dir / f'ufc_winner_model_{self.timestamp}.joblib')
                    },
                    'tuned': {
                        'accuracy': training_results['rf_tuned_score']['accuracy'],
                        'location': str(self.training_dir / f'ufc_winner_model_tuned_{self.timestamp}.joblib')
                    }
                }
            }
            
            if optimization_results:
                summary['optimized'] = {
                    'accuracy': optimization_results['metrics']['accuracy'],
                    'n_features': len(optimization_results['features']),
                    'speed_gain': f"{speed_improvement:.1f}x",
                    'location': str(self.optimized_dir / 'ufc_model_optimized_latest.joblib')
                }
            
            # Save summary
            with open(self.model_dir / 'latest_training_summary.json', 'w') as f:
                json.dump(summary, f, indent=2)
            
            logger.info("\n✅ PIPELINE COMPLETE!")
            logger.info(f"  Training directory: {self.training_dir}")
            if optimize:
                logger.info(f"  Optimized model: {self.optimized_dir}/ufc_model_optimized_latest.joblib")
                logger.info(f"  Ready for production use!")
            
            return summary
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise


def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Complete UFC Training Pipeline')
    parser.add_argument('--tune', action='store_true', help='Tune hyperparameters')
    parser.add_argument('--no-optimize', action='store_true', help='Skip optimization step')
    parser.add_argument('--n-features', type=int, default=32, 
                       help='Number of features for optimized model (default: 32)')
    
    args = parser.parse_args()
    
    # Run pipeline
    pipeline = CompletePipeline()
    results = pipeline.run_complete_pipeline(
        tune=args.tune,
        optimize=not args.no_optimize,
        n_features=args.n_features
    )
    
    # Print final summary
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()