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

# Import ELO system for automatic updates
try:
    from src.ufc_predictor.utils.ufc_elo_system import UFCELOSystem
    from src.ufc_predictor.utils.elo_historical_processor import UFCELOHistoricalProcessor
    ELO_AVAILABLE = True
except ImportError:
    ELO_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class CompletePipeline:
    """Complete training pipeline with automatic optimization."""
    
    def __init__(self, base_dir: Path = None, use_temporal_split: bool = True, 
                 production_mode: bool = False):
        """Initialize the pipeline.
        
        Args:
            base_dir: Base directory for the project
            use_temporal_split: Whether to use temporal splitting for evaluation
            production_mode: If True, trains on all data (no holdout for testing)
        """
        self.base_dir = base_dir or project_root
        self.model_dir = self.base_dir / 'model'
        self.data_dir = self.base_dir / 'data'
        self.optimized_dir = self.model_dir / 'optimized'
        self.use_temporal_split = use_temporal_split
        self.production_mode = production_mode
        
        # Create directories if they don't exist
        self.optimized_dir.mkdir(parents=True, exist_ok=True)
        
        # Timestamp for this training run
        self.timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M')
        self.training_dir = self.model_dir / f'training_{self.timestamp}'
        self.training_dir.mkdir(parents=True, exist_ok=True)
        
        mode_str = "PRODUCTION" if production_mode else "EVALUATION"
        split_str = "temporal" if use_temporal_split and not production_mode else "all data" if production_mode else "random"
        logger.info(f"Pipeline initialized. Mode: {mode_str}, Split: {split_str}")
        logger.info(f"Training directory: {self.training_dir}")
    
    def check_elo_ratings_status(self) -> bool:
        """Check if ELO ratings exist and log their status."""
        elo_ratings_path = self.base_dir / "ufc_fighter_elo_ratings.csv"
        
        if not elo_ratings_path.exists():
            logger.warning("⚠️ ELO ratings file not found!")
            logger.warning("   Run the fast scraper to generate ELO ratings:")
            logger.warning("   python3 src/ufc_predictor/scrapers/fast_scraping.py")
            return False
        
        try:
            # Check when ELO ratings were last updated
            import os
            from datetime import datetime
            
            file_stats = os.stat(elo_ratings_path)
            last_modified = datetime.fromtimestamp(file_stats.st_mtime)
            days_old = (datetime.now() - last_modified).days
            
            # Load and check content
            elo_df = pd.read_csv(elo_ratings_path)
            
            logger.info(f"✅ ELO ratings found:")
            logger.info(f"   - File: {elo_ratings_path}")
            logger.info(f"   - Last updated: {days_old} days ago")
            logger.info(f"   - Contains: {len(elo_df)} fighters")
            
            if days_old > 7:
                logger.warning("   ⚠️ ELO ratings may be stale (>7 days old)")
                logger.warning("   Consider running the scraper for latest data:")
                logger.warning("   python3 src/ufc_predictor/scrapers/fast_scraping.py")
            
            return True
            
        except Exception as e:
            logger.error(f"Error reading ELO ratings: {e}")
            return False
    
    def load_and_prepare_data(self) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """Load and prepare data for training."""
        logger.info("Loading and preparing data...")
        
        # Check ELO ratings status (don't rebuild, just check and warn if needed)
        self.check_elo_ratings_status()
        
        # Load raw fight data for temporal information
        fights_path = self.data_dir / 'ufc_fights.csv'
        if not fights_path.exists():
            raise FileNotFoundError(f"No fight data found at {fights_path}")
        
        fights_df = pd.read_csv(fights_path)
        
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
            # Process raw data
            logger.info("Processing raw fight data...")
            X, y = prepare_modeling_data(fights_df)
            # Create data dataframe for consistency
            data = pd.concat([X, y], axis=1)
            data['Event'] = fights_df.get('Event', '')  # Add Event column if available
            data['Date'] = fights_df.get('Date', '')    # Add Date column if available
        else:
            # Load processed data
            logger.info(f"Loading processed data from {processed_path}")
            data = pd.read_csv(processed_path)
            
            # Handle different target column names
            if 'Winner' in data.columns:
                target_col = 'Winner'
                y = data[target_col]
            elif 'winner' in data.columns:
                target_col = 'winner'
                y = data[target_col]
            elif 'Outcome' in data.columns:
                # Convert win/loss to binary
                target_col = 'Outcome'
                y = (data['Outcome'] == 'win').astype(int)
            else:
                raise ValueError(f"Could not find target column. Available columns: {data.columns.tolist()[:10]}...")
            
            # Exclude non-feature columns (including any string columns)
            exclude_cols = [target_col, 'Outcome', 'Winner', 'winner', 'Method', 'Method_Cleaned', 
                          'Round', 'Title_fight', 'Event', 'Date', 'Fighter', 'Opponent', 
                          'fighter_url', 'opponent_url', 'blue_fighter_url', 'red_fighter_url', 
                          'blue_Name', 'red_Name', 'Time']
            
            # Get potential feature columns
            feature_cols = [col for col in data.columns if col not in exclude_cols]
            X_temp = data[feature_cols]
            
            # Keep only numeric columns
            numeric_cols = X_temp.select_dtypes(include=['int64', 'float64', 'int32', 'float32', 'bool']).columns
            X = data[numeric_cols]
        
        # Handle different splitting strategies
        if self.production_mode:
            # PRODUCTION MODE: Use ALL data for training (no test set)
            logger.info("🚀 PRODUCTION MODE: Training on ALL available data")
            X_train, y_train = X, y
            X_test, y_test = X.iloc[:0], y.iloc[:0]  # Empty test set
            logger.info(f"Training on all {len(X_train)} samples")
            
        elif self.use_temporal_split:
            # EVALUATION MODE with temporal split
            logger.info("⏰ Using TEMPORAL SPLIT for realistic evaluation")
            
            # Try to use dates from the original fights data for temporal ordering
            temporal_split_successful = False
            
            try:
                # Parse dates from original fight data
                if 'Date' in fights_df.columns:
                    fights_df['date'] = pd.to_datetime(fights_df['Date'], errors='coerce')
                    
                    # We need to align the processed data with the original dates
                    # This is a simplified approach - just use the index order
                    if len(X) == len(fights_df):
                        # Data aligns with fights_df
                        dates = fights_df['date']
                        valid_dates = dates.notna()
                        
                        if valid_dates.sum() > 100:  # Need enough valid dates
                            X_with_dates = X[valid_dates]
                            y_with_dates = y[valid_dates]
                            dates = dates[valid_dates]
                            
                            # Sort by date
                            sorted_idx = dates.argsort()
                            X_sorted = X_with_dates.iloc[sorted_idx]
                            y_sorted = y_with_dates.iloc[sorted_idx]
                            dates_sorted = dates.iloc[sorted_idx]
                            
                            # Use 80% for training, 20% for testing (temporal split)
                            split_idx = int(len(X_sorted) * 0.8)
                            
                            X_train = X_sorted.iloc[:split_idx]
                            X_test = X_sorted.iloc[split_idx:]
                            y_train = y_sorted.iloc[:split_idx]
                            y_test = y_sorted.iloc[split_idx:]
                            
                            train_end_date = dates_sorted.iloc[split_idx - 1]
                            test_start_date = dates_sorted.iloc[split_idx]
                            
                            logger.info(f"Temporal split successful:")
                            logger.info(f"  Training: {len(X_train)} fights (up to {train_end_date.strftime('%Y-%m-%d')})")
                            logger.info(f"  Testing:  {len(X_test)} fights (from {test_start_date.strftime('%Y-%m-%d')})")
                            
                            # Calculate temporal gap
                            gap_days = (test_start_date - train_end_date).days
                            logger.info(f"  Temporal gap: {gap_days} days")
                            
                            temporal_split_successful = True
                    
                if not temporal_split_successful:
                    # Try a simpler approach - just sort by index assuming chronological order
                    logger.info("Using simplified temporal split (assuming chronological order by index)")
                    split_idx = int(len(X) * 0.8)
                    
                    X_train = X.iloc[:split_idx]
                    X_test = X.iloc[split_idx:]
                    y_train = y.iloc[:split_idx]
                    y_test = y.iloc[split_idx:]
                    
                    logger.info(f"  Training: First {len(X_train)} fights (80%)")
                    logger.info(f"  Testing:  Last {len(X_test)} fights (20%)")
                    temporal_split_successful = True
                    
            except Exception as e:
                logger.warning(f"Temporal split failed: {e}")
                temporal_split_successful = False
            
            if not temporal_split_successful:
                # Fall back to random split
                logger.warning("Could not perform temporal split - falling back to random split")
                logger.warning("Consider updating your data to include proper date information")
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y
                )
        else:
            # EVALUATION MODE with random split (old behavior)
            logger.info("Using RANDOM SPLIT (Note: Consider temporal split for time-series data)")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
        
        # Save processed data for reference
        if len(X_train) > 0:
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
        
        # Initialize feature selector with importance-based method
        selector = UFCFeatureSelector(
            method='importance_based',
            n_features=n_features
        )
        
        # Fit the selector on training data
        selector.fit(X_train, y_train)
        
        # Get selected features
        selected_features = selector.selected_features
        
        logger.info(f"Selected {len(selected_features)} features")
        
        # Transform data to selected features
        X_train_opt = selector.transform(X_train)
        X_test_opt = selector.transform(X_test)
        
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
        
        # Use the same timestamp as the rest of the pipeline
        # Save with timestamp
        optimized_model_path = self.optimized_dir / f'ufc_model_optimized_{self.timestamp}.joblib'
        selector_path = self.optimized_dir / f'feature_selector_{self.timestamp}.json'
        metrics_path = self.optimized_dir / f'metrics_{self.timestamp}.json'
        
        joblib.dump(optimized_model, optimized_model_path)
        selector.save(str(selector_path))
        
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
        
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        logger.info(f"Optimized model saved:")
        logger.info(f"  - Model: {optimized_model_path.name}")
        logger.info(f"  - Selector: {selector_path.name}")
        logger.info(f"  - Directory: {self.optimized_dir}")
        
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
                    'location': str(self.optimized_dir / f'ufc_model_optimized_{self.timestamp}.joblib')
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