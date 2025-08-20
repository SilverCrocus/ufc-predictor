#!/usr/bin/env python3
"""
Enhanced Training Pipeline with Walk-Forward Validation
Integrates the new walk-forward validation system to address overfitting.
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

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.ufc_predictor.models.model_training import UFCModelTrainer
from src.ufc_predictor.models.feature_selection import UFCFeatureSelector
from src.ufc_predictor.data.feature_engineering import prepare_modeling_data
from src.ufc_predictor.evaluation.walk_forward_validator import WalkForwardValidator, RetrainingConfig, compare_validation_methods

# Import existing pipeline for compatibility
from .complete_training_pipeline import CompletePipeline

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class EnhancedTrainingPipeline(CompletePipeline):
    """
    Enhanced training pipeline that extends CompletePipeline with walk-forward validation.
    Addresses overfitting through realistic temporal validation and model retraining.
    """
    
    def __init__(self, base_dir: Path = None, validation_mode: str = 'walk_forward', 
                 production_mode: bool = False):
        """
        Initialize enhanced training pipeline.
        
        Args:
            base_dir: Base directory for the project
            validation_mode: 'walk_forward', 'static_temporal', or 'comparison'
            production_mode: If True, trains on all data (no holdout for testing)
        """
        super().__init__(base_dir, use_temporal_split=True, production_mode=production_mode)
        
        self.validation_mode = validation_mode
        self.walk_forward_results = None
        self.validation_comparison = None
        
        # Create validation output directory
        self.validation_dir = self.model_dir / 'validation_analysis'
        self.validation_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Enhanced pipeline initialized with validation mode: {validation_mode}")
    
    def run_enhanced_pipeline(self, tune: bool = True, optimize: bool = True, 
                             n_features: int = 32, retrain_frequency_months: int = 3) -> Dict:
        """
        Run the enhanced training pipeline with walk-forward validation.
        
        Args:
            tune: Whether to tune hyperparameters
            optimize: Whether to create optimized models
            n_features: Number of features for optimized model
            retrain_frequency_months: How often to retrain models (in months)
            
        Returns:
            Enhanced training summary with validation analysis
        """
        logger.info("="*60)
        logger.info("ENHANCED UFC TRAINING PIPELINE WITH WALK-FORWARD VALIDATION")
        logger.info("="*60)
        
        try:
            # Step 1: Load and prepare data (same as parent)
            logger.info("\n📊 STEP 1: Data Preparation")
            X_train, X_test, y_train, y_test = self.load_and_prepare_data()
            
            # Load original fights data for temporal information
            fights_path = self.data_dir / 'ufc_fights.csv'
            if not fights_path.exists():
                raise FileNotFoundError(f"Original fights data not found at {fights_path}")
            
            fights_df = pd.read_csv(fights_path)
            logger.info(f"Loaded {len(fights_df)} fights for temporal analysis")
            
            # Step 2: Run validation analysis
            if self.validation_mode == 'walk_forward':
                logger.info("\n🔄 STEP 2: Walk-Forward Validation")
                self._run_walk_forward_validation(X_train, y_train, fights_df, tune, retrain_frequency_months)
                
            elif self.validation_mode == 'comparison':
                logger.info("\n📊 STEP 2: Validation Method Comparison")
                self._run_validation_comparison(X_train, y_train, fights_df)
                
            else:  # static_temporal
                logger.info("\n⏰ STEP 2: Traditional Temporal Validation")
                # Use parent class method
                pass
            
            # Step 3: Train final models (if not production mode or if comparison shows static is better)
            if not self.production_mode:
                logger.info("\n🎯 STEP 3: Final Model Training")
                training_results = self.train_models(X_train, X_test, y_train, y_test, tune=tune)
            else:
                # In production mode, use all data
                X_all = pd.concat([X_train, X_test], axis=0)
                y_all = pd.concat([y_train, y_test], axis=0)
                training_results = self.train_models(X_all, X_all.iloc[:0], y_all, y_all.iloc[:0], tune=tune)
            
            # Step 4: Create optimized model if requested
            if optimize:
                logger.info("\n⚡ STEP 4: Model Optimization")
                
                if not self.production_mode:
                    optimization_results = self.optimize_model(
                        training_results['rf_tuned'], X_train, y_train, X_test, y_test,
                        n_features=n_features
                    )
                else:
                    # Use cross-validation for optimization in production mode
                    X_all = pd.concat([X_train, X_test], axis=0)
                    y_all = pd.concat([y_train, y_test], axis=0)
                    optimization_results = self.optimize_model(
                        training_results['rf_tuned'], X_all, y_all, X_all.iloc[:10], y_all.iloc[:10],
                        n_features=n_features
                    )
            else:
                optimization_results = None
            
            # Step 5: Generate comprehensive summary
            logger.info("\n📋 STEP 5: Enhanced Pipeline Summary")
            
            summary = self._create_enhanced_summary(
                training_results, optimization_results, X_train, X_test, n_features
            )
            
            # Save enhanced summary
            enhanced_summary_path = self.model_dir / 'enhanced_training_summary.json'
            with open(enhanced_summary_path, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            
            # Create validation report
            if self.walk_forward_results:
                self._create_validation_report()
            
            logger.info("\n✅ ENHANCED PIPELINE COMPLETE!")
            self._print_final_recommendations(summary)
            
            return summary
            
        except Exception as e:
            logger.error(f"Enhanced pipeline failed: {e}")
            raise
    
    def _run_walk_forward_validation(self, X: pd.DataFrame, y: pd.Series, fights_df: pd.DataFrame, 
                                   tune: bool, retrain_frequency_months: int):
        """Run walk-forward validation analysis."""
        
        # Configure retraining strategy
        retrain_config = RetrainingConfig(
            retrain_frequency_months=retrain_frequency_months,
            min_new_samples=50,  # Retrain if 50+ new samples
            performance_threshold=0.05,  # Retrain if performance drops >5%
            max_training_window_years=7,  # Use max 7 years of data
            use_expanding_window=True  # Use expanding windows
        )
        
        # Initialize validator
        validator = WalkForwardValidator(
            retrain_config=retrain_config,
            base_output_dir=self.validation_dir
        )
        
        # Run walk-forward validation
        logger.info(f"Running walk-forward validation with {retrain_frequency_months}-month retraining...")
        self.walk_forward_results = validator.run_walk_forward_validation(
            X, y, fights_df, 
            tune_hyperparameters=tune,
            save_models=True  # Save models for analysis
        )
        
        # Log key results
        wf_results = self.walk_forward_results
        logger.info(f"Walk-forward validation completed:")
        logger.info(f"  Mean test accuracy: {wf_results.aggregate_metrics['mean_test_accuracy']:.4f}")
        logger.info(f"  Mean overfitting gap: {wf_results.overfitting_analysis['mean_overfitting_gap']:.4f}")
        logger.info(f"  Model retrains: {wf_results.metadata['total_retrains']}")
        
        # Create plots
        plot_path = self.validation_dir / f"walk_forward_plots_{self.timestamp}.png"
        validator.plot_validation_results(wf_results, plot_path)
    
    def _run_validation_comparison(self, X: pd.DataFrame, y: pd.Series, fights_df: pd.DataFrame):
        """Run comparison between validation methods."""
        
        logger.info("Comparing validation methods: static temporal split vs walk-forward...")
        
        self.validation_comparison = compare_validation_methods(
            X, y, fights_df, output_dir=self.validation_dir
        )
        
        # Log comparison results
        comp = self.validation_comparison['comparison']
        logger.info(f"Validation method comparison:")
        logger.info(f"  Static split overfitting gap: {comp['static_overfitting_gap']:.4f}")
        logger.info(f"  Walk-forward overfitting gap: {comp['walk_forward_overfitting_gap']:.4f}")
        logger.info(f"  Overfitting improvement: {comp['overfitting_improvement']:.4f} ({comp['overfitting_improvement_pct']:.1f}%)")
        logger.info(f"  Recommended method: {comp['recommendation']}")
    
    def _create_enhanced_summary(self, training_results: Dict, optimization_results: Optional[Dict],
                               X_train: pd.DataFrame, X_test: pd.DataFrame, n_features: int) -> Dict:
        """Create enhanced summary with validation analysis."""
        
        # Start with base summary
        summary = {
            'timestamp': self.timestamp,
            'training_dir': str(self.training_dir),
            'validation_mode': self.validation_mode,
            'production_mode': self.production_mode,
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
        
        # Add optimization results
        if optimization_results:
            speed_improvement = X_train.shape[1] / n_features
            summary['optimized'] = {
                'accuracy': optimization_results['metrics']['accuracy'],
                'n_features': len(optimization_results['features']),
                'speed_gain': f"{speed_improvement:.1f}x",
                'location': str(self.optimized_dir / f'ufc_model_optimized_{self.timestamp}.joblib')
            }
        
        # Add walk-forward validation results
        if self.walk_forward_results:
            wf = self.walk_forward_results
            summary['walk_forward_validation'] = {
                'mean_test_accuracy': wf.aggregate_metrics['mean_test_accuracy'],
                'std_test_accuracy': wf.aggregate_metrics['std_test_accuracy'],
                'mean_overfitting_gap': wf.overfitting_analysis['mean_overfitting_gap'],
                'performance_degradation_pct': wf.performance_degradation.get('degradation_pct', 0),
                'model_stability_score': wf.model_stability['performance_stability_score'],
                'n_folds': wf.metadata['n_folds'],
                'total_retrains': wf.metadata['total_retrains'],
                'validation_report_path': str(self.validation_dir / f'validation_report_{self.timestamp}.txt')
            }
        
        # Add validation comparison results
        if self.validation_comparison:
            comp = self.validation_comparison['comparison']
            summary['validation_comparison'] = {
                'static_overfitting_gap': comp['static_overfitting_gap'],
                'walk_forward_overfitting_gap': comp['walk_forward_overfitting_gap'],
                'overfitting_improvement': comp['overfitting_improvement'],
                'overfitting_improvement_pct': comp['overfitting_improvement_pct'],
                'recommended_method': comp['recommendation']
            }
        
        return summary
    
    def _create_validation_report(self):
        """Create and save validation report."""
        if not self.walk_forward_results:
            return
        
        # Create comprehensive validation report
        validator = WalkForwardValidator()  # Temporary instance for report generation
        report_text = validator.create_validation_report(self.walk_forward_results)
        
        # Save report
        report_path = self.validation_dir / f'validation_report_{self.timestamp}.txt'
        with open(report_path, 'w') as f:
            f.write(report_text)
        
        logger.info(f"Validation report saved to {report_path}")
        
        # Print key findings
        logger.info("\n📊 VALIDATION ANALYSIS SUMMARY:")
        wf = self.walk_forward_results
        logger.info(f"  Average overfitting gap: {wf.overfitting_analysis['mean_overfitting_gap']:.4f}")
        logger.info(f"  Performance stability: {wf.model_stability['performance_stability_score']:.4f}")
        
        if wf.performance_degradation.get('degradation_pct', 0) > 10:
            logger.warning(f"  ⚠️ Performance degradation detected: {wf.performance_degradation['degradation_pct']:.1f}%")
        
        if wf.overfitting_analysis['mean_overfitting_gap'] > 0.15:
            logger.warning(f"  ⚠️ High overfitting detected (>{wf.overfitting_analysis['mean_overfitting_gap']:.1%})")
    
    def _print_final_recommendations(self, summary: Dict):
        """Print final recommendations based on analysis."""
        
        logger.info(f"\n🎯 FINAL RECOMMENDATIONS:")
        logger.info(f"  Training directory: {self.training_dir}")
        
        # Model recommendations
        if 'optimized' in summary:
            logger.info(f"  ✅ Use optimized model for production: {self.optimized_dir}/ufc_model_optimized_latest.joblib")
            logger.info(f"     - {summary['optimized']['speed_gain']} faster inference")
            logger.info(f"     - {summary['optimized']['n_features']} features vs {summary['data']['n_features_original']} original")
        
        # Validation recommendations
        if 'walk_forward_validation' in summary:
            wf_val = summary['walk_forward_validation']
            
            if wf_val['mean_overfitting_gap'] < 0.10:
                logger.info(f"  ✅ Overfitting is manageable ({wf_val['mean_overfitting_gap']:.1%})")
            else:
                logger.info(f"  ⚠️ Consider reducing model complexity (overfitting: {wf_val['mean_overfitting_gap']:.1%})")
            
            if wf_val['model_stability_score'] > 0.8:
                logger.info(f"  ✅ Model stability is good ({wf_val['model_stability_score']:.3f})")
            else:
                logger.info(f"  ⚠️ Consider ensemble methods for better stability")
            
            if abs(wf_val['performance_degradation_pct']) > 10:
                logger.info(f"  ⚠️ Consider more frequent retraining (degradation: {wf_val['performance_degradation_pct']:.1f}%)")
        
        # Comparison recommendations
        if 'validation_comparison' in summary:
            comp = summary['validation_comparison']
            if comp['overfitting_improvement'] > 0.05:
                logger.info(f"  ✅ Walk-forward validation shows significant improvement over static split")
                logger.info(f"     - Overfitting reduced by {comp['overfitting_improvement']:.1%} ({comp['overfitting_improvement_pct']:.1f}%)")
            
        logger.info(f"\n📁 All results saved to: {self.model_dir}")


def main():
    """Main execution function with enhanced validation options."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Enhanced UFC Training Pipeline with Walk-Forward Validation')
    
    # Basic training options
    parser.add_argument('--tune', action='store_true', help='Tune hyperparameters')
    parser.add_argument('--no-optimize', action='store_true', help='Skip optimization step')
    parser.add_argument('--n-features', type=int, default=32, 
                       help='Number of features for optimized model (default: 32)')
    
    # Validation options
    parser.add_argument('--validation-mode', choices=['walk_forward', 'static_temporal', 'comparison'], 
                       default='walk_forward', help='Validation method to use')
    parser.add_argument('--retrain-months', type=int, default=3,
                       help='Retrain frequency in months for walk-forward validation (default: 3)')
    
    # Mode options
    parser.add_argument('--production', action='store_true',
                       help='Production mode: train on ALL data (no test set)')
    
    args = parser.parse_args()
    
    # Run enhanced pipeline
    pipeline = EnhancedTrainingPipeline(
        validation_mode=args.validation_mode,
        production_mode=args.production
    )
    
    results = pipeline.run_enhanced_pipeline(
        tune=args.tune,
        optimize=not args.no_optimize,
        n_features=args.n_features,
        retrain_frequency_months=args.retrain_months
    )
    
    # Print final summary
    print("\n" + "="*60)
    print("ENHANCED TRAINING SUMMARY")
    print("="*60)
    print(json.dumps(results, indent=2, default=str))


if __name__ == '__main__':
    main()