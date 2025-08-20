"""
Enhanced Walk-Forward Validation for UFC Prediction Models
Addresses overfitting by providing realistic time-series validation with model retraining.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
from pathlib import Path
import json
import joblib
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss
import matplotlib.pyplot as plt
import seaborn as sns

from .temporal_split import TemporalWalkForwardSplitter, TemporalFold
from .metrics import UFCMetricsCalculator, MetricsResult
from ..models.model_training import UFCModelTrainer

logger = logging.getLogger(__name__)


@dataclass
class WalkForwardResults:
    """Container for walk-forward validation results."""
    fold_results: List[Dict[str, Any]]
    aggregate_metrics: Dict[str, float]
    performance_degradation: Dict[str, float]
    overfitting_analysis: Dict[str, float]
    temporal_trends: pd.DataFrame
    model_stability: Dict[str, float]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrainingConfig:
    """Configuration for model retraining strategy."""
    retrain_frequency_months: int = 3  # Retrain every 3 months
    min_new_samples: int = 50  # Minimum new samples before retraining
    performance_threshold: float = 0.05  # Retrain if performance drops > 5%
    max_training_window_years: int = 7  # Maximum training data to use
    use_expanding_window: bool = True  # True for expanding, False for rolling


class WalkForwardValidator:
    """
    Enhanced walk-forward validator that addresses overfitting through:
    1. Realistic temporal validation
    2. Periodic model retraining
    3. Performance degradation tracking
    4. Overfitting measurement over time
    """
    
    def __init__(
        self,
        retrain_config: Optional[RetrainingConfig] = None,
        base_output_dir: Optional[Path] = None
    ):
        """
        Initialize walk-forward validator.
        
        Args:
            retrain_config: Configuration for retraining strategy
            base_output_dir: Base directory for saving results
        """
        self.retrain_config = retrain_config or RetrainingConfig()
        self.base_output_dir = base_output_dir or Path("model/walk_forward_validation")
        self.base_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.metrics_calculator = UFCMetricsCalculator()
        self.temporal_splitter = TemporalWalkForwardSplitter(
            train_years=5,  # Initial training window
            test_months=3,  # Test period length
            gap_days=14     # Gap to prevent leakage
        )
        
        # Tracking variables
        self.fold_results = []
        self.models_trained = {}
        self.performance_history = []
        
    def run_walk_forward_validation(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        fights_df: pd.DataFrame,
        tune_hyperparameters: bool = True,
        save_models: bool = False
    ) -> WalkForwardResults:
        """
        Run complete walk-forward validation with retraining.
        
        Args:
            X: Feature matrix
            y: Target variable
            fights_df: Original fights DataFrame with dates
            tune_hyperparameters: Whether to tune hyperparameters during training
            save_models: Whether to save trained models
            
        Returns:
            WalkForwardResults with comprehensive analysis
        """
        logger.info("Starting walk-forward validation with retraining")
        
        # Ensure temporal alignment
        if len(X) != len(fights_df):
            raise ValueError(f"Feature matrix ({len(X)}) and fights DataFrame ({len(fights_df)}) length mismatch")
        
        # Add dates to feature matrix for temporal splitting
        X_with_dates = X.copy()
        X_with_dates['date'] = pd.to_datetime(fights_df['Date'])
        
        # Create temporal folds using expanding windows
        folds = self.temporal_splitter.make_expanding_folds(
            X_with_dates,
            test_months=self.retrain_config.retrain_frequency_months,
            step_months=1  # Move forward 1 month at a time
        )
        
        logger.info(f"Created {len(folds)} temporal folds for validation")
        
        # Initialize tracking
        current_model = None
        last_retrain_fold = -1
        fold_results = []
        
        for fold_idx, fold in enumerate(folds):
            logger.info(f"Processing fold {fold_idx + 1}/{len(folds)}")
            
            # Get training and test data
            train_indices = fold.train_indices
            test_indices = fold.test_indices
            
            # Remove date column for modeling
            X_train = X.iloc[train_indices]
            X_test = X.iloc[test_indices]
            y_train = y.iloc[train_indices]
            y_test = y.iloc[test_indices]
            
            # Determine if we need to retrain
            should_retrain = self._should_retrain_model(
                fold_idx, last_retrain_fold, len(test_indices), 
                current_model, X_test, y_test
            )
            
            if should_retrain or current_model is None:
                logger.info(f"Retraining model for fold {fold_idx + 1}")
                
                # Limit training window if configured
                X_train_limited, y_train_limited = self._limit_training_window(
                    X_train, y_train, fold.train_start
                )
                
                # Train new model
                current_model = self._train_model(
                    X_train_limited, y_train_limited, 
                    fold_idx, tune_hyperparameters, save_models
                )
                last_retrain_fold = fold_idx
            
            # Evaluate on test set
            fold_result = self._evaluate_fold(
                current_model, X_train, y_train, X_test, y_test,
                fold, fold_idx
            )
            
            fold_results.append(fold_result)
            self.performance_history.append({
                'fold': fold_idx,
                'date': fold.test_start,
                'test_accuracy': fold_result['test_metrics']['accuracy'],
                'train_accuracy': fold_result['train_metrics']['accuracy'],
                'overfitting_gap': fold_result['overfitting_gap'],
                'retrained': should_retrain
            })
            
            logger.info(f"Fold {fold_idx + 1} - Test Accuracy: {fold_result['test_metrics']['accuracy']:.4f}, "
                       f"Overfitting Gap: {fold_result['overfitting_gap']:.4f}")
        
        # Analyze results
        results = self._analyze_walk_forward_results(fold_results, folds)
        
        # Save results
        self._save_results(results)
        
        return results
    
    def _should_retrain_model(
        self,
        fold_idx: int,
        last_retrain_fold: int,
        test_size: int,
        current_model: Optional[UFCModelTrainer],
        X_test: pd.DataFrame,
        y_test: pd.Series
    ) -> bool:
        """Determine if model should be retrained."""
        
        # Always retrain on first fold
        if current_model is None:
            return True
        
        # Check if enough folds have passed
        folds_since_retrain = fold_idx - last_retrain_fold
        months_since_retrain = folds_since_retrain  # Assuming monthly folds
        
        if months_since_retrain >= self.retrain_config.retrain_frequency_months:
            return True
        
        # Check if we have enough new samples
        if test_size >= self.retrain_config.min_new_samples:
            return True
        
        # Check performance degradation
        if len(self.performance_history) >= 3:  # Need some history
            # Get recent performance
            recent_performance = [h['test_accuracy'] for h in self.performance_history[-3:]]
            avg_recent = np.mean(recent_performance)
            
            # Get performance at last retrain
            retrain_performance = [h['test_accuracy'] for h in self.performance_history 
                                 if h['retrained']]
            if retrain_performance:
                last_retrain_performance = retrain_performance[-1]
                performance_drop = last_retrain_performance - avg_recent
                
                if performance_drop > self.retrain_config.performance_threshold:
                    logger.info(f"Performance dropped by {performance_drop:.4f}, triggering retrain")
                    return True
        
        return False
    
    def _limit_training_window(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        current_date: datetime
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Limit training window to maximum configured years."""
        
        if not self.retrain_config.use_expanding_window:
            # Use rolling window
            cutoff_date = current_date - timedelta(days=365 * self.retrain_config.max_training_window_years)
            
            # This is simplified - in practice you'd need date information in the training data
            # For now, just use the last N samples as an approximation
            max_samples = int(len(X_train) * 0.8)  # Use 80% as rolling window
            
            if len(X_train) > max_samples:
                X_train = X_train.iloc[-max_samples:]
                y_train = y_train.iloc[-max_samples:]
                logger.info(f"Limited training window to {len(X_train)} samples")
        
        return X_train, y_train
    
    def _train_model(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        fold_idx: int,
        tune_hyperparameters: bool,
        save_models: bool
    ) -> UFCModelTrainer:
        """Train a new model for the current fold."""
        
        trainer = UFCModelTrainer(random_state=42)
        
        # Train Random Forest (primary model)
        trainer.train_random_forest(X_train, y_train)
        
        # Tune if requested
        if tune_hyperparameters:
            try:
                trainer.tune_random_forest(X_train, y_train, n_iter=50, cv=3)
            except Exception as e:
                logger.warning(f"Hyperparameter tuning failed: {e}")
        
        # Save model if requested
        if save_models:
            model_path = self.base_output_dir / f"model_fold_{fold_idx}.joblib"
            best_model_name = 'random_forest_tuned' if 'random_forest_tuned' in trainer.models else 'random_forest'
            joblib.dump(trainer.models[best_model_name], model_path)
            logger.info(f"Saved model for fold {fold_idx} to {model_path}")
        
        self.models_trained[fold_idx] = trainer
        return trainer
    
    def _evaluate_fold(
        self,
        model: UFCModelTrainer,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        fold: TemporalFold,
        fold_idx: int
    ) -> Dict[str, Any]:
        """Evaluate model performance on a single fold."""
        
        # Get best model
        best_model_name = 'random_forest_tuned' if 'random_forest_tuned' in model.models else 'random_forest'
        best_model = model.models[best_model_name]
        
        # Predictions
        y_train_pred = best_model.predict(X_train)
        y_test_pred = best_model.predict(X_test)
        
        y_train_proba = best_model.predict_proba(X_train)[:, 1]
        y_test_proba = best_model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        train_metrics = {
            'accuracy': accuracy_score(y_train, y_train_pred),
            'auc': roc_auc_score(y_train, y_train_proba),
            'log_loss': log_loss(y_train, y_train_proba)
        }
        
        test_metrics = {
            'accuracy': accuracy_score(y_test, y_test_pred),
            'auc': roc_auc_score(y_test, y_test_proba),
            'log_loss': log_loss(y_test, y_test_proba)
        }
        
        # Calculate overfitting gap
        overfitting_gap = train_metrics['accuracy'] - test_metrics['accuracy']
        
        # Detailed calibration analysis
        calibration_metrics = self.metrics_calculator.calculate_calibration_metrics(
            y_test.values, y_test_proba
        )
        
        return {
            'fold_id': fold_idx,
            'train_period': (fold.train_start, fold.train_end),
            'test_period': (fold.test_start, fold.test_end),
            'train_size': len(X_train),
            'test_size': len(X_test),
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'calibration_metrics': calibration_metrics,
            'overfitting_gap': overfitting_gap,
            'model_name': best_model_name
        }
    
    def _analyze_walk_forward_results(
        self,
        fold_results: List[Dict[str, Any]],
        folds: List[TemporalFold]
    ) -> WalkForwardResults:
        """Analyze and aggregate walk-forward validation results."""
        
        # Aggregate metrics
        test_accuracies = [r['test_metrics']['accuracy'] for r in fold_results]
        test_aucs = [r['test_metrics']['auc'] for r in fold_results]
        overfitting_gaps = [r['overfitting_gap'] for r in fold_results]
        
        aggregate_metrics = {
            'mean_test_accuracy': np.mean(test_accuracies),
            'std_test_accuracy': np.std(test_accuracies),
            'mean_test_auc': np.mean(test_aucs),
            'std_test_auc': np.std(test_aucs),
            'mean_overfitting_gap': np.mean(overfitting_gaps),
            'std_overfitting_gap': np.std(overfitting_gaps),
            'min_test_accuracy': np.min(test_accuracies),
            'max_test_accuracy': np.max(test_accuracies),
            'accuracy_range': np.max(test_accuracies) - np.min(test_accuracies)
        }
        
        # Performance degradation analysis
        if len(test_accuracies) >= 3:
            early_performance = np.mean(test_accuracies[:3])
            late_performance = np.mean(test_accuracies[-3:])
            performance_degradation = {
                'early_accuracy': early_performance,
                'late_accuracy': late_performance,
                'degradation': early_performance - late_performance,
                'degradation_pct': (early_performance - late_performance) / early_performance * 100
            }
        else:
            performance_degradation = {'degradation': 0, 'degradation_pct': 0}
        
        # Overfitting analysis
        overfitting_analysis = {
            'consistent_overfitting': np.mean([gap > 0.05 for gap in overfitting_gaps]),
            'severe_overfitting': np.mean([gap > 0.15 for gap in overfitting_gaps]),
            'median_overfitting_gap': np.median(overfitting_gaps),
            'max_overfitting_gap': np.max(overfitting_gaps),
            'improving_over_time': self._analyze_overfitting_trend(overfitting_gaps)
        }
        
        # Create temporal trends DataFrame
        temporal_trends = pd.DataFrame([
            {
                'fold': r['fold_id'],
                'test_start': r['test_period'][0],
                'test_accuracy': r['test_metrics']['accuracy'],
                'test_auc': r['test_metrics']['auc'],
                'overfitting_gap': r['overfitting_gap'],
                'train_size': r['train_size'],
                'test_size': r['test_size']
            }
            for r in fold_results
        ])
        
        # Model stability analysis
        model_stability = {
            'accuracy_cv': np.std(test_accuracies) / np.mean(test_accuracies),
            'auc_cv': np.std(test_aucs) / np.mean(test_aucs),
            'performance_stability_score': 1 - (np.std(test_accuracies) / np.mean(test_accuracies))
        }
        
        return WalkForwardResults(
            fold_results=fold_results,
            aggregate_metrics=aggregate_metrics,
            performance_degradation=performance_degradation,
            overfitting_analysis=overfitting_analysis,
            temporal_trends=temporal_trends,
            model_stability=model_stability,
            metadata={
                'n_folds': len(fold_results),
                'total_retrains': len(self.models_trained),
                'validation_period': (folds[0].test_start, folds[-1].test_end),
                'retrain_config': self.retrain_config.__dict__
            }
        )
    
    def _analyze_overfitting_trend(self, overfitting_gaps: List[float]) -> float:
        """Analyze if overfitting is improving over time (negative slope is good)."""
        if len(overfitting_gaps) < 3:
            return 0
        
        x = np.arange(len(overfitting_gaps))
        slope = np.polyfit(x, overfitting_gaps, 1)[0]
        return slope  # Negative slope means improving over time
    
    def _save_results(self, results: WalkForwardResults):
        """Save walk-forward validation results."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save aggregate results
        results_path = self.base_output_dir / f"walk_forward_results_{timestamp}.json"
        with open(results_path, 'w') as f:
            # Convert datetime objects to strings for JSON serialization
            serializable_results = self._make_json_serializable(results.__dict__)
            json.dump(serializable_results, f, indent=2)
        
        # Save temporal trends
        trends_path = self.base_output_dir / f"temporal_trends_{timestamp}.csv"
        results.temporal_trends.to_csv(trends_path, index=False)
        
        logger.info(f"Walk-forward validation results saved to {self.base_output_dir}")
    
    def _make_json_serializable(self, obj):
        """Convert object to JSON-serializable format."""
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(v) for v in obj]
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict('records')
        elif isinstance(obj, (pd.Timestamp, datetime)):
            return obj.isoformat()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        else:
            return obj
    
    def create_validation_report(self, results: WalkForwardResults) -> str:
        """Create a comprehensive validation report."""
        
        report_lines = [
            "UFC Walk-Forward Validation Report",
            "=" * 50,
            "",
            f"Validation Period: {results.metadata['validation_period'][0]} to {results.metadata['validation_period'][1]}",
            f"Number of Folds: {results.metadata['n_folds']}",
            f"Total Model Retrains: {results.metadata['total_retrains']}",
            "",
            "AGGREGATE PERFORMANCE METRICS",
            "-" * 30,
            f"Mean Test Accuracy: {results.aggregate_metrics['mean_test_accuracy']:.4f} ± {results.aggregate_metrics['std_test_accuracy']:.4f}",
            f"Mean Test AUC: {results.aggregate_metrics['mean_test_auc']:.4f} ± {results.aggregate_metrics['std_test_auc']:.4f}",
            f"Accuracy Range: {results.aggregate_metrics['accuracy_range']:.4f}",
            "",
            "OVERFITTING ANALYSIS",
            "-" * 20,
            f"Mean Overfitting Gap: {results.overfitting_analysis['mean_overfitting_gap']:.4f} ± {results.overfitting_analysis['std_overfitting_gap']:.4f}",
            f"Median Overfitting Gap: {results.overfitting_analysis['median_overfitting_gap']:.4f}",
            f"Max Overfitting Gap: {results.overfitting_analysis['max_overfitting_gap']:.4f}",
            f"Consistent Overfitting (>5%): {results.overfitting_analysis['consistent_overfitting']:.1%} of folds",
            f"Severe Overfitting (>15%): {results.overfitting_analysis['severe_overfitting']:.1%} of folds",
            f"Overfitting Trend: {'Improving' if results.overfitting_analysis['improving_over_time'] < 0 else 'Worsening'}",
            "",
            "PERFORMANCE DEGRADATION",
            "-" * 25,
            f"Early Performance: {results.performance_degradation.get('early_accuracy', 0):.4f}",
            f"Late Performance: {results.performance_degradation.get('late_accuracy', 0):.4f}",
            f"Degradation: {results.performance_degradation.get('degradation', 0):.4f} ({results.performance_degradation.get('degradation_pct', 0):.2f}%)",
            "",
            "MODEL STABILITY",
            "-" * 15,
            f"Accuracy CV: {results.model_stability['accuracy_cv']:.4f}",
            f"AUC CV: {results.model_stability['auc_cv']:.4f}",
            f"Stability Score: {results.model_stability['performance_stability_score']:.4f}",
            "",
            "RECOMMENDATIONS",
            "-" * 15
        ]
        
        # Add recommendations based on results
        if results.overfitting_analysis['mean_overfitting_gap'] > 0.15:
            report_lines.append("⚠️  HIGH OVERFITTING DETECTED - Consider regularization or feature reduction")
        
        if results.performance_degradation.get('degradation_pct', 0) > 10:
            report_lines.append("⚠️  SIGNIFICANT PERFORMANCE DEGRADATION - Consider more frequent retraining")
        
        if results.model_stability['performance_stability_score'] < 0.8:
            report_lines.append("⚠️  LOW MODEL STABILITY - Consider ensemble methods or cross-validation")
        
        if results.overfitting_analysis['improving_over_time'] < -0.01:
            report_lines.append("✅ OVERFITTING IMPROVING OVER TIME - Current strategy is working")
        
        return "\n".join(report_lines)
    
    def plot_validation_results(self, results: WalkForwardResults, save_path: Optional[Path] = None):
        """Create visualization plots for walk-forward validation results."""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Walk-Forward Validation Results', fontsize=16)
        
        # Plot 1: Accuracy over time
        axes[0, 0].plot(results.temporal_trends['test_start'], 
                       results.temporal_trends['test_accuracy'], 
                       marker='o', linewidth=2, markersize=6)
        axes[0, 0].set_title('Test Accuracy Over Time')
        axes[0, 0].set_xlabel('Date')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Plot 2: Overfitting gap over time
        axes[0, 1].plot(results.temporal_trends['test_start'], 
                       results.temporal_trends['overfitting_gap'], 
                       marker='s', linewidth=2, markersize=6, color='red')
        axes[0, 1].axhline(y=0.05, color='orange', linestyle='--', alpha=0.7, label='5% threshold')
        axes[0, 1].axhline(y=0.15, color='red', linestyle='--', alpha=0.7, label='15% threshold')
        axes[0, 1].set_title('Overfitting Gap Over Time')
        axes[0, 1].set_xlabel('Date')
        axes[0, 1].set_ylabel('Train - Test Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Plot 3: Training data size growth
        axes[1, 0].plot(results.temporal_trends['test_start'], 
                       results.temporal_trends['train_size'], 
                       marker='^', linewidth=2, markersize=6, color='green')
        axes[1, 0].set_title('Training Set Size Over Time')
        axes[1, 0].set_xlabel('Date')
        axes[1, 0].set_ylabel('Number of Training Samples')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Plot 4: Accuracy distribution
        axes[1, 1].hist(results.temporal_trends['test_accuracy'], 
                       bins=15, alpha=0.7, edgecolor='black')
        axes[1, 1].axvline(results.aggregate_metrics['mean_test_accuracy'], 
                          color='red', linestyle='--', linewidth=2, label='Mean')
        axes[1, 1].set_title('Test Accuracy Distribution')
        axes[1, 1].set_xlabel('Accuracy')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Validation plots saved to {save_path}")
        
        plt.show()


def compare_validation_methods(
    X: pd.DataFrame,
    y: pd.Series,
    fights_df: pd.DataFrame,
    output_dir: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Compare walk-forward validation with traditional static split.
    
    Returns:
        Comparison results showing the benefits of walk-forward validation
    """
    
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    # 1. Traditional static temporal split (current method)
    logger.info("Running traditional 80/20 temporal split...")
    split_idx = int(len(X) * 0.8)
    X_train_static = X.iloc[:split_idx]
    X_test_static = X.iloc[split_idx:]
    y_train_static = y.iloc[:split_idx]
    y_test_static = y.iloc[split_idx:]
    
    # Train model on static split
    trainer_static = UFCModelTrainer(random_state=42)
    trainer_static.train_random_forest(X_train_static, y_train_static)
    trainer_static.tune_random_forest(X_train_static, y_train_static, n_iter=50, cv=3)
    
    # Evaluate static split
    best_model = trainer_static.models['random_forest_tuned']
    y_train_pred_static = best_model.predict(X_train_static)
    y_test_pred_static = best_model.predict(X_test_static)
    
    static_results = {
        'train_accuracy': accuracy_score(y_train_static, y_train_pred_static),
        'test_accuracy': accuracy_score(y_test_static, y_test_pred_static),
        'overfitting_gap': accuracy_score(y_train_static, y_train_pred_static) - accuracy_score(y_test_static, y_test_pred_static),
        'method': 'static_split'
    }
    
    results['static_split'] = static_results
    
    # 2. Walk-forward validation
    logger.info("Running walk-forward validation...")
    validator = WalkForwardValidator(
        retrain_config=RetrainingConfig(retrain_frequency_months=3),
        base_output_dir=output_dir / "walk_forward" if output_dir else None
    )
    
    walk_forward_results = validator.run_walk_forward_validation(
        X, y, fights_df, tune_hyperparameters=True, save_models=False
    )
    
    results['walk_forward'] = {
        'mean_test_accuracy': walk_forward_results.aggregate_metrics['mean_test_accuracy'],
        'std_test_accuracy': walk_forward_results.aggregate_metrics['std_test_accuracy'],
        'mean_overfitting_gap': walk_forward_results.overfitting_analysis['mean_overfitting_gap'],
        'method': 'walk_forward'
    }
    
    # 3. Analysis and comparison
    overfitting_improvement = static_results['overfitting_gap'] - walk_forward_results.overfitting_analysis['mean_overfitting_gap']
    
    comparison_summary = {
        'overfitting_improvement': overfitting_improvement,
        'overfitting_improvement_pct': (overfitting_improvement / static_results['overfitting_gap']) * 100,
        'static_overfitting_gap': static_results['overfitting_gap'],
        'walk_forward_overfitting_gap': walk_forward_results.overfitting_analysis['mean_overfitting_gap'],
        'recommendation': 'walk_forward' if overfitting_improvement > 0.02 else 'static_split'
    }
    
    results['comparison'] = comparison_summary
    
    # Save comparison results
    if output_dir:
        comparison_path = output_dir / "validation_method_comparison.json"
        with open(comparison_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Comparison results saved to {comparison_path}")
    
    return results