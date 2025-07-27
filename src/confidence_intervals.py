"""
Bootstrap Confidence Intervals for UFC Predictions

Provides uncertainty quantification for machine learning models through
bootstrap resampling and percentile-based confidence intervals.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional, Any, Callable
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from dataclasses import dataclass
import psutil
import os
import time

logger = logging.getLogger(__name__)


@dataclass
class ConfidenceIntervalResult:
    """Result from confidence interval calculation"""
    predictions: np.ndarray
    lower_bounds: np.ndarray
    upper_bounds: np.ndarray
    confidence_level: float
    n_bootstrap: int
    uncertainty_scores: np.ndarray


class BootstrapConfidenceCalculator:
    """
    Bootstrap-based confidence interval calculator for UFC predictions
    
    Supports multiple models and provides both prediction intervals
    and uncertainty quantification for betting decisions.
    """
    
    def __init__(self, confidence_level: float = 0.95, n_bootstrap: int = 100, 
                 n_jobs: int = -1, random_state: int = 42, max_memory_mb: int = 4096):
        """
        Initialize production bootstrap confidence calculator
        
        Args:
            confidence_level: Confidence level for intervals (e.g., 0.95 for 95%)
            n_bootstrap: Number of bootstrap samples
            n_jobs: Number of parallel jobs (-1 for all cores)
            random_state: Random seed for reproducibility
            max_memory_mb: Maximum memory usage in MB
            
        Raises:
            ValueError: If parameters are invalid
            RuntimeError: If system resources are insufficient
        """
        # Strict validation - NO fallbacks
        if not 0.5 <= confidence_level <= 0.999:
            raise ValueError(f"Confidence level must be between 0.5 and 0.999, got {confidence_level}")
        
        if n_bootstrap < 10 or n_bootstrap > 10000:
            raise ValueError(f"n_bootstrap must be between 10 and 10000, got {n_bootstrap}")
        
        if n_jobs < -1 or n_jobs == 0:
            raise ValueError(f"n_jobs must be -1 or positive, got {n_jobs}")
        
        if random_state < 0:
            raise ValueError(f"random_state must be non-negative, got {random_state}")
        
        if max_memory_mb < 512:
            raise ValueError(f"max_memory_mb must be at least 512MB, got {max_memory_mb}")
        
        # Check available system memory
        available_memory_mb = psutil.virtual_memory().available / (1024 * 1024)
        if max_memory_mb > available_memory_mb * 0.8:  # Don't use more than 80% of available
            raise RuntimeError(f"Requested memory {max_memory_mb}MB exceeds safe limit "
                             f"({available_memory_mb * 0.8:.0f}MB of {available_memory_mb:.0f}MB available)")
        
        self.confidence_level = confidence_level
        self.n_bootstrap = n_bootstrap
        self.n_jobs = min(n_jobs if n_jobs != -1 else mp.cpu_count(), 8, mp.cpu_count())
        self.random_state = random_state
        self.max_memory_mb = max_memory_mb
        
        # Create thread-safe random number generator - NO global state
        self.rng = np.random.Generator(np.random.PCG64(random_state))
        
        logger.info(f"Initialized production BootstrapConfidenceCalculator: "
                   f"{n_bootstrap} samples, {confidence_level:.1%} confidence, "
                   f"{self.n_jobs} workers, {max_memory_mb}MB memory limit")
    
    def calculate_intervals(self, models: Dict[str, Any], X: pd.DataFrame,
                          model_weights: Optional[Dict[str, float]] = None) -> ConfidenceIntervalResult:
        """
        Calculate bootstrap confidence intervals with memory monitoring and strict validation
        
        Args:
            models: Dictionary of trained models
            X: Feature matrix for predictions
            model_weights: Optional weights for ensemble averaging
            
        Returns:
            ConfidenceIntervalResult: Complete confidence interval analysis
            
        Raises:
            ValueError: If inputs are invalid
            RuntimeError: If memory limits exceeded or computation fails
        """
        # Memory monitoring start
        process = psutil.Process(os.getpid())
        memory_start = process.memory_info().rss / (1024 * 1024)
        
        if memory_start > self.max_memory_mb * 0.8:
            raise RuntimeError(f"Memory usage {memory_start:.0f}MB exceeds limit before starting "
                             f"(limit: {self.max_memory_mb}MB)")
        
        # Strict input validation - NO fallbacks
        if not models:
            raise ValueError("Models dictionary cannot be empty")
        
        if X.empty:
            raise ValueError("Feature matrix X cannot be empty")
        
        if X.isnull().any().any():
            raise ValueError("Feature matrix X contains null values")
        
        if not isinstance(X, pd.DataFrame):
            raise ValueError(f"X must be pandas DataFrame, got {type(X)}")
        
        n_samples = len(X)
        if n_samples == 0:
            raise ValueError("Feature matrix must have at least 1 sample")
        
        logger.info(f"Starting bootstrap confidence calculation: {n_samples} samples, "
                   f"{self.n_bootstrap} bootstrap iterations, {memory_start:.0f}MB memory")
        
        # Default equal weights if not provided
        if model_weights is None:
            model_weights = {name: 1.0 / len(models) for name in models.keys()}
        
        # Validate weights
        total_weight = sum(model_weights.values())
        if abs(total_weight - 1.0) > 0.001:
            raise ValueError(f"Model weights must sum to 1.0, got {total_weight}")
        
        n_samples = len(X)
        bootstrap_predictions = []
        
        # Parallel bootstrap sampling
        if self.n_jobs > 1:
            bootstrap_predictions = self._parallel_bootstrap(models, X, model_weights)
        else:
            bootstrap_predictions = self._sequential_bootstrap(models, X, model_weights)
        
        if not bootstrap_predictions:
            raise RuntimeError("All bootstrap samples failed")
        
        # Convert to numpy array for easier manipulation
        bootstrap_predictions = np.array(bootstrap_predictions)
        logger.debug(f"Bootstrap predictions shape: {bootstrap_predictions.shape}")
        
        # Calculate percentiles for confidence intervals
        alpha = 1 - self.confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        # Calculate confidence intervals for each sample
        mean_predictions = np.mean(bootstrap_predictions, axis=0)
        lower_bounds = np.percentile(bootstrap_predictions, lower_percentile, axis=0)
        upper_bounds = np.percentile(bootstrap_predictions, upper_percentile, axis=0)
        
        # Calculate uncertainty scores (width of confidence intervals)
        uncertainty_scores = upper_bounds - lower_bounds
        
        result = ConfidenceIntervalResult(
            predictions=mean_predictions,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            confidence_level=self.confidence_level,
            n_bootstrap=len(bootstrap_predictions),
            uncertainty_scores=uncertainty_scores
        )
        
        # Final memory monitoring
        memory_end = process.memory_info().rss / (1024 * 1024)
        memory_used = memory_end - memory_start
        
        if memory_end > self.max_memory_mb:
            logger.warning(f"Memory usage {memory_end:.0f}MB exceeded limit {self.max_memory_mb}MB")
        
        logger.info(f"Bootstrap confidence intervals completed: {n_samples} predictions, "
                   f"{len(bootstrap_predictions)} samples, {memory_used:.0f}MB used")
        logger.info(f"Mean uncertainty score: {np.mean(uncertainty_scores):.4f}")
        
        return result
    
    def _sequential_bootstrap(self, models: Dict[str, Any], X: pd.DataFrame,
                            model_weights: Dict[str, float]) -> List[np.ndarray]:
        """Sequential bootstrap sampling with thread-safe random generation"""
        bootstrap_predictions = []
        n_samples = len(X)
        failed_samples = []
        
        for i in range(self.n_bootstrap):
            try:
                # Generate bootstrap sample using thread-safe RNG
                bootstrap_indices = self.rng.choice(n_samples, size=n_samples, replace=True)
                X_bootstrap = X.iloc[bootstrap_indices]
                
                # Get ensemble prediction for this bootstrap sample
                ensemble_pred = self._get_ensemble_prediction(models, X_bootstrap, model_weights)
                
                # Validate prediction output
                if ensemble_pred is None or len(ensemble_pred) != n_samples:
                    raise ValueError(f"Invalid ensemble prediction shape: {ensemble_pred}")
                
                if np.any(np.isnan(ensemble_pred)) or np.any(np.isinf(ensemble_pred)):
                    raise ValueError("Ensemble prediction contains NaN or Inf values")
                
                bootstrap_predictions.append(ensemble_pred)
                
            except Exception as e:
                failed_samples.append((i, str(e)))
                logger.error(f"Bootstrap sample {i} failed: {str(e)}")
                # NO FALLBACKS - strict error handling
        
        # Strict validation - fail if too many samples failed
        failure_rate = len(failed_samples) / self.n_bootstrap
        if failure_rate > 0.05:  # Allow max 5% failure rate
            raise RuntimeError(
                f"Bootstrap sampling failed: {len(failed_samples)}/{self.n_bootstrap} samples failed "
                f"(failure rate: {failure_rate:.1%}). Failed samples: {failed_samples[:5]}"
            )
        
        if not bootstrap_predictions:
            raise RuntimeError("All bootstrap samples failed - no predictions generated")
        
        return bootstrap_predictions
    
    def _parallel_bootstrap(self, models: Dict[str, Any], X: pd.DataFrame,
                          model_weights: Dict[str, float]) -> List[np.ndarray]:
        """Parallel bootstrap sampling with strict error handling"""
        bootstrap_predictions = []
        n_samples = len(X)
        failed_tasks = []
        
        # Create tasks for parallel execution with unique seeds
        tasks = []
        with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
            for i in range(self.n_bootstrap):
                # Generate unique seed for each worker
                worker_seed = self.random_state + i * 10000
                
                # Submit task with unique seed
                future = executor.submit(
                    self._bootstrap_worker,
                    models, X, model_weights, i, worker_seed, n_samples
                )
                tasks.append((future, i))
            
            # Collect results with strict validation
            for future, task_id in tasks:
                try:
                    result = future.result(timeout=60)  # 1-minute timeout per task
                    
                    if result is None:
                        failed_tasks.append((task_id, "Worker returned None"))
                        continue
                    
                    # Validate result
                    if len(result) != n_samples:
                        failed_tasks.append((task_id, f"Invalid result length: {len(result)} != {n_samples}"))
                        continue
                    
                    if np.any(np.isnan(result)) or np.any(np.isinf(result)):
                        failed_tasks.append((task_id, "Result contains NaN or Inf values"))
                        continue
                    
                    bootstrap_predictions.append(result)
                    
                except Exception as e:
                    failed_tasks.append((task_id, str(e)))
                    logger.error(f"Parallel bootstrap task {task_id} failed: {str(e)}")
        
        # Strict error handling - fail if too many tasks failed
        total_tasks = self.n_bootstrap
        failure_rate = len(failed_tasks) / total_tasks
        
        if failure_rate > 0.05:  # Allow max 5% failure rate
            error_summary = "; ".join([f"Task {tid}: {err}" for tid, err in failed_tasks[:3]])
            raise RuntimeError(
                f"Parallel bootstrap failed: {len(failed_tasks)}/{total_tasks} tasks failed "
                f"(failure rate: {failure_rate:.1%}). Sample errors: {error_summary}"
            )
        
        if not bootstrap_predictions:
            raise RuntimeError("All parallel bootstrap tasks failed - no predictions generated")
        
        return bootstrap_predictions
    
    @staticmethod
    def _bootstrap_worker(models: Dict[str, Any], X: pd.DataFrame,
                         model_weights: Dict[str, float], task_id: int, 
                         worker_seed: int, n_samples: int) -> Optional[np.ndarray]:
        """Worker function for parallel bootstrap sampling with thread-safe RNG"""
        try:
            # Create thread-safe RNG for this worker
            worker_rng = np.random.Generator(np.random.PCG64(worker_seed))
            
            # Generate bootstrap indices
            bootstrap_indices = worker_rng.choice(n_samples, size=n_samples, replace=True)
            X_bootstrap = X.iloc[bootstrap_indices]
            
            # Get ensemble prediction
            result = BootstrapConfidenceCalculator._get_ensemble_prediction(
                models, X_bootstrap, model_weights
            )
            
            # Validate result before returning
            if result is None:
                raise ValueError("Ensemble prediction returned None")
            
            if len(result) != n_samples:
                raise ValueError(f"Invalid prediction length: {len(result)} != {n_samples}")
            
            if np.any(np.isnan(result)) or np.any(np.isinf(result)):
                raise ValueError("Prediction contains NaN or Inf values")
            
            return result
            
        except Exception as e:
            logger.error(f"Bootstrap worker {task_id} failed: {str(e)}")
            raise  # Re-raise to ensure error propagation
    
    @staticmethod
    def _get_ensemble_prediction(models: Dict[str, Any], X: pd.DataFrame,
                               model_weights: Dict[str, float]) -> np.ndarray:
        """Get weighted ensemble prediction from multiple models"""
        ensemble_pred = None
        
        for model_name, model in models.items():
            if model_name not in model_weights:
                continue
            
            weight = model_weights[model_name]
            
            # Get model prediction (probability of positive class)
            if hasattr(model, 'predict_proba'):
                pred = model.predict_proba(X)[:, 1]  # Probability of class 1
            else:
                pred = model.predict(X)  # Direct prediction
            
            # Weighted sum
            if ensemble_pred is None:
                ensemble_pred = weight * pred
            else:
                ensemble_pred += weight * pred
        
        return ensemble_pred
    
    def calculate_prediction_intervals(self, model: Any, X: pd.DataFrame,
                                     prediction_func: Optional[Callable] = None) -> ConfidenceIntervalResult:
        """
        Calculate prediction intervals for a single model
        
        Args:
            model: Trained model
            X: Feature matrix
            prediction_func: Optional custom prediction function
            
        Returns:
            ConfidenceIntervalResult: Confidence intervals for single model
        """
        if prediction_func is None:
            prediction_func = lambda m, x: m.predict_proba(x)[:, 1] if hasattr(m, 'predict_proba') else m.predict(x)
        
        models = {'model': model}
        model_weights = {'model': 1.0}
        
        return self.calculate_intervals(models, X, model_weights)
    
    def assess_prediction_reliability(self, confidence_result: ConfidenceIntervalResult,
                                    threshold: float = 0.1) -> Dict[str, Any]:
        """
        Assess prediction reliability based on confidence interval width
        
        Args:
            confidence_result: Result from calculate_intervals
            threshold: Uncertainty threshold for reliable predictions
            
        Returns:
            Dictionary with reliability assessment
        """
        uncertainty_scores = confidence_result.uncertainty_scores
        
        # Classify predictions by reliability
        reliable_mask = uncertainty_scores <= threshold
        unreliable_mask = uncertainty_scores > threshold
        
        n_reliable = np.sum(reliable_mask)
        n_unreliable = np.sum(unreliable_mask)
        total_predictions = len(uncertainty_scores)
        
        assessment = {
            'total_predictions': total_predictions,
            'reliable_predictions': int(n_reliable),
            'unreliable_predictions': int(n_unreliable),
            'reliability_rate': n_reliable / total_predictions,
            'mean_uncertainty': float(np.mean(uncertainty_scores)),
            'median_uncertainty': float(np.median(uncertainty_scores)),
            'max_uncertainty': float(np.max(uncertainty_scores)),
            'uncertainty_threshold': threshold,
            'reliable_indices': np.where(reliable_mask)[0].tolist(),
            'unreliable_indices': np.where(unreliable_mask)[0].tolist()
        }
        
        logger.info(f"Reliability assessment: {n_reliable}/{total_predictions} "
                   f"({assessment['reliability_rate']:.1%}) reliable predictions")
        
        return assessment
    
    def calibrate_intervals(self, true_outcomes: np.ndarray,
                          confidence_result: ConfidenceIntervalResult) -> Dict[str, float]:
        """
        Assess calibration of confidence intervals against true outcomes
        
        Args:
            true_outcomes: Actual binary outcomes (0 or 1)
            confidence_result: Predicted confidence intervals
            
        Returns:
            Dictionary with calibration metrics
        """
        predictions = confidence_result.predictions
        lower_bounds = confidence_result.lower_bounds
        upper_bounds = confidence_result.upper_bounds
        
        # Check how many true outcomes fall within confidence intervals
        within_intervals = ((true_outcomes >= lower_bounds) & 
                           (true_outcomes <= upper_bounds))
        coverage = np.mean(within_intervals)
        
        # Calculate interval widths
        interval_widths = upper_bounds - lower_bounds
        mean_width = np.mean(interval_widths)
        
        # Calculate prediction accuracy
        predicted_classes = (predictions > 0.5).astype(int)
        accuracy = np.mean(predicted_classes == true_outcomes)
        
        calibration_metrics = {
            'coverage': float(coverage),
            'expected_coverage': self.confidence_level,
            'coverage_error': float(abs(coverage - self.confidence_level)),
            'mean_interval_width': float(mean_width),
            'prediction_accuracy': float(accuracy),
            'calibration_score': float(1.0 - abs(coverage - self.confidence_level))
        }
        
        logger.info(f"Interval calibration: {coverage:.1%} coverage "
                   f"(expected {self.confidence_level:.1%})")
        
        return calibration_metrics


def create_ufc_confidence_calculator(n_bootstrap: int = 100,
                                   confidence_level: float = 0.95,
                                   n_jobs: int = -1) -> BootstrapConfidenceCalculator:
    """
    Factory function for UFC-specific confidence interval calculator
    
    Args:
        n_bootstrap: Number of bootstrap samples (100 for speed, 1000 for precision)
        confidence_level: Confidence level (0.95 for 95% intervals)
        n_jobs: Parallel jobs (-1 for all cores)
        
    Returns:
        BootstrapConfidenceCalculator: Configured calculator
    """
    return BootstrapConfidenceCalculator(
        confidence_level=confidence_level,
        n_bootstrap=n_bootstrap,
        n_jobs=n_jobs,
        random_state=42
    )


def quick_confidence_intervals(models: Dict[str, Any], X: pd.DataFrame,
                             model_weights: Optional[Dict[str, float]] = None,
                             n_bootstrap: int = 50) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Quick calculation of confidence intervals with minimal setup
    
    Args:
        models: Dictionary of trained models
        X: Feature matrix
        model_weights: Optional model weights
        n_bootstrap: Number of bootstrap samples (reduced for speed)
        
    Returns:
        Tuple of (predictions, lower_bounds, upper_bounds)
    """
    calculator = BootstrapConfidenceCalculator(n_bootstrap=n_bootstrap, n_jobs=1)
    result = calculator.calculate_intervals(models, X, model_weights)
    
    return result.predictions, result.lower_bounds, result.upper_bounds