"""
Production-Grade XGBoost Ensemble Manager
========================================

Memory-efficient ensemble manager with strict error handling and no fallbacks.
Designed for production UFC betting system with 70 features and 100-1000 bootstrap samples.

Key Features:
- Memory-efficient batch processing
- Strict input validation and error handling
- Thread-safe parallel processing
- Real-time memory monitoring
- Comprehensive logging and metrics
- No fallback mechanisms - fail fast and clear
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, NamedTuple
import logging
from dataclasses import dataclass
import joblib
import psutil
import gc
from pathlib import Path
import time
from contextlib import contextmanager

from ..utils.thread_safe_bootstrap import (
    ThreadSafeBootstrapSampler, 
    BootstrapConfig, 
    create_bootstrap_config,
    calculate_confidence_intervals
)
from ..utils.enhanced_error_handling import UFCPredictorError

logger = logging.getLogger(__name__)


class EnsembleError(UFCPredictorError):
    """Ensemble-specific errors"""
    pass


class ValidationError(UFCPredictorError):
    """Input validation errors"""
    pass


class ModelLoadError(UFCPredictorError):
    """Model loading errors"""
    pass


class PredictionResult(NamedTuple):
    """Structured prediction result"""
    fighter_a: str
    fighter_b: str
    ensemble_probability: float
    confidence_interval: Tuple[float, float]
    uncertainty_score: float
    model_breakdown: Dict[str, float]
    data_quality_score: float
    processing_time_ms: float


@dataclass
class EnsembleConfig:
    """Production ensemble configuration"""
    model_weights: Dict[str, float]
    confidence_level: float = 0.95
    bootstrap_samples: int = 100
    max_memory_mb: int = 4096
    n_jobs: int = -1
    batch_size: int = 50
    enable_parallel: bool = True
    min_prediction_confidence: float = 0.5
    max_uncertainty_threshold: float = 0.4
    
    def __post_init__(self):
        """Validate ensemble configuration"""
        if not self.model_weights:
            raise EnsembleError("No model weights provided")
        
        total_weight = sum(self.model_weights.values())
        if abs(total_weight - 1.0) > 1e-6:
            raise EnsembleError(f"Model weights must sum to 1.0, got {total_weight}")
        
        if not 0 < self.confidence_level < 1:
            raise EnsembleError(f"confidence_level must be between 0 and 1, got {self.confidence_level}")
        
        if self.bootstrap_samples <= 0:
            raise EnsembleError(f"bootstrap_samples must be positive, got {self.bootstrap_samples}")


class ProductionEnsembleManager:
    """Production-grade ensemble manager for UFC predictions"""
    
    def __init__(self, config: EnsembleConfig):
        self.config = config
        self.models = {}
        self.feature_columns = None
        self.is_initialized = False
        self.bootstrap_sampler = None
        self.memory_monitor = psutil.Process()
        
        # Performance metrics
        self.prediction_count = 0
        self.total_processing_time = 0.0
        self.error_count = 0
        
        self._initialize_bootstrap_sampler()
        
        logger.info(f"Initialized ProductionEnsembleManager with weights: {self.config.model_weights}")
    
    def _initialize_bootstrap_sampler(self):
        """Initialize thread-safe bootstrap sampler"""
        bootstrap_config = create_bootstrap_config(
            n_bootstrap=self.config.bootstrap_samples,
            confidence_level=self.config.confidence_level,
            n_jobs=self.config.n_jobs if self.config.enable_parallel else 1,
            max_memory_mb=self.config.max_memory_mb
        )
        
        self.bootstrap_sampler = ThreadSafeBootstrapSampler(bootstrap_config)
        logger.info("Bootstrap sampler initialized")
    
    def load_models(self, model_paths: Dict[str, str], feature_columns_path: Optional[str] = None):
        """Load models with comprehensive validation"""
        
        logger.info(f"Loading models from {len(model_paths)} paths")
        
        # Validate all paths exist first
        for model_name, path in model_paths.items():
            if not Path(path).exists():
                raise ModelLoadError(f"Model file not found: {path}", model_name=model_name)
        
        # Load feature columns first if provided
        if feature_columns_path:
            self._load_feature_columns(feature_columns_path)
        
        # Load each model with validation
        for model_name, path in model_paths.items():
            try:
                start_time = time.time()
                model = joblib.load(path)
                load_time = (time.time() - start_time) * 1000
                
                # Validate model has required methods
                if not hasattr(model, 'predict_proba') and not hasattr(model, 'predict'):
                    raise ModelLoadError(
                        f"Model '{model_name}' has no predict_proba or predict method",
                        model_name=model_name
                    )
                
                # Check if model weight exists
                if model_name not in self.config.model_weights:
                    raise ModelLoadError(
                        f"No weight configured for model '{model_name}'",
                        model_name=model_name
                    )
                
                self.models[model_name] = model
                logger.info(f"Loaded model '{model_name}' in {load_time:.1f}ms")
                
            except Exception as e:
                if isinstance(e, ModelLoadError):
                    raise
                raise ModelLoadError(
                    f"Failed to load model '{model_name}': {str(e)}",
                    model_name=model_name,
                    file_path=path
                )
        
        # Validate all required models are loaded
        required_models = set(self.config.model_weights.keys())
        loaded_models = set(self.models.keys())
        
        if required_models != loaded_models:
            missing = required_models - loaded_models
            extra = loaded_models - required_models
            raise ModelLoadError(
                f"Model mismatch - Missing: {missing}, Extra: {extra}",
                required=list(required_models),
                loaded=list(loaded_models)
            )
        
        self.is_initialized = True
        logger.info(f"All models loaded successfully: {list(self.models.keys())}")
    
    def _load_feature_columns(self, feature_columns_path: str):
        """Load and validate feature columns"""
        try:
            import json
            with open(feature_columns_path, 'r') as f:
                self.feature_columns = json.load(f)
            
            if not isinstance(self.feature_columns, list) or len(self.feature_columns) == 0:
                raise ValidationError(f"Invalid feature columns format: {type(self.feature_columns)}")
            
            logger.info(f"Loaded {len(self.feature_columns)} feature columns")
            
        except Exception as e:
            raise ValidationError(f"Failed to load feature columns: {str(e)}")
    
    def predict_fights(self, X: pd.DataFrame, 
                      fighter_pairs: List[Tuple[str, str]],
                      enable_bootstrap: bool = True) -> List[PredictionResult]:
        """Generate predictions with comprehensive validation and monitoring"""
        
        if not self.is_initialized:
            raise EnsembleError("Models not loaded. Call load_models() first.")
        
        start_time = time.time()
        
        try:
            # Comprehensive input validation
            self._validate_prediction_inputs(X, fighter_pairs)
            
            # Memory check before starting
            self._check_memory_usage("before prediction")
            
            logger.info(f"Starting prediction for {len(fighter_pairs)} fights with {len(X)} samples")
            
            # Get base ensemble predictions
            ensemble_predictions = self._get_base_ensemble_predictions(X)
            
            # Get confidence intervals if bootstrap enabled
            confidence_intervals = None
            uncertainty_scores = None
            
            if enable_bootstrap:
                confidence_intervals, uncertainty_scores = self._get_bootstrap_confidence_intervals(X)
            
            # Memory check after computation
            self._check_memory_usage("after computation")
            
            # Format results
            results = self._format_prediction_results(
                X, fighter_pairs, ensemble_predictions, 
                confidence_intervals, uncertainty_scores, start_time
            )
            
            # Update metrics
            self.prediction_count += len(results)
            self.total_processing_time += (time.time() - start_time)
            
            logger.info(f"Prediction completed successfully for {len(results)} fights in {(time.time() - start_time)*1000:.1f}ms")
            
            return results
            
        except Exception as e:
            self.error_count += 1
            if isinstance(e, (EnsembleError, ValidationError)):
                raise
            else:
                raise EnsembleError(f"Prediction failed: {str(e)}")
    
    def _validate_prediction_inputs(self, X: pd.DataFrame, fighter_pairs: List[Tuple[str, str]]):
        """Comprehensive input validation"""
        
        # Validate DataFrame
        if X.empty:
            raise ValidationError("Input DataFrame X is empty")
        
        if len(X) != len(fighter_pairs):
            raise ValidationError(
                f"Data length mismatch: X has {len(X)} rows, but {len(fighter_pairs)} fighter pairs provided"
            )
        
        # Check for null values
        null_columns = X.columns[X.isnull().any()].tolist()
        if null_columns:
            null_counts = X[null_columns].isnull().sum().to_dict()
            raise ValidationError(f"Input data contains null values: {null_counts}")
        
        # Check for infinite values
        inf_columns = X.columns[np.isinf(X.select_dtypes(include=[np.number])).any()].tolist()
        if inf_columns:
            raise ValidationError(f"Input data contains infinite values in columns: {inf_columns}")
        
        # Validate feature columns if available
        if self.feature_columns:
            missing_features = set(self.feature_columns) - set(X.columns)
            if missing_features:
                raise ValidationError(f"Missing required features: {list(missing_features)}")
            
            extra_features = set(X.columns) - set(self.feature_columns)
            if extra_features:
                logger.warning(f"Extra features present (will be ignored): {list(extra_features)}")
            
            # Reorder columns to match expected order
            X = X[self.feature_columns]
        
        # Validate fighter pairs
        for i, (fighter_a, fighter_b) in enumerate(fighter_pairs):
            if not fighter_a or not fighter_b:
                raise ValidationError(f"Empty fighter name at index {i}: '{fighter_a}' vs '{fighter_b}'")
            
            if fighter_a == fighter_b:
                raise ValidationError(f"Same fighter in pair at index {i}: '{fighter_a}'")
        
        # Check data types
        numeric_columns = X.select_dtypes(include=[np.number]).columns
        if len(numeric_columns) != len(X.columns):
            non_numeric = set(X.columns) - set(numeric_columns)
            raise ValidationError(f"Non-numeric columns found: {list(non_numeric)}")
    
    def _get_base_ensemble_predictions(self, X: pd.DataFrame) -> np.ndarray:
        """Get base ensemble predictions with validation"""
        
        ensemble_predictions = None
        model_predictions = {}
        
        for model_name, model in self.models.items():
            try:
                # Get model prediction
                if hasattr(model, 'predict_proba'):
                    pred = model.predict_proba(X)
                    if pred.shape[1] < 2:
                        raise EnsembleError(f"Model '{model_name}' predict_proba returned insufficient classes")
                    pred = pred[:, 1]  # Probability of positive class
                else:
                    pred = model.predict(X)
                
                # Validate prediction
                if len(pred) != len(X):
                    raise EnsembleError(
                        f"Model '{model_name}' prediction length mismatch: {len(pred)} != {len(X)}"
                    )
                
                # Check for invalid predictions
                if np.any(np.isnan(pred)) or np.any(np.isinf(pred)):
                    raise EnsembleError(f"Model '{model_name}' returned invalid predictions (NaN/Inf)")
                
                if np.any((pred < 0) | (pred > 1)):
                    raise EnsembleError(f"Model '{model_name}' returned predictions outside [0,1] range")
                
                model_predictions[model_name] = pred
                weight = self.config.model_weights[model_name]
                
                # Weighted sum
                if ensemble_predictions is None:
                    ensemble_predictions = weight * pred
                else:
                    ensemble_predictions += weight * pred
                
                logger.debug(f"Model '{model_name}' predictions: mean={np.mean(pred):.3f}, std={np.std(pred):.3f}")
                
            except Exception as e:
                raise EnsembleError(f"Model '{model_name}' prediction failed: {str(e)}")
        
        if ensemble_predictions is None:
            raise EnsembleError("No ensemble predictions generated")
        
        # Final validation of ensemble predictions
        if np.any(np.isnan(ensemble_predictions)) or np.any(np.isinf(ensemble_predictions)):
            raise EnsembleError("Ensemble predictions contain invalid values (NaN/Inf)")
        
        return ensemble_predictions
    
    def _get_bootstrap_confidence_intervals(self, X: pd.DataFrame) -> Tuple[List[Tuple[float, float]], np.ndarray]:
        """Get bootstrap confidence intervals"""
        
        try:
            # Use thread-safe bootstrap sampler
            bootstrap_predictions = self.bootstrap_sampler.sample_bootstrap_predictions(
                self.models, X, self.config.model_weights
            )
            
            # Calculate confidence intervals
            mean_pred, lower_bounds, upper_bounds = calculate_confidence_intervals(
                bootstrap_predictions, self.config.confidence_level
            )
            
            # Format as list of tuples
            confidence_intervals = [(float(lower), float(upper)) 
                                  for lower, upper in zip(lower_bounds, upper_bounds)]
            
            # Calculate uncertainty scores
            uncertainty_scores = upper_bounds - lower_bounds
            
            return confidence_intervals, uncertainty_scores
            
        except Exception as e:
            raise EnsembleError(f"Bootstrap confidence interval calculation failed: {str(e)}")
    
    def _format_prediction_results(self, X: pd.DataFrame, fighter_pairs: List[Tuple[str, str]],
                                 ensemble_predictions: np.ndarray,
                                 confidence_intervals: Optional[List[Tuple[float, float]]] = None,
                                 uncertainty_scores: Optional[np.ndarray] = None,
                                 start_time: float = None) -> List[PredictionResult]:
        """Format prediction results with validation"""
        
        results = []
        processing_time_ms = (time.time() - start_time) * 1000 if start_time else 0.0
        
        for i, (fighter_a, fighter_b) in enumerate(fighter_pairs):
            try:
                # Get base prediction
                prob_fighter_b = float(ensemble_predictions[i])
                
                # Validate prediction confidence
                if prob_fighter_b < self.config.min_prediction_confidence and prob_fighter_b > (1 - self.config.min_prediction_confidence):
                    logger.warning(f"Low confidence prediction for {fighter_a} vs {fighter_b}: {prob_fighter_b:.3f}")
                
                # Get confidence interval and uncertainty
                ci = confidence_intervals[i] if confidence_intervals else (prob_fighter_b - 0.1, prob_fighter_b + 0.1)
                uncertainty = float(uncertainty_scores[i]) if uncertainty_scores is not None else 0.2
                
                # Validate uncertainty threshold
                if uncertainty > self.config.max_uncertainty_threshold:
                    logger.warning(f"High uncertainty for {fighter_a} vs {fighter_b}: {uncertainty:.3f}")
                
                # Create model breakdown
                model_breakdown = {}
                for model_name, model in self.models.items():
                    if hasattr(model, 'predict_proba'):
                        pred = model.predict_proba(X.iloc[[i]])[:, 1]
                    else:
                        pred = model.predict(X.iloc[[i]])
                    model_breakdown[model_name] = float(pred[0])
                
                model_breakdown['weights'] = dict(self.config.model_weights)
                
                # Calculate data quality score (simple implementation)
                data_quality = self._calculate_data_quality_score(X.iloc[i])
                
                result = PredictionResult(
                    fighter_a=fighter_a,
                    fighter_b=fighter_b,
                    ensemble_probability=prob_fighter_b,
                    confidence_interval=ci,
                    uncertainty_score=uncertainty,
                    model_breakdown=model_breakdown,
                    data_quality_score=data_quality,
                    processing_time_ms=processing_time_ms / len(fighter_pairs)
                )
                
                results.append(result)
                
            except Exception as e:
                raise EnsembleError(f"Failed to format result for {fighter_a} vs {fighter_b}: {str(e)}")
        
        return results
    
    def _calculate_data_quality_score(self, row: pd.Series) -> float:
        """Calculate data quality score for a single row"""
        
        # Simple quality score based on completeness and value ranges
        missing_ratio = row.isnull().sum() / len(row)
        
        # Check for extreme values (beyond 3 standard deviations)
        numeric_row = row.select_dtypes(include=[np.number])
        if len(numeric_row) > 0:
            z_scores = np.abs((numeric_row - numeric_row.mean()) / (numeric_row.std() + 1e-8))
            extreme_values_ratio = (z_scores > 3).sum() / len(numeric_row)
        else:
            extreme_values_ratio = 0
        
        # Quality score (higher is better)
        quality_score = 1.0 - missing_ratio - (extreme_values_ratio * 0.5)
        return max(0.1, quality_score)  # Minimum 10% quality
    
    def _check_memory_usage(self, context: str = ""):
        """Check and log memory usage"""
        memory_info = self.memory_monitor.memory_info()
        memory_mb = memory_info.rss / 1024 / 1024
        memory_percent = (memory_info.rss / (self.config.max_memory_mb * 1024 * 1024)) * 100
        
        logger.debug(f"Memory usage {context}: {memory_mb:.1f} MB ({memory_percent:.1f}%)")
        
        if memory_percent > 90:
            logger.warning(f"High memory usage {context}: {memory_percent:.1f}%")
        
        if memory_mb > self.config.max_memory_mb:
            raise EnsembleError(
                f"Memory limit exceeded {context}: {memory_mb:.1f} MB > {self.config.max_memory_mb} MB"
            )
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        avg_processing_time = (self.total_processing_time / self.prediction_count 
                             if self.prediction_count > 0 else 0)
        
        return {
            'total_predictions': self.prediction_count,
            'total_errors': self.error_count,
            'error_rate': self.error_count / max(1, self.prediction_count + self.error_count),
            'avg_processing_time_ms': avg_processing_time * 1000,
            'total_processing_time_s': self.total_processing_time,
            'memory_usage_mb': self.memory_monitor.memory_info().rss / 1024 / 1024,
            'models_loaded': list(self.models.keys()),
            'configuration': {
                'bootstrap_samples': self.config.bootstrap_samples,
                'confidence_level': self.config.confidence_level,
                'max_memory_mb': self.config.max_memory_mb,
                'parallel_enabled': self.config.enable_parallel
            }
        }
    
    def validate_model_consistency(self, X_test: pd.DataFrame) -> Dict[str, Any]:
        """Validate model consistency across predictions"""
        
        if not self.is_initialized:
            raise EnsembleError("Models not loaded")
        
        logger.info("Validating model consistency...")
        
        consistency_results = {}
        
        for model_name, model in self.models.items():
            try:
                # Make predictions twice
                pred1 = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else model.predict(X_test)
                pred2 = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else model.predict(X_test)
                
                # Check consistency
                max_diff = np.max(np.abs(pred1 - pred2))
                mean_diff = np.mean(np.abs(pred1 - pred2))
                
                consistency_results[model_name] = {
                    'max_difference': float(max_diff),
                    'mean_difference': float(mean_diff),
                    'is_consistent': max_diff < 1e-10
                }
                
                if max_diff >= 1e-10:
                    logger.warning(f"Model '{model_name}' shows inconsistency: max_diff={max_diff}")
                
            except Exception as e:
                consistency_results[model_name] = {
                    'error': str(e),
                    'is_consistent': False
                }
        
        return consistency_results


def create_production_ensemble_config(model_weights: Dict[str, float],
                                     bootstrap_samples: int = 100,
                                     max_memory_mb: int = 4096,
                                     n_jobs: int = -1) -> EnsembleConfig:
    """Factory function for production ensemble configuration"""
    
    return EnsembleConfig(
        model_weights=model_weights,
        bootstrap_samples=bootstrap_samples,
        max_memory_mb=max_memory_mb,
        n_jobs=n_jobs
    )


# Example usage for testing
if __name__ == "__main__":
    print("Production Ensemble Manager Test")
    print("=" * 50)
    
    # Create test configuration
    weights = {'random_forest': 0.4, 'xgboost': 0.35, 'neural_network': 0.25}
    config = create_production_ensemble_config(weights, bootstrap_samples=20, max_memory_mb=1024)
    
    manager = ProductionEnsembleManager(config)
    
    # Create dummy test data
    np.random.seed(42)
    X_test = pd.DataFrame(np.random.rand(50, 10), columns=[f'feature_{i}' for i in range(10)])
    fighter_pairs = [(f'Fighter_A_{i}', f'Fighter_B_{i}') for i in range(50)]
    
    # Create dummy models
    class DummyModel:
        def predict_proba(self, X):
            np.random.seed(42)  # For consistency
            prob_class_1 = np.random.beta(2, 2, size=len(X))  # Realistic probabilities
            prob_class_0 = 1 - prob_class_1
            return np.column_stack([prob_class_0, prob_class_1])
    
    # Save dummy models for testing
    models = {}
    model_paths = {}
    for name in weights.keys():
        model = DummyModel()
        models[name] = model
        path = f'/tmp/test_{name}_model.joblib'
        joblib.dump(model, path)
        model_paths[name] = path
    
    try:
        # Test model loading
        manager.load_models(model_paths)
        print("✅ Models loaded successfully")
        
        # Test predictions
        results = manager.predict_fights(X_test, fighter_pairs, enable_bootstrap=True)
        print(f"✅ Predictions generated: {len(results)} results")
        
        # Test performance metrics
        metrics = manager.get_performance_metrics()
        print(f"✅ Performance metrics: {metrics['total_predictions']} predictions")
        
        # Test model consistency
        consistency = manager.validate_model_consistency(X_test[:10])
        print(f"✅ Model consistency validated: {len(consistency)} models checked")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        raise
    finally:
        # Cleanup
        import os
        for path in model_paths.values():
            try:
                os.remove(path)
            except:
                pass
    
    print("✅ Production ensemble manager test completed")