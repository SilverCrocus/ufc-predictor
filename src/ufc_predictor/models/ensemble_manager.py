"""
UFC Ensemble Manager for Phase 2A Enhanced ML Pipeline

Coordinates Random Forest and XGBoost models with bootstrap confidence intervals
and data quality integration for improved UFC fight predictions.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import logging
from dataclasses import dataclass
from configs.model_config import ENSEMBLE_CONFIG
import joblib
import psutil
import gc
import time

logger = logging.getLogger(__name__)


@dataclass
class EnsemblePrediction:
    """Result from ensemble prediction with confidence intervals"""
    fighter_a: str
    fighter_b: str
    ensemble_probability: float
    predicted_winner: str
    confidence_interval: Tuple[float, float]
    data_quality_score: float
    model_breakdown: Dict[str, float]
    uncertainty_score: float


class UFCEnsembleManager:
    """
    Specialized ensemble manager for UFC predictions with 40% RF + 35% XGBoost + 25% NN
    Integrates with Phase 2A data quality confidence scoring
    """
    
    def __init__(self, data_confidence_scorer=None, config: Dict = None):
        """
        Initialize ensemble manager with memory monitoring
        
        Args:
            data_confidence_scorer: Optional scorer for data quality assessment
            config: Optional configuration override
        """
        self.config = config or ENSEMBLE_CONFIG
        self.models = {}
        self.ensemble_weights = self.config['weights']
        self.data_confidence_scorer = data_confidence_scorer
        self.confidence_intervals = {}
        self.is_trained = False
        
        # Memory management
        self.max_memory_mb = self.config.get('max_memory_mb', 4096)
        self.memory_monitor = psutil.Process()
        self.prediction_count = 0
        self.total_processing_time = 0.0
        
        logger.info(f"Initialized UFC Ensemble Manager with weights: {self.ensemble_weights}, "
                   f"memory limit: {self.max_memory_mb} MB")
    
    def add_trained_models(self, rf_model, xgb_model, nn_model=None):
        """Add pre-trained models to ensemble"""
        self.models['random_forest'] = rf_model
        self.models['xgboost'] = xgb_model
        
        if nn_model:
            self.models['neural_network'] = nn_model
        else:
            logger.info("Neural network not provided, adjusting weights for RF+XGBoost only")
            # Adjust weights when neural network is not available
            total_weight = self.ensemble_weights['random_forest'] + self.ensemble_weights['xgboost']
            self.ensemble_weights['random_forest'] = self.ensemble_weights['random_forest'] / total_weight
            self.ensemble_weights['xgboost'] = self.ensemble_weights['xgboost'] / total_weight
        
        self.is_trained = True
        logger.info(f"Added models to ensemble: {list(self.models.keys())}")
    
    def load_models_from_paths(self, rf_path: str, xgb_path: str, nn_path: Optional[str] = None):
        """Load trained models from file paths"""
        try:
            self.models['random_forest'] = joblib.load(rf_path)
            logger.info(f"Loaded Random Forest model from {rf_path}")
            
            self.models['xgboost'] = joblib.load(xgb_path)
            logger.info(f"Loaded XGBoost model from {xgb_path}")
            
            if nn_path:
                self.models['neural_network'] = joblib.load(nn_path)
                logger.info(f"Loaded Neural Network model from {nn_path}")
            else:
                # Adjust weights for RF+XGBoost only
                total_weight = self.ensemble_weights['random_forest'] + self.ensemble_weights['xgboost']
                self.ensemble_weights['random_forest'] = self.ensemble_weights['random_forest'] / total_weight
                self.ensemble_weights['xgboost'] = self.ensemble_weights['xgboost'] / total_weight
                logger.info("Neural network not loaded, adjusted weights for RF+XGBoost")
            
            self.is_trained = True
            return True
            
        except Exception as e:
            logger.error(f"Failed to load models: {str(e)}")
            return False
    
    def predict_with_confidence(self, X: pd.DataFrame, 
                              fighter_pairs: List[Tuple[str, str]],
                              bootstrap_samples: Optional[int] = None) -> List[EnsemblePrediction]:
        """Generate ensemble predictions with bootstrap confidence intervals"""
        
        if not self.is_trained:
            raise ValueError("Models not loaded. Call add_trained_models() or load_models_from_paths() first.")
        
        start_time = time.time()
        
        # Check memory before starting
        self._check_memory_usage("before prediction")
        
        # Strict input validation with NO fallbacks
        from ufc_predictor.utils.validation import validate_ufc_prediction_dataframe, validate_ufc_fighter_pair, DataFrameValidationError, FighterNameValidationError
        
        # Validate DataFrame structure and content
        try:
            validated_X = validate_ufc_prediction_dataframe(X, strict_mode=True)
            logger.debug(f"Input DataFrame validated: {len(validated_X)} rows, {len(validated_X.columns)} columns")
        except DataFrameValidationError as e:
            raise DataFrameValidationError(f"Ensemble input validation failed: {e}")
        
        # Validate fighter pairs
        try:
            validated_pairs = []
            for i, (fighter_a, fighter_b) in enumerate(fighter_pairs):
                validated_a, validated_b = validate_ufc_fighter_pair(fighter_a, fighter_b, strict_mode=True)
                validated_pairs.append((validated_a, validated_b))
            
            fighter_pairs = validated_pairs
            logger.debug(f"Fighter pairs validated: {len(fighter_pairs)} pairs")
        except FighterNameValidationError as e:
            raise FighterNameValidationError(f"Fighter pair validation failed: {e}")
        
        # Validate consistency between DataFrame and fighter pairs
        if len(validated_X) != len(fighter_pairs):
            raise ValueError(f"Mismatch between DataFrame rows ({len(validated_X)}) and fighter pairs ({len(fighter_pairs)})")
        
        bootstrap_samples = bootstrap_samples or self.config['bootstrap_samples']
        
        logger.info(f"Generating ensemble predictions for {len(fighter_pairs)} fights with {bootstrap_samples} bootstrap samples")
        
        # Get individual model predictions using validated data
        rf_proba = self.models['random_forest'].predict_proba(validated_X)
        xgb_proba = self.models['xgboost'].predict_proba(validated_X)
        
        # Handle neural network if available
        nn_proba = None
        if 'neural_network' in self.models:
            nn_proba = self.models['neural_network'].predict_proba(validated_X)
        
        # Calculate weighted ensemble prediction
        ensemble_proba = self._calculate_weighted_ensemble(rf_proba, xgb_proba, nn_proba)
        
        # Check memory after model predictions
        self._check_memory_usage("after model predictions")
        
        # Generate bootstrap confidence intervals
        confidence_intervals = self._bootstrap_confidence_intervals(validated_X, bootstrap_samples)
        
        # Check memory after bootstrap
        self._check_memory_usage("after bootstrap")
        
        # Assess data quality
        data_quality_scores = self._assess_data_quality(validated_X, fighter_pairs)
        
        # Format results
        results = []
        for i, (fighter_a, fighter_b) in enumerate(fighter_pairs):
            # Calculate uncertainty score from confidence interval width
            ci_lower, ci_upper = confidence_intervals[i]
            uncertainty_score = ci_upper - ci_lower
            
            # Determine predicted winner
            prob_fighter_b = ensemble_proba[i, 1]
            predicted_winner = fighter_b if prob_fighter_b > 0.5 else fighter_a
            
            # Create model breakdown
            model_breakdown = {
                'random_forest': float(rf_proba[i, 1]),
                'xgboost': float(xgb_proba[i, 1]),
                'weights_applied': dict(self.ensemble_weights)
            }
            
            if nn_proba is not None:
                model_breakdown['neural_network'] = float(nn_proba[i, 1])
            
            result = EnsemblePrediction(
                fighter_a=fighter_a,
                fighter_b=fighter_b,
                ensemble_probability=float(prob_fighter_b),
                predicted_winner=predicted_winner,
                confidence_interval=confidence_intervals[i],
                data_quality_score=data_quality_scores[i],
                model_breakdown=model_breakdown,
                uncertainty_score=uncertainty_score
            )
            
            results.append(result)
        
        # Update performance metrics
        processing_time = time.time() - start_time
        self.prediction_count += len(results)
        self.total_processing_time += processing_time
        
        # Final memory check
        self._check_memory_usage("after prediction complete")
        
        logger.info(f"Generated {len(results)} ensemble predictions in {processing_time:.2f}s")
        return results
    
    def _calculate_weighted_ensemble(self, rf_proba, xgb_proba, nn_proba=None):
        """Calculate weighted ensemble prediction from individual model probabilities"""
        ensemble_proba = (
            rf_proba * self.ensemble_weights['random_forest'] +
            xgb_proba * self.ensemble_weights['xgboost']
        )
        
        if nn_proba is not None:
            ensemble_proba += nn_proba * self.ensemble_weights['neural_network']
        
        return ensemble_proba
    
    def _bootstrap_confidence_intervals(self, X: pd.DataFrame, 
                                      n_bootstrap: int) -> List[Tuple[float, float]]:
        """Generate bootstrap confidence intervals for predictions"""
        logger.debug(f"Calculating bootstrap confidence intervals with {n_bootstrap} samples")
        
        n_samples = len(X)
        bootstrap_predictions = []
        
        for i in range(n_bootstrap):
            # Bootstrap sample indices
            bootstrap_indices = np.random.choice(n_samples, size=n_samples, replace=True)
            X_bootstrap = X.iloc[bootstrap_indices]
            
            try:
                # Get predictions on bootstrap sample
                rf_proba = self.models['random_forest'].predict_proba(X_bootstrap)
                xgb_proba = self.models['xgboost'].predict_proba(X_bootstrap)
                
                nn_proba = None
                if 'neural_network' in self.models:
                    nn_proba = self.models['neural_network'].predict_proba(X_bootstrap)
                
                # Calculate weighted ensemble
                ensemble_proba = self._calculate_weighted_ensemble(rf_proba, xgb_proba, nn_proba)
                bootstrap_predictions.append(ensemble_proba[:, 1])
                
            except Exception as e:
                logger.warning(f"Bootstrap sample {i} failed: {str(e)}")
                continue
        
        if not bootstrap_predictions:
            raise RuntimeError(
                "All bootstrap samples failed - cannot generate confidence intervals. "
                "Check model inputs and data quality."
            )
        
        # Calculate confidence intervals
        bootstrap_predictions = np.array(bootstrap_predictions)
        confidence_intervals = []
        confidence_level = self.config['confidence_level']
        alpha = 1 - confidence_level
        
        for i in range(n_samples):
            sample_predictions = bootstrap_predictions[:, i]
            lower_bound = np.percentile(sample_predictions, (alpha / 2) * 100)
            upper_bound = np.percentile(sample_predictions, (1 - alpha / 2) * 100)
            confidence_intervals.append((float(lower_bound), float(upper_bound)))
        
        logger.debug(f"Generated {len(confidence_intervals)} confidence intervals")
        return confidence_intervals
    
    def _assess_data_quality(self, X: pd.DataFrame, 
                           fighter_pairs: List[Tuple[str, str]]) -> List[float]:
        """Assess data quality for each prediction"""
        if self.data_confidence_scorer:
            return self.data_confidence_scorer.score_predictions(X, fighter_pairs)
        
        # Default quality assessment based on missing values
        data_quality_scores = []
        for i in range(len(fighter_pairs)):
            row = X.iloc[i]
            missing_ratio = row.isnull().sum() / len(row)
            quality_score = max(0.1, 1.0 - missing_ratio)  # Minimum 10% quality
            data_quality_scores.append(quality_score)
        
        return data_quality_scores
    
    def _check_memory_usage(self, context: str = "") -> None:
        """Check and enforce memory limits"""
        memory_info = self.memory_monitor.memory_info()
        memory_mb = memory_info.rss / 1024 / 1024
        memory_percent = (memory_mb / self.max_memory_mb) * 100
        
        logger.debug(f"Memory usage {context}: {memory_mb:.1f} MB ({memory_percent:.1f}%)")
        
        if memory_mb > self.max_memory_mb:
            gc.collect()  # Force garbage collection
            
            # Check again after garbage collection
            memory_info = self.memory_monitor.memory_info()
            memory_mb = memory_info.rss / 1024 / 1024
            
            if memory_mb > self.max_memory_mb:
                raise RuntimeError(
                    f"Memory limit exceeded {context}: {memory_mb:.1f} MB > {self.max_memory_mb} MB. "
                    "Reduce batch size or bootstrap samples."
                )
        
        if memory_percent > 90:
            logger.warning(f"High memory usage {context}: {memory_percent:.1f}%")
    
    def get_model_weights(self) -> Dict[str, float]:
        """Get current ensemble weights"""
        return dict(self.ensemble_weights)
    
    def update_weights(self, new_weights: Dict[str, float]):
        """Update ensemble weights"""
        total_weight = sum(new_weights.values())
        if abs(total_weight - 1.0) > 0.01:
            raise ValueError(f"Weights must sum to 1.0, got {total_weight}")
        
        self.ensemble_weights.update(new_weights)
        logger.info(f"Updated ensemble weights: {self.ensemble_weights}")
    
    def get_ensemble_summary(self) -> Dict[str, Any]:
        """Get summary of ensemble configuration and status"""
        return {
            'models_loaded': list(self.models.keys()),
            'ensemble_weights': dict(self.ensemble_weights),
            'is_trained': self.is_trained,
            'config': dict(self.config),
            'available_methods': [
                'predict_with_confidence',
                'bootstrap_confidence_intervals',
                'data_quality_assessment'
            ]
        }


def create_default_ensemble_manager(rf_model, xgb_model, nn_model=None, 
                                   data_confidence_scorer=None) -> UFCEnsembleManager:
    """
    Factory function to create a default UFC ensemble manager
    
    Args:
        rf_model: Trained Random Forest model
        xgb_model: Trained XGBoost model
        nn_model: Optional trained Neural Network model
        data_confidence_scorer: Optional data quality scorer
        
    Returns:
        UFCEnsembleManager: Configured ensemble manager
    """
    manager = UFCEnsembleManager(data_confidence_scorer=data_confidence_scorer)
    manager.add_trained_models(rf_model, xgb_model, nn_model)
    return manager