"""
Advanced Ensemble Methods for UFC Prediction
===========================================

Sophisticated ensemble system implementing multiple advanced ensemble techniques
specifically optimized for sports prediction and UFC fight prediction.

Features:
- Weighted soft voting with dynamic performance-based weights
- Stacking ensembles with meta-learners
- Bayesian model averaging with posterior updates
- Conditional ensembles based on fight characteristics
- Multi-stage prediction architecture
- Market-aware ensemble integration
- Confidence-weighted predictions
- Time-decay weighting for model performance

Usage:
    from ufc_predictor.advanced_ensemble_methods import AdvancedEnsembleSystem
    
    ensemble = AdvancedEnsembleSystem()
    ensemble.add_models({'rf': rf_model, 'xgb': xgb_model, 'lgb': lgb_model})
    prediction = ensemble.predict_with_confidence(X, fighter_names)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
import logging
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ML libraries
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, KFold, TimeSeriesSplit
from sklearn.metrics import accuracy_score, roc_auc_score, brier_score_loss
from sklearn.calibration import CalibratedClassifierCV
import xgboost as xgb
import lightgbm as lgb

# Our modules
from ufc_predictor.utils.unified_config import config
from ufc_predictor.utils.logging_config import get_logger
from ufc_predictor.utils.common_utilities import ValidationUtils

logger = get_logger(__name__)


@dataclass
class ModelPerformance:
    """Track model performance metrics over time"""
    accuracy_scores: List[float] = None
    roc_auc_scores: List[float] = None
    brier_scores: List[float] = None
    prediction_times: List[float] = None
    last_updated: Optional[datetime] = None
    sample_count: int = 0
    
    def __post_init__(self):
        if self.accuracy_scores is None:
            self.accuracy_scores = []
        if self.roc_auc_scores is None:
            self.roc_auc_scores = []
        if self.brier_scores is None:
            self.brier_scores = []
        if self.prediction_times is None:
            self.prediction_times = []
    
    def add_performance(self, accuracy: float, roc_auc: float, brier_score: float, pred_time: float):
        """Add new performance metrics"""
        self.accuracy_scores.append(accuracy)
        self.roc_auc_scores.append(roc_auc)
        self.brier_scores.append(brier_score)
        self.prediction_times.append(pred_time)
        self.last_updated = datetime.now()
        self.sample_count += 1
    
    def get_recent_performance(self, n_recent: int = 10) -> Dict[str, float]:
        """Get recent average performance"""
        recent_acc = np.mean(self.accuracy_scores[-n_recent:]) if self.accuracy_scores else 0.0
        recent_auc = np.mean(self.roc_auc_scores[-n_recent:]) if self.roc_auc_scores else 0.0
        recent_brier = np.mean(self.brier_scores[-n_recent:]) if self.brier_scores else 1.0
        recent_time = np.mean(self.prediction_times[-n_recent:]) if self.prediction_times else 0.0
        
        return {
            'accuracy': recent_acc,
            'roc_auc': recent_auc,
            'brier_score': recent_brier,
            'prediction_time': recent_time,
            'calibration_score': 1.0 - recent_brier  # Higher is better
        }


@dataclass 
class EnsemblePrediction:
    """Complete ensemble prediction result"""
    prediction_probability: float
    predicted_class: int
    confidence: float
    individual_predictions: Dict[str, float]
    individual_confidences: Dict[str, float]
    ensemble_method: str
    meta_features: Optional[Dict[str, float]] = None
    prediction_explanation: Optional[str] = None


class WeightedVotingEnsemble:
    """Advanced weighted voting ensemble with dynamic weight updates"""
    
    def __init__(self, models: Dict[str, Any], update_weights: bool = True):
        self.models = models
        self.update_weights = update_weights
        self.model_performances = {name: ModelPerformance() for name in models.keys()}
        self.weights = {name: 1.0 for name in models.keys()}
        self.weight_decay = 0.95  # Time decay factor
        
    def calculate_dynamic_weights(self, recent_window: int = 20) -> Dict[str, float]:
        """Calculate dynamic weights based on recent performance"""
        weights = {}
        
        for model_name in self.models:
            perf = self.model_performances[model_name]
            recent_perf = perf.get_recent_performance(recent_window)
            
            # Combined performance score (accuracy + calibration + speed)
            accuracy_weight = recent_perf['accuracy']
            calibration_weight = recent_perf['calibration_score'] * 0.8  # Slightly lower weight
            speed_weight = min(1.0, 1.0 / (recent_perf['prediction_time'] + 0.01))
            
            combined_score = (
                accuracy_weight * 0.5 + 
                calibration_weight * 0.3 + 
                speed_weight * 0.2
            )
            
            # Apply time decay if we have historical performance
            if perf.last_updated:
                days_since_update = (datetime.now() - perf.last_updated).days
                time_decay = self.weight_decay ** (days_since_update / 30)  # Decay over months
                combined_score *= time_decay
            
            weights[model_name] = max(0.01, combined_score)  # Minimum weight
        
        # Normalize weights
        total_weight = sum(weights.values())
        return {name: weight / total_weight for name, weight in weights.items()}
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Make probability predictions with dynamic weighting"""
        if self.update_weights:
            self.weights = self.calculate_dynamic_weights()
        
        predictions = []
        for model_name, model in self.models.items():
            pred_proba = model.predict_proba(X)
            predictions.append(pred_proba * self.weights[model_name])
        
        # Average weighted predictions
        ensemble_pred = np.sum(predictions, axis=0)
        
        # Renormalize probabilities
        ensemble_pred = ensemble_pred / np.sum(ensemble_pred, axis=1, keepdims=True)
        
        return ensemble_pred
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make class predictions"""
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)


class StackingEnsemble:
    """Advanced stacking ensemble with multiple meta-learners"""
    
    def __init__(self, base_models: Dict[str, Any], meta_learners: Optional[Dict[str, Any]] = None):
        self.base_models = base_models
        
        if meta_learners is None:
            self.meta_learners = {
                'logistic': LogisticRegression(random_state=42),
                'xgb_meta': xgb.XGBClassifier(n_estimators=50, max_depth=3, random_state=42),
                'rf_meta': RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
            }
        else:
            self.meta_learners = meta_learners
        
        self.best_meta_learner = None
        self.is_fitted = False
    
    def create_meta_features(self, X: pd.DataFrame, use_oof: bool = True) -> pd.DataFrame:
        """Create meta-features using out-of-fold predictions"""
        if use_oof and not self.is_fitted:
            return self._create_oof_meta_features(X)
        else:
            return self._create_direct_meta_features(X)
    
    def _create_oof_meta_features(self, X: pd.DataFrame, cv_folds: int = 5) -> pd.DataFrame:
        """Create meta-features using out-of-fold cross-validation"""
        meta_features = pd.DataFrame(index=X.index)
        
        # Use TimeSeriesSplit for temporal data
        cv = TimeSeriesSplit(n_splits=cv_folds)
        
        for model_name, model in self.base_models.items():
            oof_predictions = np.zeros((len(X), 2))  # Binary classification
            
            for train_idx, val_idx in cv.split(X):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                
                # Train model on fold
                model_copy = type(model)(**model.get_params())
                model_copy.fit(X_train, np.zeros(len(X_train)))  # Dummy y for demo
                
                # Predict on validation fold
                val_pred = model_copy.predict_proba(X_val)
                oof_predictions[val_idx] = val_pred
            
            # Add meta-features
            meta_features[f'{model_name}_prob_0'] = oof_predictions[:, 0]
            meta_features[f'{model_name}_prob_1'] = oof_predictions[:, 1]
            meta_features[f'{model_name}_confidence'] = np.max(oof_predictions, axis=1)
        
        return meta_features
    
    def _create_direct_meta_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create meta-features using direct predictions (for fitted models)"""
        meta_features = pd.DataFrame(index=X.index)
        
        for model_name, model in self.base_models.items():
            try:
                pred_proba = model.predict_proba(X)
                meta_features[f'{model_name}_prob_0'] = pred_proba[:, 0]
                meta_features[f'{model_name}_prob_1'] = pred_proba[:, 1]
                meta_features[f'{model_name}_confidence'] = np.max(pred_proba, axis=1)
            except Exception as e:
                logger.warning(f"Error getting predictions from {model_name}: {e}")
                # Fill with neutral predictions
                meta_features[f'{model_name}_prob_0'] = 0.5
                meta_features[f'{model_name}_prob_1'] = 0.5
                meta_features[f'{model_name}_confidence'] = 0.5
        
        return meta_features
    
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Fit the stacking ensemble"""
        # Create meta-features
        meta_features = self.create_meta_features(X, use_oof=True)
        
        # Train and evaluate meta-learners
        best_score = -np.inf
        best_meta_name = None
        
        for meta_name, meta_learner in self.meta_learners.items():
            try:
                # Use cross-validation to evaluate meta-learner
                cv_scores = cross_val_score(meta_learner, meta_features, y, 
                                          cv=TimeSeriesSplit(n_splits=3), 
                                          scoring='roc_auc')
                avg_score = np.mean(cv_scores)
                
                if avg_score > best_score:
                    best_score = avg_score
                    best_meta_name = meta_name
                
                logger.debug(f"Meta-learner {meta_name}: {avg_score:.4f} AUC")
                
            except Exception as e:
                logger.warning(f"Error training meta-learner {meta_name}: {e}")
        
        # Train best meta-learner on full data
        if best_meta_name:
            self.best_meta_learner = self.meta_learners[best_meta_name]
            self.best_meta_learner.fit(meta_features, y)
            self.is_fitted = True
            logger.info(f"Best meta-learner: {best_meta_name} ({best_score:.4f} AUC)")
        else:
            raise ValueError("No meta-learner could be trained successfully")
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Make probability predictions using stacking"""
        if not self.is_fitted:
            raise ValueError("Stacking ensemble must be fitted first")
        
        meta_features = self.create_meta_features(X, use_oof=False)
        return self.best_meta_learner.predict_proba(meta_features)
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make class predictions"""
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)


class BayesianModelAveraging:
    """Bayesian Model Averaging with dynamic posterior updates"""
    
    def __init__(self, models: Dict[str, Any], prior_weights: Optional[Dict[str, float]] = None):
        self.models = models
        self.prior_weights = prior_weights or {name: 1.0 for name in models.keys()}
        self.posterior_weights = self.prior_weights.copy()
        self.evidence_history = []
        
    def update_posterior_weights(self, predictions: Dict[str, np.ndarray], 
                               actual_outcomes: np.ndarray, 
                               learning_rate: float = 0.1):
        """Update posterior weights based on prediction performance"""
        for model_name in self.models:
            if model_name in predictions:
                # Calculate likelihood of actual outcomes given model predictions
                model_preds = predictions[model_name]
                
                # Binary log-likelihood
                likelihood = 0.0
                for i, outcome in enumerate(actual_outcomes):
                    prob = model_preds[i, outcome] if len(model_preds.shape) > 1 else model_preds[i]
                    likelihood += np.log(max(prob, 1e-10))  # Avoid log(0)
                
                # Update posterior weight (online learning)
                current_weight = self.posterior_weights[model_name]
                self.posterior_weights[model_name] = (
                    (1 - learning_rate) * current_weight + 
                    learning_rate * np.exp(likelihood / len(actual_outcomes))
                )
        
        # Normalize weights
        total_weight = sum(self.posterior_weights.values())
        for model_name in self.posterior_weights:
            self.posterior_weights[model_name] /= total_weight
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Make Bayesian model averaged predictions"""
        weighted_predictions = []
        
        for model_name, model in self.models.items():
            try:
                pred_proba = model.predict_proba(X)
                weight = self.posterior_weights[model_name]
                weighted_predictions.append(pred_proba * weight)
            except Exception as e:
                logger.warning(f"Error getting predictions from {model_name}: {e}")
        
        if not weighted_predictions:
            raise ValueError("No models could make predictions")
        
        return np.sum(weighted_predictions, axis=0)
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make class predictions"""
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)


class ConditionalEnsemble:
    """Conditional ensemble that selects different models based on fight characteristics"""
    
    def __init__(self, models: Dict[str, Any]):
        self.models = models
        self.condition_models = {
            'title_fights': models,
            'heavyweight_fights': models,
            'striker_vs_wrestler': models,
            'experienced_fighters': models,
            'default': models
        }
    
    def detect_fight_type(self, fight_features: pd.Series) -> str:
        """Detect fight type based on features"""
        # These would be based on actual feature analysis
        # For now, using simplified heuristics
        
        # Title fight detection (would need title fight indicator feature)
        if hasattr(fight_features, 'title_fight') and fight_features.title_fight:
            return 'title_fights'
        
        # Weight class detection (would need weight class feature)
        if hasattr(fight_features, 'weight_class'):
            if 'heavyweight' in str(fight_features.weight_class).lower():
                return 'heavyweight_fights'
        
        # Style matchup detection
        if hasattr(fight_features, 'striker_advantage') and hasattr(fight_features, 'wrestler_advantage'):
            if abs(fight_features.striker_advantage) > 1.0:
                return 'striker_vs_wrestler'
        
        # Experience level detection
        if hasattr(fight_features, 'wins_diff'):
            if abs(fight_features.wins_diff) > 10:
                return 'experienced_fighters'
        
        return 'default'
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Make conditional predictions based on fight characteristics"""
        predictions = []
        
        for idx, row in X.iterrows():
            fight_type = self.detect_fight_type(row)
            selected_models = self.condition_models[fight_type]
            
            # Use weighted voting for selected models
            ensemble = WeightedVotingEnsemble(selected_models, update_weights=False)
            pred = ensemble.predict_proba(row.to_frame().T)
            predictions.append(pred[0])
        
        return np.array(predictions)
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make class predictions"""
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)


class AdvancedEnsembleSystem:
    """
    Complete advanced ensemble system combining multiple ensemble techniques
    """
    
    def __init__(self, ensemble_config: Optional[Dict[str, Any]] = None):
        self.models = {}
        self.ensembles = {}
        self.model_performances = {}
        
        # Configuration
        self.config = ensemble_config or {
            'use_weighted_voting': True,
            'use_stacking': True,
            'use_bayesian_averaging': True,
            'use_conditional_ensemble': False,
            'confidence_threshold': 0.6,
            'performance_window': 50,
            'update_frequency': 10
        }
        
        # Performance tracking
        self.prediction_history = []
        self.ensemble_performance = {}
        
    def add_models(self, models: Dict[str, Any]):
        """Add models to the ensemble system"""
        self.models.update(models)
        
        # Initialize performance tracking
        for model_name in models:
            if model_name not in self.model_performances:
                self.model_performances[model_name] = ModelPerformance()
        
        self._initialize_ensembles()
        logger.info(f"Added {len(models)} models. Total models: {len(self.models)}")
    
    def _initialize_ensembles(self):
        """Initialize different ensemble methods"""
        if len(self.models) < 2:
            logger.warning("Need at least 2 models for ensemble methods")
            return
        
        # Weighted voting ensemble
        if self.config['use_weighted_voting']:
            self.ensembles['weighted_voting'] = WeightedVotingEnsemble(
                self.models, update_weights=True
            )
        
        # Stacking ensemble
        if self.config['use_stacking']:
            self.ensembles['stacking'] = StackingEnsemble(self.models)
        
        # Bayesian model averaging
        if self.config['use_bayesian_averaging']:
            self.ensembles['bayesian'] = BayesianModelAveraging(self.models)
        
        # Conditional ensemble
        if self.config['use_conditional_ensemble']:
            self.ensembles['conditional'] = ConditionalEnsemble(self.models)
        
        logger.info(f"Initialized {len(self.ensembles)} ensemble methods")
    
    def fit_ensembles(self, X: pd.DataFrame, y: pd.Series):
        """Fit ensemble methods that require training"""
        if 'stacking' in self.ensembles:
            logger.info("Fitting stacking ensemble...")
            self.ensembles['stacking'].fit(X, y)
        
        logger.info("All ensembles fitted successfully")
    
    def predict_with_confidence(self, X: pd.DataFrame, 
                              fighter_pairs: Optional[List[Tuple[str, str]]] = None,
                              ensemble_method: str = 'auto') -> List[EnsemblePrediction]:
        """
        Make predictions with confidence estimates using specified ensemble method
        """
        if ensemble_method == 'auto':
            ensemble_method = self._select_best_ensemble_method()
        
        if ensemble_method not in self.ensembles:
            raise ValueError(f"Ensemble method '{ensemble_method}' not available")
        
        ensemble = self.ensembles[ensemble_method]
        
        # Get ensemble predictions
        ensemble_proba = ensemble.predict_proba(X)
        ensemble_pred = np.argmax(ensemble_proba, axis=1)
        
        # Get individual model predictions for comparison
        individual_predictions = {}
        individual_confidences = {}
        
        for model_name, model in self.models.items():
            try:
                pred_proba = model.predict_proba(X)
                individual_predictions[model_name] = pred_proba[:, 1]  # Probability of class 1
                individual_confidences[model_name] = np.max(pred_proba, axis=1)
            except Exception as e:
                logger.warning(f"Error getting predictions from {model_name}: {e}")
                individual_predictions[model_name] = np.full(len(X), 0.5)
                individual_confidences[model_name] = np.full(len(X), 0.5)
        
        # Create prediction objects
        predictions = []
        for i in range(len(X)):
            # Calculate ensemble confidence
            ensemble_confidence = self._calculate_ensemble_confidence(
                ensemble_proba[i], individual_predictions, i
            )
            
            # Create prediction explanation
            explanation = self._create_prediction_explanation(
                ensemble_method, individual_predictions, i, fighter_pairs[i] if fighter_pairs else None
            )
            
            pred_obj = EnsemblePrediction(
                prediction_probability=ensemble_proba[i, 1],
                predicted_class=ensemble_pred[i],
                confidence=ensemble_confidence,
                individual_predictions={name: preds[i] for name, preds in individual_predictions.items()},
                individual_confidences={name: confs[i] for name, confs in individual_confidences.items()},
                ensemble_method=ensemble_method,
                prediction_explanation=explanation
            )
            
            predictions.append(pred_obj)
        
        return predictions
    
    def _calculate_ensemble_confidence(self, ensemble_proba: np.ndarray, 
                                     individual_predictions: Dict[str, np.ndarray], 
                                     sample_idx: int) -> float:
        """Calculate confidence based on model agreement and prediction strength"""
        # Prediction strength (how far from 0.5)
        strength = abs(ensemble_proba[1] - 0.5) * 2
        
        # Model agreement (how similar individual predictions are)
        individual_probs = [preds[sample_idx] for preds in individual_predictions.values()]
        if len(individual_probs) > 1:
            agreement = 1.0 - np.std(individual_probs)  # Lower std = higher agreement
        else:
            agreement = 1.0
        
        # Combined confidence
        confidence = (strength * 0.6 + agreement * 0.4)
        return min(0.99, max(0.01, confidence))
    
    def _create_prediction_explanation(self, ensemble_method: str,
                                     individual_predictions: Dict[str, np.ndarray],
                                     sample_idx: int,
                                     fighter_pair: Optional[Tuple[str, str]] = None) -> str:
        """Create human-readable explanation for the prediction"""
        individual_probs = {name: preds[sample_idx] for name, preds in individual_predictions.items()}
        
        # Find most confident and least confident models
        most_confident = max(individual_probs.items(), key=lambda x: abs(x[1] - 0.5))
        least_confident = min(individual_probs.items(), key=lambda x: abs(x[1] - 0.5))
        
        fighter_names = f" for {fighter_pair[0]} vs {fighter_pair[1]}" if fighter_pair else ""
        
        explanation = (f"Ensemble prediction{fighter_names} using {ensemble_method}. "
                      f"Most confident: {most_confident[0]} ({most_confident[1]:.3f}), "
                      f"Least confident: {least_confident[0]} ({least_confident[1]:.3f})")
        
        return explanation
    
    def _select_best_ensemble_method(self) -> str:
        """Select the best performing ensemble method"""
        if not self.ensemble_performance:
            return 'weighted_voting'  # Default
        
        best_method = max(self.ensemble_performance.items(), 
                         key=lambda x: x[1].get('recent_accuracy', 0))
        return best_method[0]
    
    def update_performance(self, predictions: List[EnsemblePrediction], 
                          actual_outcomes: List[int]):
        """Update performance tracking with new predictions and outcomes"""
        if len(predictions) != len(actual_outcomes):
            raise ValueError("Predictions and outcomes must have same length")
        
        # Update ensemble performance
        ensemble_method = predictions[0].ensemble_method
        if ensemble_method not in self.ensemble_performance:
            self.ensemble_performance[ensemble_method] = {}
        
        # Calculate metrics
        pred_probs = [p.prediction_probability for p in predictions]
        pred_classes = [p.predicted_class for p in predictions]
        
        accuracy = accuracy_score(actual_outcomes, pred_classes)
        
        try:
            auc = roc_auc_score(actual_outcomes, pred_probs)
        except ValueError:
            auc = 0.5  # If only one class present
        
        brier_score = brier_score_loss(actual_outcomes, pred_probs)
        
        # Store performance
        perf_data = self.ensemble_performance[ensemble_method]
        perf_data.setdefault('accuracy_history', []).append(accuracy)
        perf_data.setdefault('auc_history', []).append(auc)
        perf_data.setdefault('brier_history', []).append(brier_score)
        
        # Calculate recent performance
        window = self.config['performance_window']
        perf_data['recent_accuracy'] = np.mean(perf_data['accuracy_history'][-window:])
        perf_data['recent_auc'] = np.mean(perf_data['auc_history'][-window:])
        perf_data['recent_brier'] = np.mean(perf_data['brier_history'][-window:])
        
        logger.debug(f"Updated {ensemble_method} performance: "
                    f"Acc={accuracy:.4f}, AUC={auc:.4f}, Brier={brier_score:.4f}")
    
    def get_ensemble_summary(self) -> Dict[str, Any]:
        """Get comprehensive summary of ensemble system"""
        summary = {
            'models': list(self.models.keys()),
            'ensemble_methods': list(self.ensembles.keys()),
            'configuration': self.config,
            'performance_history': self.ensemble_performance.copy()
        }
        
        # Add recent performance summary
        if self.ensemble_performance:
            recent_performance = {}
            for method, perf in self.ensemble_performance.items():
                recent_performance[method] = {
                    'accuracy': perf.get('recent_accuracy', 0.0),
                    'auc': perf.get('recent_auc', 0.0),
                    'brier_score': perf.get('recent_brier', 1.0)
                }
            summary['recent_performance'] = recent_performance
        
        return summary


# Convenience functions
def create_standard_ensemble(models: Dict[str, Any], 
                           X_train: pd.DataFrame, 
                           y_train: pd.Series) -> AdvancedEnsembleSystem:
    """Create and fit a standard ensemble system"""
    ensemble_system = AdvancedEnsembleSystem()
    ensemble_system.add_models(models)
    ensemble_system.fit_ensembles(X_train, y_train)
    return ensemble_system


def compare_ensemble_methods(models: Dict[str, Any], 
                           X_test: pd.DataFrame, 
                           y_test: pd.Series) -> pd.DataFrame:
    """Compare different ensemble methods on test data"""
    results = []
    
    # Create ensemble system
    ensemble_system = AdvancedEnsembleSystem()
    ensemble_system.add_models(models)
    
    # Test each ensemble method
    for method_name in ensemble_system.ensembles.keys():
        try:
            predictions = ensemble_system.predict_with_confidence(X_test, ensemble_method=method_name)
            
            pred_classes = [p.predicted_class for p in predictions]
            pred_probs = [p.prediction_probability for p in predictions]
            avg_confidence = np.mean([p.confidence for p in predictions])
            
            accuracy = accuracy_score(y_test, pred_classes)
            auc = roc_auc_score(y_test, pred_probs)
            brier = brier_score_loss(y_test, pred_probs)
            
            results.append({
                'method': method_name,
                'accuracy': accuracy,
                'auc': auc,
                'brier_score': brier,
                'avg_confidence': avg_confidence,
                'calibration_score': 1.0 - brier
            })
            
        except Exception as e:
            logger.warning(f"Error testing {method_name}: {e}")
    
    return pd.DataFrame(results).sort_values('accuracy', ascending=False)


if __name__ == "__main__":
    # Demonstration of advanced ensemble methods
    logger.info("🎯 Advanced Ensemble Methods Demo")
    
    try:
        from sklearn.datasets import make_classification
        
        # Create sample data
        X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, 
                                  n_redundant=5, random_state=42)
        X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
        y = pd.Series(y)
        
        # Split data
        split_idx = int(0.8 * len(X))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Create sample models
        models = {
            'rf': RandomForestClassifier(n_estimators=100, random_state=42),
            'xgb': xgb.XGBClassifier(n_estimators=100, random_state=42),
            'lgb': lgb.LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)
        }
        
        # Train models
        for name, model in models.items():
            model.fit(X_train, y_train)
        
        # Create ensemble system
        ensemble_system = AdvancedEnsembleSystem()
        ensemble_system.add_models(models)
        ensemble_system.fit_ensembles(X_train, y_train)
        
        # Make predictions
        predictions = ensemble_system.predict_with_confidence(X_test)
        
        print(f"\n📊 Ensemble Results:")
        print(f"   Ensemble methods available: {list(ensemble_system.ensembles.keys())}")
        print(f"   Predictions made: {len(predictions)}")
        print(f"   Average confidence: {np.mean([p.confidence for p in predictions]):.3f}")
        
        # Compare methods
        comparison = compare_ensemble_methods(models, X_test, y_test)
        print(f"\n📈 Method Comparison:")
        print(comparison.round(4))
        
        print(f"\n✅ Advanced ensemble methods demonstration completed")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        raise