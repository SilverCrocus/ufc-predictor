"""
Production-Grade Stacking Ensemble for UFC Predictions
=====================================================

Advanced stacking ensemble implementation with out-of-fold prediction generation,
temporal validation, and integration with the existing production ensemble system.

Key Features:
- Out-of-fold prediction generation with TimeSeriesSplit
- Multiple meta-learner selection and optimization
- Integration with ProductionEnsembleManager
- Thread-safe parallel processing
- Comprehensive validation and error handling
- Model persistence and versioning
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union, NamedTuple
import logging
from dataclasses import dataclass
import joblib
import json
from pathlib import Path
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import threading
from contextlib import contextmanager

# ML libraries
from sklearn.model_selection import TimeSeriesSplit, cross_val_score, ParameterGrid
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_auc_score, brier_score_loss, log_loss
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
import xgboost as xgb
import lightgbm as lgb

# Our modules
from .production_ensemble_manager import ProductionEnsembleManager, EnsembleConfig, PredictionResult
from ..utils.enhanced_error_handling import UFCPredictorError
from ..utils.thread_safe_bootstrap import ThreadSafeBootstrapSampler, BootstrapConfig

logger = logging.getLogger(__name__)


class StackingError(Exception):
    """Stacking-specific errors"""
    pass


class OOFGenerationError(Exception):
    """Out-of-fold generation errors"""
    pass


class MetaLearnerError(Exception):
    """Meta-learner training errors"""
    pass


@dataclass
class StackingConfig:
    """Configuration for stacking ensemble"""
    base_model_weights: Dict[str, float]
    meta_learner_candidates: List[str] = None
    cv_splits: int = 5
    confidence_level: float = 0.95
    bootstrap_samples: int = 100
    enable_meta_optimization: bool = True
    meta_optimization_trials: int = 50
    temporal_validation: bool = True
    max_memory_mb: int = 4096
    n_jobs: int = -1
    random_state: int = 42
    
    def __post_init__(self):
        if self.meta_learner_candidates is None:
            self.meta_learner_candidates = [
                'logistic_regression',
                'ridge_classifier', 
                'xgb_meta',
                'lgb_meta',
                'rf_meta'
            ]
        
        # Validation
        if not self.base_model_weights:
            raise StackingError("base_model_weights cannot be empty")
        
        total_weight = sum(self.base_model_weights.values())
        if abs(total_weight - 1.0) > 1e-6:
            raise StackingError(f"Base model weights must sum to 1.0, got {total_weight}")
        
        if self.cv_splits < 2:
            raise StackingError(f"cv_splits must be >= 2, got {self.cv_splits}")


class StackingResult(NamedTuple):
    """Result from stacking ensemble prediction"""
    stacked_probability: float
    base_predictions: Dict[str, float]
    meta_learner_name: str
    confidence_interval: Tuple[float, float]
    oof_validation_score: float
    meta_learner_confidence: float


class OOFPredictionGenerator:
    """Generate out-of-fold predictions for meta-learner training"""
    
    def __init__(self, config: StackingConfig):
        self.config = config
        self.oof_predictions = {}
        self.oof_targets = None
        self.fold_indices = []
        
    def generate_oof_predictions(self, base_models: Dict[str, Any], 
                                X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Generate out-of-fold predictions using temporal cross-validation
        
        Args:
            base_models: Dictionary of trained base models
            X: Feature matrix
            y: Target vector
            
        Returns:
            Tuple of (meta_features_df, y_aligned)
        """
        logger.info(f"Generating OOF predictions with {self.config.cv_splits}-fold CV")
        
        # Input validation
        self._validate_inputs(base_models, X, y)
        
        # Use TimeSeriesSplit for temporal validation
        if self.config.temporal_validation:
            cv = TimeSeriesSplit(n_splits=self.config.cv_splits)
        else:
            from sklearn.model_selection import KFold
            cv = KFold(n_splits=self.config.cv_splits, shuffle=True, 
                      random_state=self.config.random_state)
        
        # Initialize OOF prediction arrays
        n_samples = len(X)
        oof_predictions = {}
        for model_name in base_models.keys():
            oof_predictions[model_name] = np.full(n_samples, np.nan)
        
        # Track fold indices for validation
        self.fold_indices = []
        
        # Generate OOF predictions for each fold
        fold_scores = {model_name: [] for model_name in base_models.keys()}
        
        for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X)):
            logger.debug(f"Processing fold {fold_idx + 1}/{self.config.cv_splits}")
            
            self.fold_indices.append((train_idx, val_idx))
            
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Process each base model
            for model_name, model in base_models.items():
                try:
                    fold_pred, fold_score = self._train_and_predict_fold(
                        model, model_name, X_train, y_train, X_val, y_val
                    )
                    
                    oof_predictions[model_name][val_idx] = fold_pred
                    fold_scores[model_name].append(fold_score)
                    
                except Exception as e:
                    logger.error(f"Fold {fold_idx} failed for model {model_name}: {str(e)}")
                    # Fill with neutral predictions to continue
                    oof_predictions[model_name][val_idx] = 0.5
                    fold_scores[model_name].append(0.5)
        
        # Log OOF predictions completeness
        for model_name, pred_array in oof_predictions.items():
            nan_count = np.isnan(pred_array).sum()
            if nan_count > 0:
                logger.warning(f"Model {model_name} has {nan_count}/{n_samples} NaN predictions, filling with 0.5")
                pred_array[np.isnan(pred_array)] = 0.5
        
        # Create meta-features DataFrame
        meta_features = self._create_meta_features_dataframe(oof_predictions, X.index)
        
        # Log OOF validation scores
        self._log_oof_scores(fold_scores)
        
        return meta_features, y
    
    def _validate_inputs(self, base_models: Dict[str, Any], X: pd.DataFrame, y: pd.Series):
        """Validate inputs for OOF generation"""
        if not base_models:
            raise OOFGenerationError("No base models provided")
        
        if X.empty or y.empty:
            raise OOFGenerationError("Empty training data provided")
        
        if len(X) != len(y):
            raise OOFGenerationError(f"X and y length mismatch: {len(X)} != {len(y)}")
        
        if X.isnull().any().any():
            null_cols = X.columns[X.isnull().any()].tolist()
            raise OOFGenerationError(f"X contains null values in columns: {null_cols}")
        
        if y.isnull().any():
            raise OOFGenerationError("y contains null values")
        
        # Check minimum samples for CV
        min_samples_per_fold = len(X) // self.config.cv_splits
        if min_samples_per_fold < 20:
            raise OOFGenerationError(
                f"Insufficient samples for {self.config.cv_splits}-fold CV: "
                f"{min_samples_per_fold} samples per fold < 20"
            )
    
    def _train_and_predict_fold(self, base_model: Any, model_name: str,
                               X_train: pd.DataFrame, y_train: pd.Series,
                               X_val: pd.DataFrame, y_val: pd.Series) -> Tuple[np.ndarray, float]:
        """Train model on fold and generate predictions"""
        
        # Create a copy of the model to avoid modifying the original
        model_params = base_model.get_params() if hasattr(base_model, 'get_params') else {}
        model_class = type(base_model)
        fold_model = model_class(**model_params)
        
        # Handle scaling for certain models
        scaler = None
        if model_name.lower() in ['logistic_regression', 'svm', 'neural_network']:
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
        else:
            X_train_scaled = X_train
            X_val_scaled = X_val
        
        # Train fold model
        fold_model.fit(X_train_scaled, y_train)
        
        # Generate predictions
        if hasattr(fold_model, 'predict_proba'):
            fold_pred_proba = fold_model.predict_proba(X_val_scaled)
            fold_pred = fold_pred_proba[:, 1]  # Probability of positive class
        else:
            fold_pred = fold_model.predict(X_val_scaled)
        
        # Calculate fold validation score
        try:
            fold_score = roc_auc_score(y_val, fold_pred)
        except ValueError:
            # Handle case where only one class is present
            fold_score = 0.5
        
        return fold_pred, fold_score
    
    def _validate_oof_completeness(self, oof_predictions: Dict[str, np.ndarray], n_samples: int):
        """Validate that all OOF predictions are complete"""
        for model_name, pred_array in oof_predictions.items():
            nan_count = np.isnan(pred_array).sum()
            if nan_count > 0:
                raise OOFGenerationError(
                    f"Incomplete OOF predictions for {model_name}: {nan_count}/{n_samples} NaN values"
                )
    
    def _create_meta_features_dataframe(self, oof_predictions: Dict[str, np.ndarray],
                                       index: pd.Index) -> pd.DataFrame:
        """Create meta-features DataFrame from OOF predictions"""
        meta_features = pd.DataFrame(index=index)
        
        for model_name, predictions in oof_predictions.items():
            # Base prediction
            meta_features[f'{model_name}_pred'] = predictions
            
            # Confidence (distance from 0.5)
            meta_features[f'{model_name}_confidence'] = np.abs(predictions - 0.5) * 2
            
            # Squared predictions (non-linearity)
            meta_features[f'{model_name}_pred_sq'] = predictions ** 2
            
            # Interaction with weighted average
            weighted_avg = np.average(
                list(oof_predictions.values()), 
                weights=list(self.config.base_model_weights.values()),
                axis=0
            )
            meta_features[f'{model_name}_vs_ensemble'] = predictions - weighted_avg
        
        return meta_features
    
    def _log_oof_scores(self, fold_scores: Dict[str, List[float]]):
        """Log out-of-fold validation scores"""
        logger.info("Out-of-fold validation scores:")
        for model_name, scores in fold_scores.items():
            mean_score = np.mean(scores)
            std_score = np.std(scores)
            logger.info(f"  {model_name}: {mean_score:.4f} ± {std_score:.4f}")


class MetaLearnerSelector:
    """Select and optimize meta-learners for stacking"""
    
    def __init__(self, config: StackingConfig):
        self.config = config
        self.meta_learner_candidates = self._create_meta_learner_candidates()
        self.best_meta_learner = None
        self.best_meta_score = -np.inf
        self.meta_learner_scores = {}
        
    def _create_meta_learner_candidates(self) -> Dict[str, Any]:
        """Create dictionary of meta-learner candidates"""
        candidates = {}
        
        if 'logistic_regression' in self.config.meta_learner_candidates:
            candidates['logistic_regression'] = LogisticRegression(
                random_state=self.config.random_state,
                max_iter=1000,
                solver='liblinear'
            )
        
        if 'ridge_classifier' in self.config.meta_learner_candidates:
            candidates['ridge_classifier'] = RidgeClassifier(
                random_state=self.config.random_state
            )
        
        if 'xgb_meta' in self.config.meta_learner_candidates:
            candidates['xgb_meta'] = xgb.XGBClassifier(
                n_estimators=50,
                max_depth=3,
                learning_rate=0.1,
                random_state=self.config.random_state,
                eval_metric='logloss'
            )
        
        if 'lgb_meta' in self.config.meta_learner_candidates:
            candidates['lgb_meta'] = lgb.LGBMClassifier(
                n_estimators=50,
                max_depth=3,
                learning_rate=0.1,
                random_state=self.config.random_state,
                verbose=-1
            )
        
        if 'rf_meta' in self.config.meta_learner_candidates:
            candidates['rf_meta'] = RandomForestClassifier(
                n_estimators=50,
                max_depth=5,
                random_state=self.config.random_state
            )
        
        return candidates
    
    def select_best_meta_learner(self, meta_features: pd.DataFrame, 
                                y: pd.Series) -> Tuple[Any, str, float]:
        """
        Select the best meta-learner using cross-validation
        
        Returns:
            Tuple of (best_model, model_name, best_score)
        """
        logger.info(f"Evaluating {len(self.meta_learner_candidates)} meta-learner candidates")
        
        # Use same CV strategy as OOF generation
        if self.config.temporal_validation:
            cv = TimeSeriesSplit(n_splits=max(3, self.config.cv_splits - 2))
        else:
            from sklearn.model_selection import KFold
            cv = KFold(n_splits=max(3, self.config.cv_splits - 2), shuffle=True,
                      random_state=self.config.random_state)
        
        best_score = -np.inf
        best_name = None
        best_model = None
        
        for name, model in self.meta_learner_candidates.items():
            try:
                # Cross-validation evaluation
                scores = cross_val_score(
                    model, meta_features, y,
                    cv=cv,
                    scoring='roc_auc',
                    n_jobs=1  # Avoid nested parallelization
                )
                
                mean_score = np.mean(scores)
                std_score = np.std(scores)
                
                self.meta_learner_scores[name] = {
                    'mean_score': mean_score,
                    'std_score': std_score,
                    'scores': scores.tolist()
                }
                
                logger.debug(f"Meta-learner {name}: {mean_score:.4f} ± {std_score:.4f}")
                
                if mean_score > best_score:
                    best_score = mean_score
                    best_name = name
                    best_model = model
                    
            except Exception as e:
                logger.warning(f"Meta-learner {name} evaluation failed: {str(e)}")
                self.meta_learner_scores[name] = {'error': str(e)}
        
        if best_model is None:
            raise MetaLearnerError("No meta-learner could be evaluated successfully")
        
        # Train best meta-learner on full data
        best_model.fit(meta_features, y)
        
        self.best_meta_learner = best_model
        self.best_meta_score = best_score
        
        logger.info(f"Selected meta-learner: {best_name} (AUC: {best_score:.4f})")
        
        return best_model, best_name, best_score
    
    def optimize_meta_learner(self, meta_features: pd.DataFrame, 
                             y: pd.Series, meta_learner_name: str) -> Any:
        """Optimize hyperparameters for selected meta-learner"""
        
        if not self.config.enable_meta_optimization:
            return self.meta_learner_candidates[meta_learner_name]
        
        logger.info(f"Optimizing hyperparameters for {meta_learner_name}")
        
        # Define hyperparameter grids
        param_grids = {
            'logistic_regression': {
                'C': [0.01, 0.1, 1.0, 10.0, 100.0],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear']
            },
            'ridge_classifier': {
                'alpha': [0.1, 1.0, 10.0, 100.0]
            },
            'xgb_meta': {
                'n_estimators': [25, 50, 100],
                'max_depth': [2, 3, 4],
                'learning_rate': [0.05, 0.1, 0.2]
            },
            'lgb_meta': {
                'n_estimators': [25, 50, 100],
                'max_depth': [2, 3, 4],
                'learning_rate': [0.05, 0.1, 0.2]
            },
            'rf_meta': {
                'n_estimators': [25, 50, 100],
                'max_depth': [3, 5, 7],
                'min_samples_split': [2, 5]
            }
        }
        
        if meta_learner_name not in param_grids:
            logger.warning(f"No parameter grid for {meta_learner_name}, using default")
            return self.meta_learner_candidates[meta_learner_name]
        
        # Use RandomizedSearchCV for efficiency
        from sklearn.model_selection import RandomizedSearchCV
        
        base_model = self.meta_learner_candidates[meta_learner_name]
        param_grid = param_grids[meta_learner_name]
        
        # Temporal CV for optimization
        if self.config.temporal_validation:
            cv = TimeSeriesSplit(n_splits=3)
        else:
            from sklearn.model_selection import KFold
            cv = KFold(n_splits=3, shuffle=True, random_state=self.config.random_state)
        
        search = RandomizedSearchCV(
            base_model,
            param_grid,
            n_iter=min(self.config.meta_optimization_trials, len(list(ParameterGrid(param_grid)))),
            cv=cv,
            scoring='roc_auc',
            random_state=self.config.random_state,
            n_jobs=1
        )
        
        search.fit(meta_features, y)
        
        logger.info(f"Optimization completed. Best score: {search.best_score_:.4f}")
        logger.debug(f"Best parameters: {search.best_params_}")
        
        return search.best_estimator_


class ProductionStackingManager(ProductionEnsembleManager):
    """Production-ready stacking ensemble manager"""
    
    def __init__(self, stacking_config: StackingConfig):
        # Initialize base ensemble config
        ensemble_config = EnsembleConfig(
            model_weights=stacking_config.base_model_weights,
            confidence_level=stacking_config.confidence_level,
            bootstrap_samples=stacking_config.bootstrap_samples,
            max_memory_mb=stacking_config.max_memory_mb,
            n_jobs=stacking_config.n_jobs
        )
        
        super().__init__(ensemble_config)
        
        self.stacking_config = stacking_config
        self.oof_generator = OOFPredictionGenerator(stacking_config)
        self.meta_selector = MetaLearnerSelector(stacking_config)
        self.meta_learner = None
        self.meta_learner_name = None
        self.meta_learner_score = None
        self.meta_features_columns = None
        self.is_stacking_fitted = False
        
    def fit_stacking_ensemble(self, X: pd.DataFrame, y: pd.Series):
        """Fit the complete stacking ensemble"""
        
        if not self.is_initialized:
            raise StackingError("Base models not loaded. Call load_models() first.")
        
        logger.info("Starting stacking ensemble training...")
        start_time = time.time()
        
        try:
            # Generate out-of-fold predictions
            logger.info("Generating out-of-fold predictions...")
            meta_features, y_aligned = self.oof_generator.generate_oof_predictions(
                self.models, X, y
            )
            
            # Store meta-features column names for prediction
            self.meta_features_columns = meta_features.columns.tolist()
            
            # Select best meta-learner
            logger.info("Selecting optimal meta-learner...")
            meta_learner, meta_name, meta_score = self.meta_selector.select_best_meta_learner(
                meta_features, y_aligned
            )
            
            # Optimize meta-learner if enabled
            if self.stacking_config.enable_meta_optimization:
                logger.info("Optimizing meta-learner hyperparameters...")
                meta_learner = self.meta_selector.optimize_meta_learner(
                    meta_features, y_aligned, meta_name
                )
            
            self.meta_learner = meta_learner
            self.meta_learner_name = meta_name
            self.meta_learner_score = meta_score
            self.is_stacking_fitted = True
            
            training_time = time.time() - start_time
            logger.info(f"Stacking ensemble training completed in {training_time:.2f}s")
            logger.info(f"Meta-learner: {meta_name} (OOF AUC: {meta_score:.4f})")
            
        except Exception as e:
            if isinstance(e, (StackingError, OOFGenerationError, MetaLearnerError)):
                raise
            else:
                raise StackingError(f"Stacking ensemble training failed: {str(e)}")
    
    def predict_stacking(self, X: pd.DataFrame, 
                        fighter_pairs: List[Tuple[str, str]],
                        enable_bootstrap: bool = True) -> List[StackingResult]:
        """Generate stacking ensemble predictions"""
        
        if not self.is_stacking_fitted:
            raise StackingError("Stacking ensemble not fitted. Call fit_stacking_ensemble() first.")
        
        start_time = time.time()
        
        try:
            # Generate base model predictions
            base_predictions = self._get_base_model_predictions(X)
            
            # Create meta-features
            meta_features = self._create_prediction_meta_features(base_predictions, X)
            
            # Generate stacking predictions
            stacking_probabilities = self.meta_learner.predict_proba(meta_features)[:, 1]
            
            # Calculate confidence intervals if requested
            confidence_intervals = None
            if enable_bootstrap:
                confidence_intervals = self._get_stacking_confidence_intervals(
                    X, meta_features, stacking_probabilities
                )
            
            # Format results
            results = []
            for i, (fighter_a, fighter_b) in enumerate(fighter_pairs):
                
                # Get base model predictions for this sample
                sample_base_preds = {
                    name: float(preds[i]) for name, preds in base_predictions.items()
                }
                
                # Confidence interval
                ci = confidence_intervals[i] if confidence_intervals else (
                    stacking_probabilities[i] - 0.1, stacking_probabilities[i] + 0.1
                )
                
                # Meta-learner confidence (based on prediction strength)
                meta_confidence = abs(stacking_probabilities[i] - 0.5) * 2
                
                result = StackingResult(
                    stacked_probability=float(stacking_probabilities[i]),
                    base_predictions=sample_base_preds,
                    meta_learner_name=self.meta_learner_name,
                    confidence_interval=ci,
                    oof_validation_score=self.meta_learner_score,
                    meta_learner_confidence=meta_confidence
                )
                
                results.append(result)
            
            processing_time = time.time() - start_time
            logger.info(f"Stacking predictions completed in {processing_time*1000:.1f}ms")
            
            return results
            
        except Exception as e:
            if isinstance(e, StackingError):
                raise
            else:
                raise StackingError(f"Stacking prediction failed: {str(e)}")
    
    def _get_base_model_predictions(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Get predictions from all base models"""
        base_predictions = {}
        
        for model_name, model in self.models.items():
            try:
                if hasattr(model, 'predict_proba'):
                    pred = model.predict_proba(X)[:, 1]
                else:
                    pred = model.predict(X)
                
                base_predictions[model_name] = pred
                
            except Exception as e:
                raise StackingError(f"Base model {model_name} prediction failed: {str(e)}")
        
        return base_predictions
    
    def _create_prediction_meta_features(self, base_predictions: Dict[str, np.ndarray],
                                        X: pd.DataFrame) -> pd.DataFrame:
        """Create meta-features for prediction (matching training format)"""
        
        meta_features = pd.DataFrame(index=X.index)
        
        # Calculate weighted average for interaction features
        weighted_avg = np.average(
            list(base_predictions.values()),
            weights=list(self.config.model_weights.values()),
            axis=0
        )
        
        for model_name, predictions in base_predictions.items():
            # Base prediction
            meta_features[f'{model_name}_pred'] = predictions
            
            # Confidence (distance from 0.5)
            meta_features[f'{model_name}_confidence'] = np.abs(predictions - 0.5) * 2
            
            # Squared predictions (non-linearity)
            meta_features[f'{model_name}_pred_sq'] = predictions ** 2
            
            # Interaction with ensemble
            meta_features[f'{model_name}_vs_ensemble'] = predictions - weighted_avg
        
        # Ensure column order matches training
        if self.meta_features_columns:
            missing_cols = set(self.meta_features_columns) - set(meta_features.columns)
            if missing_cols:
                raise StackingError(f"Missing meta-features columns: {missing_cols}")
            
            meta_features = meta_features[self.meta_features_columns]
        
        return meta_features
    
    def _get_stacking_confidence_intervals(self, X: pd.DataFrame,
                                         meta_features: pd.DataFrame,
                                         stacking_probabilities: np.ndarray) -> List[Tuple[float, float]]:
        """Generate confidence intervals for stacking predictions using bootstrap"""
        
        try:
            # Use bootstrap sampling on meta-features
            bootstrap_config = BootstrapConfig(
                n_bootstrap=self.stacking_config.bootstrap_samples,
                confidence_level=self.stacking_config.confidence_level,
                n_jobs=self.stacking_config.n_jobs,
                random_state=self.stacking_config.random_state
            )
            
            bootstrap_sampler = ThreadSafeBootstrapSampler(bootstrap_config)
            
            # Create pseudo-models for bootstrap (meta-learner only)
            meta_models = {'meta_learner': self.meta_learner}
            meta_weights = {'meta_learner': 1.0}
            
            # Generate bootstrap predictions on meta-features
            bootstrap_predictions = bootstrap_sampler.sample_bootstrap_predictions(
                meta_models, meta_features, meta_weights
            )
            
            # Calculate confidence intervals
            from ..utils.thread_safe_bootstrap import calculate_confidence_intervals
            mean_pred, lower_bounds, upper_bounds = calculate_confidence_intervals(
                bootstrap_predictions, self.stacking_config.confidence_level
            )
            
            return [(float(lower), float(upper)) 
                   for lower, upper in zip(lower_bounds, upper_bounds)]
            
        except Exception as e:
            logger.warning(f"Bootstrap confidence intervals failed: {str(e)}")
            # Fallback to simple confidence intervals
            margin = 0.1
            return [(prob - margin, prob + margin) for prob in stacking_probabilities]
    
    def save_stacking_ensemble(self, filepath: str):
        """Save complete stacking ensemble to disk"""
        
        if not self.is_stacking_fitted:
            raise StackingError("No fitted stacking ensemble to save")
        
        save_data = {
            'meta_learner': self.meta_learner,
            'meta_learner_name': self.meta_learner_name,
            'meta_learner_score': self.meta_learner_score,
            'meta_features_columns': self.meta_features_columns,
            'stacking_config': self.stacking_config,
            'meta_learner_scores': self.meta_selector.meta_learner_scores,
            'oof_fold_indices': self.oof_generator.fold_indices
        }
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        joblib.dump(save_data, filepath)
        
        # Save configuration as JSON
        config_path = filepath.with_suffix('.json')
        config_data = {
            'meta_learner_name': self.meta_learner_name,
            'meta_learner_score': self.meta_learner_score,
            'base_model_weights': self.stacking_config.base_model_weights,
            'cv_splits': self.stacking_config.cv_splits,
            'meta_features_columns': self.meta_features_columns
        }
        
        with open(config_path, 'w') as f:
            json.dump(config_data, f, indent=2)
        
        logger.info(f"Stacking ensemble saved to {filepath}")
    
    @classmethod
    def load_stacking_ensemble(cls, filepath: str, base_models: Dict[str, Any]) -> 'ProductionStackingManager':
        """Load stacking ensemble from disk"""
        
        filepath = Path(filepath)
        if not filepath.exists():
            raise StackingError(f"Stacking ensemble file not found: {filepath}")
        
        save_data = joblib.load(filepath)
        
        # Recreate stacking config
        stacking_config = save_data['stacking_config']
        
        # Create manager instance
        manager = cls(stacking_config)
        
        # Load base models
        model_paths = {name: f"dummy_path_{name}" for name in base_models.keys()}
        manager.models = base_models
        manager.is_initialized = True
        
        # Restore stacking components
        manager.meta_learner = save_data['meta_learner']
        manager.meta_learner_name = save_data['meta_learner_name']
        manager.meta_learner_score = save_data['meta_learner_score']
        manager.meta_features_columns = save_data['meta_features_columns']
        manager.meta_selector.meta_learner_scores = save_data['meta_learner_scores']
        manager.oof_generator.fold_indices = save_data['oof_fold_indices']
        manager.is_stacking_fitted = True
        
        logger.info(f"Stacking ensemble loaded from {filepath}")
        logger.info(f"Meta-learner: {manager.meta_learner_name} (AUC: {manager.meta_learner_score:.4f})")
        
        return manager
    
    def get_stacking_summary(self) -> Dict[str, Any]:
        """Get comprehensive summary of stacking ensemble"""
        
        summary = {
            'is_fitted': self.is_stacking_fitted,
            'base_models': list(self.models.keys()) if self.models else [],
            'base_model_weights': self.stacking_config.base_model_weights,
            'meta_learner_name': self.meta_learner_name,
            'meta_learner_score': self.meta_learner_score,
            'meta_features_count': len(self.meta_features_columns) if self.meta_features_columns else 0,
            'cv_splits': self.stacking_config.cv_splits,
            'temporal_validation': self.stacking_config.temporal_validation,
            'meta_optimization_enabled': self.stacking_config.enable_meta_optimization
        }
        
        if self.meta_selector.meta_learner_scores:
            summary['all_meta_learner_scores'] = self.meta_selector.meta_learner_scores
        
        return summary


def create_stacking_config(base_model_weights: Dict[str, float],
                          cv_splits: int = 5,
                          temporal_validation: bool = True,
                          enable_optimization: bool = True) -> StackingConfig:
    """Factory function for creating stacking configuration"""
    
    return StackingConfig(
        base_model_weights=base_model_weights,
        cv_splits=cv_splits,
        temporal_validation=temporal_validation,
        enable_meta_optimization=enable_optimization
    )


# Example usage and testing
if __name__ == "__main__":
    print("Production Stacking Ensemble Test")
    print("=" * 50)
    
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
        
        # Create base models
        base_models = {
            'rf': RandomForestClassifier(n_estimators=100, random_state=42),
            'xgb': xgb.XGBClassifier(n_estimators=100, random_state=42),
            'lgb': lgb.LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)
        }
        
        # Train base models
        for model in base_models.values():
            model.fit(X_train, y_train)
        
        # Create stacking configuration
        base_weights = {'rf': 0.4, 'xgb': 0.35, 'lgb': 0.25}
        stacking_config = create_stacking_config(
            base_weights, 
            cv_splits=5,
            temporal_validation=True,
            enable_optimization=True
        )
        
        # Create stacking manager
        stacking_manager = ProductionStackingManager(stacking_config)
        stacking_manager.models = base_models
        stacking_manager.is_initialized = True
        
        # Fit stacking ensemble
        stacking_manager.fit_stacking_ensemble(X_train, y_train)
        
        # Generate predictions
        fighter_pairs = [(f'Fighter_A_{i}', f'Fighter_B_{i}') for i in range(len(X_test))]
        stacking_results = stacking_manager.predict_stacking(X_test, fighter_pairs)
        
        # Evaluate performance
        stacked_preds = [r.stacked_probability for r in stacking_results]
        stacked_classes = [1 if p > 0.5 else 0 for p in stacked_preds]
        
        accuracy = accuracy_score(y_test, stacked_classes)
        auc = roc_auc_score(y_test, stacked_preds)
        
        print(f"✅ Stacking ensemble test completed")
        print(f"   Meta-learner: {stacking_results[0].meta_learner_name}")
        print(f"   OOF validation AUC: {stacking_results[0].oof_validation_score:.4f}")
        print(f"   Test accuracy: {accuracy:.4f}")
        print(f"   Test AUC: {auc:.4f}")
        print(f"   Predictions generated: {len(stacking_results)}")
        
        # Test saving and loading
        save_path = "/tmp/test_stacking_ensemble.joblib"
        stacking_manager.save_stacking_ensemble(save_path)
        
        loaded_manager = ProductionStackingManager.load_stacking_ensemble(save_path, base_models)
        print(f"✅ Save/load test completed")
        
        # Cleanup
        import os
        for ext in ['.joblib', '.json']:
            try:
                os.remove(save_path.replace('.joblib', ext))
            except:
                pass
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        raise
