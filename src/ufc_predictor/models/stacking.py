"""
Stacking ensemble with out-of-fold predictions for UFC fight predictions.
Implements temporal-aware stacking with proper cross-validation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from sklearn.model_selection import TimeSeriesSplit, StratifiedKFold
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb
from sklearn.base import BaseEstimator, ClassifierMixin, clone
import joblib
import logging
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class StackingResult:
    """Container for stacking ensemble results."""
    oof_predictions: np.ndarray
    test_predictions: Optional[np.ndarray]
    meta_model: Any
    base_models: Dict[str, Any]
    cv_scores: Dict[str, float]
    feature_importance: Optional[pd.DataFrame]


class OOFStackingEnsemble(BaseEstimator, ClassifierMixin):
    """
    Out-of-fold stacking ensemble for temporal UFC predictions.
    Generates OOF predictions from base models and trains meta-learner.
    """
    
    def __init__(
        self,
        base_models: Optional[Dict[str, BaseEstimator]] = None,
        meta_model: Optional[BaseEstimator] = None,
        cv_strategy: str = 'temporal',
        n_splits: int = 5,
        use_probas: bool = True,
        use_features_in_meta: bool = False,
        random_state: int = 42
    ):
        """
        Initialize stacking ensemble.
        
        Args:
            base_models: Dictionary of base models
            meta_model: Meta-learner model
            cv_strategy: 'temporal' or 'stratified'
            n_splits: Number of CV splits
            use_probas: Use probabilities (True) or predictions (False)
            use_features_in_meta: Include original features in meta-model
            random_state: Random seed
        """
        self.random_state = random_state  # Set this first
        self.base_models = base_models or self._get_default_base_models()
        self.meta_model = meta_model or LogisticRegression(C=1.0, random_state=random_state)
        self.cv_strategy = cv_strategy
        self.n_splits = n_splits
        self.use_probas = use_probas
        self.use_features_in_meta = use_features_in_meta
        
        self.oof_predictions_ = None
        self.fitted_base_models_ = {}
        self.meta_model_ = None
        self.feature_importance_ = None
    
    def _get_default_base_models(self) -> Dict[str, BaseEstimator]:
        """Get default base models for ensemble."""
        return {
            'rf': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=20,
                random_state=self.random_state
            ),
            'xgb': xgb.XGBClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=self.random_state,
                use_label_encoder=False,
                eval_metric='logloss'
            ),
            'lgb': lgb.LGBMClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=self.random_state,
                verbosity=-1
            )
        }
    
    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        sample_weight: Optional[np.ndarray] = None,
        eval_set: Optional[Tuple] = None
    ) -> 'OOFStackingEnsemble':
        """
        Fit stacking ensemble with OOF predictions.
        
        Args:
            X: Training features
            y: Training labels
            sample_weight: Sample weights
            eval_set: Validation set for early stopping
            
        Returns:
            Fitted ensemble
        """
        X = self._validate_input(X)
        y = np.array(y)
        
        n_samples = X.shape[0]
        n_models = len(self.base_models)
        
        # Initialize OOF predictions array
        if self.use_probas:
            self.oof_predictions_ = np.zeros((n_samples, n_models))
        else:
            self.oof_predictions_ = np.zeros((n_samples, n_models))
        
        # Get CV splits
        cv_splits = self._get_cv_splits(X, y)
        
        # Generate OOF predictions for each base model
        cv_scores = {}
        
        for model_idx, (name, base_model) in enumerate(self.base_models.items()):
            logger.info(f"Training base model: {name}")
            
            model_oof = np.zeros(n_samples)
            fold_models = []
            
            for fold_idx, (train_idx, val_idx) in enumerate(cv_splits):
                X_train_fold, X_val_fold = X[train_idx], X[val_idx]
                y_train_fold, y_val_fold = y[train_idx], y[val_idx]
                
                # Clone model for this fold
                fold_model = clone(base_model)
                
                # Handle sample weights if provided
                if sample_weight is not None:
                    weights_fold = sample_weight[train_idx]
                else:
                    weights_fold = None
                
                # Train model
                if hasattr(fold_model, 'fit') and 'sample_weight' in fold_model.fit.__code__.co_varnames:
                    fold_model.fit(X_train_fold, y_train_fold, sample_weight=weights_fold)
                else:
                    fold_model.fit(X_train_fold, y_train_fold)
                
                # Generate OOF predictions
                if self.use_probas:
                    val_preds = fold_model.predict_proba(X_val_fold)[:, 1]
                else:
                    val_preds = fold_model.predict(X_val_fold)
                
                model_oof[val_idx] = val_preds
                fold_models.append(fold_model)
            
            # Store OOF predictions
            self.oof_predictions_[:, model_idx] = model_oof
            
            # Calculate CV score
            from sklearn.metrics import roc_auc_score
            cv_scores[name] = roc_auc_score(y, model_oof)
            logger.info(f"{name} CV AUC: {cv_scores[name]:.4f}")
            
            # Train final model on full data
            final_model = clone(base_model)
            if sample_weight is not None and hasattr(final_model, 'fit'):
                if 'sample_weight' in final_model.fit.__code__.co_varnames:
                    final_model.fit(X, y, sample_weight=sample_weight)
                else:
                    final_model.fit(X, y)
            else:
                final_model.fit(X, y)
            
            self.fitted_base_models_[name] = final_model
        
        # Train meta-model on OOF predictions
        logger.info("Training meta-model on OOF predictions")
        
        if self.use_features_in_meta:
            # Combine OOF predictions with original features
            meta_features = np.hstack([self.oof_predictions_, X])
        else:
            meta_features = self.oof_predictions_
        
        self.meta_model_ = clone(self.meta_model)
        self.meta_model_.fit(meta_features, y)
        
        # Calculate feature importance if meta-model supports it
        self._calculate_feature_importance()
        
        # Store CV scores
        self.cv_scores_ = cv_scores
        
        return self
    
    def predict_proba(
        self,
        X: Union[np.ndarray, pd.DataFrame]
    ) -> np.ndarray:
        """
        Predict probabilities using stacking ensemble.
        
        Args:
            X: Features to predict
            
        Returns:
            Predicted probabilities
        """
        X = self._validate_input(X)
        
        # Generate base model predictions
        base_predictions = self._get_base_predictions(X)
        
        # Prepare meta features
        if self.use_features_in_meta:
            meta_features = np.hstack([base_predictions, X])
        else:
            meta_features = base_predictions
        
        # Get meta-model predictions
        if hasattr(self.meta_model_, 'predict_proba'):
            probas = self.meta_model_.predict_proba(meta_features)
        else:
            # For regression meta-models, clip to [0, 1]
            probas = self.meta_model_.predict(meta_features)
            probas = np.clip(probas, 0, 1)
            probas = np.vstack([1 - probas, probas]).T
        
        return probas
    
    def predict(
        self,
        X: Union[np.ndarray, pd.DataFrame]
    ) -> np.ndarray:
        """
        Predict classes using stacking ensemble.
        
        Args:
            X: Features to predict
            
        Returns:
            Predicted classes
        """
        probas = self.predict_proba(X)
        return (probas[:, 1] >= 0.5).astype(int)
    
    def _get_cv_splits(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: np.ndarray
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Get cross-validation splits based on strategy."""
        if self.cv_strategy == 'temporal':
            cv = TimeSeriesSplit(n_splits=self.n_splits)
        else:  # stratified
            cv = StratifiedKFold(
                n_splits=self.n_splits,
                shuffle=True,
                random_state=self.random_state
            )
        
        return list(cv.split(X, y))
    
    def _get_base_predictions(
        self,
        X: Union[np.ndarray, pd.DataFrame]
    ) -> np.ndarray:
        """Get predictions from all base models."""
        n_samples = X.shape[0]
        n_models = len(self.fitted_base_models_)
        
        if self.use_probas:
            base_predictions = np.zeros((n_samples, n_models))
        else:
            base_predictions = np.zeros((n_samples, n_models))
        
        for idx, (name, model) in enumerate(self.fitted_base_models_.items()):
            if self.use_probas:
                preds = model.predict_proba(X)[:, 1]
            else:
                preds = model.predict(X)
            
            base_predictions[:, idx] = preds
        
        return base_predictions
    
    def _validate_input(
        self,
        X: Union[np.ndarray, pd.DataFrame]
    ) -> np.ndarray:
        """Validate and convert input to numpy array."""
        if isinstance(X, pd.DataFrame):
            return X.values
        return np.array(X)
    
    def _calculate_feature_importance(self):
        """Calculate feature importance from meta-model."""
        if hasattr(self.meta_model_, 'coef_'):
            # Linear model coefficients
            importance = np.abs(self.meta_model_.coef_).flatten()
            
            # Create feature names
            feature_names = list(self.base_models.keys())
            if self.use_features_in_meta:
                feature_names += [f'original_feature_{i}' for i in range(
                    len(importance) - len(self.base_models)
                )]
            
            # Trim to match importance length
            feature_names = feature_names[:len(importance)]
            
            self.feature_importance_ = pd.DataFrame({
                'feature': feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False)
    
    def get_oof_predictions(self) -> np.ndarray:
        """Get out-of-fold predictions from training."""
        return self.oof_predictions_
    
    def save(self, filepath: str):
        """Save stacking ensemble to disk."""
        save_dict = {
            'base_models': self.fitted_base_models_,
            'meta_model': self.meta_model_,
            'oof_predictions': self.oof_predictions_,
            'cv_scores': self.cv_scores_,
            'feature_importance': self.feature_importance_,
            'params': {
                'cv_strategy': self.cv_strategy,
                'n_splits': self.n_splits,
                'use_probas': self.use_probas,
                'use_features_in_meta': self.use_features_in_meta
            }
        }
        joblib.dump(save_dict, filepath)
        logger.info(f"Saved stacking ensemble to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'OOFStackingEnsemble':
        """Load stacking ensemble from disk."""
        save_dict = joblib.load(filepath)
        
        # Create instance with saved parameters
        ensemble = cls(**save_dict['params'])
        
        # Restore fitted models
        ensemble.fitted_base_models_ = save_dict['base_models']
        ensemble.meta_model_ = save_dict['meta_model']
        ensemble.oof_predictions_ = save_dict['oof_predictions']
        ensemble.cv_scores_ = save_dict['cv_scores']
        ensemble.feature_importance_ = save_dict['feature_importance']
        
        return ensemble


class TemporalStackingEnsemble(OOFStackingEnsemble):
    """
    Temporal-aware stacking ensemble specifically for UFC predictions.
    Handles time-based features and ensures no future data leakage.
    """
    
    def __init__(
        self,
        base_models: Optional[Dict[str, BaseEstimator]] = None,
        meta_model: Optional[BaseEstimator] = None,
        date_col: str = 'date',
        gap_days: int = 14,
        **kwargs
    ):
        """
        Initialize temporal stacking ensemble.
        
        Args:
            base_models: Dictionary of base models
            meta_model: Meta-learner model
            date_col: Column containing dates
            gap_days: Gap between train and validation
            **kwargs: Additional arguments for parent class
        """
        super().__init__(
            base_models=base_models,
            meta_model=meta_model,
            cv_strategy='temporal',
            **kwargs
        )
        self.date_col = date_col
        self.gap_days = gap_days
    
    def fit_with_dates(
        self,
        X: pd.DataFrame,
        y: Union[np.ndarray, pd.Series],
        dates: pd.Series,
        sample_weight: Optional[np.ndarray] = None
    ) -> 'TemporalStackingEnsemble':
        """
        Fit with temporal awareness using dates.
        
        Args:
            X: Training features
            y: Training labels
            dates: Fight dates for temporal splitting
            sample_weight: Sample weights
            
        Returns:
            Fitted ensemble
        """
        # Sort by date
        sort_idx = dates.argsort()
        X = X.iloc[sort_idx]
        y = y[sort_idx]
        dates = dates.iloc[sort_idx]
        
        if sample_weight is not None:
            sample_weight = sample_weight[sort_idx]
        
        # Store dates for later use
        self.dates_ = dates
        
        # Fit using parent method
        return self.fit(X, y, sample_weight)
    
    def _get_cv_splits(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: np.ndarray
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Get temporal CV splits with gap."""
        if not hasattr(self, 'dates_'):
            # Fallback to parent method
            return super()._get_cv_splits(X, y)
        
        n_samples = len(y)
        splits = []
        
        # Create temporal splits with gap
        split_size = n_samples // (self.n_splits + 1)
        
        for i in range(self.n_splits):
            train_end = (i + 1) * split_size
            val_start = train_end + self.gap_days  # Gap to prevent leakage
            val_end = min(val_start + split_size, n_samples)
            
            if val_start < n_samples:
                train_idx = np.arange(0, train_end)
                val_idx = np.arange(val_start, val_end)
                splits.append((train_idx, val_idx))
        
        return splits


def create_meta_features(
    base_predictions: np.ndarray,
    include_interactions: bool = True,
    include_statistics: bool = True
) -> np.ndarray:
    """
    Create enhanced meta-features from base predictions.
    
    Args:
        base_predictions: Array of base model predictions
        include_interactions: Include interaction features
        include_statistics: Include statistical features
        
    Returns:
        Enhanced meta-features
    """
    features = [base_predictions]
    
    if include_interactions:
        # Pairwise products
        n_models = base_predictions.shape[1]
        for i in range(n_models):
            for j in range(i + 1, n_models):
                interaction = base_predictions[:, i] * base_predictions[:, j]
                features.append(interaction.reshape(-1, 1))
    
    if include_statistics:
        # Statistical features
        features.append(np.mean(base_predictions, axis=1).reshape(-1, 1))
        features.append(np.std(base_predictions, axis=1).reshape(-1, 1))
        features.append(np.max(base_predictions, axis=1).reshape(-1, 1))
        features.append(np.min(base_predictions, axis=1).reshape(-1, 1))
    
    return np.hstack(features)