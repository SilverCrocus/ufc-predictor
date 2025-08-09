#!/usr/bin/env python3
"""
Feature Selection Module for UFC Predictor.

Provides consistent feature selection between training and inference,
with support for multiple selection methods and persistence.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json
import joblib
from typing import List, Dict, Tuple, Optional, Union
from sklearn.feature_selection import (
    SelectKBest, 
    f_classif, 
    mutual_info_classif,
    RFE
)
from sklearn.ensemble import RandomForestClassifier
import logging

logger = logging.getLogger(__name__)


class UFCFeatureSelector:
    """
    Manages feature selection for UFC fight prediction models.
    Ensures consistency between training and inference.
    """
    
    def __init__(
        self,
        method: str = 'importance_based',
        n_features: int = 32,
        importance_file: Optional[str] = None
    ):
        """
        Initialize feature selector.
        
        Args:
            method: Selection method ('importance_based', 'mutual_info', 'f_classif', 'recursive')
            n_features: Number of features to select
            importance_file: Path to feature importance CSV (for importance_based method)
        """
        self.method = method
        self.n_features = n_features
        self.importance_file = importance_file
        self.selected_features = None
        self.selector = None
        self.feature_importance_df = None
        
    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: np.ndarray,
        feature_names: Optional[List[str]] = None
    ) -> 'UFCFeatureSelector':
        """
        Fit the feature selector on training data.
        
        Args:
            X: Feature matrix
            y: Target labels
            feature_names: Names of features (if X is numpy array)
            
        Returns:
            Self for chaining
        """
        # Convert to DataFrame if needed
        if isinstance(X, np.ndarray):
            if feature_names is None:
                feature_names = [f'feature_{i}' for i in range(X.shape[1])]
            X = pd.DataFrame(X, columns=feature_names)
        
        self.feature_names = list(X.columns)
        
        if self.method == 'importance_based':
            self._fit_importance_based(X, y)
        elif self.method == 'mutual_info':
            self._fit_mutual_info(X, y)
        elif self.method == 'f_classif':
            self._fit_f_classif(X, y)
        elif self.method == 'recursive':
            self._fit_recursive(X, y)
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        logger.info(f"Selected {len(self.selected_features)} features using {self.method}")
        return self
    
    def _fit_importance_based(self, X: pd.DataFrame, y: np.ndarray):
        """Fit using pre-computed feature importance."""
        if self.importance_file and Path(self.importance_file).exists():
            # Load importance from file
            self.feature_importance_df = pd.read_csv(self.importance_file)
        else:
            # Calculate importance using Random Forest
            logger.info("Calculating feature importance with Random Forest...")
            rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            rf.fit(X, y)
            
            importance = rf.feature_importances_
            self.feature_importance_df = pd.DataFrame({
                'feature': X.columns,
                'importance': importance,
                'importance_pct': importance / importance.sum() * 100
            }).sort_values('importance', ascending=False)
        
        # Select top N features
        self.selected_features = list(
            self.feature_importance_df.head(self.n_features)['feature']
        )
        
        # Calculate cumulative importance
        cumulative_importance = (
            self.feature_importance_df.head(self.n_features)['importance_pct'].sum()
        )
        logger.info(f"Top {self.n_features} features capture {cumulative_importance:.1f}% of importance")
    
    def _fit_mutual_info(self, X: pd.DataFrame, y: np.ndarray):
        """Fit using mutual information."""
        self.selector = SelectKBest(
            score_func=lambda X, y: mutual_info_classif(X, y, random_state=42),
            k=self.n_features
        )
        self.selector.fit(X, y)
        
        # Get selected feature names
        mask = self.selector.get_support()
        self.selected_features = [
            feat for feat, selected in zip(X.columns, mask) if selected
        ]
    
    def _fit_f_classif(self, X: pd.DataFrame, y: np.ndarray):
        """Fit using ANOVA F-statistic."""
        self.selector = SelectKBest(score_func=f_classif, k=self.n_features)
        self.selector.fit(X, y)
        
        # Get selected feature names
        mask = self.selector.get_support()
        self.selected_features = [
            feat for feat, selected in zip(X.columns, mask) if selected
        ]
    
    def _fit_recursive(self, X: pd.DataFrame, y: np.ndarray):
        """Fit using Recursive Feature Elimination."""
        estimator = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
        self.selector = RFE(
            estimator,
            n_features_to_select=self.n_features,
            step=1
        )
        self.selector.fit(X, y)
        
        # Get selected feature names
        mask = self.selector.get_support()
        self.selected_features = [
            feat for feat, selected in zip(X.columns, mask) if selected
        ]
    
    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> Union[pd.DataFrame, np.ndarray]:
        """
        Transform data to selected features.
        
        Args:
            X: Feature matrix to transform
            
        Returns:
            Transformed feature matrix with selected features only
        """
        if self.selected_features is None:
            raise ValueError("Selector must be fitted first")
        
        # Handle numpy arrays
        if isinstance(X, np.ndarray):
            if hasattr(self, 'feature_names'):
                X_df = pd.DataFrame(X, columns=self.feature_names)
                return self._select_features(X_df).values
            else:
                raise ValueError("Cannot transform numpy array without feature names")
        
        return self._select_features(X)
    
    def _select_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Select features from DataFrame."""
        # Find available features
        available_features = [f for f in self.selected_features if f in X.columns]
        
        if len(available_features) < len(self.selected_features):
            missing = set(self.selected_features) - set(available_features)
            logger.warning(f"Missing {len(missing)} features: {missing}")
        
        return X[available_features]
    
    def fit_transform(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: np.ndarray,
        feature_names: Optional[List[str]] = None
    ) -> Union[pd.DataFrame, np.ndarray]:
        """
        Fit selector and transform data in one step.
        
        Args:
            X: Feature matrix
            y: Target labels
            feature_names: Names of features (if X is numpy array)
            
        Returns:
            Transformed feature matrix
        """
        self.fit(X, y, feature_names)
        return self.transform(X)
    
    def save(self, filepath: str):
        """
        Save feature selector configuration.
        
        Args:
            filepath: Path to save configuration
        """
        config = {
            'method': self.method,
            'n_features': self.n_features,
            'selected_features': self.selected_features,
            'feature_names': getattr(self, 'feature_names', None)
        }
        
        # Save importance data if available
        if self.feature_importance_df is not None:
            config['feature_importance'] = self.feature_importance_df.to_dict('records')
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Feature selector saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'UFCFeatureSelector':
        """
        Load feature selector from saved configuration.
        
        Args:
            filepath: Path to configuration file
            
        Returns:
            Loaded feature selector
        """
        with open(filepath, 'r') as f:
            config = json.load(f)
        
        selector = cls(
            method=config['method'],
            n_features=config['n_features']
        )
        selector.selected_features = config['selected_features']
        selector.feature_names = config.get('feature_names')
        
        # Load importance data if available
        if 'feature_importance' in config:
            selector.feature_importance_df = pd.DataFrame(config['feature_importance'])
        
        logger.info(f"Feature selector loaded from {filepath}")
        return selector
    
    def get_feature_importance_summary(self) -> pd.DataFrame:
        """
        Get summary of feature importance.
        
        Returns:
            DataFrame with feature importance information
        """
        if self.feature_importance_df is not None:
            return self.feature_importance_df.head(self.n_features)
        elif self.selector and hasattr(self.selector, 'scores_'):
            # For SelectKBest methods
            scores = self.selector.scores_
            return pd.DataFrame({
                'feature': self.feature_names,
                'score': scores
            }).sort_values('score', ascending=False).head(self.n_features)
        else:
            return pd.DataFrame({
                'feature': self.selected_features,
                'selected': True
            })
    
    def analyze_feature_reduction_impact(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_test: pd.DataFrame,
        y_test: np.ndarray,
        model_class=RandomForestClassifier,
        **model_kwargs
    ) -> Dict:
        """
        Analyze the impact of feature reduction on model performance.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_test: Test features
            y_test: Test labels
            model_class: Model class to use for evaluation
            **model_kwargs: Arguments for model initialization
            
        Returns:
            Dictionary with performance comparison
        """
        from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
        
        # Default model parameters
        if not model_kwargs:
            model_kwargs = {'n_estimators': 100, 'random_state': 42, 'n_jobs': -1}
        
        # Train with all features
        model_all = model_class(**model_kwargs)
        model_all.fit(X_train, y_train)
        
        y_pred_all = model_all.predict(X_test)
        y_proba_all = model_all.predict_proba(X_test)[:, 1]
        
        metrics_all = {
            'accuracy': accuracy_score(y_test, y_pred_all),
            'auc': roc_auc_score(y_test, y_proba_all),
            'f1': f1_score(y_test, y_pred_all),
            'n_features': X_train.shape[1]
        }
        
        # Train with selected features
        X_train_selected = self.transform(X_train)
        X_test_selected = self.transform(X_test)
        
        model_selected = model_class(**model_kwargs)
        model_selected.fit(X_train_selected, y_train)
        
        y_pred_selected = model_selected.predict(X_test_selected)
        y_proba_selected = model_selected.predict_proba(X_test_selected)[:, 1]
        
        metrics_selected = {
            'accuracy': accuracy_score(y_test, y_pred_selected),
            'auc': roc_auc_score(y_test, y_proba_selected),
            'f1': f1_score(y_test, y_pred_selected),
            'n_features': X_train_selected.shape[1]
        }
        
        # Calculate impact
        impact = {
            'accuracy_diff': metrics_selected['accuracy'] - metrics_all['accuracy'],
            'auc_diff': metrics_selected['auc'] - metrics_all['auc'],
            'f1_diff': metrics_selected['f1'] - metrics_all['f1'],
            'feature_reduction': X_train.shape[1] - X_train_selected.shape[1],
            'feature_reduction_pct': (1 - X_train_selected.shape[1] / X_train.shape[1]) * 100
        }
        
        return {
            'all_features': metrics_all,
            'selected_features': metrics_selected,
            'impact': impact,
            'selected_feature_names': self.selected_features
        }


def create_optimized_feature_selector(
    data_path: str = None,
    importance_path: str = None,
    n_features: int = 32
) -> UFCFeatureSelector:
    """
    Create an optimized feature selector for UFC predictions.
    
    Args:
        data_path: Path to training data
        importance_path: Path to feature importance CSV
        n_features: Number of features to select
        
    Returns:
        Configured UFCFeatureSelector
    """
    # Default paths
    if importance_path is None:
        importance_path = 'artifacts/feature_importance/feature_importance.csv'
    
    # Create selector
    selector = UFCFeatureSelector(
        method='importance_based',
        n_features=n_features,
        importance_file=importance_path
    )
    
    # If data provided, fit the selector
    if data_path:
        df = pd.read_csv(data_path)
        
        # Prepare features and target
        exclude_cols = ['Date', 'date', 'Winner', 'winner', 'Outcome', 'outcome',
                       'blue_fighter', 'red_fighter', 'Fighter', 'Opponent',
                       'loser_fighter', 'Event', 'event', 'Method', 'method', 
                       'Time', 'time', 'fighter_a', 'fighter_b', 'Round',
                       'fighter_url', 'opponent_url', 'blue_fighter_url', 'red_fighter_url',
                       'blue_Name', 'red_Name']
        
        feature_cols = [col for col in df.columns 
                       if col not in exclude_cols and pd.api.types.is_numeric_dtype(df[col])]
        
        X = df[feature_cols]
        # Get target - handle different column names
        if 'Winner' in df.columns:
            y = df['Winner']
        elif 'Outcome' in df.columns:
            y = (df['Outcome'] == 'W').astype(int)
        else:
            y = df.get('winner', pd.Series(dtype=int))
        
        selector.fit(X, y)
    
    return selector