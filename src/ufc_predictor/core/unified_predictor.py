#!/usr/bin/env python3
"""
Unified UFC Fight Predictor - Consolidated implementation
Combines the best of all predictor modules into a single, clean interface.
"""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Optional, List, Any
from dataclasses import dataclass

from ..data.feature_engineering import (
    engineer_features_final,
    create_differential_features
)


@dataclass
class PredictionResult:
    """Standardized prediction result."""
    fighter1: str
    fighter2: str
    fighter1_prob: float
    fighter2_prob: float
    predicted_winner: str
    confidence: float
    method_prediction: Optional[Dict[str, float]] = None
    metadata: Optional[Dict[str, Any]] = None


class UnifiedUFCPredictor:
    """
    Unified predictor that consolidates all prediction functionality.
    Supports both winner and method predictions with a clean interface.
    """
    
    def __init__(self, 
                 winner_model_path: Optional[str] = None,
                 method_model_path: Optional[str] = None,
                 fighters_data_path: Optional[str] = None):
        """
        Initialize predictor with optional model paths.
        If not provided, will auto-detect latest models.
        """
        self.winner_model_path = winner_model_path or self._find_latest_model('winner')
        self.method_model_path = method_model_path or self._find_latest_model('method')
        self.fighters_data_path = fighters_data_path or self._find_latest_data()
        
        self._load_models()
        self._load_data()
    
    def _find_latest_model(self, model_type: str) -> str:
        """Auto-detect latest trained model."""
        model_dir = Path('model')
        
        # Try versioned directories first
        training_dirs = sorted([d for d in model_dir.iterdir() 
                               if d.is_dir() and d.name.startswith('training_')],
                              reverse=True)
        
        for train_dir in training_dirs:
            if model_type == 'winner':
                model_files = list(train_dir.glob('ufc_winner_model_tuned_*.joblib'))
                if not model_files:
                    model_files = list(train_dir.glob('ufc_winner_model_*.joblib'))
            else:
                model_files = list(train_dir.glob('ufc_method_model_*.joblib'))
            
            if model_files:
                return str(model_files[0])
        
        # Fallback to root model directory
        if model_type == 'winner':
            default = model_dir / 'ufc_random_forest_model_tuned.joblib'
        else:
            default = model_dir / 'ufc_multiclass_model.joblib'
        
        if default.exists():
            return str(default)
        
        raise FileNotFoundError(f"No {model_type} model found")
    
    def _find_latest_data(self) -> str:
        """Auto-detect latest fighter data."""
        model_dir = Path('model')
        
        # Try versioned directories first
        training_dirs = sorted([d for d in model_dir.iterdir() 
                               if d.is_dir() and d.name.startswith('training_')],
                              reverse=True)
        
        for train_dir in training_dirs:
            data_files = list(train_dir.glob('ufc_fighters_engineered_*.csv'))
            if data_files:
                return str(data_files[0])
        
        # Fallback to root model directory
        default = model_dir / 'ufc_fighters_engineered_corrected.csv'
        if default.exists():
            return str(default)
        
        raise FileNotFoundError("No fighters data found")
    
    def _load_models(self):
        """Load prediction models."""
        self.winner_model = joblib.load(self.winner_model_path)
        self.method_model = None
        
        if self.method_model_path and Path(self.method_model_path).exists():
            self.method_model = joblib.load(self.method_model_path)
        
        # Load feature columns
        winner_cols_path = Path(self.winner_model_path).parent / 'winner_model_columns.json'
        if winner_cols_path.exists():
            import json
            with open(winner_cols_path) as f:
                self.feature_columns = json.load(f)
        else:
            # Will be determined from data
            self.feature_columns = None
    
    def _load_data(self):
        """Load fighters data."""
        self.fighters_df = pd.read_csv(self.fighters_data_path)
        
        # Ensure fighter names are strings
        if 'fighter' in self.fighters_df.columns:
            self.fighters_df['fighter'] = self.fighters_df['fighter'].astype(str)
    
    def predict_fight(self, 
                     fighter1: str, 
                     fighter2: str,
                     include_method: bool = True) -> PredictionResult:
        """
        Predict a single fight outcome.
        
        Args:
            fighter1: Name of first fighter
            fighter2: Name of second fighter
            include_method: Whether to include method prediction
            
        Returns:
            PredictionResult with all prediction details
        """
        # Get fighter data
        f1_data = self._get_fighter_data(fighter1)
        f2_data = self._get_fighter_data(fighter2)
        
        if f1_data is None or f2_data is None:
            missing = fighter1 if f1_data is None else fighter2
            raise ValueError(f"Fighter not found: {missing}")
        
        # Create differential features
        features_df = create_differential_features(f1_data, f2_data)
        
        # Ensure correct feature columns
        if self.feature_columns:
            features_df = features_df[self.feature_columns]
        
        # Make winner prediction
        winner_prob = self.winner_model.predict_proba(features_df)[0]
        
        # Average with reverse prediction for symmetry
        features_rev = create_differential_features(f2_data, f1_data)
        if self.feature_columns:
            features_rev = features_rev[self.feature_columns]
        winner_prob_rev = self.winner_model.predict_proba(features_rev)[0]
        
        # Average predictions
        f1_prob = (winner_prob[1] + (1 - winner_prob_rev[1])) / 2
        f2_prob = 1 - f1_prob
        
        predicted_winner = fighter1 if f1_prob > 0.5 else fighter2
        confidence = max(f1_prob, f2_prob)
        
        # Method prediction if requested
        method_pred = None
        if include_method and self.method_model:
            method_pred = self._predict_method(features_df, features_rev)
        
        return PredictionResult(
            fighter1=fighter1,
            fighter2=fighter2,
            fighter1_prob=f1_prob,
            fighter2_prob=f2_prob,
            predicted_winner=predicted_winner,
            confidence=confidence,
            method_prediction=method_pred
        )
    
    def predict_card(self, fights: List[Tuple[str, str]], 
                    include_method: bool = True) -> List[PredictionResult]:
        """
        Predict multiple fights (e.g., full card).
        
        Args:
            fights: List of (fighter1, fighter2) tuples
            include_method: Whether to include method predictions
            
        Returns:
            List of PredictionResult objects
        """
        results = []
        for fighter1, fighter2 in fights:
            try:
                result = self.predict_fight(fighter1, fighter2, include_method)
                results.append(result)
            except Exception as e:
                print(f"⚠️  Error predicting {fighter1} vs {fighter2}: {e}")
        
        return results
    
    def _get_fighter_data(self, fighter_name: str) -> Optional[pd.Series]:
        """Get fighter data with fuzzy name matching."""
        # Exact match first
        exact = self.fighters_df[self.fighters_df['fighter'] == fighter_name]
        if not exact.empty:
            return exact.iloc[-1]  # Latest record
        
        # Fuzzy match
        fighter_lower = fighter_name.lower()
        for idx, row in self.fighters_df.iterrows():
            if fighter_lower in row['fighter'].lower() or row['fighter'].lower() in fighter_lower:
                return row
        
        return None
    
    def _predict_method(self, features_df: pd.DataFrame, 
                       features_rev: pd.DataFrame) -> Dict[str, float]:
        """Predict fight ending method."""
        method_probs = self.method_model.predict_proba(features_df)[0]
        method_probs_rev = self.method_model.predict_proba(features_rev)[0]
        
        # Average predictions
        avg_probs = (method_probs + method_probs_rev) / 2
        
        # Map to method names
        methods = ['Decision', 'KO/TKO', 'Submission']
        return {method: prob for method, prob in zip(methods, avg_probs)}
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded models."""
        return {
            'winner_model': self.winner_model_path,
            'method_model': self.method_model_path,
            'fighters_data': self.fighters_data_path,
            'num_fighters': len(self.fighters_df),
            'features': len(self.feature_columns) if self.feature_columns else 'Auto'
        }