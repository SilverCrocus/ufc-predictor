"""
Survival analysis and competing risks model for UFC method of victory prediction.
Predicts KO/TKO, Submission, and Decision outcomes using discrete-time survival models.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import warnings
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class MethodPrediction:
    """Container for method of victory predictions."""
    ko_tko_prob: float
    submission_prob: float
    decision_prob: float
    expected_round: float
    confidence: float


class CompetingRisksModel(BaseEstimator, ClassifierMixin):
    """
    Competing risks model for method of victory prediction.
    Models KO/TKO and Submission as competing events with Decision as censoring.
    """
    
    def __init__(
        self,
        n_rounds: int = 5,
        use_round_features: bool = True,
        regularization: float = 1.0,
        random_state: int = 42
    ):
        """
        Initialize competing risks model.
        
        Args:
            n_rounds: Maximum number of rounds (3 or 5)
            use_round_features: Include round-specific features
            regularization: Regularization strength for logistic models
            random_state: Random seed
        """
        self.n_rounds = n_rounds
        self.use_round_features = use_round_features
        self.regularization = regularization
        self.random_state = random_state
        
        # Separate models for each outcome type
        self.ko_model = None
        self.sub_model = None
        self.round_model = None
        self.scaler = None
        
        self.is_fitted = False
    
    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y_method: Union[np.ndarray, pd.Series],
        y_round: Optional[Union[np.ndarray, pd.Series]] = None,
        sample_weight: Optional[np.ndarray] = None
    ) -> 'CompetingRisksModel':
        """
        Fit competing risks model.
        
        Args:
            X: Features
            y_method: Method outcomes ('KO/TKO', 'SUB', 'DEC')
            y_round: Round of finish (optional)
            sample_weight: Sample weights
            
        Returns:
            Fitted model
        """
        X = self._validate_input(X)
        
        # Standardize features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Create binary targets for each outcome
        y_ko = (y_method == 'KO/TKO').astype(int)
        y_sub = (y_method == 'SUB').astype(int)
        y_dec = (y_method == 'DEC').astype(int)
        
        # Fit KO/TKO model
        self.ko_model = LogisticRegression(
            C=self.regularization,
            random_state=self.random_state,
            max_iter=1000
        )
        self.ko_model.fit(X_scaled, y_ko, sample_weight=sample_weight)
        
        # Fit Submission model
        self.sub_model = LogisticRegression(
            C=self.regularization,
            random_state=self.random_state,
            max_iter=1000
        )
        self.sub_model.fit(X_scaled, y_sub, sample_weight=sample_weight)
        
        # Fit round prediction model if round data provided
        if y_round is not None:
            # Only use finishes for round prediction
            finish_mask = ~y_dec
            if finish_mask.sum() > 0:
                X_finish = X_scaled[finish_mask]
                y_round_finish = y_round[finish_mask]
                
                self.round_model = LogisticRegression(
                    C=self.regularization,
                    random_state=self.random_state,
                    max_iter=1000,
                    multi_class='multinomial'
                )
                self.round_model.fit(X_finish, y_round_finish)
        
        self.is_fitted = True
        return self
    
    def predict_proba(
        self,
        X: Union[np.ndarray, pd.DataFrame]
    ) -> np.ndarray:
        """
        Predict method probabilities.
        
        Args:
            X: Features
            
        Returns:
            Array of shape (n_samples, 3) with columns [KO/TKO, SUB, DEC]
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X = self._validate_input(X)
        X_scaled = self.scaler.transform(X)
        
        # Get raw probabilities
        ko_prob = self.ko_model.predict_proba(X_scaled)[:, 1]
        sub_prob = self.sub_model.predict_proba(X_scaled)[:, 1]
        
        # Apply competing risks adjustment
        # P(KO) and P(SUB) compete, P(DEC) is complement
        probas = np.zeros((X.shape[0], 3))
        
        for i in range(X.shape[0]):
            # Normalize to ensure probabilities sum to 1
            raw_ko = ko_prob[i]
            raw_sub = sub_prob[i]
            
            # Simple normalization approach
            total_finish = raw_ko + raw_sub
            
            if total_finish > 1:
                # Scale down if total exceeds 1
                probas[i, 0] = raw_ko / total_finish
                probas[i, 1] = raw_sub / total_finish
                probas[i, 2] = 0
            else:
                probas[i, 0] = raw_ko
                probas[i, 1] = raw_sub
                probas[i, 2] = 1 - total_finish
        
        # Ensure probabilities are valid
        probas = np.clip(probas, 0, 1)
        probas = probas / probas.sum(axis=1, keepdims=True)
        
        return probas
    
    def predict(
        self,
        X: Union[np.ndarray, pd.DataFrame]
    ) -> np.ndarray:
        """
        Predict most likely method.
        
        Returns:
            Array of predicted methods
        """
        probas = self.predict_proba(X)
        classes = np.array(['KO/TKO', 'SUB', 'DEC'])
        return classes[np.argmax(probas, axis=1)]
    
    def predict_round(
        self,
        X: Union[np.ndarray, pd.DataFrame]
    ) -> np.ndarray:
        """
        Predict expected round of finish.
        
        Returns:
            Array of expected rounds
        """
        if self.round_model is None:
            # Default to middle round if no round model
            return np.full(X.shape[0], (self.n_rounds + 1) / 2)
        
        X = self._validate_input(X)
        X_scaled = self.scaler.transform(X)
        
        # Get round probabilities
        round_probs = self.round_model.predict_proba(X_scaled)
        
        # Calculate expected round
        rounds = np.arange(1, round_probs.shape[1] + 1)
        expected_rounds = np.sum(round_probs * rounds, axis=1)
        
        return expected_rounds
    
    def predict_detailed(
        self,
        X: Union[np.ndarray, pd.DataFrame]
    ) -> List[MethodPrediction]:
        """
        Get detailed predictions including confidence and expected round.
        
        Returns:
            List of MethodPrediction objects
        """
        probas = self.predict_proba(X)
        expected_rounds = self.predict_round(X)
        
        predictions = []
        for i in range(X.shape[0]):
            # Calculate confidence as max probability minus second max
            sorted_probs = np.sort(probas[i])[::-1]
            confidence = sorted_probs[0] - sorted_probs[1]
            
            pred = MethodPrediction(
                ko_tko_prob=probas[i, 0],
                submission_prob=probas[i, 1],
                decision_prob=probas[i, 2],
                expected_round=expected_rounds[i],
                confidence=confidence
            )
            predictions.append(pred)
        
        return predictions
    
    def _validate_input(
        self,
        X: Union[np.ndarray, pd.DataFrame]
    ) -> np.ndarray:
        """Validate and convert input."""
        if isinstance(X, pd.DataFrame):
            return X.values
        return np.array(X)


class DiscreteTimeSurvivalModel(BaseEstimator):
    """
    Discrete-time survival model for round-by-round finish probability.
    Models the hazard of fight ending in each round.
    """
    
    def __init__(
        self,
        max_rounds: int = 5,
        baseline_hazard: str = 'logistic',
        include_time_varying: bool = True,
        random_state: int = 42
    ):
        """
        Initialize discrete-time survival model.
        
        Args:
            max_rounds: Maximum number of rounds
            baseline_hazard: Type of baseline hazard ('constant', 'logistic', 'flexible')
            include_time_varying: Include time-varying covariates
            random_state: Random seed
        """
        self.max_rounds = max_rounds
        self.baseline_hazard = baseline_hazard
        self.include_time_varying = include_time_varying
        self.random_state = random_state
        
        self.hazard_models = {}
        self.is_fitted = False
    
    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        finish_round: Union[np.ndarray, pd.Series],
        finished: Union[np.ndarray, pd.Series],
        sample_weight: Optional[np.ndarray] = None
    ) -> 'DiscreteTimeSurvivalModel':
        """
        Fit survival model.
        
        Args:
            X: Features
            finish_round: Round of finish (1-5, or max_rounds+1 for decision)
            finished: Whether fight finished before decision
            sample_weight: Sample weights
            
        Returns:
            Fitted model
        """
        X = self._validate_input(X)
        
        # Create discrete-time dataset
        dt_data = self._create_discrete_time_data(X, finish_round, finished)
        
        # Fit hazard model for each round
        for round_num in range(1, self.max_rounds + 1):
            round_data = dt_data[dt_data['round'] == round_num]
            
            if len(round_data) > 0:
                # Features for this round
                X_round = round_data.drop(['round', 'event'], axis=1)
                y_round = round_data['event']
                
                # Fit logistic regression for hazard
                hazard_model = LogisticRegression(
                    random_state=self.random_state,
                    max_iter=1000
                )
                hazard_model.fit(X_round, y_round)
                
                self.hazard_models[round_num] = hazard_model
        
        self.is_fitted = True
        return self
    
    def predict_survival_curve(
        self,
        X: Union[np.ndarray, pd.DataFrame]
    ) -> np.ndarray:
        """
        Predict survival curve (probability of reaching each round).
        
        Returns:
            Array of shape (n_samples, max_rounds+1) with survival probabilities
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X = self._validate_input(X)
        n_samples = X.shape[0]
        
        # Initialize survival probabilities
        survival_probs = np.ones((n_samples, self.max_rounds + 1))
        
        # Calculate cumulative survival
        for round_num in range(1, self.max_rounds + 1):
            if round_num in self.hazard_models:
                # Get hazard for this round
                hazard = self.hazard_models[round_num].predict_proba(X)[:, 1]
                
                # Update survival probability
                survival_probs[:, round_num] = survival_probs[:, round_num - 1] * (1 - hazard)
        
        return survival_probs
    
    def predict_hazard_curve(
        self,
        X: Union[np.ndarray, pd.DataFrame]
    ) -> np.ndarray:
        """
        Predict hazard curve (probability of finish in each round).
        
        Returns:
            Array of shape (n_samples, max_rounds) with hazard probabilities
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X = self._validate_input(X)
        n_samples = X.shape[0]
        
        # Initialize hazard probabilities
        hazard_probs = np.zeros((n_samples, self.max_rounds))
        
        for round_num in range(1, self.max_rounds + 1):
            if round_num in self.hazard_models:
                hazard_probs[:, round_num - 1] = self.hazard_models[round_num].predict_proba(X)[:, 1]
        
        return hazard_probs
    
    def predict_expected_duration(
        self,
        X: Union[np.ndarray, pd.DataFrame]
    ) -> np.ndarray:
        """
        Predict expected fight duration in rounds.
        
        Returns:
            Array of expected rounds
        """
        survival_curve = self.predict_survival_curve(X)
        
        # Calculate expected value
        expected_rounds = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            # Sum of survival probabilities gives expected duration
            expected_rounds[i] = np.sum(survival_curve[i, :self.max_rounds])
        
        return expected_rounds
    
    def _create_discrete_time_data(
        self,
        X: np.ndarray,
        finish_round: Union[np.ndarray, pd.Series],
        finished: Union[np.ndarray, pd.Series]
    ) -> pd.DataFrame:
        """
        Create discrete-time dataset for survival analysis.
        Each fight contributes one row per round until finish/censoring.
        """
        dt_rows = []
        
        for i in range(X.shape[0]):
            fight_rounds = int(finish_round[i]) if finished[i] else self.max_rounds
            
            for round_num in range(1, min(fight_rounds + 1, self.max_rounds + 1)):
                row = {
                    'round': round_num,
                    'event': 1 if (finished[i] and round_num == fight_rounds) else 0
                }
                
                # Add features
                for j in range(X.shape[1]):
                    row[f'feature_{j}'] = X[i, j]
                
                # Add time-varying features if requested
                if self.include_time_varying:
                    row['round_squared'] = round_num ** 2
                    row['is_championship_round'] = 1 if round_num > 3 else 0
                
                dt_rows.append(row)
        
        return pd.DataFrame(dt_rows)
    
    def _validate_input(
        self,
        X: Union[np.ndarray, pd.DataFrame]
    ) -> np.ndarray:
        """Validate and convert input."""
        if isinstance(X, pd.DataFrame):
            return X.values
        return np.array(X)


def create_method_features(
    fighter_stats: pd.DataFrame,
    feature_names: List[str] = None
) -> pd.DataFrame:
    """
    Create features specifically for method prediction.
    
    Args:
        fighter_stats: DataFrame with fighter statistics
        feature_names: List of features to create
        
    Returns:
        DataFrame with method-specific features
    """
    features = fighter_stats.copy()
    
    # KO/TKO indicators
    if 'strikes_landed_per_min' in features.columns:
        features['ko_power_score'] = (
            features['strikes_landed_per_min'] * 
            features.get('knockdowns_landed', 0)
        )
    
    # Submission indicators
    if 'submission_attempts_per_15min' in features.columns:
        features['sub_threat_score'] = (
            features['submission_attempts_per_15min'] * 
            features.get('takedowns_per_15min', 1)
        )
    
    # Decision indicators
    if 'control_time_pct' in features.columns:
        features['decision_score'] = (
            features['control_time_pct'] * 
            features.get('significant_strikes_defense', 0.5)
        )
    
    # Finish rate features
    if 'total_fights' in features.columns:
        features['ko_rate'] = features.get('ko_wins', 0) / features['total_fights'].clip(lower=1)
        features['sub_rate'] = features.get('sub_wins', 0) / features['total_fights'].clip(lower=1)
        features['dec_rate'] = features.get('dec_wins', 0) / features['total_fights'].clip(lower=1)
    
    return features