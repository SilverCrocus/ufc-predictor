"""
Bradley-Terry model for UFC fighter ratings and opponent quality adjustment.
Provides head-to-head probability predictions and strength ratings.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from scipy.optimize import minimize
from scipy.special import expit
import logging
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class BradleyTerryResult:
    """Container for Bradley-Terry model results."""
    ratings: Dict[str, float]
    log_likelihood: float
    n_iterations: int
    converged: bool
    metadata: Dict[str, Any]


class BradleyTerryModel:
    """
    Bradley-Terry model for fighter ratings.
    
    The Bradley-Terry model estimates strength parameters for each fighter
    such that P(i beats j) = exp(θ_i) / (exp(θ_i) + exp(θ_j))
    """
    
    def __init__(
        self,
        home_advantage: bool = False,
        regularization: float = 0.01,
        max_iterations: int = 1000,
        convergence_tol: float = 1e-6
    ):
        """
        Initialize Bradley-Terry model.
        
        Args:
            home_advantage: Whether to include home advantage parameter
            regularization: L2 regularization strength
            max_iterations: Maximum optimization iterations
            convergence_tol: Convergence tolerance
        """
        self.home_advantage = home_advantage
        self.regularization = regularization
        self.max_iterations = max_iterations
        self.convergence_tol = convergence_tol
        self.ratings: Dict[str, float] = {}
        self.is_fitted = False
        self.home_advantage_param = 0.0
    
    def fit(self, fights_df: pd.DataFrame) -> BradleyTerryResult:
        """
        Fit Bradley-Terry model to fight results.
        
        Args:
            fights_df: DataFrame with columns:
                - fighter_a: Name of fighter A
                - fighter_b: Name of fighter B
                - winner: 1 if fighter_a won, 0 if fighter_b won
                - date: Fight date (optional, for temporal weighting)
                - is_home_a: Whether fighter_a is fighting at home (optional)
                
        Returns:
            BradleyTerryResult with fitted ratings
        """
        # Prepare data
        fighters = self._get_unique_fighters(fights_df)
        fighter_to_idx = {f: i for i, f in enumerate(fighters)}
        n_fighters = len(fighters)
        
        logger.info(f"Fitting Bradley-Terry model for {n_fighters} fighters")
        
        # Initialize parameters (log-scale for numerical stability)
        if self.home_advantage:
            params = np.zeros(n_fighters + 1)  # Extra param for home advantage
        else:
            params = np.zeros(n_fighters)
        
        # Set first fighter as reference (rating = 0)
        params[0] = 0
        
        # Prepare fight data
        fight_data = self._prepare_fight_data(fights_df, fighter_to_idx)
        
        # Optimize using maximum likelihood
        result = minimize(
            fun=self._negative_log_likelihood,
            x0=params,
            args=(fight_data, n_fighters),
            method='L-BFGS-B',
            options={
                'maxiter': self.max_iterations,
                'ftol': self.convergence_tol
            }
        )
        
        # Extract ratings
        self._extract_ratings(result.x, fighters)
        
        # Create result object
        bt_result = BradleyTerryResult(
            ratings=self.ratings,
            log_likelihood=-result.fun,
            n_iterations=result.nit,
            converged=result.success,
            metadata={
                'n_fighters': n_fighters,
                'n_fights': len(fights_df),
                'home_advantage': self.home_advantage_param if self.home_advantage else None
            }
        )
        
        self.is_fitted = True
        logger.info(f"Bradley-Terry fitting complete. Converged: {result.success}")
        
        return bt_result
    
    def _get_unique_fighters(self, fights_df: pd.DataFrame) -> List[str]:
        """Get unique fighter names from fights DataFrame."""
        fighters_a = fights_df['fighter_a'].unique()
        fighters_b = fights_df['fighter_b'].unique()
        return sorted(list(set(fighters_a) | set(fighters_b)))
    
    def _prepare_fight_data(
        self,
        fights_df: pd.DataFrame,
        fighter_to_idx: Dict[str, int]
    ) -> List[Tuple[int, int, int, float, int]]:
        """
        Prepare fight data for optimization.
        
        Returns:
            List of tuples (fighter_a_idx, fighter_b_idx, winner, weight, is_home_a)
        """
        fight_data = []
        
        for _, row in fights_df.iterrows():
            fighter_a_idx = fighter_to_idx[row['fighter_a']]
            fighter_b_idx = fighter_to_idx[row['fighter_b']]
            winner = int(row['winner'])  # 1 if fighter_a won, 0 otherwise
            
            # Temporal weighting (recent fights more important)
            if 'date' in row and pd.notna(row['date']):
                days_ago = (datetime.now() - pd.to_datetime(row['date'])).days
                weight = np.exp(-days_ago / 365)  # Exponential decay over years
            else:
                weight = 1.0
            
            # Home advantage
            is_home_a = int(row.get('is_home_a', 0))
            
            fight_data.append((fighter_a_idx, fighter_b_idx, winner, weight, is_home_a))
        
        return fight_data
    
    def _negative_log_likelihood(
        self,
        params: np.ndarray,
        fight_data: List[Tuple],
        n_fighters: int
    ) -> float:
        """
        Calculate negative log-likelihood for optimization.
        
        Args:
            params: Parameter vector (fighter ratings + optional home advantage)
            fight_data: Prepared fight data
            n_fighters: Number of fighters
            
        Returns:
            Negative log-likelihood
        """
        # Extract ratings
        ratings = params[:n_fighters]
        
        # Constrain first fighter rating to 0 (reference)
        ratings[0] = 0
        
        # Extract home advantage if applicable
        if self.home_advantage and len(params) > n_fighters:
            home_param = params[n_fighters]
        else:
            home_param = 0
        
        # Calculate log-likelihood
        log_likelihood = 0
        
        for fighter_a_idx, fighter_b_idx, winner, weight, is_home_a in fight_data:
            # Calculate probability of fighter_a winning
            rating_diff = ratings[fighter_a_idx] - ratings[fighter_b_idx]
            
            if self.home_advantage:
                rating_diff += home_param * is_home_a
            
            prob_a_wins = expit(rating_diff)  # Sigmoid function
            
            # Add to log-likelihood
            if winner == 1:
                log_likelihood += weight * np.log(prob_a_wins + 1e-10)
            else:
                log_likelihood += weight * np.log(1 - prob_a_wins + 1e-10)
        
        # Add L2 regularization
        regularization_term = self.regularization * np.sum(ratings ** 2)
        
        return -(log_likelihood - regularization_term)
    
    def _extract_ratings(self, params: np.ndarray, fighters: List[str]):
        """Extract and store fighter ratings from optimized parameters."""
        n_fighters = len(fighters)
        ratings = params[:n_fighters]
        
        # Normalize ratings (mean = 0)
        ratings = ratings - np.mean(ratings)
        
        # Store ratings
        self.ratings = {fighter: float(rating) for fighter, rating in zip(fighters, ratings)}
        
        # Store home advantage if applicable
        if self.home_advantage and len(params) > n_fighters:
            self.home_advantage_param = float(params[n_fighters])
    
    def predict_probability(
        self,
        fighter_a: str,
        fighter_b: str,
        is_home_a: bool = False
    ) -> float:
        """
        Predict probability that fighter_a beats fighter_b.
        
        Args:
            fighter_a: Name of fighter A
            fighter_b: Name of fighter B
            is_home_a: Whether fighter_a is fighting at home
            
        Returns:
            Probability that fighter_a wins
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        # Get ratings (default to 0 for unknown fighters)
        rating_a = self.ratings.get(fighter_a, 0)
        rating_b = self.ratings.get(fighter_b, 0)
        
        # Calculate rating difference
        rating_diff = rating_a - rating_b
        
        # Add home advantage if applicable
        if self.home_advantage and is_home_a:
            rating_diff += self.home_advantage_param
        
        # Calculate probability
        prob = expit(rating_diff)
        
        return prob
    
    def get_top_fighters(self, n: int = 10) -> List[Tuple[str, float]]:
        """
        Get top n fighters by rating.
        
        Returns:
            List of (fighter_name, rating) tuples
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        sorted_fighters = sorted(
            self.ratings.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return sorted_fighters[:n]
    
    def get_fighter_rating(self, fighter: str) -> float:
        """Get rating for a specific fighter."""
        return self.ratings.get(fighter, 0.0)


class EloRatingSystem:
    """
    Elo rating system as an alternative to Bradley-Terry.
    Provides online updates after each fight.
    """
    
    def __init__(
        self,
        initial_rating: float = 1500,
        k_factor: float = 32,
        home_advantage: float = 50
    ):
        """
        Initialize Elo rating system.
        
        Args:
            initial_rating: Starting rating for new fighters
            k_factor: Rating change factor
            home_advantage: Home advantage in rating points
        """
        self.initial_rating = initial_rating
        self.k_factor = k_factor
        self.home_advantage = home_advantage
        self.ratings: Dict[str, float] = {}
        self.rating_history: Dict[str, List[Tuple[datetime, float]]] = {}
    
    def update_ratings(
        self,
        fighter_a: str,
        fighter_b: str,
        winner: int,
        date: Optional[datetime] = None,
        is_home_a: bool = False
    ):
        """
        Update ratings after a fight.
        
        Args:
            fighter_a: Name of fighter A
            fighter_b: Name of fighter B
            winner: 1 if fighter_a won, 0 if fighter_b won
            date: Fight date
            is_home_a: Whether fighter_a is fighting at home
        """
        # Get current ratings
        rating_a = self.ratings.get(fighter_a, self.initial_rating)
        rating_b = self.ratings.get(fighter_b, self.initial_rating)
        
        # Apply home advantage
        if is_home_a:
            rating_a += self.home_advantage
        
        # Calculate expected scores
        expected_a = 1 / (1 + 10 ** ((rating_b - rating_a) / 400))
        expected_b = 1 - expected_a
        
        # Calculate new ratings
        actual_a = winner
        actual_b = 1 - winner
        
        new_rating_a = rating_a + self.k_factor * (actual_a - expected_a)
        new_rating_b = rating_b + self.k_factor * (actual_b - expected_b)
        
        # Remove home advantage from stored rating
        if is_home_a:
            new_rating_a -= self.home_advantage
        
        # Update ratings
        self.ratings[fighter_a] = new_rating_a
        self.ratings[fighter_b] = new_rating_b
        
        # Store history
        if date:
            if fighter_a not in self.rating_history:
                self.rating_history[fighter_a] = []
            if fighter_b not in self.rating_history:
                self.rating_history[fighter_b] = []
            
            self.rating_history[fighter_a].append((date, new_rating_a))
            self.rating_history[fighter_b].append((date, new_rating_b))
    
    def predict_probability(
        self,
        fighter_a: str,
        fighter_b: str,
        is_home_a: bool = False
    ) -> float:
        """
        Predict probability that fighter_a beats fighter_b.
        
        Returns:
            Probability that fighter_a wins
        """
        rating_a = self.ratings.get(fighter_a, self.initial_rating)
        rating_b = self.ratings.get(fighter_b, self.initial_rating)
        
        if is_home_a:
            rating_a += self.home_advantage
        
        prob_a = 1 / (1 + 10 ** ((rating_b - rating_a) / 400))
        
        return prob_a
    
    def get_rating(self, fighter: str) -> float:
        """Get current rating for a fighter."""
        return self.ratings.get(fighter, self.initial_rating)


def fit_bradley_terry_temporal(
    fights_df: pd.DataFrame,
    cutoff_date: datetime,
    **kwargs
) -> BradleyTerryModel:
    """
    Fit Bradley-Terry model using only fights before cutoff date.
    
    Args:
        fights_df: DataFrame with fight data
        cutoff_date: Only use fights before this date
        **kwargs: Additional arguments for BradleyTerryModel
        
    Returns:
        Fitted BradleyTerryModel
    """
    # Filter fights before cutoff
    fights_df['date'] = pd.to_datetime(fights_df['date'])
    train_fights = fights_df[fights_df['date'] < cutoff_date].copy()
    
    if len(train_fights) == 0:
        raise ValueError(f"No fights found before {cutoff_date}")
    
    # Fit model
    model = BradleyTerryModel(**kwargs)
    model.fit(train_fights)
    
    return model