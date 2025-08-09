"""
Odds utilities for UFC betting analysis.
Includes vig removal, implied probability conversion, and fair odds calculation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from scipy import optimize
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class OddsInfo:
    """Container for odds information."""
    decimal_odds: float
    american_odds: int
    implied_prob: float
    fair_prob: Optional[float]
    vig: Optional[float]


class OddsConverter:
    """
    Convert between different odds formats and probabilities.
    """
    
    @staticmethod
    def decimal_to_american(decimal_odds: float) -> int:
        """Convert decimal odds to American odds."""
        if decimal_odds < 1:
            raise ValueError(f"Invalid decimal odds: {decimal_odds}")
        
        if decimal_odds >= 2.0:
            # Positive American odds
            return int((decimal_odds - 1) * 100)
        else:
            # Negative American odds
            return int(-100 / (decimal_odds - 1))
    
    @staticmethod
    def american_to_decimal(american_odds: int) -> float:
        """Convert American odds to decimal odds."""
        if american_odds > 0:
            return 1 + (american_odds / 100)
        else:
            return 1 + (100 / abs(american_odds))
    
    @staticmethod
    def decimal_to_implied_prob(decimal_odds: float) -> float:
        """Convert decimal odds to implied probability."""
        if decimal_odds <= 1:
            raise ValueError(f"Invalid decimal odds: {decimal_odds}")
        return 1 / decimal_odds
    
    @staticmethod
    def implied_prob_to_decimal(prob: float) -> float:
        """Convert implied probability to decimal odds."""
        if prob <= 0 or prob >= 1:
            raise ValueError(f"Probability must be between 0 and 1: {prob}")
        return 1 / prob
    
    @staticmethod
    def fractional_to_decimal(fractional: str) -> float:
        """
        Convert fractional odds (e.g., "5/2") to decimal.
        
        Args:
            fractional: Fractional odds as string
            
        Returns:
            Decimal odds
        """
        try:
            numerator, denominator = map(float, fractional.split('/'))
            return 1 + (numerator / denominator)
        except:
            raise ValueError(f"Invalid fractional odds format: {fractional}")


class VigCalculator:
    """
    Calculate and remove vigorish (bookmaker margin) from odds.
    """
    
    @staticmethod
    def calculate_vig_two_way(
        odds_a: float,
        odds_b: float,
        format: str = 'decimal'
    ) -> float:
        """
        Calculate vig for two-way market.
        
        Args:
            odds_a: Odds for outcome A
            odds_b: Odds for outcome B
            format: Odds format ('decimal', 'american')
            
        Returns:
            Vig as percentage
        """
        if format == 'american':
            odds_a = OddsConverter.american_to_decimal(odds_a)
            odds_b = OddsConverter.american_to_decimal(odds_b)
        
        implied_a = OddsConverter.decimal_to_implied_prob(odds_a)
        implied_b = OddsConverter.decimal_to_implied_prob(odds_b)
        
        total_implied = implied_a + implied_b
        vig = (total_implied - 1) * 100
        
        return vig
    
    @staticmethod
    def remove_vig_two_way(
        odds_a: float,
        odds_b: float,
        method: str = 'balanced',
        format: str = 'decimal'
    ) -> Tuple[float, float]:
        """
        Remove vig from two-way market to get fair probabilities.
        
        Args:
            odds_a: Odds for outcome A
            odds_b: Odds for outcome B
            method: Vig removal method ('balanced', 'proportional', 'logarithmic')
            format: Odds format
            
        Returns:
            Tuple of (fair_prob_a, fair_prob_b)
        """
        if format == 'american':
            odds_a = OddsConverter.american_to_decimal(odds_a)
            odds_b = OddsConverter.american_to_decimal(odds_b)
        
        implied_a = OddsConverter.decimal_to_implied_prob(odds_a)
        implied_b = OddsConverter.decimal_to_implied_prob(odds_b)
        
        if method == 'balanced':
            # Equal distribution of vig
            total_implied = implied_a + implied_b
            fair_a = implied_a / total_implied
            fair_b = implied_b / total_implied
            
        elif method == 'proportional':
            # Proportional to implied probabilities
            total_implied = implied_a + implied_b
            vig = total_implied - 1
            fair_a = implied_a - (implied_a * vig / total_implied)
            fair_b = implied_b - (implied_b * vig / total_implied)
            
        elif method == 'logarithmic':
            # Shin method (more sophisticated)
            fair_a, fair_b = VigCalculator._shin_method(implied_a, implied_b)
            
        else:
            raise ValueError(f"Unknown vig removal method: {method}")
        
        # Ensure probabilities sum to 1
        total = fair_a + fair_b
        return fair_a / total, fair_b / total
    
    @staticmethod
    def _shin_method(p1: float, p2: float) -> Tuple[float, float]:
        """
        Shin's method for vig removal.
        Accounts for favorite-longshot bias.
        """
        # Solve quadratic equation for z (proportion of insider trading)
        total = p1 + p2
        b = total - 1
        
        if b <= 0:
            return p1, p2
        
        # Quadratic formula
        discriminant = b * b + 4 * p1 * p2
        z = (np.sqrt(discriminant) - 1) / (total - 2)
        
        # Calculate fair probabilities
        fair_p1 = (p1 - z) / (1 - z)
        fair_p2 = (p2 - z) / (1 - z)
        
        return fair_p1, fair_p2
    
    @staticmethod
    def calculate_fair_odds(
        odds_list: List[float],
        method: str = 'balanced'
    ) -> List[float]:
        """
        Calculate fair odds for multi-way market.
        
        Args:
            odds_list: List of decimal odds
            method: Vig removal method
            
        Returns:
            List of fair decimal odds
        """
        # Convert to implied probabilities
        implied_probs = [OddsConverter.decimal_to_implied_prob(odds) for odds in odds_list]
        
        # Remove vig
        total_implied = sum(implied_probs)
        
        if method == 'balanced':
            fair_probs = [p / total_implied for p in implied_probs]
        else:
            # For multi-way, default to balanced
            fair_probs = [p / total_implied for p in implied_probs]
        
        # Convert back to odds
        fair_odds = [OddsConverter.implied_prob_to_decimal(p) for p in fair_probs]
        
        return fair_odds


class OddsAggregator:
    """
    Aggregate odds from multiple bookmakers.
    """
    
    @staticmethod
    def get_best_odds(
        odds_dict: Dict[str, Dict[str, float]],
        outcome: str
    ) -> Tuple[float, str]:
        """
        Get best odds for an outcome across bookmakers.
        
        Args:
            odds_dict: Dict of {bookmaker: {outcome: odds}}
            outcome: Outcome to find best odds for
            
        Returns:
            Tuple of (best_odds, bookmaker)
        """
        best_odds = 0
        best_book = None
        
        for book, odds in odds_dict.items():
            if outcome in odds and odds[outcome] > best_odds:
                best_odds = odds[outcome]
                best_book = book
        
        return best_odds, best_book
    
    @staticmethod
    def calculate_consensus_odds(
        odds_dict: Dict[str, Dict[str, float]],
        weights: Optional[Dict[str, float]] = None,
        method: str = 'mean'
    ) -> Dict[str, float]:
        """
        Calculate consensus odds from multiple bookmakers.
        
        Args:
            odds_dict: Dict of {bookmaker: {outcome: odds}}
            weights: Optional weights for each bookmaker
            method: Aggregation method ('mean', 'median', 'weighted')
            
        Returns:
            Dict of consensus odds by outcome
        """
        # Collect all outcomes
        all_outcomes = set()
        for book_odds in odds_dict.values():
            all_outcomes.update(book_odds.keys())
        
        consensus = {}
        
        for outcome in all_outcomes:
            outcome_odds = []
            outcome_weights = []
            
            for book, odds in odds_dict.items():
                if outcome in odds:
                    outcome_odds.append(odds[outcome])
                    if weights:
                        outcome_weights.append(weights.get(book, 1.0))
                    else:
                        outcome_weights.append(1.0)
            
            if outcome_odds:
                if method == 'mean':
                    consensus[outcome] = np.mean(outcome_odds)
                elif method == 'median':
                    consensus[outcome] = np.median(outcome_odds)
                elif method == 'weighted':
                    consensus[outcome] = np.average(outcome_odds, weights=outcome_weights)
                else:
                    raise ValueError(f"Unknown aggregation method: {method}")
        
        return consensus
    
    @staticmethod
    def detect_arbitrage(
        odds_dict: Dict[str, Dict[str, float]]
    ) -> Optional[Dict[str, Any]]:
        """
        Detect arbitrage opportunities across bookmakers.
        
        Args:
            odds_dict: Dict of {bookmaker: {outcome: odds}}
            
        Returns:
            Arbitrage info if opportunity exists, None otherwise
        """
        # Get all outcomes
        all_outcomes = set()
        for book_odds in odds_dict.values():
            all_outcomes.update(book_odds.keys())
        
        # Find best odds for each outcome
        best_odds = {}
        best_books = {}
        
        for outcome in all_outcomes:
            odds, book = OddsAggregator.get_best_odds(odds_dict, outcome)
            if odds > 0:
                best_odds[outcome] = odds
                best_books[outcome] = book
        
        # Check if arbitrage exists
        if len(best_odds) >= 2:
            total_implied = sum(1 / odds for odds in best_odds.values())
            
            if total_implied < 1:
                # Arbitrage opportunity exists
                profit_pct = (1 / total_implied - 1) * 100
                
                # Calculate optimal stakes
                stakes = {}
                for outcome, odds in best_odds.items():
                    stakes[outcome] = (1 / odds) / total_implied
                
                return {
                    'exists': True,
                    'profit_pct': profit_pct,
                    'best_odds': best_odds,
                    'best_books': best_books,
                    'optimal_stakes': stakes,
                    'total_implied': total_implied
                }
        
        return None


class LineMovementAnalyzer:
    """
    Analyze odds movements and line shopping opportunities.
    """
    
    @staticmethod
    def calculate_line_movement(
        open_odds: float,
        current_odds: float,
        format: str = 'decimal'
    ) -> Dict[str, float]:
        """
        Calculate line movement metrics.
        
        Args:
            open_odds: Opening odds
            current_odds: Current odds
            format: Odds format
            
        Returns:
            Dict with movement metrics
        """
        if format == 'american':
            open_odds = OddsConverter.american_to_decimal(open_odds)
            current_odds = OddsConverter.american_to_decimal(current_odds)
        
        # Calculate implied probability change
        open_prob = OddsConverter.decimal_to_implied_prob(open_odds)
        current_prob = OddsConverter.decimal_to_implied_prob(current_odds)
        
        prob_change = current_prob - open_prob
        prob_change_pct = (prob_change / open_prob) * 100 if open_prob > 0 else 0
        
        # Calculate odds movement
        odds_change = current_odds - open_odds
        odds_change_pct = (odds_change / open_odds) * 100 if open_odds > 0 else 0
        
        return {
            'odds_change': odds_change,
            'odds_change_pct': odds_change_pct,
            'prob_change': prob_change,
            'prob_change_pct': prob_change_pct,
            'steam_move': abs(prob_change_pct) > 5,  # Significant movement
            'direction': 'shortening' if odds_change < 0 else 'drifting'
        }
    
    @staticmethod
    def identify_sharp_action(
        odds_history: pd.DataFrame,
        volume_threshold: float = 0.7
    ) -> bool:
        """
        Identify potential sharp action based on line movement.
        
        Args:
            odds_history: DataFrame with odds history
            volume_threshold: Threshold for volume spike
            
        Returns:
            Whether sharp action is detected
        """
        if len(odds_history) < 2:
            return False
        
        # Look for sudden line moves
        odds_changes = odds_history['odds'].pct_change()
        
        # Sharp action indicators:
        # 1. Large single move (>3% change)
        large_moves = (abs(odds_changes) > 0.03).any()
        
        # 2. Consistent direction with acceleration
        if len(odds_changes) >= 3:
            recent_moves = odds_changes.tail(3)
            consistent = ((recent_moves > 0).all() or (recent_moves < 0).all())
            accelerating = abs(recent_moves.iloc[-1]) > abs(recent_moves.iloc[0])
            sharp_pattern = consistent and accelerating
        else:
            sharp_pattern = False
        
        return large_moves or sharp_pattern


def calculate_expected_value(
    prob: float,
    odds: float,
    format: str = 'decimal'
) -> float:
    """
    Calculate expected value of a bet.
    
    Args:
        prob: True probability of winning
        odds: Offered odds
        format: Odds format
        
    Returns:
        Expected value as percentage
    """
    if format == 'american':
        odds = OddsConverter.american_to_decimal(odds)
    
    ev = (prob * odds) - 1
    return ev * 100  # Return as percentage


def required_probability(
    odds: float,
    format: str = 'decimal'
) -> float:
    """
    Calculate minimum probability needed for positive EV.
    
    Args:
        odds: Offered odds
        format: Odds format
        
    Returns:
        Required probability
    """
    if format == 'american':
        odds = OddsConverter.american_to_decimal(odds)
    
    return 1 / odds