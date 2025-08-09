"""
Parlay correlation estimation for UFC multi-bet analysis.
Estimates correlation between fight outcomes on the same card.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics.pairwise import cosine_similarity
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ParlayAnalysis:
    """Container for parlay analysis results."""
    legs: List[Dict[str, Any]]
    correlation_matrix: np.ndarray
    combined_probability: float
    adjusted_probability: float
    expected_value: float
    correlation_penalty: float
    recommended: bool


class ParlayCorrelationEstimator:
    """
    Estimates correlation between UFC fight outcomes for parlay betting.
    """
    
    def __init__(
        self,
        base_correlation: float = 0.08,
        same_division_bonus: float = 0.05,
        teammate_correlation: float = 0.15,
        max_correlation: float = 0.30,
        min_legs: int = 2,
        max_legs: int = 4
    ):
        """
        Initialize parlay correlation estimator.
        
        Args:
            base_correlation: Base correlation for same-event fights
            same_division_bonus: Additional correlation for same division
            teammate_correlation: Correlation for teammates/training partners
            max_correlation: Maximum allowed correlation
            min_legs: Minimum parlay legs
            max_legs: Maximum parlay legs
        """
        self.base_correlation = base_correlation
        self.same_division_bonus = same_division_bonus
        self.teammate_correlation = teammate_correlation
        self.max_correlation = max_correlation
        self.min_legs = min_legs
        self.max_legs = max_legs
        
        # Historical correlation data (would be fitted on historical data)
        self.historical_correlations = {}
    
    def estimate_correlation(
        self,
        fight1: Dict[str, Any],
        fight2: Dict[str, Any]
    ) -> float:
        """
        Estimate correlation between two fight outcomes.
        
        Args:
            fight1: Dict with fight 1 details
            fight2: Dict with fight 2 details
            
        Returns:
            Estimated correlation coefficient
        """
        correlation = 0.0
        
        # Same event correlation
        if fight1.get('event') == fight2.get('event'):
            correlation += self.base_correlation
            
            # Same division bonus
            if fight1.get('division') == fight2.get('division'):
                correlation += self.same_division_bonus
            
            # Teammate/training partner correlation
            if self._are_teammates(fight1, fight2):
                correlation += self.teammate_correlation
            
            # Style matchup correlation
            style_corr = self._calculate_style_correlation(fight1, fight2)
            correlation += style_corr
            
            # Betting line correlation
            line_corr = self._calculate_line_correlation(fight1, fight2)
            correlation += line_corr
        
        # Cap at maximum
        correlation = min(correlation, self.max_correlation)
        
        return correlation
    
    def _are_teammates(
        self,
        fight1: Dict[str, Any],
        fight2: Dict[str, Any]
    ) -> bool:
        """Check if fighters are teammates or training partners."""
        # Check if fighters share a camp/team
        camp1 = fight1.get('camp', '')
        camp2 = fight2.get('camp', '')
        
        if camp1 and camp2 and camp1 == camp2:
            return True
        
        # Could expand with known teammate relationships
        known_teams = {
            'American Top Team': ['Poirier', 'Masvidal', 'Covington'],
            'City Kickboxing': ['Adesanya', 'Volkanovski', 'Hooker'],
            'Team Alpha Male': ['Faber', 'Garbrandt', 'Emmett'],
        }
        
        for team, fighters in known_teams.items():
            fighter1_in_team = any(f in fight1.get('fighter', '') for f in fighters)
            fighter2_in_team = any(f in fight2.get('fighter', '') for f in fighters)
            if fighter1_in_team and fighter2_in_team:
                return True
        
        return False
    
    def _calculate_style_correlation(
        self,
        fight1: Dict[str, Any],
        fight2: Dict[str, Any]
    ) -> float:
        """Calculate correlation based on fighting styles."""
        # Simplified style correlation
        style1 = fight1.get('style', '')
        style2 = fight2.get('style', '')
        
        # Wrestlers tend to have correlated outcomes
        if 'wrestler' in style1.lower() and 'wrestler' in style2.lower():
            return 0.03
        
        # Strikers vs grapplers may have negative correlation
        if ('striker' in style1.lower() and 'grappler' in style2.lower()) or \
           ('grappler' in style1.lower() and 'striker' in style2.lower()):
            return -0.02
        
        return 0.0
    
    def _calculate_line_correlation(
        self,
        fight1: Dict[str, Any],
        fight2: Dict[str, Any]
    ) -> float:
        """Calculate correlation based on betting lines."""
        # Heavy favorites may have correlated outcomes
        odds1 = fight1.get('odds', 2.0)
        odds2 = fight2.get('odds', 2.0)
        
        # Both heavy favorites
        if odds1 < 1.3 and odds2 < 1.3:
            return 0.05
        
        # Both underdogs
        if odds1 > 3.0 and odds2 > 3.0:
            return 0.03
        
        # One favorite, one dog
        if (odds1 < 1.5 and odds2 > 2.5) or (odds2 < 1.5 and odds1 > 2.5):
            return -0.02
        
        return 0.0
    
    def analyze_parlay(
        self,
        legs: List[Dict[str, Any]],
        correlation_penalty_factor: float = 0.08
    ) -> ParlayAnalysis:
        """
        Analyze a parlay bet with correlation adjustment.
        
        Args:
            legs: List of parlay legs with fight details and probabilities
            correlation_penalty_factor: Penalty factor for correlation
            
        Returns:
            ParlayAnalysis with adjusted probabilities and recommendation
        """
        n_legs = len(legs)
        
        if n_legs < self.min_legs or n_legs > self.max_legs:
            raise ValueError(f"Parlay must have {self.min_legs}-{self.max_legs} legs")
        
        # Build correlation matrix
        correlation_matrix = np.eye(n_legs)
        
        for i in range(n_legs):
            for j in range(i + 1, n_legs):
                corr = self.estimate_correlation(legs[i], legs[j])
                correlation_matrix[i, j] = corr
                correlation_matrix[j, i] = corr
        
        # Calculate combined probability (assuming independence)
        probs = [leg['probability'] for leg in legs]
        combined_prob_independent = np.prod(probs)
        
        # Adjust for correlation
        avg_correlation = (correlation_matrix.sum() - n_legs) / (n_legs * (n_legs - 1))
        correlation_penalty = avg_correlation * correlation_penalty_factor
        
        # Adjusted probability (simplified adjustment)
        adjusted_prob = combined_prob_independent * (1 - correlation_penalty)
        
        # Calculate expected value
        combined_odds = np.prod([leg['odds'] for leg in legs])
        expected_value = (adjusted_prob * combined_odds) - 1
        
        # Recommendation based on EV
        min_edge = 0.08  # 8% minimum edge for parlays
        recommended = expected_value > min_edge
        
        return ParlayAnalysis(
            legs=legs,
            correlation_matrix=correlation_matrix,
            combined_probability=combined_prob_independent,
            adjusted_probability=adjusted_prob,
            expected_value=expected_value,
            correlation_penalty=correlation_penalty,
            recommended=recommended
        )
    
    def find_best_parlays(
        self,
        opportunities: pd.DataFrame,
        max_parlays: int = 10,
        min_combined_odds: float = 2.0,
        max_combined_odds: float = 20.0
    ) -> List[ParlayAnalysis]:
        """
        Find best parlay combinations from available opportunities.
        
        Args:
            opportunities: DataFrame with betting opportunities
            max_parlays: Maximum number of parlays to return
            min_combined_odds: Minimum combined odds
            max_combined_odds: Maximum combined odds
            
        Returns:
            List of best parlay combinations
        """
        parlays = []
        n_opps = len(opportunities)
        
        # Generate all valid combinations
        from itertools import combinations
        
        for n_legs in range(self.min_legs, min(self.max_legs + 1, n_opps + 1)):
            for combo in combinations(range(n_opps), n_legs):
                legs = []
                combined_odds = 1.0
                
                for idx in combo:
                    opp = opportunities.iloc[idx]
                    legs.append({
                        'fighter': opp.get('fighter', f'Fighter_{idx}'),
                        'probability': opp['probability'],
                        'odds': opp['odds'],
                        'event': opp.get('event', 'Unknown'),
                        'division': opp.get('division', 'Unknown')
                    })
                    combined_odds *= opp['odds']
                
                # Check odds constraints
                if min_combined_odds <= combined_odds <= max_combined_odds:
                    try:
                        analysis = self.analyze_parlay(legs)
                        if analysis.recommended:
                            parlays.append(analysis)
                    except Exception as e:
                        logger.debug(f"Error analyzing parlay: {e}")
        
        # Sort by expected value
        parlays.sort(key=lambda x: x.expected_value, reverse=True)
        
        return parlays[:max_parlays]


class FeatureCorrelationAnalyzer:
    """
    Analyzes correlation between fighter features for multi-bet risk assessment.
    """
    
    def __init__(self):
        """Initialize feature correlation analyzer."""
        self.feature_weights = {
            'stance': 0.1,
            'age': 0.05,
            'height': 0.05,
            'reach': 0.05,
            'style': 0.15,
            'camp': 0.2,
            'recent_form': 0.1,
            'ranking': 0.15,
            'experience': 0.15
        }
    
    def calculate_fighter_similarity(
        self,
        fighter1_features: Dict[str, Any],
        fighter2_features: Dict[str, Any]
    ) -> float:
        """
        Calculate similarity between two fighters based on features.
        
        Args:
            fighter1_features: Features for fighter 1
            fighter2_features: Features for fighter 2
            
        Returns:
            Similarity score [0, 1]
        """
        similarity = 0.0
        total_weight = 0.0
        
        for feature, weight in self.feature_weights.items():
            if feature in fighter1_features and feature in fighter2_features:
                feat1 = fighter1_features[feature]
                feat2 = fighter2_features[feature]
                
                if isinstance(feat1, (int, float)) and isinstance(feat2, (int, float)):
                    # Numerical features
                    diff = abs(feat1 - feat2)
                    max_val = max(abs(feat1), abs(feat2))
                    if max_val > 0:
                        feat_similarity = 1 - (diff / max_val)
                    else:
                        feat_similarity = 1.0
                elif isinstance(feat1, str) and isinstance(feat2, str):
                    # Categorical features
                    feat_similarity = 1.0 if feat1 == feat2 else 0.0
                else:
                    continue
                
                similarity += weight * feat_similarity
                total_weight += weight
        
        if total_weight > 0:
            similarity /= total_weight
        
        return similarity
    
    def estimate_outcome_correlation(
        self,
        similarity_score: float,
        same_event: bool = True
    ) -> float:
        """
        Estimate outcome correlation based on fighter similarity.
        
        Args:
            similarity_score: Fighter similarity score
            same_event: Whether fights are on same event
            
        Returns:
            Estimated correlation
        """
        if not same_event:
            return 0.0
        
        # Map similarity to correlation (non-linear relationship)
        if similarity_score < 0.3:
            correlation = 0.0
        elif similarity_score < 0.5:
            correlation = 0.05
        elif similarity_score < 0.7:
            correlation = 0.10
        elif similarity_score < 0.9:
            correlation = 0.15
        else:
            correlation = 0.20
        
        return correlation


def calculate_parlay_variance(
    probabilities: List[float],
    correlation_matrix: np.ndarray
) -> float:
    """
    Calculate variance of parlay outcome considering correlation.
    
    Args:
        probabilities: List of individual win probabilities
        correlation_matrix: Correlation matrix between outcomes
        
    Returns:
        Parlay variance
    """
    n = len(probabilities)
    
    # Individual variances
    variances = [p * (1 - p) for p in probabilities]
    
    # Total variance with correlation
    total_variance = 0.0
    
    for i in range(n):
        total_variance += variances[i]
        
        for j in range(i + 1, n):
            covariance = correlation_matrix[i, j] * np.sqrt(variances[i] * variances[j])
            total_variance += 2 * covariance
    
    return total_variance