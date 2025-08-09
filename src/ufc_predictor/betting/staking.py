"""
Advanced staking strategies for UFC betting.
Includes pessimistic Kelly, fractional Kelly, and portfolio optimization.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from scipy.optimize import minimize
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class StakeRecommendation:
    """Container for staking recommendation."""
    bet_amount: float
    kelly_fraction: float
    edge: float
    expected_value: float
    confidence_level: float
    risk_adjusted: bool


class KellyStaking:
    """
    Kelly criterion staking with various modifications.
    """
    
    def __init__(
        self,
        kelly_fraction: float = 0.25,
        max_bet_pct: float = 0.05,
        min_bet_pct: float = 0.005,
        use_pessimistic: bool = True,
        confidence_level: float = 0.20
    ):
        """
        Initialize Kelly staking calculator.
        
        Args:
            kelly_fraction: Fraction of full Kelly to use
            max_bet_pct: Maximum bet as percentage of bankroll
            min_bet_pct: Minimum bet as percentage of bankroll
            use_pessimistic: Use pessimistic probability estimates
            confidence_level: Confidence level for pessimistic Kelly (e.g., 0.20 for 20th percentile)
        """
        self.kelly_fraction = kelly_fraction
        self.max_bet_pct = max_bet_pct
        self.min_bet_pct = min_bet_pct
        self.use_pessimistic = use_pessimistic
        self.confidence_level = confidence_level
    
    def calculate_kelly_stake(
        self,
        prob: float,
        odds: float,
        bankroll: float,
        prob_lower: Optional[float] = None,
        prob_upper: Optional[float] = None
    ) -> StakeRecommendation:
        """
        Calculate Kelly stake with safety constraints.
        
        Args:
            prob: Estimated probability of winning
            odds: Decimal odds offered
            bankroll: Current bankroll
            prob_lower: Lower confidence bound for probability
            prob_upper: Upper confidence bound for probability
            
        Returns:
            StakeRecommendation with bet details
        """
        # Use pessimistic probability if available and requested
        if self.use_pessimistic and prob_lower is not None:
            kelly_prob = prob_lower
            risk_adjusted = True
        else:
            kelly_prob = prob
            risk_adjusted = False
        
        # Calculate edge
        edge = (kelly_prob * odds) - 1
        
        # No bet if negative edge
        if edge <= 0:
            return StakeRecommendation(
                bet_amount=0,
                kelly_fraction=0,
                edge=edge,
                expected_value=edge * 100,
                confidence_level=kelly_prob,
                risk_adjusted=risk_adjusted
            )
        
        # Full Kelly calculation
        # f = (p * b - q) / b, where b = odds - 1, q = 1 - p
        b = odds - 1
        q = 1 - kelly_prob
        full_kelly = (kelly_prob * b - q) / b
        
        # Apply fractional Kelly
        kelly_fraction = full_kelly * self.kelly_fraction
        
        # Apply constraints
        kelly_fraction = max(0, kelly_fraction)  # No negative bets
        kelly_fraction = min(kelly_fraction, self.max_bet_pct)  # Max cap
        
        # Check minimum threshold
        if kelly_fraction < self.min_bet_pct and kelly_fraction > 0:
            kelly_fraction = 0  # Don't bet if below minimum
        
        # Calculate bet amount
        bet_amount = bankroll * kelly_fraction
        
        return StakeRecommendation(
            bet_amount=bet_amount,
            kelly_fraction=kelly_fraction,
            edge=edge,
            expected_value=edge * 100,
            confidence_level=kelly_prob,
            risk_adjusted=risk_adjusted
        )
    
    def calculate_pessimistic_kelly(
        self,
        prob_samples: np.ndarray,
        odds: float,
        bankroll: float
    ) -> StakeRecommendation:
        """
        Calculate pessimistic Kelly using bootstrap samples.
        
        Args:
            prob_samples: Array of probability samples (e.g., from bootstrap)
            odds: Decimal odds
            bankroll: Current bankroll
            
        Returns:
            StakeRecommendation
        """
        # Get pessimistic probability estimate
        prob_lower = np.percentile(prob_samples, self.confidence_level * 100)
        prob_mean = np.mean(prob_samples)
        prob_upper = np.percentile(prob_samples, (1 - self.confidence_level) * 100)
        
        return self.calculate_kelly_stake(
            prob=prob_mean,
            odds=odds,
            bankroll=bankroll,
            prob_lower=prob_lower,
            prob_upper=prob_upper
        )
    
    def calculate_simultaneous_kelly(
        self,
        probs: List[float],
        odds: List[float],
        bankroll: float,
        correlation_matrix: Optional[np.ndarray] = None
    ) -> List[float]:
        """
        Calculate Kelly stakes for multiple simultaneous bets.
        
        Args:
            probs: List of probabilities
            odds: List of decimal odds
            bankroll: Current bankroll
            correlation_matrix: Correlation between bets
            
        Returns:
            List of recommended stakes
        """
        n_bets = len(probs)
        
        if correlation_matrix is None:
            # Assume independence
            correlation_matrix = np.eye(n_bets)
        
        # Objective function (negative expected log growth)
        def objective(stakes):
            # Expected returns for each outcome combination
            expected_growth = 0
            
            # For simplicity, use mean-variance approximation
            returns = np.array([p * (o - 1) - (1 - p) for p, o in zip(probs, odds)])
            expected_return = np.dot(stakes, returns)
            
            # Variance calculation with correlation
            variances = [(o - 1) ** 2 * p * (1 - p) for p, o in zip(probs, odds)]
            variance_matrix = np.outer(variances, variances) * correlation_matrix
            portfolio_variance = np.dot(stakes, np.dot(variance_matrix, stakes))
            
            # Kelly objective (log utility)
            growth = expected_return - 0.5 * portfolio_variance
            return -growth  # Minimize negative growth
        
        # Constraints
        constraints = [
            {'type': 'ineq', 'fun': lambda x: x},  # Non-negative stakes
            {'type': 'ineq', 'fun': lambda x: self.max_bet_pct * n_bets - np.sum(x)}  # Total stake limit
        ]
        
        # Initial guess (equal stakes)
        x0 = np.ones(n_bets) * self.kelly_fraction * self.max_bet_pct
        
        # Optimize
        result = minimize(
            objective,
            x0,
            method='SLSQP',
            constraints=constraints,
            bounds=[(0, self.max_bet_pct) for _ in range(n_bets)]
        )
        
        if result.success:
            stakes = result.x * bankroll
        else:
            # Fallback to individual Kelly
            stakes = []
            for p, o in zip(probs, odds):
                rec = self.calculate_kelly_stake(p, o, bankroll)
                stakes.append(rec.bet_amount)
        
        return stakes


class DynamicStaking:
    """
    Dynamic staking based on confidence and market conditions.
    """
    
    def __init__(
        self,
        confidence_buckets: List[float] = None,
        stake_multipliers: List[float] = None,
        base_stake_pct: float = 0.02
    ):
        """
        Initialize dynamic staking.
        
        Args:
            confidence_buckets: Confidence thresholds
            stake_multipliers: Stake multipliers for each bucket
            base_stake_pct: Base stake as percentage of bankroll
        """
        self.confidence_buckets = confidence_buckets or [0.55, 0.60, 0.65, 0.70, 0.75]
        self.stake_multipliers = stake_multipliers or [0.5, 0.75, 1.0, 1.25, 1.5]
        self.base_stake_pct = base_stake_pct
    
    def calculate_dynamic_stake(
        self,
        prob: float,
        odds: float,
        bankroll: float,
        model_confidence: Optional[float] = None,
        market_conditions: Optional[str] = None
    ) -> float:
        """
        Calculate dynamic stake based on confidence.
        
        Args:
            prob: Predicted probability
            odds: Decimal odds
            bankroll: Current bankroll
            model_confidence: Model confidence score
            market_conditions: Market condition ('bull', 'bear', 'neutral')
            
        Returns:
            Recommended stake
        """
        # Determine confidence level
        if model_confidence is not None:
            confidence = model_confidence
        else:
            # Use probability as proxy for confidence
            confidence = abs(prob - 0.5) * 2  # Scale to [0, 1]
        
        # Find appropriate bucket
        multiplier = self.stake_multipliers[0]  # Default to lowest
        for threshold, mult in zip(self.confidence_buckets, self.stake_multipliers):
            if confidence >= threshold:
                multiplier = mult
        
        # Adjust for market conditions
        if market_conditions == 'bear':
            multiplier *= 0.8  # Reduce stakes in unfavorable conditions
        elif market_conditions == 'bull':
            multiplier *= 1.1  # Slightly increase in favorable conditions
        
        # Calculate stake
        base_stake = bankroll * self.base_stake_pct
        stake = base_stake * multiplier
        
        return stake


class PortfolioOptimizer:
    """
    Portfolio optimization for multiple UFC bets.
    """
    
    def __init__(
        self,
        max_bets: int = 5,
        max_exposure_pct: float = 0.15,
        diversification_penalty: float = 0.02,
        correlation_threshold: float = 0.5
    ):
        """
        Initialize portfolio optimizer.
        
        Args:
            max_bets: Maximum number of bets in portfolio
            max_exposure_pct: Maximum exposure as percentage of bankroll
            diversification_penalty: Penalty for concentrated portfolio
            correlation_threshold: Threshold for high correlation
        """
        self.max_bets = max_bets
        self.max_exposure_pct = max_exposure_pct
        self.diversification_penalty = diversification_penalty
        self.correlation_threshold = correlation_threshold
    
    def optimize_portfolio(
        self,
        opportunities: pd.DataFrame,
        bankroll: float,
        correlation_matrix: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Optimize bet portfolio using mean-variance optimization.
        
        Args:
            opportunities: DataFrame with columns ['fighter', 'prob', 'odds', 'edge']
            bankroll: Current bankroll
            correlation_matrix: Correlation between bets
            
        Returns:
            Dict with optimal portfolio
        """
        n_opps = len(opportunities)
        
        if n_opps == 0:
            return {'bets': [], 'total_stake': 0, 'expected_return': 0}
        
        # Sort by edge and take top opportunities
        opportunities = opportunities.nlargest(min(n_opps, self.max_bets * 2), 'edge')
        
        probs = opportunities['prob'].values
        odds = opportunities['odds'].values
        edges = opportunities['edge'].values
        
        # Expected returns
        returns = edges
        
        # Covariance matrix (simplified)
        if correlation_matrix is None:
            # Estimate correlation based on event
            correlation_matrix = self._estimate_correlation(opportunities)
        
        # Optimization problem
        def objective(weights):
            # Expected portfolio return
            portfolio_return = np.dot(weights, returns)
            
            # Portfolio variance
            variances = [(o - 1) ** 2 * p * (1 - p) for p, o in zip(probs, odds)]
            cov_matrix = np.outer(variances, variances) * correlation_matrix
            portfolio_variance = np.dot(weights, np.dot(cov_matrix, weights))
            
            # Diversification penalty
            concentration = np.sum(weights ** 2)
            div_penalty = self.diversification_penalty * concentration
            
            # Maximize Sharpe-like ratio (return / risk)
            if portfolio_variance > 0:
                sharpe = portfolio_return / np.sqrt(portfolio_variance)
            else:
                sharpe = portfolio_return
            
            return -(sharpe - div_penalty)  # Minimize negative Sharpe
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # Weights sum to 1
            {'type': 'ineq', 'fun': lambda x: x},  # Non-negative weights
        ]
        
        # Initial guess (equal weights)
        x0 = np.ones(len(opportunities)) / len(opportunities)
        
        # Optimize
        result = minimize(
            objective,
            x0,
            method='SLSQP',
            constraints=constraints,
            bounds=[(0, 1) for _ in range(len(opportunities))]
        )
        
        if result.success:
            weights = result.x
            
            # Convert weights to stakes
            total_exposure = bankroll * self.max_exposure_pct
            stakes = weights * total_exposure
            
            # Filter out tiny bets
            min_stake = bankroll * 0.005
            mask = stakes >= min_stake
            
            selected_opps = opportunities[mask].copy()
            selected_opps['stake'] = stakes[mask]
            
            # Limit to max bets
            if len(selected_opps) > self.max_bets:
                selected_opps = selected_opps.nlargest(self.max_bets, 'stake')
            
            return {
                'bets': selected_opps.to_dict('records'),
                'total_stake': selected_opps['stake'].sum(),
                'expected_return': np.dot(selected_opps['stake'], selected_opps['edge']),
                'n_bets': len(selected_opps),
                'optimization_success': True
            }
        else:
            # Fallback to simple selection
            logger.warning("Portfolio optimization failed, using simple selection")
            return self._simple_selection(opportunities, bankroll)
    
    def _estimate_correlation(self, opportunities: pd.DataFrame) -> np.ndarray:
        """
        Estimate correlation between bets based on event and fighter characteristics.
        """
        n = len(opportunities)
        correlation = np.eye(n)
        
        # Same event bets are correlated
        if 'event' in opportunities.columns:
            for i in range(n):
                for j in range(i + 1, n):
                    if opportunities.iloc[i]['event'] == opportunities.iloc[j]['event']:
                        correlation[i, j] = self.correlation_threshold
                        correlation[j, i] = self.correlation_threshold
        
        return correlation
    
    def _simple_selection(
        self,
        opportunities: pd.DataFrame,
        bankroll: float
    ) -> Dict[str, Any]:
        """
        Simple bet selection based on edge.
        """
        # Select top bets by edge
        selected = opportunities.nlargest(self.max_bets, 'edge').copy()
        
        # Equal stakes
        stake_per_bet = (bankroll * self.max_exposure_pct) / len(selected)
        selected['stake'] = stake_per_bet
        
        return {
            'bets': selected.to_dict('records'),
            'total_stake': selected['stake'].sum(),
            'expected_return': np.dot(selected['stake'], selected['edge']),
            'n_bets': len(selected),
            'optimization_success': False
        }