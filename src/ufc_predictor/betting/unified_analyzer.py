#!/usr/bin/env python3
"""
Unified Betting Analyzer - Consolidated profitability analysis
Combines all betting analysis functionality into a single module.
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import numpy as np


@dataclass
class BettingOpportunity:
    """Represents a betting opportunity."""
    fighter: str
    opponent: str
    win_probability: float
    odds: float
    expected_value: float
    recommended_bet: float
    potential_return: float
    confidence: str
    
    def to_dict(self):
        return asdict(self)


@dataclass
class MultiBetOpportunity:
    """Represents a multi-bet (parlay) opportunity."""
    legs: List[str]
    combined_odds: float
    combined_probability: float
    expected_value: float
    recommended_bet: float
    potential_return: float
    num_legs: int
    
    def to_dict(self):
        return asdict(self)


class UnifiedBettingAnalyzer:
    """
    Unified betting analyzer that handles all profitability calculations.
    Supports single bets, multi-bets, and various betting strategies.
    """
    
    def __init__(self, 
                 bankroll: float = 1000,
                 min_ev_threshold: float = 0.05,
                 max_bet_pct: float = 0.05,
                 kelly_fraction: float = 0.25):
        """
        Initialize analyzer with betting parameters.
        
        Args:
            bankroll: Starting bankroll
            min_ev_threshold: Minimum expected value to consider bet
            max_bet_pct: Maximum percentage of bankroll for single bet
            kelly_fraction: Fraction of Kelly criterion to use (conservative)
        """
        self.bankroll = bankroll
        self.min_ev_threshold = min_ev_threshold
        self.max_bet_pct = max_bet_pct
        self.kelly_fraction = kelly_fraction
        
        # Track betting history
        self.betting_history = []
        self.current_exposure = 0
    
    def analyze_single_bets(self, 
                           predictions: Dict[str, float],
                           odds: Dict[str, float]) -> List[BettingOpportunity]:
        """
        Analyze single betting opportunities.
        
        Args:
            predictions: Fighter win probabilities
            odds: Decimal odds for each fighter
            
        Returns:
            List of profitable betting opportunities
        """
        opportunities = []
        
        for fighter, prob in predictions.items():
            if fighter not in odds:
                continue
            
            decimal_odds = odds[fighter]
            
            # Calculate expected value
            ev = (prob * decimal_odds) - 1
            
            if ev >= self.min_ev_threshold:
                # Calculate optimal bet size (Kelly criterion)
                kelly_bet = self._calculate_kelly_bet(prob, decimal_odds)
                
                # Apply constraints
                recommended_bet = min(
                    kelly_bet * self.bankroll,
                    self.bankroll * self.max_bet_pct
                )
                
                # Determine confidence level
                confidence = self._get_confidence_level(ev, prob)
                
                opportunity = BettingOpportunity(
                    fighter=fighter,
                    opponent=self._find_opponent(fighter, predictions),
                    win_probability=prob,
                    odds=decimal_odds,
                    expected_value=ev,
                    recommended_bet=round(recommended_bet, 2),
                    potential_return=round(recommended_bet * decimal_odds, 2),
                    confidence=confidence
                )
                
                opportunities.append(opportunity)
        
        # Sort by expected value
        opportunities.sort(key=lambda x: x.expected_value, reverse=True)
        
        return opportunities
    
    def analyze_multi_bets(self,
                          single_bets: List[BettingOpportunity],
                          max_legs: int = 4) -> List[MultiBetOpportunity]:
        """
        Analyze multi-bet (parlay) opportunities.
        
        Args:
            single_bets: List of profitable single bets
            max_legs: Maximum number of legs in multi-bet
            
        Returns:
            List of profitable multi-bet opportunities
        """
        multi_opportunities = []
        
        # Generate combinations
        from itertools import combinations
        
        for num_legs in range(2, min(len(single_bets) + 1, max_legs + 1)):
            for combo in combinations(single_bets, num_legs):
                # Calculate combined probability
                combined_prob = np.prod([bet.win_probability for bet in combo])
                
                # Calculate combined odds
                combined_odds = np.prod([bet.odds for bet in combo])
                
                # Apply correlation penalty (same event)
                correlation_penalty = self._calculate_correlation_penalty(combo)
                adjusted_prob = combined_prob * (1 - correlation_penalty)
                
                # Calculate expected value
                ev = (adjusted_prob * combined_odds) - 1
                
                if ev >= self.min_ev_threshold * 1.5:  # Higher threshold for multis
                    # Conservative bet sizing for multis
                    recommended_bet = min(
                        self.bankroll * 0.02,  # Max 2% for multis
                        50  # Cap at $50
                    )
                    
                    multi_opp = MultiBetOpportunity(
                        legs=[bet.fighter for bet in combo],
                        combined_odds=round(combined_odds, 2),
                        combined_probability=round(adjusted_prob, 4),
                        expected_value=round(ev, 4),
                        recommended_bet=recommended_bet,
                        potential_return=round(recommended_bet * combined_odds, 2),
                        num_legs=num_legs
                    )
                    
                    multi_opportunities.append(multi_opp)
        
        # Sort by expected value
        multi_opportunities.sort(key=lambda x: x.expected_value, reverse=True)
        
        return multi_opportunities[:10]  # Return top 10
    
    def calculate_optimal_portfolio(self,
                                  opportunities: List[BettingOpportunity]) -> Dict[str, Any]:
        """
        Calculate optimal betting portfolio considering correlation.
        
        Args:
            opportunities: List of betting opportunities
            
        Returns:
            Optimal portfolio allocation
        """
        if not opportunities:
            return {'bets': [], 'total_stake': 0, 'expected_return': 0}
        
        # Simple greedy allocation for now
        portfolio_bets = []
        total_stake = 0
        expected_return = 0
        remaining_bankroll = self.bankroll * 0.2  # Max 20% exposure
        
        for opp in opportunities:
            if opp.recommended_bet <= remaining_bankroll:
                portfolio_bets.append({
                    'fighter': opp.fighter,
                    'stake': opp.recommended_bet,
                    'expected_return': opp.recommended_bet * (1 + opp.expected_value)
                })
                total_stake += opp.recommended_bet
                expected_return += opp.recommended_bet * (1 + opp.expected_value)
                remaining_bankroll -= opp.recommended_bet
        
        return {
            'bets': portfolio_bets,
            'total_stake': round(total_stake, 2),
            'expected_return': round(expected_return, 2),
            'expected_profit': round(expected_return - total_stake, 2),
            'roi': round((expected_return - total_stake) / total_stake * 100, 2) if total_stake > 0 else 0
        }
    
    def _calculate_kelly_bet(self, prob: float, odds: float) -> float:
        """Calculate Kelly criterion bet size."""
        q = 1 - prob
        kelly = (prob * odds - q) / odds
        
        # Apply Kelly fraction for conservative betting
        return max(0, kelly * self.kelly_fraction)
    
    def _get_confidence_level(self, ev: float, prob: float) -> str:
        """Determine confidence level based on EV and probability."""
        if ev > 0.15 and prob > 0.65:
            return "HIGH"
        elif ev > 0.08 or prob > 0.60:
            return "MEDIUM"
        else:
            return "LOW"
    
    def _find_opponent(self, fighter: str, predictions: Dict[str, float]) -> str:
        """Find opponent fighter name."""
        fighters = list(predictions.keys())
        for f in fighters:
            if f != fighter:
                return f
        return "Unknown"
    
    def _calculate_correlation_penalty(self, bets: List[BettingOpportunity]) -> float:
        """Calculate correlation penalty for multi-bets."""
        # Simple penalty - could be enhanced with actual correlation data
        return 0.08 * (len(bets) - 1)  # 8% penalty per additional leg
    
    def track_bet_result(self, bet: BettingOpportunity, won: bool):
        """Track betting result for performance analysis."""
        result = {
            'timestamp': datetime.now().isoformat(),
            'fighter': bet.fighter,
            'stake': bet.recommended_bet,
            'odds': bet.odds,
            'probability': bet.win_probability,
            'expected_value': bet.expected_value,
            'won': won,
            'profit': bet.recommended_bet * (bet.odds - 1) if won else -bet.recommended_bet
        }
        
        self.betting_history.append(result)
        self.bankroll += result['profit']
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary of betting history."""
        if not self.betting_history:
            return {'message': 'No betting history'}
        
        df = pd.DataFrame(self.betting_history)
        
        return {
            'total_bets': len(df),
            'winning_bets': df['won'].sum(),
            'win_rate': df['won'].mean(),
            'total_staked': df['stake'].sum(),
            'total_profit': df['profit'].sum(),
            'roi': df['profit'].sum() / df['stake'].sum() * 100,
            'current_bankroll': self.bankroll,
            'avg_odds': df['odds'].mean(),
            'avg_ev': df['expected_value'].mean()
        }
    
    def export_analysis(self, 
                       opportunities: List[BettingOpportunity],
                       multi_bets: List[MultiBetOpportunity],
                       filepath: Optional[str] = None) -> str:
        """Export analysis to JSON file."""
        if not filepath:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filepath = f"analysis_{timestamp}.json"
        
        analysis = {
            'timestamp': datetime.now().isoformat(),
            'bankroll': self.bankroll,
            'single_bets': [opp.to_dict() for opp in opportunities],
            'multi_bets': [multi.to_dict() for multi in multi_bets],
            'portfolio': self.calculate_optimal_portfolio(opportunities),
            'parameters': {
                'min_ev_threshold': self.min_ev_threshold,
                'max_bet_pct': self.max_bet_pct,
                'kelly_fraction': self.kelly_fraction
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(analysis, f, indent=2)
        
        return filepath