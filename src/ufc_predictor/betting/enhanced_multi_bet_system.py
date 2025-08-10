"""
Enhanced Multi-Bet UFC Betting System
Production-ready implementation with sophisticated correlation analysis and portfolio management.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from scipy import stats
from scipy.optimize import minimize
from itertools import combinations
import logging
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from .parlay_correlation import ParlayCorrelationAnalyzer
from .staking import KellyStaking, StakeRecommendation
from .correlation import ParlayCorrelationEstimator

logger = logging.getLogger(__name__)


@dataclass
class BetLeg:
    """Individual bet leg for multi-bet analysis."""
    fighter: str
    opponent: str
    probability: float
    odds: float
    edge: float
    event: str
    division: Optional[str] = None
    card_position: Optional[int] = None
    features: Optional[Dict[str, float]] = field(default_factory=dict)
    confidence_interval: Optional[Tuple[float, float]] = None


@dataclass
class MultiBetRecommendation:
    """Container for multi-bet recommendation with full analysis."""
    legs: List[BetLeg]
    bet_type: str  # 'single', 'parlay_2', 'parlay_3'
    combined_probability: float
    adjusted_probability: float
    combined_odds: float
    expected_value: float
    correlation_penalty: float
    kelly_fraction: float
    recommended_stake: float
    risk_score: float
    confidence_score: float
    correlation_matrix: np.ndarray


class EnhancedMultiBetSystem:
    """
    Production-ready multi-bet system with conditional parlay logic,
    correlation estimation, and portfolio management.
    """
    
    def __init__(
        self,
        bankroll: float,
        single_bet_threshold: int = 2,  # Activate parlays only when <2 qualified singles
        max_parlay_legs: int = 3,
        min_single_edge: float = 0.05,  # 5% minimum edge for singles
        min_parlay_edge: float = 0.10,  # 10% minimum edge for parlays
        max_single_stake_pct: float = 0.05,  # 5% max for singles
        max_parlay_stake_pct: float = 0.005,  # 0.5% max for parlays
        max_total_parlay_exposure: float = 0.015,  # 1.5% total parlay exposure
        max_parlays: int = 2,
        correlation_penalty_alpha: float = 1.2,
        confidence_threshold: float = 0.60
    ):
        """
        Initialize enhanced multi-bet system.
        
        Args:
            bankroll: Current bankroll
            single_bet_threshold: Activate parlays only when <N qualified single bets
            max_parlay_legs: Maximum legs in a parlay
            min_single_edge: Minimum edge for single bets
            min_parlay_edge: Minimum edge for parlay bets
            max_single_stake_pct: Max stake % for single bets
            max_parlay_stake_pct: Max stake % per parlay
            max_total_parlay_exposure: Max total parlay exposure
            max_parlays: Maximum number of parlays
            correlation_penalty_alpha: Correlation penalty factor
            confidence_threshold: Minimum confidence for recommendations
        """
        self.bankroll = bankroll
        self.single_bet_threshold = single_bet_threshold
        self.max_parlay_legs = max_parlay_legs
        self.min_single_edge = min_single_edge
        self.min_parlay_edge = min_parlay_edge
        self.max_single_stake_pct = max_single_stake_pct
        self.max_parlay_stake_pct = max_parlay_stake_pct
        self.max_total_parlay_exposure = max_total_parlay_exposure
        self.max_parlays = max_parlays
        self.correlation_penalty_alpha = correlation_penalty_alpha
        self.confidence_threshold = confidence_threshold
        
        # Initialize components
        self.correlation_analyzer = ParlayCorrelationAnalyzer(
            correlation_penalty_alpha=correlation_penalty_alpha
        )
        self.kelly_calculator = KellyStaking(
            kelly_fraction=0.25,
            max_bet_pct=max_single_stake_pct,
            use_pessimistic=True,
            confidence_level=0.20
        )
        
        # Portfolio tracking
        self.active_bets = []
        self.current_exposure = 0.0
        self.parlay_exposure = 0.0
        
    def analyze_betting_opportunities(
        self,
        bet_legs: List[BetLeg],
        live_features: Optional[pd.DataFrame] = None
    ) -> Dict[str, List[MultiBetRecommendation]]:
        """
        Main entry point: analyze all betting opportunities with conditional logic.
        
        Args:
            bet_legs: List of potential bet legs
            live_features: Optional feature DataFrame for correlation analysis
            
        Returns:
            Dict with 'single_bets' and 'parlays' recommendations
        """
        logger.info(f"🎯 Analyzing {len(bet_legs)} betting opportunities")
        
        # Step 1: Identify qualified single bets
        qualified_singles = self._identify_qualified_singles(bet_legs)
        
        logger.info(f"📊 Found {len(qualified_singles)} qualified single bets")
        
        # Step 2: Apply conditional parlay logic
        recommendations = {
            'single_bets': [],
            'parlays': []
        }
        
        if len(qualified_singles) >= self.single_bet_threshold:
            # Sufficient single bets - focus on singles
            logger.info("✅ Sufficient single bets found - recommending single bet strategy")
            recommendations['single_bets'] = qualified_singles
            
        else:
            # Insufficient singles - activate parlay system
            logger.info("🔄 Insufficient single bets - activating parlay system")
            
            # Include remaining single bets
            recommendations['single_bets'] = qualified_singles
            
            # Generate parlay recommendations
            parlay_candidates = bet_legs if len(qualified_singles) == 0 else \
                              [leg for leg in bet_legs if leg not in [s.legs[0] for s in qualified_singles]]
                              
            parlays = self._generate_optimal_parlays(parlay_candidates, live_features)
            recommendations['parlays'] = parlays
            
        # Step 3: Apply portfolio management
        final_recommendations = self._apply_portfolio_management(recommendations)
        
        return final_recommendations
    
    def _identify_qualified_singles(self, bet_legs: List[BetLeg]) -> List[MultiBetRecommendation]:
        """Identify single bets that meet our quality criteria."""
        qualified_singles = []
        
        for leg in bet_legs:
            # Check minimum edge requirement
            if leg.edge < self.min_single_edge:
                continue
                
            # Calculate Kelly stake
            prob_lower = leg.confidence_interval[0] if leg.confidence_interval else None
            
            kelly_rec = self.kelly_calculator.calculate_kelly_stake(
                prob=leg.probability,
                odds=leg.odds,
                bankroll=self.bankroll,
                prob_lower=prob_lower
            )
            
            if kelly_rec.bet_amount > 0:
                # Create single bet recommendation
                single_rec = MultiBetRecommendation(
                    legs=[leg],
                    bet_type='single',
                    combined_probability=leg.probability,
                    adjusted_probability=leg.probability,
                    combined_odds=leg.odds,
                    expected_value=leg.edge,
                    correlation_penalty=0.0,
                    kelly_fraction=kelly_rec.kelly_fraction,
                    recommended_stake=kelly_rec.bet_amount,
                    risk_score=self._calculate_risk_score([leg]),
                    confidence_score=leg.probability,
                    correlation_matrix=np.array([[1.0]])
                )
                
                qualified_singles.append(single_rec)
        
        # Sort by expected value
        qualified_singles.sort(key=lambda x: x.expected_value, reverse=True)
        
        return qualified_singles
    
    def _generate_optimal_parlays(
        self,
        candidate_legs: List[BetLeg],
        live_features: Optional[pd.DataFrame] = None
    ) -> List[MultiBetRecommendation]:
        """Generate optimal parlay combinations using advanced selection algorithm."""
        if len(candidate_legs) < 2:
            return []
        
        parlay_recommendations = []
        
        # Generate 2-leg and 3-leg combinations
        for n_legs in range(2, min(self.max_parlay_legs + 1, len(candidate_legs) + 1)):
            
            for combo in combinations(candidate_legs, n_legs):
                parlay_analysis = self._analyze_parlay_combination(
                    list(combo), live_features
                )
                
                if parlay_analysis and parlay_analysis.expected_value >= self.min_parlay_edge:
                    parlay_recommendations.append(parlay_analysis)
        
        # Sort by risk-adjusted expected value
        parlay_recommendations.sort(
            key=lambda x: x.expected_value / (1 + x.risk_score), 
            reverse=True
        )
        
        # Limit to max parlays
        return parlay_recommendations[:self.max_parlays]
    
    def _analyze_parlay_combination(
        self,
        legs: List[BetLeg],
        live_features: Optional[pd.DataFrame] = None
    ) -> Optional[MultiBetRecommendation]:
        """Analyze a specific parlay combination with full correlation adjustment."""
        if len(legs) < 2:
            return None
            
        # Build correlation matrix
        n_legs = len(legs)
        correlation_matrix = np.eye(n_legs)
        
        # Calculate pairwise correlations
        for i in range(n_legs):
            for j in range(i + 1, n_legs):
                # Convert legs to dict format for correlation analyzer
                bet1 = self._leg_to_bet_dict(legs[i])
                bet2 = self._leg_to_bet_dict(legs[j])
                
                correlation = self.correlation_analyzer.estimate_correlation(
                    bet1, bet2, live_features
                )
                
                correlation_matrix[i, j] = correlation
                correlation_matrix[j, i] = correlation
        
        # Calculate combined probabilities
        individual_probs = [leg.probability for leg in legs]
        combined_prob_independent = np.prod(individual_probs)
        
        # Apply correlation adjustment using enhanced formula
        avg_correlation = np.mean(correlation_matrix[np.triu_indices(n_legs, k=1)])
        legs_adjustment = np.sqrt(n_legs - 1) / n_legs
        
        correlation_penalty = (
            self.correlation_penalty_alpha * 
            avg_correlation * 
            legs_adjustment * 
            0.1  # Scale factor
        )
        
        adjusted_probability = combined_prob_independent * (1 - correlation_penalty)
        adjusted_probability = max(adjusted_probability, combined_prob_independent * 0.5)  # Floor
        
        # Calculate expected value
        combined_odds = np.prod([leg.odds for leg in legs])
        expected_value = (adjusted_probability * combined_odds) - 1
        
        if expected_value < self.min_parlay_edge:
            return None
        
        # Calculate pessimistic Kelly stake for parlays
        kelly_fraction = self._calculate_parlay_kelly(
            adjusted_probability, combined_odds, avg_correlation
        )
        
        recommended_stake = min(
            kelly_fraction * self.bankroll,
            self.bankroll * self.max_parlay_stake_pct
        )
        
        # Calculate risk and confidence scores
        risk_score = self._calculate_risk_score(legs, avg_correlation)
        confidence_score = self._calculate_confidence_score(legs, correlation_penalty)
        
        return MultiBetRecommendation(
            legs=legs,
            bet_type=f'parlay_{n_legs}',
            combined_probability=combined_prob_independent,
            adjusted_probability=adjusted_probability,
            combined_odds=combined_odds,
            expected_value=expected_value,
            correlation_penalty=correlation_penalty,
            kelly_fraction=kelly_fraction,
            recommended_stake=recommended_stake,
            risk_score=risk_score,
            confidence_score=confidence_score,
            correlation_matrix=correlation_matrix
        )
    
    def _calculate_parlay_kelly(
        self,
        probability: float,
        combined_odds: float,
        avg_correlation: float
    ) -> float:
        """Calculate pessimistic Kelly fraction for parlay bets."""
        # Use more conservative Kelly for parlays
        base_kelly = (probability * combined_odds - 1) / (combined_odds - 1)
        
        # Apply pessimistic adjustment
        pessimistic_factor = 0.5  # Use half Kelly for parlays
        correlation_adjustment = 1 - (avg_correlation * 0.2)  # Reduce further for correlation
        
        kelly_fraction = base_kelly * pessimistic_factor * correlation_adjustment
        
        return max(0, min(kelly_fraction, self.max_parlay_stake_pct))
    
    def _calculate_risk_score(self, legs: List[BetLeg], avg_correlation: float = 0.0) -> float:
        """Calculate risk score for bet combination."""
        # Base risk from individual probabilities
        prob_variance = np.var([leg.probability for leg in legs])
        odds_risk = np.mean([1/leg.odds for leg in legs])  # Higher odds = higher risk
        
        # Correlation risk
        correlation_risk = avg_correlation * 0.5
        
        # Combine risk factors
        risk_score = prob_variance + odds_risk + correlation_risk
        
        return min(risk_score, 1.0)  # Cap at 1.0
    
    def _calculate_confidence_score(self, legs: List[BetLeg], correlation_penalty: float) -> float:
        """Calculate confidence score for bet combination."""
        # Average individual confidence
        avg_confidence = np.mean([leg.probability for leg in legs])
        
        # Penalty for correlation
        confidence_adjustment = 1 - correlation_penalty
        
        return avg_confidence * confidence_adjustment
    
    def _leg_to_bet_dict(self, leg: BetLeg) -> Dict[str, Any]:
        """Convert BetLeg to dict format for correlation analyzer."""
        return {
            'fighter': leg.fighter,
            'opponent': leg.opponent,
            'event': leg.event,
            'probability': leg.probability,
            'odds': leg.odds,
            'weight_class': leg.division,
            'card_position': leg.card_position or 0,
            'features': leg.features or {}
        }
    
    def _apply_portfolio_management(
        self,
        recommendations: Dict[str, List[MultiBetRecommendation]]
    ) -> Dict[str, List[MultiBetRecommendation]]:
        """Apply portfolio-level constraints and optimization."""
        
        # Calculate total exposure
        total_single_exposure = sum(
            rec.recommended_stake for rec in recommendations['single_bets']
        )
        total_parlay_exposure = sum(
            rec.recommended_stake for rec in recommendations['parlays']
        )
        
        # Check parlay exposure limit
        if total_parlay_exposure > self.bankroll * self.max_total_parlay_exposure:
            # Scale down parlay stakes proportionally
            scale_factor = (self.bankroll * self.max_total_parlay_exposure) / total_parlay_exposure
            
            for parlay in recommendations['parlays']:
                parlay.recommended_stake *= scale_factor
                parlay.kelly_fraction *= scale_factor
        
        # Final filtering based on confidence threshold
        recommendations['single_bets'] = [
            rec for rec in recommendations['single_bets']
            if rec.confidence_score >= self.confidence_threshold
        ]
        
        recommendations['parlays'] = [
            rec for rec in recommendations['parlays']
            if rec.confidence_score >= self.confidence_threshold
        ]
        
        return recommendations
    
    def generate_betting_report(
        self,
        recommendations: Dict[str, List[MultiBetRecommendation]]
    ) -> Dict[str, Any]:
        """Generate comprehensive betting analysis report."""
        
        total_single_stake = sum(rec.recommended_stake for rec in recommendations['single_bets'])
        total_parlay_stake = sum(rec.recommended_stake for rec in recommendations['parlays'])
        total_stake = total_single_stake + total_parlay_stake
        
        single_ev = sum(
            rec.expected_value * rec.recommended_stake 
            for rec in recommendations['single_bets']
        )
        parlay_ev = sum(
            rec.expected_value * rec.recommended_stake 
            for rec in recommendations['parlays']
        )
        
        report = {
            'strategy_type': 'single_focused' if len(recommendations['single_bets']) >= self.single_bet_threshold else 'parlay_activated',
            'total_recommendations': len(recommendations['single_bets']) + len(recommendations['parlays']),
            'single_bets': {
                'count': len(recommendations['single_bets']),
                'total_stake': total_single_stake,
                'total_ev': single_ev,
                'avg_edge': np.mean([r.expected_value for r in recommendations['single_bets']]) if recommendations['single_bets'] else 0
            },
            'parlays': {
                'count': len(recommendations['parlays']),
                'total_stake': total_parlay_stake,
                'total_ev': parlay_ev,
                'avg_correlation': np.mean([
                    np.mean(r.correlation_matrix[np.triu_indices(len(r.legs), k=1)])
                    for r in recommendations['parlays']
                ]) if recommendations['parlays'] else 0
            },
            'portfolio': {
                'total_stake': total_stake,
                'total_exposure_pct': (total_stake / self.bankroll) * 100,
                'expected_return': single_ev + parlay_ev,
                'risk_adjusted_return': (single_ev + parlay_ev) / max(total_stake, 1)
            }
        }
        
        return report


def create_sample_legs() -> List[BetLeg]:
    """Create sample bet legs for testing."""
    return [
        BetLeg(
            fighter="Jon Jones",
            opponent="Stipe Miocic", 
            probability=0.75,
            odds=1.8,
            edge=0.08,
            event="UFC 309",
            division="heavyweight",
            card_position=1,
            confidence_interval=(0.70, 0.80)
        ),
        BetLeg(
            fighter="Islam Makhachev",
            opponent="Charles Oliveira",
            probability=0.65,
            odds=2.2,
            edge=0.12,
            event="UFC 309", 
            division="lightweight",
            card_position=2,
            confidence_interval=(0.60, 0.70)
        ),
        BetLeg(
            fighter="Sean O'Malley", 
            opponent="Marlon Vera",
            probability=0.62,
            odds=2.1,
            edge=0.06,
            event="UFC 310",
            division="bantamweight",
            card_position=1,
            confidence_interval=(0.58, 0.66)
        )
    ]


if __name__ == "__main__":
    # Demo the enhanced multi-bet system
    print("🚀 Enhanced Multi-Bet UFC System Demo")
    print("=" * 50)
    
    # Initialize system
    system = EnhancedMultiBetSystem(
        bankroll=1000.0,
        single_bet_threshold=2
    )
    
    # Create sample data
    sample_legs = create_sample_legs()
    
    # Analyze opportunities
    recommendations = system.analyze_betting_opportunities(sample_legs)
    
    # Generate report
    report = system.generate_betting_report(recommendations)
    
    # Display results
    print(f"\n📊 ANALYSIS RESULTS:")
    print(f"Strategy: {report['strategy_type']}")
    print(f"Total recommendations: {report['total_recommendations']}")
    
    print(f"\n💰 SINGLE BETS:")
    print(f"Count: {report['single_bets']['count']}")
    print(f"Total stake: ${report['single_bets']['total_stake']:.2f}")
    print(f"Expected value: {report['single_bets']['avg_edge']:.1%}")
    
    print(f"\n🎯 PARLAYS:")
    print(f"Count: {report['parlays']['count']}")
    print(f"Total stake: ${report['parlays']['total_stake']:.2f}")
    print(f"Avg correlation: {report['parlays']['avg_correlation']:.2f}")
    
    print(f"\n📈 PORTFOLIO:")
    print(f"Total exposure: {report['portfolio']['total_exposure_pct']:.1f}%")
    print(f"Expected return: ${report['portfolio']['expected_return']:.2f}")
    print(f"Risk-adjusted return: {report['portfolio']['risk_adjusted_return']:.1%}")
    
    print("\n✅ Enhanced multi-bet system ready for production!")