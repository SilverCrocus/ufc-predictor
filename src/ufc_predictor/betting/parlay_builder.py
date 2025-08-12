"""
Sophisticated Parlay Builder with Portfolio Management

Implements the complete parlay construction system from the multi-bet plan:
- Correlation-adjusted probability calculations
- Pessimistic Kelly staking with strict caps (0.5% bankroll)
- Portfolio limits (max 2 parlays, 1.5% total exposure)
- Risk-adjusted EV optimization
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, NamedTuple
from dataclasses import dataclass, field
from itertools import combinations
import logging
from pathlib import Path

from .selection import BettingOpportunity, BettingSelector
from .enhanced_correlation import EnhancedCorrelationEngine

logger = logging.getLogger(__name__)


@dataclass
class ParlayLeg:
    """Individual leg of a parlay bet."""
    opportunity: BettingOpportunity
    adjusted_prob: float = field(init=False)
    
    def __post_init__(self):
        """Calculate adjusted probability with conservative factor."""
        self.adjusted_prob = self.opportunity.model_prob * 0.85  # Conservative adjustment


@dataclass 
class Parlay:
    """Complete parlay with all legs and calculated metrics."""
    legs: List[ParlayLeg]
    correlation: float
    combined_prob: float
    combined_odds: float
    combined_ev: float
    kelly_fraction: float
    stake_amount: float
    expected_return: float
    risk_adjusted_ev: float
    confidence_score: float
    metadata: Dict = field(default_factory=dict)
    
    @property
    def n_legs(self) -> int:
        """Number of legs in the parlay."""
        return len(self.legs)
    
    @property
    def fighters(self) -> List[str]:
        """List of fighter names in the parlay."""
        return [leg.opportunity.fighter for leg in self.legs]
    
    @property
    def events(self) -> List[str]:
        """List of events represented in the parlay."""
        return [leg.opportunity.event for leg in self.legs]


class ParlayBuilder:
    """
    Sophisticated parlay builder implementing the complete multi-bet strategy.
    
    Features:
    - Correlation-adjusted probability calculations using Gaussian copula
    - Pessimistic Kelly staking with configurable caps
    - Portfolio optimization with exposure limits
    - Risk-adjusted EV maximization
    """
    
    def __init__(self,
                 config: Optional[Dict] = None,
                 correlation_engine: Optional[EnhancedCorrelationEngine] = None,
                 fighter_features: Optional[pd.DataFrame] = None,
                 historical_results: Optional[pd.DataFrame] = None):
        """Initialize the parlay builder with configuration and data sources."""
        self.config = config or self._default_config()
        
        # Initialize correlation engine
        if correlation_engine:
            self.correlation_engine = correlation_engine
        else:
            self.correlation_engine = EnhancedCorrelationEngine(
                fighter_features=fighter_features,
                historical_results=historical_results
            )
    
    def _default_config(self) -> Dict:
        """Default configuration implementing multi-bet plan specifications."""
        return {
            'parlay_limits': {
                'max_legs': 3,              # Maximum legs per parlay (proven limit)
                'max_parlays': 2,           # Maximum parlays per card
                'max_parlay_exposure': 0.0025,  # 0.25% total parlay exposure
                'max_single_parlay': 0.0025,    # 0.25% per individual parlay (max)
                'stake_min': 0.001,         # 0.1% of bankroll (minimum)
                'stake_max': 0.0025         # 0.25% of bankroll (maximum)
            },
            'correlation': {
                'max_allowed': 0.40,        # Maximum correlation allowed
                'penalty_threshold': 0.20,  # Apply penalties above this
                'penalty_factor': 1.20      # Correlation penalty multiplier
            },
            'probability': {
                'method': 'gaussian_copula', # Correlation adjustment method
                'conservative_factor': 0.85, # Pessimistic probability adjustment
                'min_combined_prob': 0.15   # Minimum combined probability
            },
            'staking': {
                'method': 'pessimistic_kelly',  # Kelly variant to use
                'kelly_fraction': 0.25,         # Base Kelly fraction (quarter Kelly)
                'confidence_scaling': True,     # Scale by confidence
                'variance_penalty': 0.10        # Penalty for correlation variance
            },
            'selection': {
                'min_combined_ev': 0.08,     # 8% minimum combined EV
                'ev_min': 0.03,              # 3% minimum individual leg EV (proven threshold)
                'confidence_min': 0.55,      # Minimum confidence per leg
                'diversification_bonus': 0.02 # Bonus for cross-division parlays
            },
            'risk_management': {
                'correlation_buffer': 0.05,  # Safety buffer for correlation estimates
                'kelly_cap': 0.005,         # Hard cap on Kelly bet size (0.5%)
                'exposure_scaling': 0.80,   # Scale down if approaching limits
                'min_expected_return': 0.10 # Minimum expected return threshold
            }
        }
    
    def build_optimal_parlays(self, 
                             parlay_pool: List[BettingOpportunity],
                             bankroll: float,
                             context: Optional[Dict] = None) -> List[Parlay]:
        """
        Build optimal parlay portfolio from the eligible pool.
        
        Args:
            parlay_pool: List of eligible betting opportunities
            bankroll: Current bankroll
            context: Additional context (event info, etc.)
            
        Returns:
            List of optimal parlays within all constraints
        """
        if len(parlay_pool) < 2:
            logger.info("Insufficient opportunities for parlay construction")
            return []
        
        logger.info(f"Building parlays from pool of {len(parlay_pool)} opportunities")
        
        # Step 1: Generate all valid combinations
        candidate_parlays = self._generate_candidates(parlay_pool, bankroll)
        logger.info(f"Generated {len(candidate_parlays)} candidate parlays")
        
        # Step 2: Apply correlation analysis
        analyzed_parlays = self._analyze_correlations(candidate_parlays, context)
        logger.info(f"Analyzed correlations for {len(analyzed_parlays)} parlays")
        
        # Step 3: Calculate metrics and probabilities
        calculated_parlays = self._calculate_parlay_metrics(analyzed_parlays, bankroll)
        logger.info(f"Calculated metrics for {len(calculated_parlays)} parlays")
        
        # Step 4: Apply filters and constraints
        filtered_parlays = self._apply_constraints(calculated_parlays, bankroll)
        logger.info(f"Applied constraints, {len(filtered_parlays)} remaining")
        
        # Step 5: Optimize portfolio selection
        optimal_parlays = self._optimize_portfolio(filtered_parlays, bankroll)
        logger.info(f"Selected {len(optimal_parlays)} optimal parlays")
        
        return optimal_parlays
    
    def _generate_candidates(self, 
                            pool: List[BettingOpportunity],
                            bankroll: float) -> List[List[ParlayLeg]]:
        """Generate all valid parlay combinations."""
        candidates = []
        max_legs = self.config['parlay_limits']['max_legs']
        
        # Generate 2-leg combinations (primary focus)
        for combo in combinations(pool, 2):
            opp1, opp2 = combo
            
            # Skip same fight (shouldn't happen but safety check)
            if opp1.fight_id == opp2.fight_id:
                continue
            
            # Basic filter checks
            if (opp1.ev >= self.config['selection']['ev_min'] and
                opp2.ev >= self.config['selection']['ev_min'] and
                opp1.confidence >= self.config['selection']['confidence_min'] and
                opp2.confidence >= self.config['selection']['confidence_min']):
                
                legs = [ParlayLeg(opp1), ParlayLeg(opp2)]
                candidates.append(legs)
        
        # Generate 3-leg combinations if configured
        if max_legs >= 3:
            for combo in combinations(pool, 3):
                if all(opp.ev >= self.config['selection']['ev_min'] and
                      opp.confidence >= self.config['selection']['confidence_min']
                      for opp in combo):
                    
                    # Check for fight diversity
                    fight_ids = set(opp.fight_id for opp in combo)
                    if len(fight_ids) == len(combo):  # All different fights
                        legs = [ParlayLeg(opp) for opp in combo]
                        candidates.append(legs)
        
        return candidates
    
    def _analyze_correlations(self, 
                             candidates: List[List[ParlayLeg]],
                             context: Optional[Dict]) -> List[Tuple[List[ParlayLeg], float]]:
        """Analyze correlations for all candidate parlays."""
        analyzed = []
        
        for legs in candidates:
            if len(legs) == 2:
                # 2-leg correlation
                bet1 = self._leg_to_bet_dict(legs[0])
                bet2 = self._leg_to_bet_dict(legs[1])
                
                correlation, _ = self.correlation_engine.estimate_correlation(bet1, bet2, context)
                
                # Apply safety buffer
                buffered_correlation = correlation + self.config['risk_management']['correlation_buffer']
                
                analyzed.append((legs, buffered_correlation))
                
            elif len(legs) == 3:
                # 3-leg correlation (average of pairwise)
                correlations = []
                for i in range(len(legs)):
                    for j in range(i + 1, len(legs)):
                        bet_i = self._leg_to_bet_dict(legs[i])
                        bet_j = self._leg_to_bet_dict(legs[j])
                        corr, _ = self.correlation_engine.estimate_correlation(bet_i, bet_j, context)
                        correlations.append(corr)
                
                avg_correlation = np.mean(correlations)
                buffered_correlation = avg_correlation + self.config['risk_management']['correlation_buffer']
                
                analyzed.append((legs, buffered_correlation))
        
        return analyzed
    
    def _calculate_parlay_metrics(self,
                                 analyzed_parlays: List[Tuple[List[ParlayLeg], float]],
                                 bankroll: float) -> List[Parlay]:
        """Calculate complete metrics for each parlay."""
        parlays = []
        
        for legs, correlation in analyzed_parlays:
            # Extract probabilities and odds
            individual_probs = [leg.adjusted_prob for leg in legs]
            individual_odds = [leg.opportunity.odds for leg in legs]
            
            # Calculate correlation-adjusted combined probability
            combined_prob = self._calculate_combined_probability(
                individual_probs, correlation
            )
            
            # Calculate combined odds
            combined_odds = np.prod(individual_odds)
            
            # Calculate combined EV
            combined_ev = (combined_prob * combined_odds) - 1
            
            # Calculate Kelly fraction
            kelly_fraction = self._calculate_kelly_fraction(
                combined_prob, combined_odds, correlation, legs
            )
            
            # Calculate stake amount
            stake_amount = self._calculate_stake(
                kelly_fraction, bankroll, combined_ev, correlation
            )
            
            # Calculate expected return
            expected_return = stake_amount * combined_ev
            
            # Calculate risk-adjusted EV
            risk_adjusted_ev = self._calculate_risk_adjusted_ev(
                combined_ev, correlation, combined_prob, len(legs)
            )
            
            # Calculate confidence score
            confidence_score = self._calculate_confidence_score(legs, correlation)
            
            # Create parlay object
            parlay = Parlay(
                legs=legs,
                correlation=correlation,
                combined_prob=combined_prob,
                combined_odds=combined_odds,
                combined_ev=combined_ev,
                kelly_fraction=kelly_fraction,
                stake_amount=stake_amount,
                expected_return=expected_return,
                risk_adjusted_ev=risk_adjusted_ev,
                confidence_score=confidence_score,
                metadata={
                    'individual_probs': individual_probs,
                    'individual_odds': individual_odds,
                    'diversification': self._calculate_diversification_score(legs)
                }
            )
            
            parlays.append(parlay)
        
        return parlays
    
    def _calculate_combined_probability(self, 
                                      individual_probs: List[float],
                                      correlation: float) -> float:
        """Calculate correlation-adjusted combined probability using Gaussian copula."""
        method = self.config['probability']['method']
        
        if method == 'gaussian_copula':
            # Base probability (independent assumption)
            base_prob = np.prod(individual_probs)
            
            # Correlation penalty
            n_legs = len(individual_probs)
            penalty_factor = self.config['correlation']['penalty_factor']
            correlation_penalty = 1 - (penalty_factor * correlation * 0.1 * n_legs)
            
            # Apply penalty
            adjusted_prob = base_prob * max(0.5, correlation_penalty)
            
        elif method == 'simple_penalty':
            # Simple correlation penalty
            base_prob = np.prod(individual_probs)
            penalty = correlation * 0.15  # 15% penalty per unit correlation
            adjusted_prob = base_prob * (1 - penalty)
            
        else:
            # Default to independent assumption
            adjusted_prob = np.prod(individual_probs)
        
        # Apply minimum threshold
        min_prob = self.config['probability']['min_combined_prob']
        return max(min_prob, adjusted_prob)
    
    def _calculate_kelly_fraction(self,
                                 combined_prob: float,
                                 combined_odds: float,
                                 correlation: float,
                                 legs: List[ParlayLeg]) -> float:
        """Calculate pessimistic Kelly fraction with correlation adjustments."""
        # Base Kelly fraction
        edge = (combined_prob * combined_odds) - 1
        if edge <= 0:
            return 0
        
        base_kelly = edge / (combined_odds - 1)
        
        # Apply configured Kelly fraction
        kelly_fraction = base_kelly * self.config['staking']['kelly_fraction']
        
        # Apply pessimistic adjustments
        # 1. Confidence scaling
        if self.config['staking']['confidence_scaling']:
            avg_confidence = np.mean([leg.opportunity.confidence for leg in legs])
            confidence_factor = 0.5 + (avg_confidence * 0.5)
            kelly_fraction *= confidence_factor
        
        # 2. Correlation penalty
        correlation_penalty = 1 - (correlation * self.config['staking']['variance_penalty'])
        kelly_fraction *= max(0.3, correlation_penalty)
        
        # 3. Multi-leg penalty
        n_legs = len(legs)
        multi_leg_penalty = 0.9 ** (n_legs - 1)  # Exponential penalty for more legs
        kelly_fraction *= multi_leg_penalty
        
        return kelly_fraction
    
    def _calculate_stake(self,
                        kelly_fraction: float,
                        bankroll: float,
                        combined_ev: float,
                        correlation: float) -> float:
        """Calculate final stake using fixed 0.1-0.25% range based on EV and correlation."""
        # Fixed stake range: 0.1% to 0.25% of bankroll
        min_stake = bankroll * self.config['parlay_limits'].get('stake_min', 0.001)  # 0.1%
        max_stake = bankroll * self.config['parlay_limits'].get('stake_max', 0.0025)  # 0.25%
        
        # Scale within range based on combined EV (higher EV = higher stake)
        # EV range typically 8% to 30% for parlays
        ev_factor = min(max(combined_ev / 0.30, 0.0), 1.0)  # Normalize to 0-1
        
        # Apply correlation penalty (higher correlation = lower stake)
        correlation_penalty = 1.0
        if correlation > self.config['correlation']['penalty_threshold']:
            correlation_penalty = 0.8  # Reduce stake by 20% for high correlation
        
        # Calculate stake within fixed range
        stake = min_stake + (max_stake - min_stake) * ev_factor * correlation_penalty
        
        # Ensure within bounds
        stake = max(min_stake, min(stake, max_stake))
        
        return round(stake, 2)
    
    def _calculate_risk_adjusted_ev(self,
                                   combined_ev: float,
                                   correlation: float,
                                   combined_prob: float,
                                   n_legs: int) -> float:
        """Calculate risk-adjusted expected value for portfolio optimization."""
        # Base EV
        risk_adjusted = combined_ev
        
        # Correlation penalty
        correlation_penalty = correlation * 0.20  # 20% penalty per unit correlation
        risk_adjusted -= correlation_penalty
        
        # Probability confidence penalty
        if combined_prob < 0.25:  # Low probability penalty
            prob_penalty = (0.25 - combined_prob) * 0.50
            risk_adjusted -= prob_penalty
        
        # Multi-leg complexity penalty
        if n_legs > 2:
            complexity_penalty = (n_legs - 2) * 0.05
            risk_adjusted -= complexity_penalty
        
        return risk_adjusted
    
    def _calculate_confidence_score(self,
                                   legs: List[ParlayLeg],
                                   correlation: float) -> float:
        """Calculate overall confidence score for the parlay."""
        # Average individual confidence
        individual_confidence = np.mean([leg.opportunity.confidence for leg in legs])
        
        # Correlation confidence penalty
        correlation_penalty = correlation * 0.30
        
        # Combined confidence
        combined_confidence = individual_confidence * (1 - correlation_penalty)
        
        # Multi-leg penalty
        n_legs = len(legs)
        multi_leg_factor = 0.95 ** (n_legs - 1)
        
        return combined_confidence * multi_leg_factor
    
    def _calculate_diversification_score(self, legs: List[ParlayLeg]) -> float:
        """Calculate diversification score for the parlay."""
        # Event diversification
        events = set(leg.opportunity.event for leg in legs)
        event_div = min(len(events) / len(legs), 1.0)
        
        # Weight class diversification
        weight_classes = set(leg.opportunity.weight_class for leg in legs)
        weight_div = min(len(weight_classes) / len(legs), 1.0)
        
        # Combined diversification
        return (event_div + weight_div) / 2
    
    def _apply_constraints(self, parlays: List[Parlay], bankroll: float) -> List[Parlay]:
        """Apply all filters and constraints."""
        filtered = []
        
        for parlay in parlays:
            # EV constraint
            if parlay.combined_ev < self.config['selection']['min_combined_ev']:
                continue
            
            # Correlation constraint
            if parlay.correlation > self.config['correlation']['max_allowed']:
                continue
            
            # Stake constraint (must be positive)
            if parlay.stake_amount <= 0:
                continue
            
            # Expected return constraint
            min_return = self.config['risk_management']['min_expected_return']
            if parlay.expected_return < min_return:
                continue
            
            # Confidence constraint
            if parlay.confidence_score < 0.40:  # Minimum confidence
                continue
            
            filtered.append(parlay)
        
        return filtered
    
    def _optimize_portfolio(self, parlays: List[Parlay], bankroll: float) -> List[Parlay]:
        """Select optimal portfolio within exposure limits."""
        if not parlays:
            return []
        
        # Sort by risk-adjusted EV
        sorted_parlays = sorted(parlays, key=lambda p: p.risk_adjusted_ev, reverse=True)
        
        # Portfolio selection with constraints
        selected = []
        total_exposure = 0
        max_total_exposure = bankroll * self.config['parlay_limits']['max_parlay_exposure']
        max_parlays = self.config['parlay_limits']['max_parlays']
        
        for parlay in sorted_parlays:
            # Check portfolio limits
            if len(selected) >= max_parlays:
                break
            
            # Check total exposure limit
            if total_exposure + parlay.stake_amount > max_total_exposure:
                # Try to scale down the stake
                remaining_exposure = max_total_exposure - total_exposure
                if remaining_exposure >= bankroll * 0.001:  # Minimum 0.1%
                    # Create scaled parlay
                    scale_factor = remaining_exposure / parlay.stake_amount
                    scaled_parlay = self._scale_parlay(parlay, scale_factor)
                    selected.append(scaled_parlay)
                    total_exposure += scaled_parlay.stake_amount
                break
            
            selected.append(parlay)
            total_exposure += parlay.stake_amount
        
        logger.info(f"Portfolio optimization: {total_exposure:.2f} total exposure ({total_exposure/bankroll:.1%})")
        return selected
    
    def _scale_parlay(self, parlay: Parlay, scale_factor: float) -> Parlay:
        """Create a scaled version of a parlay with adjusted stake."""
        scaled_parlay = Parlay(
            legs=parlay.legs,
            correlation=parlay.correlation,
            combined_prob=parlay.combined_prob,
            combined_odds=parlay.combined_odds,
            combined_ev=parlay.combined_ev,
            kelly_fraction=parlay.kelly_fraction * scale_factor,
            stake_amount=round(parlay.stake_amount * scale_factor, 2),
            expected_return=parlay.expected_return * scale_factor,
            risk_adjusted_ev=parlay.risk_adjusted_ev,  # EV doesn't scale
            confidence_score=parlay.confidence_score,
            metadata={**parlay.metadata, 'scaled': True, 'scale_factor': scale_factor}
        )
        return scaled_parlay
    
    def _leg_to_bet_dict(self, leg: ParlayLeg) -> Dict:
        """Convert ParlayLeg to dictionary format for correlation analysis."""
        opp = leg.opportunity
        return {
            'fighter': opp.fighter,
            'opponent': opp.opponent,
            'event': opp.event,
            'probability': opp.model_prob,
            'odds': opp.odds,
            'weight_class': opp.weight_class,
            'card_position': opp.card_position,
            'confidence': opp.confidence,
            'is_favorite': opp.is_favorite
        }
    
    def generate_parlay_report(self, parlays: List[Parlay]) -> str:
        """Generate comprehensive parlay portfolio report."""
        report = []
        report.append("=" * 70)
        report.append("PARLAY PORTFOLIO REPORT")
        report.append("=" * 70)
        
        if not parlays:
            report.append("No parlays selected")
            return "\n".join(report)
        
        total_stake = sum(p.stake_amount for p in parlays)
        total_expected = sum(p.expected_return for p in parlays)
        avg_correlation = np.mean([p.correlation for p in parlays])
        
        report.append(f"Portfolio Summary:")
        report.append(f"  Parlays: {len(parlays)}")
        report.append(f"  Total stake: ${total_stake:.2f}")
        report.append(f"  Expected return: ${total_expected:.2f}")
        report.append(f"  Average correlation: {avg_correlation:.3f}")
        report.append("")
        
        for i, parlay in enumerate(parlays, 1):
            report.append(f"Parlay {i}:")
            report.append(f"  Legs: {' + '.join(parlay.fighters)}")
            report.append(f"  Odds: {parlay.combined_odds:.2f}")
            report.append(f"  Probability: {parlay.combined_prob:.1%}")
            report.append(f"  EV: {parlay.combined_ev:.1%}")
            report.append(f"  Correlation: {parlay.correlation:.3f}")
            report.append(f"  Stake: ${parlay.stake_amount:.2f}")
            report.append(f"  Expected return: ${parlay.expected_return:.2f}")
            report.append(f"  Confidence: {parlay.confidence_score:.1%}")
            report.append("")
        
        return "\n".join(report)


if __name__ == "__main__":
    print("Sophisticated Parlay Builder")
    print("Implements correlation-adjusted probability calculations")
    print("with pessimistic Kelly staking and portfolio management")