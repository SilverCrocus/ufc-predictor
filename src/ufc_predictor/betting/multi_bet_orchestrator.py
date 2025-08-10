"""
Multi-Bet Orchestrator - Complete Implementation

Main orchestrator that implements the full multi-bet strategy from the plan:
- Conditional parlay activation when <2 qualified singles
- Sophisticated correlation analysis and probability adjustments  
- Portfolio management with strict exposure limits
- Integration with existing UFC predictor pipeline
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from pathlib import Path
import logging
import json
import warnings

from .selection import BettingSelector, BettingOpportunity
from .enhanced_correlation import EnhancedCorrelationEngine
from .parlay_builder import ParlayBuilder, Parlay

logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')


@dataclass
class MultiBetResult:
    """Complete result of multi-bet analysis."""
    strategy_used: str  # 'singles_only', 'conditional_parlays', 'no_bets'
    singles: List[BettingOpportunity]
    parlays: List[Parlay]
    total_exposure: float
    expected_return: float
    portfolio_risk: float
    activation_reason: str
    metadata: Dict
    

class MultiBetOrchestrator:
    """
    Complete multi-bet system orchestrator.
    
    Implements the sophisticated conditional parlay strategy:
    1. Primary: Single bets with strict filters (5-15% EV, 5% market gap)
    2. Conditional: 2-leg parlays when singles < 2 (relaxed filters)
    3. Portfolio: Risk-managed exposure with correlation analysis
    """
    
    def __init__(self,
                 config_path: Optional[str] = None,
                 fighter_features: Optional[pd.DataFrame] = None,
                 historical_results: Optional[pd.DataFrame] = None):
        """Initialize the multi-bet orchestrator."""
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Initialize components
        self.selector = BettingSelector(self.config['selection'])
        
        self.correlation_engine = EnhancedCorrelationEngine(
            config=self.config,  # Pass full config instead of just correlation section
            fighter_features=fighter_features,
            historical_results=historical_results
        )
        
        # Create parlay builder config with all required sections
        parlay_config = {
            **self.config['parlay'],
            'selection': {
                **self.config['selection']['parlays'],
                'min_combined_ev': 0.08  # 8% minimum combined EV
            },
            'risk_management': {
                'correlation_buffer': 0.05,
                'kelly_cap': 0.005,
                'exposure_scaling': 0.80,
                'min_expected_return': 0.10
            },
            'probability': {
                'method': 'gaussian_copula',
                'conservative_factor': 0.85,
                'min_combined_prob': 0.15
            }
        }
        
        self.parlay_builder = ParlayBuilder(
            config=parlay_config,
            correlation_engine=self.correlation_engine,
            fighter_features=fighter_features,
            historical_results=historical_results
        )
        
        # Tracking variables
        self.execution_history = []
        self.performance_metrics = {
            'total_bets_placed': 0,
            'singles_count': 0,
            'parlays_count': 0,
            'total_staked': 0,
            'total_returned': 0,
            'roi': 0,
            'strategy_distribution': {'singles_only': 0, 'conditional_parlays': 0, 'no_bets': 0}
        }
    
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load configuration from file or use defaults."""
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                user_config = json.load(f)
            
            # Merge with defaults
            default_config = self._default_config()
            return self._deep_merge(default_config, user_config)
        
        return self._default_config()
    
    def _default_config(self) -> Dict:
        """Complete default configuration for the multi-bet system."""
        return {
            'selection': {
                'singles': {
                    'ev_min': 0.05,
                    'ev_max': 0.15,
                    'market_gap_min': 0.05,
                    'confidence_min': 0.60,
                    'max_exposure': 0.10
                },
                'parlays': {
                    'ev_min': 0.02,
                    'market_gap_min': 0.03,
                    'confidence_min': 0.55,
                    'max_legs': 2,
                    'max_parlays': 2,
                    'max_exposure': 0.015
                },
                'activation': {
                    'min_singles_threshold': 2
                }
            },
            'correlation_sources': {
                'same_event': {'weight': 0.40, 'base_correlation': 0.15},
                'feature_similarity': {'weight': 0.30, 'scaling_factor': 0.25},
                'historical_residuals': {'weight': 0.20, 'min_observations': 10},
                'advanced_heuristics': {'weight': 0.10, 'penalties': {
                    'same_camp': 0.30,
                    'same_division': 0.15,
                    'teammates': 0.40,
                    'similar_style': 0.20
                }}
            },
            'parlay': {
                'parlay_limits': {
                    'max_legs': 2,
                    'max_parlays': 2,
                    'max_parlay_exposure': 0.015,
                    'max_single_parlay': 0.005
                },
                'correlation': {
                    'max_allowed': 0.40,
                    'penalty_threshold': 0.20,
                    'penalty_factor': 1.20
                },
                'staking': {
                    'method': 'pessimistic_kelly',
                    'kelly_fraction': 0.25,
                    'confidence_scaling': True,
                    'variance_penalty': 0.10
                }
            },
            'portfolio': {
                'max_total_exposure': 0.12,  # 12% maximum total exposure
                'diversification_bonus': 0.02,
                'correlation_penalty': 0.05,
                'risk_tolerance': 'conservative'
            },
            'safety': {
                'max_correlation': 0.50,
                'min_expected_return': 0.05,
                'exposure_scaling_threshold': 0.08,
                'emergency_stop_loss': 0.25
            },
            'blending': {
                'confidence_threshold': 0.50,
                'min_total_confidence': 1.0,
                'uncertainty_penalty': 0.05
            },
            'validation': {
                'max_correlation': 0.60,
                'min_correlation': 0.00,
                'sanity_check': True
            }
        }
    
    def analyze_betting_opportunities(self,
                                    predictions: pd.DataFrame,
                                    odds_data: Dict[str, Dict],
                                    bankroll: float,
                                    context: Optional[Dict] = None) -> MultiBetResult:
        """
        Main entry point for multi-bet analysis.
        
        Args:
            predictions: Model predictions DataFrame
            odds_data: Market odds dictionary
            bankroll: Current bankroll
            context: Additional context (event info, etc.)
            
        Returns:
            Complete MultiBetResult with strategy and recommendations
        """
        logger.info(f"Starting multi-bet analysis with bankroll ${bankroll:,.2f}")
        
        # Step 1: Initial opportunity selection
        selection_result = self.selector.select_opportunities(predictions, odds_data, bankroll)
        
        qualified_singles = selection_result['singles']
        parlay_activated = selection_result['parlay_activated']
        parlay_pool = selection_result['parlay_pool']
        
        logger.info(f"Initial selection: {len(qualified_singles)} singles, parlay_activated={parlay_activated}")
        
        # Step 2: Determine strategy
        if len(qualified_singles) >= self.config['selection']['activation']['min_singles_threshold']:
            # Primary strategy: Singles only
            strategy = 'singles_only'
            selected_singles = self._optimize_singles_portfolio(qualified_singles, bankroll)
            selected_parlays = []
            activation_reason = f"Sufficient singles ({len(qualified_singles)} >= 2)"
            
        elif parlay_activated and len(parlay_pool) >= 2:
            # Conditional strategy: Parlays
            strategy = 'conditional_parlays'
            selected_singles = qualified_singles  # Keep any qualified singles
            selected_parlays = self.parlay_builder.build_optimal_parlays(
                parlay_pool, bankroll, context
            )
            activation_reason = f"Insufficient singles ({len(qualified_singles)} < 2), activated parlays"
            
        else:
            # No betting strategy
            strategy = 'no_bets'
            selected_singles = []
            selected_parlays = []
            activation_reason = "No qualified opportunities found"
        
        # Step 3: Portfolio optimization
        optimized_result = self._optimize_complete_portfolio(
            selected_singles, selected_parlays, bankroll
        )
        
        # Step 4: Calculate portfolio metrics
        portfolio_metrics = self._calculate_portfolio_metrics(
            optimized_result['singles'], optimized_result['parlays'], bankroll
        )
        
        # Step 5: Create final result
        result = MultiBetResult(
            strategy_used=strategy,
            singles=optimized_result['singles'],
            parlays=optimized_result['parlays'],
            total_exposure=portfolio_metrics['total_exposure'],
            expected_return=portfolio_metrics['expected_return'],
            portfolio_risk=portfolio_metrics['portfolio_risk'],
            activation_reason=activation_reason,
            metadata={
                'bankroll': bankroll,
                'selection_metrics': selection_result['metrics'],
                'portfolio_metrics': portfolio_metrics,
                'config_used': strategy,
                'context': context or {}
            }
        )
        
        # Update tracking
        self._update_performance_tracking(result)
        
        logger.info(f"Multi-bet analysis complete: {strategy} - ${result.total_exposure:.2f} exposure")
        return result
    
    def _optimize_singles_portfolio(self,
                                   singles: List[BettingOpportunity],
                                   bankroll: float) -> List[BettingOpportunity]:
        """Optimize singles portfolio with exposure limits."""
        if not singles:
            return []
        
        # Sort by EV descending
        sorted_singles = sorted(singles, key=lambda x: x.ev, reverse=True)
        
        # Apply exposure limits
        max_total_exposure = bankroll * self.config['portfolio']['max_total_exposure']
        max_single_exposure = bankroll * self.config['selection']['singles']['max_exposure']
        
        selected = []
        total_exposure = 0
        
        for single in sorted_singles:
            # Calculate stake for this single
            stake = self._calculate_single_stake(single, bankroll)
            
            if total_exposure + stake > max_total_exposure:
                # Try to fit remaining exposure
                remaining = max_total_exposure - total_exposure
                if remaining > bankroll * 0.01:  # Minimum 1%
                    stake = min(stake, remaining)
                else:
                    break
            
            selected.append(single)
            total_exposure += stake
        
        logger.info(f"Singles portfolio: {len(selected)} selected, ${total_exposure:.2f} total exposure")
        return selected
    
    def _calculate_single_stake(self, single: BettingOpportunity, bankroll: float) -> float:
        """Calculate stake for a single bet using conservative Kelly."""
        # Pessimistic Kelly calculation
        prob = single.adjusted_prob
        odds = single.odds
        
        edge = (prob * odds) - 1
        if edge <= 0:
            return 0
        
        kelly_full = edge / (odds - 1)
        kelly_quarter = kelly_full * 0.25  # Quarter Kelly
        
        # Apply confidence scaling
        confidence_factor = 0.5 + (single.confidence * 0.5)
        kelly_adjusted = kelly_quarter * confidence_factor
        
        # Calculate stake
        stake = bankroll * kelly_adjusted
        
        # Apply limits
        max_single = bankroll * self.config['selection']['singles']['max_exposure']
        stake = min(stake, max_single)
        
        # Minimum threshold
        if stake < bankroll * 0.005:  # 0.5% minimum
            return 0
        
        return stake
    
    def _optimize_complete_portfolio(self,
                                    singles: List[BettingOpportunity],
                                    parlays: List[Parlay],
                                    bankroll: float) -> Dict:
        """Optimize the complete portfolio considering all constraints."""
        
        # Calculate total exposure
        singles_exposure = sum(self._calculate_single_stake(s, bankroll) for s in singles)
        parlays_exposure = sum(p.stake_amount for p in parlays)
        total_exposure = singles_exposure + parlays_exposure
        
        max_total = bankroll * self.config['portfolio']['max_total_exposure']
        
        # Check if scaling needed
        if total_exposure > max_total:
            scale_factor = max_total / total_exposure
            
            # Scale parlays first (they have lower priority)
            scaled_parlays = []
            for parlay in parlays:
                if parlay.stake_amount * scale_factor >= bankroll * 0.001:  # Min 0.1%
                    scaled_parlay = self._scale_parlay_stake(parlay, scale_factor)
                    scaled_parlays.append(scaled_parlay)
            
            # Keep singles as is unless extreme scaling needed
            if scale_factor < 0.7:
                # Scale singles too
                scaled_singles = singles[:int(len(singles) * scale_factor)]
            else:
                scaled_singles = singles
            
            logger.info(f"Portfolio scaled by {scale_factor:.2f}")
            return {'singles': scaled_singles, 'parlays': scaled_parlays}
        
        return {'singles': singles, 'parlays': parlays}
    
    def _scale_parlay_stake(self, parlay: Parlay, scale_factor: float) -> Parlay:
        """Create scaled version of parlay with adjusted stake."""
        # Create a copy with scaled stake
        scaled_parlay = Parlay(
            legs=parlay.legs,
            correlation=parlay.correlation,
            combined_prob=parlay.combined_prob,
            combined_odds=parlay.combined_odds,
            combined_ev=parlay.combined_ev,
            kelly_fraction=parlay.kelly_fraction * scale_factor,
            stake_amount=round(parlay.stake_amount * scale_factor, 2),
            expected_return=parlay.expected_return * scale_factor,
            risk_adjusted_ev=parlay.risk_adjusted_ev,
            confidence_score=parlay.confidence_score,
            metadata={**parlay.metadata, 'scaled': True, 'scale_factor': scale_factor}
        )
        return scaled_parlay
    
    def _calculate_portfolio_metrics(self,
                                    singles: List[BettingOpportunity],
                                    parlays: List[Parlay],
                                    bankroll: float) -> Dict:
        """Calculate comprehensive portfolio metrics."""
        
        # Singles metrics
        singles_stakes = [self._calculate_single_stake(s, bankroll) for s in singles]
        singles_exposure = sum(stakes for stakes in singles_stakes if stakes > 0)
        singles_expected = sum(s.ev * stake for s, stake in zip(singles, singles_stakes) if stake > 0)
        
        # Parlays metrics
        parlays_exposure = sum(p.stake_amount for p in parlays)
        parlays_expected = sum(p.expected_return for p in parlays)
        
        # Combined metrics
        total_exposure = singles_exposure + parlays_exposure
        total_expected = singles_expected + parlays_expected
        
        # Risk calculations
        portfolio_risk = self._calculate_portfolio_risk(singles, parlays)
        
        # Diversification
        diversification = self._calculate_diversification(singles, parlays)
        
        metrics = {
            'n_singles': len([s for s, stake in zip(singles, singles_stakes) if stake > 0]),
            'n_parlays': len(parlays),
            'singles_exposure': singles_exposure,
            'parlays_exposure': parlays_exposure,
            'total_exposure': total_exposure,
            'expected_return': total_expected,
            'portfolio_risk': portfolio_risk,
            'exposure_pct': total_exposure / bankroll if bankroll > 0 else 0,
            'expected_roi': total_expected / total_exposure if total_exposure > 0 else 0,
            'diversification': diversification,
            'risk_adjusted_return': total_expected - (portfolio_risk * 0.1)
        }
        
        return metrics
    
    def _calculate_portfolio_risk(self,
                                 singles: List[BettingOpportunity],
                                 parlays: List[Parlay]) -> float:
        """Calculate portfolio-level risk score."""
        risk_score = 0
        
        # Singles risk (lower)
        for single in singles:
            prob_risk = max(0, 0.7 - single.model_prob)  # Risk for low probability
            confidence_risk = max(0, 0.6 - single.confidence)  # Risk for low confidence
            risk_score += prob_risk + confidence_risk
        
        # Parlays risk (higher)
        for parlay in parlays:
            correlation_risk = parlay.correlation * 2  # Correlation penalty
            prob_risk = max(0, 0.4 - parlay.combined_prob) * 2  # Low prob penalty
            complexity_risk = len(parlay.legs) * 0.1  # Multi-leg penalty
            risk_score += correlation_risk + prob_risk + complexity_risk
        
        return risk_score
    
    def _calculate_diversification(self,
                                  singles: List[BettingOpportunity],
                                  parlays: List[Parlay]) -> float:
        """Calculate portfolio diversification score."""
        all_events = set()
        all_weight_classes = set()
        
        # Singles
        for single in singles:
            all_events.add(single.event)
            all_weight_classes.add(single.weight_class)
        
        # Parlays
        for parlay in parlays:
            for leg in parlay.legs:
                all_events.add(leg.opportunity.event)
                all_weight_classes.add(leg.opportunity.weight_class)
        
        # Diversification score
        event_div = min(len(all_events) / 3, 1.0)  # Max 3 events
        weight_div = min(len(all_weight_classes) / 5, 1.0)  # Max 5 weight classes
        
        return (event_div + weight_div) / 2
    
    def _update_performance_tracking(self, result: MultiBetResult):
        """Update internal performance tracking."""
        self.performance_metrics['strategy_distribution'][result.strategy_used] += 1
        
        if result.singles:
            self.performance_metrics['singles_count'] += len(result.singles)
        
        if result.parlays:
            self.performance_metrics['parlays_count'] += len(result.parlays)
        
        self.performance_metrics['total_staked'] += result.total_exposure
        self.execution_history.append(result)
    
    def _deep_merge(self, dict1: Dict, dict2: Dict) -> Dict:
        """Deep merge two dictionaries."""
        result = dict1.copy()
        for key, value in dict2.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        return result
    
    def generate_comprehensive_report(self, result: MultiBetResult) -> str:
        """Generate detailed analysis report."""
        report = []
        report.append("=" * 80)
        report.append("MULTI-BET ANALYSIS REPORT")
        report.append("=" * 80)
        
        # Strategy summary
        report.append(f"\n🎯 STRATEGY: {result.strategy_used.upper()}")
        report.append(f"Reason: {result.activation_reason}")
        report.append(f"Bankroll: ${result.metadata['bankroll']:,.2f}")
        
        # Portfolio overview
        report.append(f"\n📊 PORTFOLIO OVERVIEW")
        report.append("-" * 50)
        report.append(f"Total exposure: ${result.total_exposure:.2f} ({result.metadata['portfolio_metrics']['exposure_pct']:.1%})")
        report.append(f"Expected return: ${result.expected_return:.2f}")
        report.append(f"Portfolio risk: {result.portfolio_risk:.2f}")
        report.append(f"Diversification: {result.metadata['portfolio_metrics']['diversification']:.1%}")
        
        # Singles section
        if result.singles:
            report.append(f"\n💰 SINGLES ({len(result.singles)})")
            report.append("-" * 40)
            for i, single in enumerate(result.singles[:5], 1):
                stake = self._calculate_single_stake(single, result.metadata['bankroll'])
                report.append(f"{i}. {single.fighter} vs {single.opponent}")
                report.append(f"   EV: {single.ev:.1%} | Stake: ${stake:.2f} | Odds: {single.odds:.2f}")
        
        # Parlays section
        if result.parlays:
            report.append(f"\n🎲 PARLAYS ({len(result.parlays)})")
            report.append("-" * 40)
            for i, parlay in enumerate(result.parlays, 1):
                report.append(f"{i}. {' + '.join(parlay.fighters)}")
                report.append(f"   EV: {parlay.combined_ev:.1%} | Stake: ${parlay.stake_amount:.2f}")
                report.append(f"   Correlation: {parlay.correlation:.3f} | Confidence: {parlay.confidence_score:.1%}")
        
        # Risk analysis
        report.append(f"\n⚠️  RISK ANALYSIS")
        report.append("-" * 40)
        max_total = result.metadata['bankroll'] * self.config['portfolio']['max_total_exposure']
        report.append(f"Exposure limit: ${max_total:.2f} ({self.config['portfolio']['max_total_exposure']:.1%})")
        report.append(f"Current exposure: ${result.total_exposure:.2f}")
        report.append(f"Remaining capacity: ${max_total - result.total_exposure:.2f}")
        
        if result.parlays:
            avg_correlation = np.mean([p.correlation for p in result.parlays])
            report.append(f"Average correlation: {avg_correlation:.3f}")
        
        return "\n".join(report)
    
    def get_performance_summary(self) -> Dict:
        """Get historical performance summary."""
        if not self.execution_history:
            return {'message': 'No execution history available'}
        
        total_executions = len(self.execution_history)
        strategy_dist = {k: v/total_executions for k, v in self.performance_metrics['strategy_distribution'].items()}
        
        return {
            'total_executions': total_executions,
            'strategy_distribution': strategy_dist,
            'total_singles': self.performance_metrics['singles_count'],
            'total_parlays': self.performance_metrics['parlays_count'],
            'total_staked': self.performance_metrics['total_staked'],
            'average_exposure': self.performance_metrics['total_staked'] / total_executions if total_executions > 0 else 0
        }


if __name__ == "__main__":
    print("Multi-Bet Orchestrator - Complete Implementation")
    print("Implements conditional parlay strategy with sophisticated correlation analysis")