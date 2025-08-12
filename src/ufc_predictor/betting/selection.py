"""
UFC Betting Selection Module with Conditional Parlay Logic

Implements the sophisticated selection criteria from the multi-bet plan:
- Singles: Primary strategy with strict filters (5-15% EV, 5% market gap)
- Parlays: Conditional activation when <2 qualified singles
- Relaxed filters for parlay pool (2% EV, 3% market gap)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class BettingOpportunity:
    """Represents a single betting opportunity."""
    fight_id: str
    fighter: str
    opponent: str
    event: str
    model_prob: float
    market_prob: float
    odds: float
    ev: float
    confidence: float
    market_gap: float
    weight_class: str
    card_position: int
    is_favorite: bool
    
    @property
    def adjusted_prob(self) -> float:
        """No calibration - use raw model probability."""
        return self.model_prob


class BettingSelector:
    """
    Sophisticated betting selection with conditional parlay activation.
    
    Primary strategy: Single bets with strict filters
    Fallback strategy: 2-leg parlays when singles < 2
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize selector with configuration."""
        self.config = config or self._default_config()
        
    def _default_config(self) -> Dict:
        """Default configuration based on proven historical performance."""
        return {
            'singles': {
                'ev_min': 0.01,          # 1% minimum EV
                'ev_max': 0.15,          # 15% maximum EV (proven sweet spot)
                'odds_min': 1.40,        # Minimum decimal odds (avoid heavy favorites)
                'odds_max': 5.00,        # Maximum decimal odds (avoid longshots)
                'market_gap_min': 0.02,  # 2% minimum market gap
                'confidence_min': 0.50,  # 50% minimum confidence
                'max_exposure': 0.10     # 10% max bankroll exposure
            },
            'parlays': {
                'ev_min': 0.03,          # 3% minimum EV per leg
                'market_gap_min': 0.03,  # 3% minimum market gap
                'confidence_min': 0.55,  # 55% minimum confidence
                'max_legs': 3,           # Maximum 3 legs
                'max_parlays': 2,        # Maximum 2 parlays per card
                'max_exposure': 0.0025   # 0.25% max exposure for parlays
            },
            'activation': {
                'min_singles_threshold': 2  # Activate parlays when singles < 2
            }
        }
    
    def select_opportunities(self, 
                            predictions: pd.DataFrame,
                            odds_data: Dict[str, Dict],
                            bankroll: float) -> Dict:
        """
        Main selection logic with conditional parlay activation.
        
        Args:
            predictions: Model predictions DataFrame
            odds_data: Market odds dictionary
            bankroll: Current bankroll
            
        Returns:
            Dictionary with selected singles and parlays
        """
        # Step 1: Identify all opportunities
        all_opportunities = self._create_opportunities(predictions, odds_data)
        
        # Step 2: Apply strict filters for singles
        qualified_singles = self._filter_singles(all_opportunities)
        
        # Step 3: Check activation condition
        activate_parlays = len(qualified_singles) < self.config['activation']['min_singles_threshold']
        
        # Step 4: Conditional parlay selection
        selected_parlays = []
        parlay_pool = []
        
        if activate_parlays:
            logger.info(f"Activating parlays: {len(qualified_singles)} singles < {self.config['activation']['min_singles_threshold']}")
            parlay_pool = self._filter_parlay_pool(all_opportunities)
            selected_parlays = self._select_optimal_parlays(parlay_pool, bankroll)
        
        # Step 5: Calculate portfolio metrics
        portfolio_metrics = self._calculate_portfolio_metrics(
            qualified_singles, selected_parlays, bankroll
        )
        
        return {
            'singles': qualified_singles,
            'parlays': selected_parlays,
            'parlay_pool': parlay_pool,
            'parlay_activated': activate_parlays,
            'metrics': portfolio_metrics,
            'config': self.config
        }
    
    def _create_opportunities(self, 
                             predictions: pd.DataFrame,
                             odds_data: Dict[str, Dict]) -> List[BettingOpportunity]:
        """Create BettingOpportunity objects from predictions and odds."""
        opportunities = []
        
        for _, pred in predictions.iterrows():
            fight_key = f"{pred['fighter_a']} vs {pred['fighter_b']}"
            
            if fight_key not in odds_data:
                continue
                
            odds = odds_data[fight_key]
            
            # Create opportunities for both fighters
            for fighter_idx in ['a', 'b']:
                fighter = pred[f'fighter_{fighter_idx}']
                opponent = pred[f'fighter_{"b" if fighter_idx == "a" else "a"}']
                model_prob = pred[f'prob_{fighter_idx}']
                market_prob = 1 / odds[f'fighter_{fighter_idx}_decimal_odds']
                decimal_odds = odds[f'fighter_{fighter_idx}_decimal_odds']
                
                # Calculate EV and market gap
                ev = (model_prob * decimal_odds) - 1
                market_gap = model_prob - market_prob
                
                # Skip negative EV
                if ev <= 0:
                    continue
                
                opportunity = BettingOpportunity(
                    fight_id=fight_key,
                    fighter=fighter,
                    opponent=opponent,
                    event=pred.get('event', 'Unknown'),
                    model_prob=model_prob,
                    market_prob=market_prob,
                    odds=decimal_odds,
                    ev=ev,
                    confidence=pred.get('confidence', 0.5),
                    market_gap=market_gap,
                    weight_class=pred.get('weight_class', 'unknown'),
                    card_position=pred.get('card_position', 0),
                    is_favorite=(market_prob > 0.5)
                )
                
                opportunities.append(opportunity)
        
        return opportunities
    
    def _filter_singles(self, opportunities: List[BettingOpportunity]) -> List[BettingOpportunity]:
        """Apply strict filters for single bet selection including odds range."""
        config = self.config['singles']
        filtered = []
        
        for opp in opportunities:
            # Apply all filters including odds range
            if (config['ev_min'] <= opp.ev <= config['ev_max'] and
                config['odds_min'] <= opp.odds <= config['odds_max'] and
                opp.market_gap >= config['market_gap_min'] and
                opp.confidence >= config['confidence_min']):
                
                filtered.append(opp)
                logger.debug(f"Single qualified: {opp.fighter} (EV: {opp.ev:.1%}, Odds: {opp.odds:.2f})")
            elif opp.ev > config['ev_max']:
                logger.debug(f"Rejected (EV too high): {opp.fighter} (EV: {opp.ev:.1%})")
            elif opp.odds > config['odds_max']:
                logger.debug(f"Rejected (odds too high): {opp.fighter} (Odds: {opp.odds:.2f})")
            elif opp.odds < config['odds_min']:
                logger.debug(f"Rejected (odds too low): {opp.fighter} (Odds: {opp.odds:.2f})")
        
        # Sort by EV descending
        filtered.sort(key=lambda x: x.ev, reverse=True)
        
        logger.info(f"Singles filter: {len(opportunities)} → {len(filtered)}")
        return filtered
    
    def _filter_parlay_pool(self, opportunities: List[BettingOpportunity]) -> List[BettingOpportunity]:
        """Apply relaxed filters for parlay pool selection."""
        config = self.config['parlays']
        singles_config = self.config['singles']  # Use same odds limits as singles
        filtered = []
        
        for opp in opportunities:
            # Apply relaxed filters including odds range check
            if (opp.ev >= config['ev_min'] and
                opp.market_gap >= config['market_gap_min'] and
                opp.confidence >= config['confidence_min'] and
                singles_config['odds_min'] <= opp.odds <= singles_config['odds_max']):
                
                filtered.append(opp)
                logger.debug(f"Parlay pool: {opp.fighter} (EV: {opp.ev:.1%}, Odds: {opp.odds:.2f})")
        
        logger.info(f"Parlay pool: {len(opportunities)} → {len(filtered)}")
        return filtered
    
    def _select_optimal_parlays(self, 
                               pool: List[BettingOpportunity],
                               bankroll: float) -> List[Dict]:
        """
        Select optimal 2-leg parlays from the pool.
        This is a placeholder - will be replaced by sophisticated correlation analysis.
        """
        if len(pool) < 2:
            return []
        
        parlays = []
        max_parlays = self.config['parlays']['max_parlays']
        
        # Generate all 2-leg combinations
        from itertools import combinations
        
        for combo in combinations(pool, 2):
            leg1, leg2 = combo
            
            # Skip same fight
            if leg1.fight_id == leg2.fight_id:
                continue
            
            # Calculate combined metrics (simplified - will use correlation)
            combined_prob = leg1.adjusted_prob * leg2.adjusted_prob
            combined_odds = leg1.odds * leg2.odds
            combined_ev = (combined_prob * combined_odds) - 1
            
            # Skip if combined EV too low
            if combined_ev < 0.10:  # 10% minimum combined EV
                continue
            
            parlay = {
                'legs': [leg1, leg2],
                'combined_prob': combined_prob,
                'combined_odds': combined_odds,
                'combined_ev': combined_ev,
                'correlation': 0.1  # Placeholder - will be calculated
            }
            
            parlays.append(parlay)
        
        # Sort by combined EV and take top N
        parlays.sort(key=lambda x: x['combined_ev'], reverse=True)
        selected = parlays[:max_parlays]
        
        logger.info(f"Selected {len(selected)} parlays from {len(parlays)} candidates")
        return selected
    
    def _calculate_portfolio_metrics(self,
                                    singles: List[BettingOpportunity],
                                    parlays: List[Dict],
                                    bankroll: float) -> Dict:
        """Calculate portfolio-level metrics."""
        metrics = {
            'n_singles': len(singles),
            'n_parlays': len(parlays),
            'total_opportunities': len(singles) + len(parlays),
            'singles_ev': np.mean([s.ev for s in singles]) if singles else 0,
            'parlays_ev': np.mean([p['combined_ev'] for p in parlays]) if parlays else 0,
            'max_single_exposure': bankroll * self.config['singles']['max_exposure'],
            'max_parlay_exposure': bankroll * self.config['parlays']['max_exposure'],
            'portfolio_diversification': self._calculate_diversification(singles, parlays)
        }
        
        return metrics
    
    def _calculate_diversification(self,
                                  singles: List[BettingOpportunity],
                                  parlays: List[Dict]) -> float:
        """Calculate portfolio diversification score (0-1)."""
        if not singles and not parlays:
            return 0
        
        # Count unique events
        events = set()
        for s in singles:
            events.add(s.event)
        for p in parlays:
            for leg in p['legs']:
                events.add(leg.event)
        
        # Count unique weight classes
        weight_classes = set()
        for s in singles:
            weight_classes.add(s.weight_class)
        for p in parlays:
            for leg in p['legs']:
                weight_classes.add(leg.weight_class)
        
        # Simple diversification score
        event_diversity = min(len(events) / 3, 1.0)  # Normalize to max 3 events
        weight_diversity = min(len(weight_classes) / 5, 1.0)  # Normalize to max 5 classes
        
        return (event_diversity + weight_diversity) / 2
    
    def generate_report(self, selection_result: Dict) -> str:
        """Generate human-readable selection report."""
        report = []
        report.append("=" * 60)
        report.append("BETTING SELECTION REPORT")
        report.append("=" * 60)
        
        # Singles section
        report.append(f"\n📊 SINGLES ({selection_result['metrics']['n_singles']})")
        report.append("-" * 40)
        
        if selection_result['singles']:
            for s in selection_result['singles'][:5]:  # Show top 5
                report.append(f"• {s.fighter} vs {s.opponent}")
                report.append(f"  EV: {s.ev:.1%} | Gap: {s.market_gap:.1%} | Odds: {s.odds:.2f}")
        else:
            report.append("No qualified singles found")
        
        # Parlay activation status
        report.append(f"\n🎰 PARLAY ACTIVATION: {'YES' if selection_result['parlay_activated'] else 'NO'}")
        
        if selection_result['parlay_activated']:
            report.append(f"Reason: Only {selection_result['metrics']['n_singles']} singles < 2 threshold")
            
            # Parlays section
            report.append(f"\n🎲 PARLAYS ({selection_result['metrics']['n_parlays']})")
            report.append("-" * 40)
            
            for i, p in enumerate(selection_result['parlays'], 1):
                leg1, leg2 = p['legs']
                report.append(f"\nParlay {i}:")
                report.append(f"  Leg 1: {leg1.fighter} ({leg1.ev:.1%} EV)")
                report.append(f"  Leg 2: {leg2.fighter} ({leg2.ev:.1%} EV)")
                report.append(f"  Combined: {p['combined_ev']:.1%} EV @ {p['combined_odds']:.2f}")
        
        # Portfolio metrics
        report.append(f"\n📈 PORTFOLIO METRICS")
        report.append("-" * 40)
        report.append(f"Diversification: {selection_result['metrics']['portfolio_diversification']:.1%}")
        report.append(f"Avg Single EV: {selection_result['metrics']['singles_ev']:.1%}")
        report.append(f"Avg Parlay EV: {selection_result['metrics']['parlays_ev']:.1%}")
        
        return "\n".join(report)


if __name__ == "__main__":
    # Demo usage
    print("UFC Betting Selection Module - Conditional Parlay Logic")
    print("This module implements sophisticated selection with fallback to parlays")
    print("when single bet opportunities are limited (<2 qualified bets)")