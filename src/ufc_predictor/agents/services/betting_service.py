"""
UFC Betting Service

Professional betting analysis and recommendation service with dynamic bankroll management.
Extracted from notebook workflow for production use in automated agent.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class BankrollStrategy:
    """Configuration for bankroll management strategy"""
    tier: str
    description: str
    kelly_multiplier: float
    max_single_bet_pct: float
    max_total_exposure: float
    min_ev_threshold: float
    use_flat_betting: bool
    strategy: str
    flat_bet_amount: Optional[float] = None


@dataclass
class BetAnalysis:
    """Result of optimal bet size calculation"""
    recommended_bet: float
    kelly_fraction: Optional[float] = None
    adjusted_kelly: Optional[float] = None
    sizing_method: str = ""
    confidence_adjustment: float = 1.0
    ev: float = 0.0
    potential_profit: float = 0.0
    roi_potential: float = 0.0
    reason: Optional[str] = None


@dataclass
class BetRecommendation:
    """Individual betting recommendation"""
    fighter: str
    opponent: str
    fight: str
    model_probability: float
    decimal_odds: float
    american_odds: int
    expected_value: float
    kelly_fraction: float
    recommended_stake: float
    potential_profit: float
    sizing_method: str
    confidence_score: float
    is_upset_opportunity: bool
    is_high_confidence: bool
    market_favorite: str
    model_favorite: str


@dataclass
class PortfolioSummary:
    """Portfolio-level betting summary"""
    total_recommended_stake: float = 0.0
    total_potential_profit: float = 0.0
    number_of_bets: int = 0
    portfolio_ev: float = 0.0
    bankroll_utilization: float = 0.0
    expected_return: float = 0.0


class BettingRecommendations:
    """Complete betting recommendations result"""
    
    def __init__(self, bankroll: float, strategy: BankrollStrategy):
        self.bankroll_info = {
            'amount': bankroll,
            'tier': strategy.tier,
            'strategy': strategy
        }
        self.single_bets: List[BetRecommendation] = []
        self.rejected_bets: List[Dict] = []
        self.portfolio_summary = PortfolioSummary()


class UFCBettingService:
    """
    Professional UFC betting service with dynamic bankroll management
    
    Extracted from notebook Cell 4 for production use in automated agent.
    Implements research-backed Kelly criterion with tier-based strategies.
    """
    
    def __init__(self):
        """Initialize betting service"""
        pass
    
    def determine_bankroll_strategy(self, bankroll: float) -> BankrollStrategy:
        """
        Determine optimal betting strategy based on bankroll size
        Research-backed approach from academic literature
        
        Args:
            bankroll: Current bankroll amount
            
        Returns:
            BankrollStrategy: Optimal strategy configuration
        """
        if bankroll < 200:
            return BankrollStrategy(
                tier='MICRO',
                description='Ultra-conservative for small bankrolls',
                kelly_multiplier=0.15,  # Heavily fractional Kelly
                max_single_bet_pct=0.02,  # 2% max per bet
                max_total_exposure=0.10,  # 10% total exposure
                min_ev_threshold=0.12,  # Higher EV required (12%)
                flat_bet_amount=max(5, bankroll * 0.01),  # $5 minimum or 1%
                use_flat_betting=bankroll < 100,  # Use flat betting for very small amounts
                strategy='Flat betting or 1/8 Kelly to preserve capital'
            )
        elif bankroll < 1000:
            return BankrollStrategy(
                tier='SMALL',
                description='Conservative growth strategy',
                kelly_multiplier=0.25,  # Quarter Kelly
                max_single_bet_pct=0.05,  # 5% max per bet
                max_total_exposure=0.20,  # 20% total exposure
                min_ev_threshold=0.08,  # Standard EV threshold
                use_flat_betting=False,
                strategy='1/4 Kelly with strict risk management'
            )
        else:
            return BankrollStrategy(
                tier='STANDARD',
                description='Moderate Kelly with professional caps',
                kelly_multiplier=0.50,  # Half Kelly
                max_single_bet_pct=0.075,  # 7.5% max per bet
                max_total_exposure=0.25,  # 25% total exposure
                min_ev_threshold=0.05,  # Lower EV acceptable
                use_flat_betting=False,
                strategy='1/2 Kelly with professional risk management'
            )
    
    def calculate_optimal_bet_size(self, model_prob: float, decimal_odds: float, 
                                  bankroll: float, strategy: BankrollStrategy, 
                                  confidence_score: float = 1.0) -> BetAnalysis:
        """
        Calculate optimal bet size using dynamic Kelly approach
        
        Args:
            model_prob: Model probability for the fighter
            decimal_odds: Decimal odds for the bet
            bankroll: Current bankroll amount
            strategy: Bankroll strategy configuration
            confidence_score: Confidence in the prediction (0-1)
            
        Returns:
            BetAnalysis: Complete bet sizing analysis
        """
        # Calculate expected value
        ev = (model_prob * decimal_odds) - 1
        
        # Check minimum EV threshold
        if ev < strategy.min_ev_threshold:
            return BetAnalysis(
                recommended_bet=0,
                reason=f"EV too low ({ev:.1%} < {strategy.min_ev_threshold:.1%})",
                ev=ev
            )
        
        # Calculate Kelly fraction
        kelly_fraction = ((model_prob * decimal_odds) - 1) / (decimal_odds - 1)
        
        if kelly_fraction <= 0:
            return BetAnalysis(
                recommended_bet=0,
                reason="Negative Kelly fraction",
                ev=ev
            )
        
        # Apply strategy-specific adjustments
        if strategy.use_flat_betting:
            # Use flat betting for micro bankrolls
            bet_size = strategy.flat_bet_amount
            sizing_method = "Flat betting (capital preservation)"
        else:
            # Use fractional Kelly
            adjusted_kelly = kelly_fraction * strategy.kelly_multiplier
            bet_size = bankroll * adjusted_kelly
            sizing_method = f"{strategy.kelly_multiplier:.0%} Kelly"
        
        # Apply confidence adjustment
        confidence_multiplier = 0.5 + (confidence_score * 0.5)  # 0.5 to 1.0 range
        bet_size *= confidence_multiplier
        
        # Apply maximum bet size cap
        max_bet = bankroll * strategy.max_single_bet_pct
        if bet_size > max_bet:
            bet_size = max_bet
            sizing_method += " (capped)"
        
        # Apply minimum bet size
        min_bet = max(5, bankroll * 0.005)  # $5 minimum or 0.5%
        if bet_size < min_bet:
            if ev > strategy.min_ev_threshold * 1.5:  # Only bet if EV is significantly higher
                bet_size = min_bet
                sizing_method += " (minimum)"
            else:
                return BetAnalysis(
                    recommended_bet=0,
                    reason=f"Bet size too small (${bet_size:.2f} < ${min_bet:.2f})",
                    ev=ev
                )
        
        return BetAnalysis(
            recommended_bet=bet_size,
            kelly_fraction=kelly_fraction,
            adjusted_kelly=kelly_fraction * strategy.kelly_multiplier,
            sizing_method=sizing_method,
            confidence_adjustment=confidence_multiplier,
            ev=ev,
            potential_profit=bet_size * (decimal_odds - 1),
            roi_potential=((bet_size * decimal_odds) - bet_size) / bet_size
        )
    
    def generate_betting_recommendations(self, predictions_analysis, 
                                       bankroll: float) -> BettingRecommendations:
        """
        Generate complete betting recommendations with dynamic bankroll management
        
        Args:
            predictions_analysis: PredictionAnalysis object from PredictionService
            bankroll: Current bankroll amount
            
        Returns:
            BettingRecommendations: Complete betting recommendations
        """
        logger.info(f"Generating betting recommendations for bankroll: ${bankroll:.2f}")
        
        strategy = self.determine_bankroll_strategy(bankroll)
        recommendations = BettingRecommendations(bankroll, strategy)
        
        logger.info(
            f"Using {strategy.tier} tier strategy: {strategy.strategy}"
        )
        
        total_stake = 0
        total_potential_profit = 0
        
        logger.info(f"Analyzing {len(predictions_analysis.fight_predictions)} fights")
        
        for fight in predictions_analysis.fight_predictions:
            fighter_a = fight.fighter_a
            fighter_b = fight.fighter_b
            
            logger.debug(f"Analyzing betting opportunities for {fight.fight_key}")
            
            # Analyze both fighters for betting opportunities
            fighters_to_analyze = [
                (fighter_a, fight.model_prediction_a, fight.market_odds_a, fight.expected_value_a),
                (fighter_b, fight.model_prediction_b, fight.market_odds_b, fight.expected_value_b)
            ]
            
            for fighter, model_prob, odds, ev in fighters_to_analyze:
                opponent = fighter_b if fighter == fighter_a else fighter_a
                
                # Check if we have enough exposure left
                remaining_exposure = (bankroll * strategy.max_total_exposure) - total_stake
                if remaining_exposure < bankroll * 0.01:  # Less than 1% remaining
                    recommendations.rejected_bets.append({
                        'fighter': fighter,
                        'reason': 'Portfolio exposure limit reached',
                        'ev': ev
                    })
                    continue
                
                # Calculate optimal bet size
                bet_analysis = self.calculate_optimal_bet_size(
                    model_prob, odds, bankroll, strategy, fight.confidence_score
                )
                
                if bet_analysis.recommended_bet == 0:
                    recommendations.rejected_bets.append({
                        'fighter': fighter,
                        'reason': bet_analysis.reason,
                        'ev': ev
                    })
                    continue
                
                # Adjust bet size if it would exceed remaining exposure
                final_bet_size = min(bet_analysis.recommended_bet, remaining_exposure)
                
                # Create betting recommendation
                bet_recommendation = BetRecommendation(
                    fighter=fighter,
                    opponent=opponent,
                    fight=fight.fight_key,
                    model_probability=model_prob,
                    decimal_odds=odds,
                    american_odds=self._decimal_to_american(odds),
                    expected_value=ev,
                    kelly_fraction=bet_analysis.kelly_fraction or 0,
                    recommended_stake=final_bet_size,
                    potential_profit=final_bet_size * (odds - 1),
                    sizing_method=bet_analysis.sizing_method,
                    confidence_score=fight.confidence_score,
                    is_upset_opportunity=fight.is_upset_opportunity,
                    is_high_confidence=fight.is_high_confidence,
                    market_favorite=fight.market_favorite,
                    model_favorite=fight.model_favorite
                )
                
                recommendations.single_bets.append(bet_recommendation)
                total_stake += final_bet_size
                total_potential_profit += bet_recommendation.potential_profit
                
                logger.info(
                    f"Recommended bet: {fighter} @ {odds:.2f} "
                    f"(${final_bet_size:.2f}, EV: {ev:+.1%})"
                )
        
        # Calculate portfolio metrics
        if recommendations.single_bets:
            total_ev = sum(
                bet.expected_value * bet.recommended_stake 
                for bet in recommendations.single_bets
            )
            portfolio_ev = total_ev / total_stake if total_stake > 0 else 0
            
            recommendations.portfolio_summary = PortfolioSummary(
                total_recommended_stake=total_stake,
                total_potential_profit=total_potential_profit,
                number_of_bets=len(recommendations.single_bets),
                portfolio_ev=portfolio_ev,
                bankroll_utilization=total_stake / bankroll,
                expected_return=total_ev
            )
        
        logger.info(
            f"Generated {len(recommendations.single_bets)} betting recommendations, "
            f"total stake: ${total_stake:.2f}"
        )
        
        return recommendations
    
    def _decimal_to_american(self, decimal_odds: float) -> int:
        """Convert decimal odds to American format"""
        if decimal_odds >= 2.0:
            return int((decimal_odds - 1) * 100)
        else:
            return int(-100 / (decimal_odds - 1))
    
    def format_recommendations_summary(self, recommendations: BettingRecommendations) -> str:
        """
        Format betting recommendations summary for display
        
        Args:
            recommendations: Complete betting recommendations
            
        Returns:
            str: Formatted summary text
        """
        output = [
            f"📊 BETTING RECOMMENDATIONS SUMMARY",
            f"=" * 45,
            f"💳 Bankroll: ${recommendations.bankroll_info['amount']:,.2f}",
            f"🎯 Tier: {recommendations.bankroll_info['tier']}",
            f"📈 Strategy: {recommendations.bankroll_info['strategy'].strategy}"
        ]
        
        if recommendations.single_bets:
            summary = recommendations.portfolio_summary
            
            # Sort recommendations by EV
            sorted_bets = sorted(
                recommendations.single_bets, 
                key=lambda x: x.expected_value, 
                reverse=True
            )
            
            output.extend([
                f"",
                f"🎯 RECOMMENDED BETS: {summary.number_of_bets}",
                f"💰 Total Stakes: ${summary.total_recommended_stake:.2f}",
                f"📈 Expected Return: ${summary.expected_return:.2f}",
                f"🎲 Portfolio EV: {summary.portfolio_ev:+.1%}",
                f"💳 Bankroll Utilization: {summary.bankroll_utilization:.1%}",
                f"",
                f"🏆 TOP BETTING OPPORTUNITIES:"
            ])
            
            for i, bet in enumerate(sorted_bets[:5], 1):  # Show top 5
                output.extend([
                    f"{i}. {bet.fighter} @ {bet.decimal_odds:.2f}",
                    f"   💰 Stake: ${bet.recommended_stake:.2f}",
                    f"   📈 EV: {bet.expected_value:+.1%}",
                    f"   🎯 Potential: ${bet.potential_profit:.2f}"
                ])
                
                if bet.is_upset_opportunity:
                    output.append(f"   🚨 UPSET OPPORTUNITY")
                if bet.is_high_confidence:
                    output.append(f"   ⭐ HIGH CONFIDENCE")
                output.append("")
            
            # Risk warning for high utilization
            if summary.bankroll_utilization > 0.20:
                output.extend([
                    f"⚠️  WARNING: High bankroll utilization ({summary.bankroll_utilization:.1%})",
                    f"   Consider reducing position sizes for first-time strategy"
                ])
        
        else:
            output.extend([
                f"",
                f"📭 NO BETTING OPPORTUNITIES FOUND",
                f"💡 Reasons:"
            ])
            
            rejection_reasons = {}
            for rejected in recommendations.rejected_bets:
                reason = rejected['reason']
                if reason not in rejection_reasons:
                    rejection_reasons[reason] = 0
                rejection_reasons[reason] += 1
            
            for reason, count in rejection_reasons.items():
                output.append(f"   • {reason}: {count} bets")
        
        return "\n".join(output)
    
    def format_betting_sheet(self, recommendations: BettingRecommendations, 
                           event_name: str) -> str:
        """
        Format professional betting sheet
        
        Args:
            recommendations: Complete betting recommendations
            event_name: Name of the UFC event
            
        Returns:
            str: Formatted betting sheet
        """
        output = [
            f"📋 FINAL BETTING SHEET",
            f"=" * 80
        ]
        
        if not recommendations.single_bets:
            output.append(f"📭 No bets recommended for this event")
            return "\n".join(output)
        
        # Header
        output.extend([
            f"Event: {event_name}",
            f"Bankroll: ${recommendations.bankroll_info['amount']:,.2f} ({recommendations.bankroll_info['tier']} tier)",
            f"Strategy: {recommendations.bankroll_info['strategy'].strategy}",
            f"Total Recommended: ${recommendations.portfolio_summary.total_recommended_stake:.2f}",
            f"Expected Return: ${recommendations.portfolio_summary.expected_return:.2f}",
            f"Portfolio EV: {recommendations.portfolio_summary.portfolio_ev:+.1%}",
            f"",
            f"{'#':<3} {'Fighter':<25} {'Odds':<8} {'Stake':<10} {'EV':<8} {'Profit':<10} {'Type'}"
        ])
        output.append("-" * 80)
        
        # Sort by EV for display
        sorted_bets = sorted(
            recommendations.single_bets, 
            key=lambda x: x.expected_value, 
            reverse=True
        )
        
        for i, bet in enumerate(sorted_bets, 1):
            bet_type = ""
            if bet.is_upset_opportunity:
                bet_type += "🚨"
            if bet.is_high_confidence:
                bet_type += "⭐"
            if not bet_type:
                bet_type = "📊"
            
            output.append(
                f"{i:<3} {bet.fighter:<25} {bet.decimal_odds:<8.2f} "
                f"${bet.recommended_stake:<9.2f} {bet.expected_value:<7.1%} "
                f"${bet.potential_profit:<9.2f} {bet_type}"
            )
        
        output.append("-" * 80)
        output.append(
            f"{'TOTAL':<39} ${recommendations.portfolio_summary.total_recommended_stake:<9.2f} "
            f"{recommendations.portfolio_summary.portfolio_ev:<7.1%} "
            f"${recommendations.portfolio_summary.total_potential_profit:<9.2f}"
        )
        
        return "\n".join(output)
    
    def get_top_opportunities(self, recommendations: BettingRecommendations, 
                            top_n: int = 5) -> List[BetRecommendation]:
        """
        Get top betting opportunities sorted by expected value
        
        Args:
            recommendations: Complete betting recommendations
            top_n: Number of top opportunities to return
            
        Returns:
            List of top BetRecommendation objects
        """
        if not recommendations.single_bets:
            return []
        
        sorted_bets = sorted(
            recommendations.single_bets, 
            key=lambda x: x.expected_value, 
            reverse=True
        )
        
        return sorted_bets[:top_n]