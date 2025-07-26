"""
Optimal Betting Strategy Implementation

This module implements the comprehensive betting strategy framework designed to maximize
profitability from the UFC prediction system while managing risk through systematic
bankroll management, market timing, and multi-market optimization.

Based on research showing:
- UFC markets have exploitable inefficiencies with 54.1% line movement tracking success
- System achieves 72.9% winner prediction accuracy vs ~65% market baseline
- Public bias creates value on underdogs and less popular fighters
- Multi-sportsbook approach essential for 30-40% better prop betting odds
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
from pathlib import Path

class RiskTier(Enum):
    """Risk tiers for Kelly multiplier adjustment"""
    CONSERVATIVE = "conservative"
    MODERATE = "moderate" 
    AGGRESSIVE = "aggressive"

class BetType(Enum):
    """Types of bets supported"""
    WINNER = "winner"
    METHOD = "method"
    PROP = "prop"
    MULTI_BET = "multi_bet"

@dataclass
class StyleMatchup:
    """Style matchup analysis for betting confidence adjustment"""
    wrestler_vs_poor_tdd: bool = False
    heavy_hands_vs_poor_chin: bool = False
    cardio_advantage: bool = False
    southpaw_advantage: bool = False
    confidence_multiplier: float = 1.0
    
    def __post_init__(self):
        """Calculate confidence multiplier based on matchup advantages"""
        advantages = sum([
            self.wrestler_vs_poor_tdd,
            self.heavy_hands_vs_poor_chin,
            self.cardio_advantage,
            self.southpaw_advantage
        ])
        
        if advantages >= 2:
            self.confidence_multiplier = 1.25  # High confidence
        elif advantages == 1:
            self.confidence_multiplier = 1.0   # Normal confidence
        else:
            self.confidence_multiplier = 0.5   # Low confidence

@dataclass
class WeightCuttingIndicators:
    """Weight cutting assessment for risk adjustment"""
    missed_weight_recently: bool = False
    visible_struggle_weigh_ins: bool = False
    large_same_day_cut: bool = False
    looks_depleted: bool = False
    has_professional_team: bool = False
    easy_cut_history: bool = False
    
    @property
    def risk_multiplier(self) -> float:
        """Calculate risk multiplier based on weight cutting indicators"""
        red_flags = sum([
            self.missed_weight_recently,
            self.visible_struggle_weigh_ins, 
            self.large_same_day_cut,
            self.looks_depleted
        ])
        
        green_flags = sum([
            self.has_professional_team,
            self.easy_cut_history
        ])
        
        if red_flags >= 2:
            return 0.5  # High risk, reduce bet size
        elif red_flags == 1 and green_flags == 0:
            return 0.75  # Moderate risk
        elif green_flags >= 1 and red_flags == 0:
            return 1.1   # Low risk, slight increase
        else:
            return 1.0   # Neutral

@dataclass
class BettingOpportunity:
    """Enhanced betting opportunity with strategy framework integration"""
    fighter: str
    opponent: str
    event: str
    bet_type: BetType
    model_prob: float
    market_odds: float  # American odds
    expected_value: float
    kelly_fraction: float
    risk_tier: RiskTier
    style_matchup: StyleMatchup
    weight_cutting: WeightCuttingIndicators
    recommended_bet_size: float
    confidence_score: float
    market_timing: str  # "opening", "closing", "live"
    correlation_penalty: float = 0.0
    
    @property
    def adjusted_bet_size(self) -> float:
        """Calculate final bet size with all adjustments"""
        base_size = self.recommended_bet_size
        
        # Apply style matchup adjustment
        base_size *= self.style_matchup.confidence_multiplier
        
        # Apply weight cutting adjustment
        base_size *= self.weight_cutting.risk_multiplier
        
        # Apply correlation penalty
        base_size *= (1 - self.correlation_penalty)
        
        return base_size

class OptimalBettingStrategy:
    """
    Implements the complete optimal betting strategy framework
    """
    
    def __init__(self, initial_bankroll: float = 1000.0, config_path: Optional[str] = None):
        """
        Initialize the betting strategy with configurable parameters
        
        Args:
            initial_bankroll: Starting bankroll amount
            config_path: Optional path to configuration JSON file
        """
        self.initial_bankroll = initial_bankroll
        self.current_bankroll = initial_bankroll
        self.bet_history = []
        self.performance_metrics = {}
        
        # Load configuration
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
        else:
            config = self._get_default_config()
            
        self.config = config
        self._validate_config()
        
        # Performance tracking
        self.daily_returns = []
        self.max_drawdown = 0.0
        self.current_drawdown = 0.0
        self.peak_bankroll = initial_bankroll
        
    def _get_default_config(self) -> Dict:
        """Get default configuration based on optimal strategy framework"""
        return {
            "risk_tiers": {
                "conservative": {
                    "kelly_multiplier": 0.25,
                    "max_single_bet_pct": 0.05,
                    "min_expected_value": 0.08
                },
                "moderate": {
                    "kelly_multiplier": 0.5,
                    "max_single_bet_pct": 0.075,
                    "min_expected_value": 0.12
                },
                "aggressive": {
                    "kelly_multiplier": 0.75,
                    "max_single_bet_pct": 0.10,
                    "min_expected_value": 0.15
                }
            },
            "portfolio_constraints": {
                "max_total_exposure_pct": 0.25,
                "max_correlated_exposure_pct": 0.15,
                "min_bankroll_reserve_pct": 0.20
            },
            "drawdown_protection": {
                "10_pct_loss_reduction": 0.25,
                "20_pct_loss_reduction": 0.50,
                "30_pct_loss_pause": True
            },
            "bet_type_thresholds": {
                "winner": {"min_ev": 0.08, "preferred_ev": 0.12, "elite_ev": 0.20},
                "method": {"min_ev": 0.15, "ko_specialist_ev": 0.20, "submission_ev": 0.18},
                "prop": {"round_totals_ev": 0.12, "performance_props_ev": 0.15}
            },
            "portfolio_allocation": {
                "single_bets_pct": 0.70,
                "multi_bets_pct": 0.30,
                "max_legs_in_parlay": 4
            },
            "correlation_penalties": {
                "same_event": 0.08,
                "same_division": 0.05,
                "main_vs_prelim": 0.03
            },
            "market_timing": {
                "opening_line_stake_pct": 0.60,
                "closing_line_stake_pct": 0.40,
                "public_fade_threshold": 0.70
            }
        }
    
    def _validate_config(self):
        """Validate configuration parameters"""
        required_keys = ["risk_tiers", "portfolio_constraints", "bet_type_thresholds"]
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required config key: {key}")
    
    def calculate_kelly_fraction(self, prob: float, american_odds: float, risk_tier: RiskTier) -> float:
        """
        Calculate risk-adjusted Kelly fraction based on tier
        
        Args:
            prob: Model probability of outcome
            american_odds: American betting odds
            risk_tier: Risk tier for Kelly multiplier
            
        Returns:
            Adjusted Kelly fraction
        """
        # Convert American odds to decimal
        if american_odds > 0:
            decimal_odds = (american_odds / 100) + 1
        else:
            decimal_odds = (100 / abs(american_odds)) + 1
        
        # Standard Kelly calculation
        kelly = (prob * decimal_odds - 1) / (decimal_odds - 1)
        
        # Apply risk tier multiplier
        tier_config = self.config["risk_tiers"][risk_tier.value]
        adjusted_kelly = kelly * tier_config["kelly_multiplier"]
        
        # Cap at maximum single bet percentage
        max_bet_pct = tier_config["max_single_bet_pct"]
        
        return min(max(adjusted_kelly, 0), max_bet_pct)
    
    def assess_expected_value(self, prob: float, american_odds: float) -> float:
        """Calculate expected value of a bet"""
        if american_odds > 0:
            decimal_odds = (american_odds / 100) + 1
        else:
            decimal_odds = (100 / abs(american_odds)) + 1
        
        return (prob * (decimal_odds - 1)) - (1 - prob)
    
    def determine_risk_tier(self, ev: float, bet_type: BetType, confidence_factors: Dict) -> RiskTier:
        """
        Determine appropriate risk tier based on EV and confidence factors
        
        Args:
            ev: Expected value of the bet
            bet_type: Type of bet being placed
            confidence_factors: Dictionary of factors affecting confidence
            
        Returns:
            Appropriate risk tier
        """
        type_key = bet_type.value if bet_type != BetType.MULTI_BET else "winner"
        thresholds = self.config["bet_type_thresholds"][type_key]
        
        # Base tier on EV
        if ev >= thresholds.get("elite_ev", 0.20):
            base_tier = RiskTier.AGGRESSIVE
        elif ev >= thresholds.get("preferred_ev", 0.12):
            base_tier = RiskTier.MODERATE
        else:
            base_tier = RiskTier.CONSERVATIVE
        
        # Adjust based on confidence factors
        high_confidence_count = sum([
            confidence_factors.get("style_advantage", False),
            confidence_factors.get("weight_cut_advantage", False),
            confidence_factors.get("line_value", False),
            confidence_factors.get("model_confidence", False)
        ])
        
        if high_confidence_count >= 3 and base_tier != RiskTier.AGGRESSIVE:
            # Upgrade tier for high confidence
            if base_tier == RiskTier.CONSERVATIVE:
                return RiskTier.MODERATE
            else:
                return RiskTier.AGGRESSIVE
        elif high_confidence_count == 0 and base_tier != RiskTier.CONSERVATIVE:
            # Downgrade tier for low confidence
            return RiskTier.CONSERVATIVE
            
        return base_tier
    
    def analyze_style_matchup(self, fighter_stats: Dict, opponent_stats: Dict) -> StyleMatchup:
        """
        Analyze style matchup to determine confidence adjustments
        
        Args:
            fighter_stats: Fighter statistics and attributes
            opponent_stats: Opponent statistics and attributes
            
        Returns:
            StyleMatchup analysis
        """
        # Wrestler vs poor takedown defense
        wrestler_advantage = (
            fighter_stats.get("td_avg", 0) > 2.0 and
            opponent_stats.get("td_def", 0) < 0.60
        )
        
        # Heavy hands vs poor chin
        ko_advantage = (
            fighter_stats.get("ko_percentage", 0) > 0.60 and
            len(opponent_stats.get("recent_ko_losses", [])) >= 2
        )
        
        # Cardio advantage
        cardio_advantage = (
            fighter_stats.get("third_round_performance", 0) > 0.70 and
            opponent_stats.get("third_round_performance", 0) < 0.50
        )
        
        # Southpaw advantage
        southpaw_advantage = (
            fighter_stats.get("stance") == "Southpaw" and
            opponent_stats.get("southpaw_experience", 0) < 0.50
        )
        
        return StyleMatchup(
            wrestler_vs_poor_tdd=wrestler_advantage,
            heavy_hands_vs_poor_chin=ko_advantage,
            cardio_advantage=cardio_advantage,
            southpaw_advantage=southpaw_advantage
        )
    
    def assess_weight_cutting(self, fighter_data: Dict) -> WeightCuttingIndicators:
        """
        Assess weight cutting risk factors
        
        Args:
            fighter_data: Fighter information including weight cut history
            
        Returns:
            WeightCuttingIndicators assessment
        """
        return WeightCuttingIndicators(
            missed_weight_recently=fighter_data.get("missed_weight_last_2", False),
            visible_struggle_weigh_ins=fighter_data.get("struggled_at_weigh_ins", False),
            large_same_day_cut=fighter_data.get("same_day_cut_pct", 0) > 0.10,
            looks_depleted=fighter_data.get("looks_depleted", False),
            has_professional_team=fighter_data.get("professional_nutrition_team", False),
            easy_cut_history=fighter_data.get("easy_cut_history", False)
        )
    
    def calculate_correlation_penalty(self, existing_bets: List[BettingOpportunity], 
                                    new_bet: BettingOpportunity) -> float:
        """
        Calculate correlation penalty for new bet based on existing positions
        
        Args:
            existing_bets: List of already placed bets
            new_bet: New bet being considered
            
        Returns:
            Correlation penalty as a decimal (0.0 to 1.0)
        """
        total_penalty = 0.0
        penalties = self.config["correlation_penalties"]
        
        for existing_bet in existing_bets:
            # Same event penalty
            if existing_bet.event == new_bet.event:
                total_penalty += penalties["same_event"]
                
                # Additional penalties for same event
                if self._same_division(existing_bet, new_bet):
                    total_penalty += penalties["same_division"]
                    
                if self._main_vs_prelim(existing_bet, new_bet):
                    total_penalty += penalties["main_vs_prelim"]
        
        return min(total_penalty, 0.50)  # Cap penalty at 50%
    
    def _same_division(self, bet1: BettingOpportunity, bet2: BettingOpportunity) -> bool:
        """Check if two bets are from the same weight division"""
        # This would need fighter weight class data
        # Placeholder implementation
        return False
    
    def _main_vs_prelim(self, bet1: BettingOpportunity, bet2: BettingOpportunity) -> bool:
        """Check if bets are from main card vs preliminary card"""
        # This would need fight card position data
        # Placeholder implementation
        return False
    
    def should_bet_opening_line(self, prediction: Dict, market_data: Dict) -> bool:
        """
        Determine if bet should be placed on opening line vs waiting for closing
        
        Args:
            prediction: Model prediction data
            market_data: Market information including public betting percentages
            
        Returns:
            True if should bet opening line, False to wait for closing
        """
        # Bet opening line for underdogs with high model probability
        model_prob = prediction.get("win_probability", 0)
        american_odds = market_data.get("opening_odds", 0)
        
        if american_odds > 150 and model_prob > 0.40:
            return True
            
        # Fade public bias on opening lines
        public_bet_pct = market_data.get("public_bet_percentage", 0.5)
        if public_bet_pct > self.config["market_timing"]["public_fade_threshold"]:
            return True
            
        return False
    
    def get_optimal_stake_split(self, total_stake: float, timing_strategy: str) -> Tuple[float, float]:
        """
        Split stake between opening and closing line based on strategy
        
        Args:
            total_stake: Total amount to bet
            timing_strategy: "opening_heavy", "closing_heavy", or "balanced"
            
        Returns:
            Tuple of (opening_stake, closing_stake)
        """
        if timing_strategy == "opening_heavy":
            opening_pct = 0.70
        elif timing_strategy == "closing_heavy":
            opening_pct = 0.30
        else:  # balanced
            opening_pct = 0.50
            
        opening_stake = total_stake * opening_pct
        closing_stake = total_stake * (1 - opening_pct)
        
        return opening_stake, closing_stake
    
    def update_bankroll(self, bet_result: float):
        """
        Update bankroll and tracking metrics after bet result
        
        Args:
            bet_result: Profit/loss from bet (negative for losses)
        """
        self.current_bankroll += bet_result
        
        # Update peak and drawdown tracking
        if self.current_bankroll > self.peak_bankroll:
            self.peak_bankroll = self.current_bankroll
            self.current_drawdown = 0.0
        else:
            self.current_drawdown = (self.peak_bankroll - self.current_bankroll) / self.peak_bankroll
            self.max_drawdown = max(self.max_drawdown, self.current_drawdown)
    
    def get_drawdown_multiplier(self) -> float:
        """
        Get bet size multiplier based on current drawdown
        
        Returns:
            Multiplier for bet sizes (0.0 to 1.0)
        """
        drawdown_config = self.config["drawdown_protection"]
        
        if self.current_drawdown >= 0.30:
            if drawdown_config["30_pct_loss_pause"]:
                return 0.0  # Pause betting
            else:
                return 0.25
        elif self.current_drawdown >= 0.20:
            return 1 - drawdown_config["20_pct_loss_reduction"]
        elif self.current_drawdown >= 0.10:
            return 1 - drawdown_config["10_pct_loss_reduction"]
        else:
            return 1.0
    
    def analyze_betting_opportunity(self, prediction: Dict, odds_data: Dict, 
                                  fighter_stats: Dict, opponent_stats: Dict,
                                  existing_bets: List[BettingOpportunity] = None) -> Optional[BettingOpportunity]:
        """
        Comprehensive analysis of a betting opportunity using the optimal strategy framework
        
        Args:
            prediction: Model prediction data
            odds_data: Market odds and timing data
            fighter_stats: Fighter statistics and information
            opponent_stats: Opponent statistics and information  
            existing_bets: List of existing bets for correlation analysis
            
        Returns:
            BettingOpportunity if profitable, None otherwise
        """
        if existing_bets is None:
            existing_bets = []
            
        # Extract basic information
        fighter = prediction["fighter"]
        opponent = prediction["opponent"]
        model_prob = prediction["win_probability"]
        american_odds = odds_data["current_odds"]
        
        # Calculate expected value
        ev = self.assess_expected_value(model_prob, american_odds)
        
        # Check minimum EV threshold
        bet_type = BetType.WINNER  # Default, could be determined from odds_data
        min_ev = self.config["bet_type_thresholds"]["winner"]["min_ev"]
        
        if ev < min_ev:
            return None
            
        # Analyze style matchup and weight cutting
        style_matchup = self.analyze_style_matchup(fighter_stats, opponent_stats)
        weight_cutting = self.assess_weight_cutting(fighter_stats)
        
        # Determine risk tier
        confidence_factors = {
            "style_advantage": style_matchup.confidence_multiplier > 1.0,
            "weight_cut_advantage": weight_cutting.risk_multiplier > 1.0,
            "line_value": ev > min_ev * 1.5,
            "model_confidence": prediction.get("confidence_score", 0.5) > 0.7
        }
        
        risk_tier = self.determine_risk_tier(ev, bet_type, confidence_factors)
        
        # Calculate Kelly fraction and bet size
        kelly_fraction = self.calculate_kelly_fraction(model_prob, american_odds, risk_tier)
        base_bet_size = kelly_fraction * self.current_bankroll
        
        # Apply drawdown protection
        drawdown_multiplier = self.get_drawdown_multiplier()
        if drawdown_multiplier == 0.0:
            return None  # Betting paused due to drawdown
            
        base_bet_size *= drawdown_multiplier
        
        # Calculate correlation penalty
        correlation_penalty = self.calculate_correlation_penalty(existing_bets, 
                                                               BettingOpportunity(
                                                                   fighter=fighter,
                                                                   opponent=opponent,
                                                                   event=odds_data.get("event", ""),
                                                                   bet_type=bet_type,
                                                                   model_prob=model_prob,
                                                                   market_odds=american_odds,
                                                                   expected_value=ev,
                                                                   kelly_fraction=kelly_fraction,
                                                                   risk_tier=risk_tier,
                                                                   style_matchup=style_matchup,
                                                                   weight_cutting=weight_cutting,
                                                                   recommended_bet_size=base_bet_size,
                                                                   confidence_score=confidence_factors.get("model_confidence", 0.5),
                                                                   market_timing=odds_data.get("timing", "closing")
                                                               ))
        
        # Create final opportunity
        opportunity = BettingOpportunity(
            fighter=fighter,
            opponent=opponent,
            event=odds_data.get("event", ""),
            bet_type=bet_type,
            model_prob=model_prob,
            market_odds=american_odds,
            expected_value=ev,
            kelly_fraction=kelly_fraction,
            risk_tier=risk_tier,
            style_matchup=style_matchup,
            weight_cutting=weight_cutting,
            recommended_bet_size=base_bet_size,
            confidence_score=sum(confidence_factors.values()) / len(confidence_factors),
            market_timing=odds_data.get("timing", "closing"),
            correlation_penalty=correlation_penalty
        )
        
        # Final bet size with all adjustments
        final_bet_size = opportunity.adjusted_bet_size
        
        # Verify portfolio constraints
        total_exposure = sum(bet.adjusted_bet_size for bet in existing_bets) + final_bet_size
        max_exposure = self.current_bankroll * self.config["portfolio_constraints"]["max_total_exposure_pct"]
        
        if total_exposure > max_exposure:
            return None  # Would exceed portfolio limits
            
        opportunity.recommended_bet_size = final_bet_size
        return opportunity
    
    def get_performance_summary(self) -> Dict:
        """
        Get comprehensive performance summary
        
        Returns:
            Dictionary with key performance metrics
        """
        if not self.bet_history:
            return {"status": "No bets recorded"}
            
        total_profit = self.current_bankroll - self.initial_bankroll
        roi = (total_profit / self.initial_bankroll) * 100
        
        winning_bets = sum(1 for bet in self.bet_history if bet.get("profit", 0) > 0)
        total_bets = len(self.bet_history)
        win_rate = (winning_bets / total_bets) * 100 if total_bets > 0 else 0
        
        return {
            "total_bets": total_bets,
            "winning_bets": winning_bets, 
            "win_rate": f"{win_rate:.1f}%",
            "total_profit": f"${total_profit:.2f}",
            "roi": f"{roi:.1f}%",
            "max_drawdown": f"{self.max_drawdown * 100:.1f}%",
            "current_bankroll": f"${self.current_bankroll:.2f}",
            "peak_bankroll": f"${self.peak_bankroll:.2f}"
        }
    
    def save_config(self, filepath: str):
        """Save current configuration to file"""
        with open(filepath, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def export_performance_data(self, filepath: str):
        """Export performance data to CSV"""
        if self.bet_history:
            df = pd.DataFrame(self.bet_history)
            df.to_csv(filepath, index=False)