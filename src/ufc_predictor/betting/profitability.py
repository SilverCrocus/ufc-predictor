"""
UFC Prediction Profitability Module

This module extends the existing UFC prediction system to optimize for profitability
by calculating expected value, implementing bankroll management, and tracking ROI.

Core functionality:
- Expected Value (EV) calculations
- Kelly Criterion bet sizing
- Profit tracking and performance metrics
- Betting opportunity identification
"""

import pandas as pd
import numpy as np
import json
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime, timedelta
import requests
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns


@dataclass
class BettingOdds:
    """Container for betting odds data."""
    fighter_a: str
    fighter_b: str
    fighter_a_odds: float  # American odds (e.g., -150, +200)
    fighter_b_odds: float
    sportsbook: str
    timestamp: datetime


@dataclass
class BettingOpportunity:
    """Container for a betting opportunity with EV calculation."""
    matchup: str
    fighter: str
    model_prob: float
    market_prob: float
    odds: float
    expected_value: float
    kelly_fraction: float
    recommended_bet: float
    confidence_score: float


class ProfitabilityOptimizer:
    """
    Optimize UFC predictions for profitability by calculating expected value
    and implementing proper bankroll management strategies.
    """
    
    def __init__(self, bankroll: float = 1000.0, max_kelly_fraction: float = 0.05):
        """
        Initialize the profitability optimizer.
        
        Args:
            bankroll: Starting bankroll amount
            max_kelly_fraction: Maximum Kelly fraction to bet (risk management)
        """
        self.bankroll = bankroll
        self.initial_bankroll = bankroll
        self.max_kelly_fraction = max_kelly_fraction
        self.bet_history = []
        self.profit_history = []
        
    def american_odds_to_probability(self, odds: float) -> float:
        """Convert American betting odds to implied probability."""
        if odds > 0:
            return 100 / (odds + 100)
        else:
            return abs(odds) / (abs(odds) + 100)
    
    def american_odds_to_decimal(self, odds: float) -> float:
        """Convert American odds to decimal odds."""
        if odds > 0:
            return (odds / 100) + 1
        else:
            return (100 / abs(odds)) + 1
    
    def calculate_expected_value(self, model_prob: float, odds: float) -> float:
        """
        Calculate expected value of a bet.
        
        Args:
            model_prob: Your model's probability of the outcome
            odds: American betting odds
            
        Returns:
            Expected value as a percentage
        """
        decimal_odds = self.american_odds_to_decimal(odds)
        ev = (model_prob * (decimal_odds - 1)) - (1 - model_prob)
        return ev
    
    def calculate_kelly_fraction(self, model_prob: float, odds: float) -> float:
        """
        Calculate optimal Kelly Criterion betting fraction.
        
        Args:
            model_prob: Your model's probability of the outcome
            odds: American betting odds
            
        Returns:
            Optimal betting fraction (capped at max_kelly_fraction)
        """
        decimal_odds = self.american_odds_to_decimal(odds)
        kelly_fraction = (model_prob * decimal_odds - 1) / (decimal_odds - 1)
        
        # Cap at maximum fraction for risk management
        return min(max(kelly_fraction, 0), self.max_kelly_fraction)
    
    def analyze_betting_opportunity(self, prediction_result: Dict, odds_data: BettingOdds) -> List[BettingOpportunity]:
        """
        Analyze a fight prediction for betting opportunities.
        
        Args:
            prediction_result: Output from your UFC prediction model
            odds_data: Betting odds for the fight
            
        Returns:
            List of betting opportunities for each fighter
        """
        opportunities = []
        
        # Extract fighter names and probabilities from prediction
        if 'fighter_a' in prediction_result:
            # Symmetrical prediction format
            fighter_a = prediction_result['fighter_a']
            fighter_b = prediction_result['fighter_b']
            prob_a = float(prediction_result[f'{fighter_a}_win_probability'].replace('%', '')) / 100
            prob_b = float(prediction_result[f'{fighter_b}_win_probability'].replace('%', '')) / 100
        else:
            # Basic prediction format
            fighter_a = prediction_result['blue_corner']
            fighter_b = prediction_result['red_corner'] 
            prob_a = float(prediction_result['blue_win_probability'].replace('%', '')) / 100
            prob_b = float(prediction_result['red_win_probability'].replace('%', '')) / 100
        
        matchup = f"{fighter_a} vs {fighter_b}"
        
        # Analyze Fighter A
        market_prob_a = self.american_odds_to_probability(odds_data.fighter_a_odds)
        ev_a = self.calculate_expected_value(prob_a, odds_data.fighter_a_odds)
        kelly_a = self.calculate_kelly_fraction(prob_a, odds_data.fighter_a_odds)
        
        if ev_a > 0:  # Only consider positive EV bets
            opportunities.append(BettingOpportunity(
                matchup=matchup,
                fighter=fighter_a,
                model_prob=prob_a,
                market_prob=market_prob_a,
                odds=odds_data.fighter_a_odds,
                expected_value=ev_a,
                kelly_fraction=kelly_a,
                recommended_bet=kelly_a * self.bankroll,
                confidence_score=abs(prob_a - market_prob_a)  # Higher difference = higher confidence
            ))
        
        # Analyze Fighter B
        market_prob_b = self.american_odds_to_probability(odds_data.fighter_b_odds)
        ev_b = self.calculate_expected_value(prob_b, odds_data.fighter_b_odds)
        kelly_b = self.calculate_kelly_fraction(prob_b, odds_data.fighter_b_odds)
        
        if ev_b > 0:  # Only consider positive EV bets
            opportunities.append(BettingOpportunity(
                matchup=matchup,
                fighter=fighter_b,
                model_prob=prob_b,
                market_prob=market_prob_b,
                odds=odds_data.fighter_b_odds,
                expected_value=ev_b,
                kelly_fraction=kelly_b,
                recommended_bet=kelly_b * self.bankroll,
                confidence_score=abs(prob_b - market_prob_b)
            ))
        
        return opportunities
    
    def analyze_fight_card(self, predictions: List[Dict], odds_list: List[BettingOdds]) -> List[BettingOpportunity]:
        """
        Analyze an entire fight card for betting opportunities.
        
        Args:
            predictions: List of fight predictions from your model
            odds_list: List of betting odds for each fight
            
        Returns:
            List of all profitable betting opportunities, sorted by EV
        """
        all_opportunities = []
        
        for prediction, odds in zip(predictions, odds_list):
            opportunities = self.analyze_betting_opportunity(prediction, odds)
            all_opportunities.extend(opportunities)
        
        # Sort by expected value (highest first)
        all_opportunities.sort(key=lambda x: x.expected_value, reverse=True)
        
        return all_opportunities
    
    def execute_bet(self, opportunity: BettingOpportunity, actual_result: bool) -> Dict:
        """
        Record the result of a placed bet.
        
        Args:
            opportunity: The betting opportunity that was executed
            actual_result: True if the bet won, False if it lost
            
        Returns:
            Dictionary with bet result details
        """
        bet_amount = opportunity.recommended_bet
        
        if actual_result:
            # Calculate winnings
            decimal_odds = self.american_odds_to_decimal(opportunity.odds)
            payout = bet_amount * decimal_odds
            profit = payout - bet_amount
        else:
            profit = -bet_amount
        
        # Update bankroll
        self.bankroll += profit
        
        # Record bet
        bet_record = {
            'timestamp': datetime.now(),
            'matchup': opportunity.matchup,
            'fighter': opportunity.fighter,
            'bet_amount': bet_amount,
            'odds': opportunity.odds,
            'model_prob': opportunity.model_prob,
            'market_prob': opportunity.market_prob,
            'expected_value': opportunity.expected_value,
            'actual_result': actual_result,
            'profit': profit,
            'bankroll_after': self.bankroll,
            'roi_current': ((self.bankroll - self.initial_bankroll) / self.initial_bankroll) * 100
        }
        
        self.bet_history.append(bet_record)
        self.profit_history.append(profit)
        
        return bet_record
    
    def get_performance_metrics(self) -> Dict:
        """Calculate comprehensive performance metrics."""
        if not self.bet_history:
            return {"error": "No bets recorded yet"}
        
        df = pd.DataFrame(self.bet_history)
        
        total_bets = len(df)
        winning_bets = len(df[df['actual_result'] == True])
        losing_bets = len(df[df['actual_result'] == False])
        win_rate = winning_bets / total_bets
        
        total_profit = sum(self.profit_history)
        total_wagered = df['bet_amount'].sum()
        roi = ((self.bankroll - self.initial_bankroll) / self.initial_bankroll) * 100
        
        avg_winning_bet = df[df['actual_result'] == True]['profit'].mean() if winning_bets > 0 else 0
        avg_losing_bet = df[df['actual_result'] == False]['profit'].mean() if losing_bets > 0 else 0
        
        # Calculate Sharpe-like ratio for betting
        profit_std = np.std(self.profit_history) if len(self.profit_history) > 1 else 0
        risk_adjusted_return = (np.mean(self.profit_history) / profit_std) if profit_std > 0 else 0
        
        metrics = {
            'total_bets': total_bets,
            'winning_bets': winning_bets,
            'losing_bets': losing_bets,
            'win_rate': f"{win_rate*100:.2f}%",
            'total_profit': f"${total_profit:.2f}",
            'total_wagered': f"${total_wagered:.2f}",
            'roi': f"{roi:.2f}%",
            'current_bankroll': f"${self.bankroll:.2f}",
            'initial_bankroll': f"${self.initial_bankroll:.2f}",
            'avg_winning_bet': f"${avg_winning_bet:.2f}",
            'avg_losing_bet': f"${avg_losing_bet:.2f}",
            'risk_adjusted_return': f"{risk_adjusted_return:.3f}",
            'max_drawdown': self._calculate_max_drawdown()
        }
        
        return metrics
    
    def _calculate_max_drawdown(self) -> str:
        """Calculate maximum drawdown from peak bankroll."""
        if not self.bet_history:
            return "0.00%"
        
        df = pd.DataFrame(self.bet_history)
        bankroll_history = df['bankroll_after'].tolist()
        bankroll_history.insert(0, self.initial_bankroll)
        
        peak = bankroll_history[0]
        max_drawdown = 0
        
        for value in bankroll_history:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            max_drawdown = max(max_drawdown, drawdown)
        
        return f"{max_drawdown*100:.2f}%"
    
    def plot_performance(self):
        """Create performance visualization charts."""
        if not self.bet_history:
            print("No betting data to plot")
            return
        
        df = pd.DataFrame(self.bet_history)
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Bankroll over time
        bankroll_history = [self.initial_bankroll] + df['bankroll_after'].tolist()
        ax1.plot(range(len(bankroll_history)), bankroll_history, 'b-', linewidth=2)
        ax1.axhline(y=self.initial_bankroll, color='r', linestyle='--', alpha=0.7, label='Starting Bankroll')
        ax1.set_title('Bankroll Over Time', fontweight='bold')
        ax1.set_xlabel('Bet Number')
        ax1.set_ylabel('Bankroll ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Cumulative profit
        cumulative_profit = np.cumsum(self.profit_history)
        ax2.plot(range(1, len(cumulative_profit) + 1), cumulative_profit, 'g-', linewidth=2)
        ax2.axhline(y=0, color='r', linestyle='--', alpha=0.7)
        ax2.set_title('Cumulative Profit', fontweight='bold')
        ax2.set_xlabel('Bet Number')
        ax2.set_ylabel('Profit ($)')
        ax2.grid(True, alpha=0.3)
        
        # 3. Win/Loss distribution
        win_loss = ['Win' if x else 'Loss' for x in df['actual_result']]
        win_loss_counts = pd.Series(win_loss).value_counts()
        colors = ['green', 'red']
        ax3.pie(win_loss_counts.values, labels=win_loss_counts.index, autopct='%1.1f%%', 
                colors=colors, startangle=90)
        ax3.set_title('Win/Loss Distribution', fontweight='bold')
        
        # 4. Expected Value vs Actual Results
        ax4.scatter(df['expected_value'], df['profit'], alpha=0.6, 
                   c=['green' if x else 'red' for x in df['actual_result']])
        ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax4.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        ax4.set_title('Expected Value vs Actual Profit', fontweight='bold')
        ax4.set_xlabel('Expected Value')
        ax4.set_ylabel('Actual Profit ($)')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def export_betting_data(self, filepath: str):
        """Export betting history to CSV for analysis."""
        if not self.bet_history:
            print("No betting data to export")
            return
        
        df = pd.DataFrame(self.bet_history)
        df.to_csv(filepath, index=False)
        print(f"Betting data exported to {filepath}")


def create_sample_odds() -> List[BettingOdds]:
    """Create sample betting odds for testing purposes."""
    sample_odds = [
        BettingOdds("Jon Jones", "Stipe Miocic", -300, +250, "DraftKings", datetime.now()),
        BettingOdds("Islam Makhachev", "Charles Oliveira", -180, +155, "FanDuel", datetime.now()),
        BettingOdds("Sean O'Malley", "Marlon Vera", -120, +100, "BetMGM", datetime.now()),
    ]
    return sample_odds


def demonstrate_profitability_analysis():
    """Demonstrate the profitability analysis workflow."""
    print("🎯 UFC Profitability Optimization Demo")
    print("=" * 50)
    
    # Initialize optimizer
    optimizer = ProfitabilityOptimizer(bankroll=1000.0, max_kelly_fraction=0.05)
    
    # Sample prediction (would come from your actual model)
    sample_prediction = {
        'fighter_a': 'Jon Jones',
        'fighter_b': 'Stipe Miocic',
        'Jon Jones_win_probability': '75.23%',
        'Stipe Miocic_win_probability': '24.77%',
        'predicted_winner': 'Jon Jones'
    }
    
    # Sample odds
    sample_odds = BettingOdds("Jon Jones", "Stipe Miocic", -300, +250, "DraftKings", datetime.now())
    
    # Analyze opportunities
    opportunities = optimizer.analyze_betting_opportunity(sample_prediction, sample_odds)
    
    print(f"\n📊 BETTING ANALYSIS RESULTS:")
    print(f"Matchup: {sample_odds.fighter_a} vs {sample_odds.fighter_b}")
    print(f"Model probabilities: Jones {sample_prediction['Jon Jones_win_probability']}, Miocic {sample_prediction['Stipe Miocic_win_probability']}")
    print(f"Market odds: Jones {sample_odds.fighter_a_odds}, Miocic {sample_odds.fighter_b_odds}")
    
    if opportunities:
        for opp in opportunities:
            print(f"\n💰 PROFITABLE OPPORTUNITY FOUND:")
            print(f"   Fighter: {opp.fighter}")
            print(f"   Expected Value: {opp.expected_value*100:.2f}%")
            print(f"   Recommended Bet: ${opp.recommended_bet:.2f}")
            print(f"   Kelly Fraction: {opp.kelly_fraction*100:.2f}%")
            print(f"   Model Prob: {opp.model_prob*100:.1f}% vs Market Prob: {opp.market_prob*100:.1f}%")
    else:
        print("\n❌ No profitable betting opportunities found for this fight.")
    
    return optimizer, opportunities


if __name__ == "__main__":
    demonstrate_profitability_analysis() 